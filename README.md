# Unified Moshi + AvatarForcing Pipeline

A latency-optimised talking-head video generation system that connects:

| Component | Role |
|---|---|
| **Moshi** (Kyutai) | Speech-to-speech generation (Helium LM + Mimi codec) |
| **moshi-wav2vec-bridge** | Converts Moshi's discrete tokens → wav2vec-like embeddings |
| **AvatarForcing** | Diffusion-based talking-head video generation |

---

## Architecture

```
User Audio (24 kHz)
       │
       ▼
  [Moshi Mimi Encoder]
       │  codes: (1, 8, 1) per step at 12.5 Hz
       ▼
  [Moshi Helium LM]  ──────────────────────────────────┐
       │  17 tokens: 1 text + 8 user + 8 response       │
       │                                                 │
       │  8 response tokens at 12.5 Hz                  │ Mimi decoder
       ▼                                                 ▼
  [moshi-wav2vec-bridge]               [Response PCM at 24 kHz]
  (accumulates 40 steps = 3.2s)                │
  (upsamples ×2 → 25 Hz)                       │
       │                                         │
       │  audio_emb (1, 81, 10752)               │
       │  = 14 × 768 concat                      │
       ▼                                         │
  [AvatarForcing Diffusion]                      │
  + reference image + text prompt                │
       │                                         │
       │  silent video (21 frames)               │
       └─────────────── ffmpeg merge ────────────┘
                              │
                        output.mp4
```

### Latency Improvement

In the original pipeline:
```
tokens → Mimi decode → raw audio → wav2vec → embeddings → AvatarForcing
```

In the unified pipeline:
```
tokens ──────────────────────────→ bridge → embeddings → AvatarForcing
       └──→ Mimi decode → raw audio (for final merge only)
```

The bridge operates on **discrete tokens before waveform decoding**, cutting the audio processing latency from:
> `Mimi decode time + wav2vec encode time`
to:
> `bridge forward time` (a single 12-layer transformer pass)

---

## Folder Structure

```
new_bridge_try/
├── moshi-inference/           # Moshi speech-to-speech system (Kyutai)
│   └── moshi/
│       ├── models/
│       │   ├── compression.py   # MimiModel
│       │   ├── lm.py            # LMModel + LMGen
│       │   └── loaders.py       # CheckpointInfo, get_mimi, get_moshi_lm
│       └── run_inference.py     # InferenceState (used by MoshiRunner)
│
├── moshi-wav2vec-bridge/      # Bridge: Mimi tokens → wav2vec-like embeddings
│   ├── model.py               # MimiWav2Vec2Bridge (12-layer transformer)
│   ├── config.yaml            # Bridge hyperparameters
│   ├── trainer.py             # DDP training code
│   └── inference.py           # Bridge-only inference + --avatarforcing flag
│
├── AvatarForcing-inference/   # Talking-head video diffusion (UNCHANGED)
│   ├── inference.py           # Original entry point (unchanged)
│   ├── pipeline/
│   │   └── avatar_forcing_inference.py  # AvatarForcingInferencePipeline
│   ├── utils/
│   │   ├── dataset.py         # TextImageAudioPairDataset (wav2vec usage)
│   │   └── inject.py          # slice_conditional_dict, _apply_lora
│   └── configs/
│       └── avatarforcing.yaml # AvatarForcing configuration
│
├── unified_pipeline/          # NEW: orchestration layer
│   ├── __init__.py
│   ├── moshi_runner.py        # Wraps Moshi, yields MoshiStep per frame
│   ├── bridge_runner.py       # Accumulates tokens, runs bridge, returns audio_emb
│   ├── avatarforcing_runner.py# Wraps AvatarForcing, accepts audio_emb directly
│   ├── pipeline.py            # Orchestrates all three components
│   └── run_unified.py         # CLI entry point
│
└── moshi_avatar_pipeline.ipynb  # RunPod notebook (all-in-one)
```

---

## Key Design Facts

### audio_emb Shape Contract

| Source | Shape | Notes |
|---|---|---|
| `wav2vec2-base` (original) | `(B, T+1, 10752)` | 14×768 concat + zero prefix |
| `bridge` output | `(B, T+1, 10752)` | **identical** |

Where:
- `T = teacher_len = 80` frames at 25 Hz
- `14 = 1 (last_hidden_state) + 13 (hidden_states[0..12])`
- `10752 = 14 × 768`
- `+1` = prepended zero-padding frame

### Frame Rate Chain

```
Mimi output:    12.5 Hz  (1 token-set per 80ms)
Bridge upsample:   ×2
Bridge output:  25.0 Hz  (matches AvatarForcing fps)
AvatarForcing:  25.0 Hz  ✓
```

### Moshi Tokens → Bridge Accumulation

```
teacher_len = 80 frames at 25 Hz
mimi_frames_needed = 80 / 2 = 40 Moshi steps
duration = 40 / 12.5 = 3.2 seconds of response audio
```

The bridge accumulates 40 token-sets (one per Moshi step), then runs a
single forward pass to produce one `audio_emb` block.

---

## Running the Pipeline

### CLI

```bash
cd /path/to/new_bridge_try

python -m unified_pipeline.run_unified \
    --user-audio  my_speech.wav \
    --image       face.jpg \
    --output      output.mp4 \
    --bridge-ckpt checkpoints/bridge_best.pt \
    --af-ckpt     checkpoints/ode_audio_init.pt \
    --af-config   AvatarForcing-inference/configs/avatarforcing.yaml \
    --teacher-len 80 \
    --num-output-frames 21
```

Full argument reference:
```
--user-audio        Path to user input audio (.wav/.mp3)        [required]
--image             Path to reference face image (.jpg/.png)    [required]
--output            Output .mp4 path                            [output.mp4]
--bridge-ckpt       Bridge checkpoint path                      [required]
--bridge-config     Bridge config.yaml (auto-detected if unset) [optional]
--af-ckpt           AvatarForcing checkpoint path               [required]
--af-config         AvatarForcing config yaml                   [avatarforcing.yaml]
--moshi-repo        HuggingFace repo for Moshi                  [kyutai/moshiko-pytorch-q8]
--moshi-weight      Local override for Moshi LM weights         [optional]
--mimi-weight       Local override for Mimi codec weights       [optional]
--teacher-len       Must be even (default 80)                   [80]
--num-output-frames Video frames to generate                    [21]
--fps               Output video FPS                            [25]
--prompt            Text conditioning prompt                    [default prompt]
--device            cuda or cpu                                 [cuda]
--use-ema           Load EMA weights from AvatarForcing ckpt    [false]
--half              Use float16 instead of bfloat16             [false]
```

### Python API

```python
import torch
from omegaconf import OmegaConf
from unified_pipeline.pipeline import UnifiedMoshiAvatarPipeline

# Load AvatarForcing config (same as inference.py)
af_config = OmegaConf.load("AvatarForcing-inference/configs/avatarforcing.yaml")
default_cfg = OmegaConf.load("AvatarForcing-inference/configs/default_config.yaml")
af_config = OmegaConf.merge(default_cfg, af_config)

# Build + load all models
pipeline = UnifiedMoshiAvatarPipeline(
    bridge_checkpoint = "checkpoints/bridge_best.pt",
    generator_ckpt    = "checkpoints/ode_audio_init.pt",
    af_config         = af_config,
    teacher_len       = 80,
    num_output_frames = 21,
    device            = "cuda",
    dtype             = torch.bfloat16,
)

# Run inference
with torch.no_grad():
    out = pipeline.run(
        user_audio_path = "my_speech.wav",
        image_path      = "face.jpg",
        output_path     = "output.mp4",
    )
```

### RunPod Notebook

Open `moshi_avatar_pipeline.ipynb` and run cells top-to-bottom:

| Cell | Action |
|---|---|
| 1 | Install ffmpeg + system libs |
| 2 | Install PyTorch + Python deps |
| 3 | Download model weights from HuggingFace |
| 4 | Configure paths |
| 5 | Load all models |
| 6 | Run single inference |
| 7 | Preview output video |
| 8 | Batch processing |
| 9 | Shape sanity check |

---

## AvatarForcing Code Changes

**Zero changes** were made to AvatarForcing. The unified pipeline feeds
`audio_emb` directly into `inference_avatar_forcing()` as a pre-computed
tensor, exactly matching `batch['audio_emb']` from the original DataLoader:

```python
# Original AvatarForcing inference.py (UNCHANGED):
video = pipeline.inference_avatar_forcing(
    noise            = sampled_noise,
    text_prompts     = prompts,
    audio_embeddings = batch['audio_emb'],  # ← (1, T+1, 10752) from wav2vec
    y                = y,
    return_latents   = False,
    initial_latent   = initial_latent,
)

# Unified pipeline (avatarforcing_runner.py):
video = pipeline.inference_avatar_forcing(
    noise            = sampled_noise,
    text_prompts     = [prompt],
    audio_embeddings = audio_emb_batch,     # ← (1, T+1, 10752) from bridge ✓
    y                = y,
    return_latents   = False,
    initial_latent   = initial_latent,
)
```

---

## Requirements

```
# Core
torch >= 2.1
torchaudio
torchvision
transformers
diffusers
safetensors
omegaconf
peft

# Moshi
sphn
sentencepiece

# Video/Audio
opencv-python-headless
ffmpeg (system)
imageio
av

# Misc
einops
Pillow
numpy
scipy
pyyaml
```

---

## Troubleshooting

### "No bridge output — audio too short"
The bridge needs `teacher_len / 2` Moshi steps before producing output.
With `teacher_len=80`, you need ≥ 3.2 seconds of input audio.

### "audio_emb shape mismatch"
Run Cell 9 in the notebook to verify the bridge output shape is exactly
`(teacher_len+1, 10752) = (81, 10752)`.

### DDP unused-parameter error during bridge training
The bridge has no `output_proj` layer — `d_model == output_dim == 768`.
This was intentionally removed so all parameters participate in the loss.

### ffmpeg not found
```bash
apt-get install -y ffmpeg   # Linux
brew install ffmpeg          # macOS
```
