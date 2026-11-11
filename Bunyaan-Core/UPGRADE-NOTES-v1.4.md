# Bunyaan v1.4 — Native Intelligence Upgrade

## Architecture
- Added Bunyaan-native token-to-spatial cross-attention at the low-resolution 3D and 2D U-Net bottlenecks.
- Added a second material bottleneck residual block and factorized linear global self-attention in both shape and material bottlenecks.
- Added bounded pooled attention for large texture feature maps to limit attention memory growth.
- Retained the native text encoder, image encoder, diffusion schedules, and Bunyaan checkpoint identities.

## Training
- Added deterministic asset/model-level validation splits to the 3D trainer.
- Added classifier-free text dropout for material guidance and per-epoch EMA validation for both 3D and material training.
- Added best-validation checkpoints and expanded 3D checkpoint state (optimizer, scheduler, scaler, EMA, split identities and RNG state).
- Kept mixed precision optional and CUDA-aware.
- Added backward-compatible non-strict loading for earlier Bunyaan checkpoints. Newly added attention weights in old checkpoints are initialized and should be fine-tuned.

## Inference
- Added deterministic DDIM sampling with configurable inference steps and classifier-free guidance for Bunyaan 3D.
- Bunyaan's generation service now defaults to 50-step DDIM; original DDPM sampling remains selectable.
- Material synthesis accepts a reproducible seed and guidance scale, and prefers EMA weights when loading compatible checkpoints.
- Material output manifests distinguish learned channels from derived normal-map proxies and neutral AO placeholders.

## Verification
- Added tests for 2D/3D spatial conditioning attention, Bunyaan 3D cross-attention, and deterministic DDIM output.
- CPU smoke tests run one epoch of both 3D and material training on tiny synthetic datasets.
- These tests verify software paths only; they do not establish model quality or competitive performance.

## Known limitations
- Shape output is still dense occupancy, with the current default at 32³. A hierarchical latent/SDF/implicit representation and high-resolution decoder remain future work.
- There are no pretrained weights in this release.
- A large licensed training corpus and controlled benchmark suite are still required before comparisons with leading open-source 3D generators.
