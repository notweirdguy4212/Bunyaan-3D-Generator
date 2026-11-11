# Bunyaan native architecture — v1.4

## Release intent

Bunyaan v1.4 strengthens the native 3D and UV-material denoisers. It remains an independent model implementation: no external 3D generator or pretrained foundation model is imported, wrapped, launched, or required. PyTorch and standard geometry/image libraries are software dependencies, not model backends.

This is an architecture and training-pipeline upgrade, not a claim that Bunyaan has surpassed established open-source generators. No pretrained weights or large-scale benchmark results are bundled.

## Shape network

`Bunyaan3D` retains a dense occupancy diffusion target for compatibility with the current voxel corpus, and upgrades its U-Net with:

- three resolution stages and residual skip paths;
- timestep and pooled text conditioning through FiLM;
- native image conditioning when paired images are available;
- token-level text cross-attention and factorized linear global self-attention at the low-resolution 3D bottleneck;
- classifier-free text dropout during training;
- EMA weights for validation and inference;
- deterministic DDIM sampling with configurable steps and guidance.

Cross-attention is applied at the low-resolution bottleneck, not at every voxel. This provides a route for spatial features to use token-level prompt information without quadratic attention at the full voxel resolution.

### Current representation boundary

The output remains a dense occupancy field. At the default 32³ resolution it cannot represent the thin structures, fine topology, sharp edges or high-frequency detail expected from leading high-resolution 3D generators. Increasing resolution raises 3D activation memory rapidly. The next major research step is a trained hierarchical latent/implicit surface representation with sampled SDF supervision and adaptive high-resolution decoding; this is not claimed as implemented in v1.4.

## Material network

`BunyaanTextureNet` predicts five UV-aligned channels in the fixed order:

1. linear base-colour R
2. linear base-colour G
3. linear base-colour B
4. roughness
5. metallic

The U-Net now has a second bottleneck residual block and token-level text cross-attention. For large UV maps, attention operates on a bounded pooled feature grid and contributes a resized global-context residual. Full-resolution convolutional skip paths remain intact.

Training maintains EMA weights. Validation is run against EMA weights, and inference checkpoint loading prefers EMA weights when present. Material inference uses deterministic seeded initialization and classifier-free guidance.

### Map truthfulness

Only the five channels above are learned by this model. A generated normal map is currently a derived proxy, and ambient occlusion is a neutral placeholder. Neither is a learned prediction. Do not present these helper outputs as physically accurate normal/AO predictions.

## Training improvements

### 3D

`training/train.py` now:

- creates deterministic, provenance-model-level train/validation splits;
- rejects a corpus with fewer than two distinct model IDs;
- supports CUDA mixed precision when enabled and available;
- trains with classifier-free text dropout;
- evaluates each epoch using EMA weights;
- records train/validation loss and split identity;
- saves latest and best-validation checkpoints with optimizer, scheduler, scaler, EMA, RNG and data-split state;
- resumes old Bunyaan v1 checkpoints with newly introduced attention blocks initialized natively.

### Materials

`training/train_texture.py` now:

- validates five-channel samples and licensing metadata;
- splits by source model identity;
- supports CUDA AMP;
- uses classifier-free text dropout to train the unconditional guidance branch;
- maintains EMA weights and evaluates the EMA network;
- saves latest and best-validation checkpoints;
- retains optimizer, scaler, random state, split IDs and loader-generator state for resume.

## Inference

The shape service defaults to deterministic DDIM sampling (50 steps) and supports the original DDPM sampler. The DDIM step count is capped at the checkpoint's trained diffusion schedule length. A lower step count is a speed/quality trade-off and must be benchmarked for the selected checkpoint.

## Checkpoint compatibility

- Shape: `bunyaan-native-3d`, version `1.4.0-dev`.
- Material: `bunyaan-texture-diffusion-v2` with optional `ema` state.
- v1 shape checkpoints can be loaded with non-strict state loading; new attention weights are initialized and therefore still require fine-tuning before quality claims.
- Older texture checkpoints can load non-strictly; newly added attention/residual parameters are initialized and require fine-tuning.

## Required evaluation before a quality claim

A meaningful comparison requires the same held-out asset split, prompt set, output resolution, hardware class and generation budget. Report:

- shape: Chamfer-L1/L2, F-score at fixed thresholds, normal consistency, watertight rate, component count, diversity, prompt alignment and human preference;
- materials: per-channel MAE/MSE, perceptual image metrics for base colour, roughness/metallic calibration, seam continuity, UV-mask validity and expert material review;
- operations: peak VRAM, seconds/sample, parameter count, training FLOPs or step time, checkpoint size and failure rate;
- data: exact dataset versions, asset-level licenses, deduplication and train/validation/test split hashes.

Never compare a Bunyaan research checkpoint against another project's pretrained model and call the result an architecture comparison. Separate model capability from training-data scale, pretraining and inference budget.
