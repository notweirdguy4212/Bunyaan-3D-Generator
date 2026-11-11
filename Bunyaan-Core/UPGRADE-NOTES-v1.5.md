# Bunyaan v1.5 — Hierarchical Field Upgrade

## Implemented
- Added a native coarse 3D feature encoder and arbitrary-coordinate implicit field decoder.
- Added continuous SDF geometry, normalized surface normals, linear base colour, roughness, metallic and opacity predictions.
- Added chunked query evaluation to bound activation memory for dense high-resolution sampling.
- Added camera-aware multi-view patch fusion using caller-supplied calibrated 3x4 camera matrices.
- Added SDF, normal and PBR supervision with optional Eikonal regularization.
- Added `training/train_implicit.py`, asset-level train/validation splits, EMA checkpoints and JSONL loss records.
- Added CPU tests for decoder contracts, chunking, multi-view masks, differentiable Eikonal loss and hierarchical forward pass.

## Verification
- 20 automated tests passed on the development environment.
- One-epoch CPU smoke training completed on synthetic field records and produced an EMA checkpoint.
- These checks verify code paths only; they do not establish perceptual quality, generalization, GPU throughput or superiority over other systems.

## Still required
- A large legally cleared dataset with aligned SDF, normal and PBR supervision.
- A full training run and independent evaluation.
- A validated sparse latent representation and adaptive high-resolution spatial allocation.
- End-to-end inference/export integration and matched TRELLIS.2 comparisons.
- GPU profiling, memory measurements, topology and PBR quality evaluation.

Bunyaan remains independently implemented. TRELLIS.2, Hunyuan3D and TRELLIS source/runtime dependencies were not added.
