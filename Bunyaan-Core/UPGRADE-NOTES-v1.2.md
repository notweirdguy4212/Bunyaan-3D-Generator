# Bunyaan v1.2 engineering upgrade

This release adds five functional engineering paths:

1. Trainable text-conditioned UV PBR texture denoiser (`bunyaan_model.texture_ai`) plus `training/train_texture.py`.
2. Sparse tiled neural inference (`bunyaan_model.sparse_inference`, `POST /inference/sparse`).
3. Calibrated multi-view visual-hull reconstruction (`bunyaan_model.multiview`, `POST /reconstruct/multiview`).
4. Blender-native FBX/USD export bridge (`bunyaan_model.surface.export_interchange`, `POST /mesh/export-bridge`).
5. Adaptive edge subdivision and Taubin surface refinement (`bunyaan_model.surface.refine_surface`, `POST /mesh/refine`).

## Important runtime setup

- Train a texture checkpoint from licensed UV-aligned PBR examples, then set `BUNYAAN_TEXTURE_CHECKPOINT` to its path.
- Multi-view NPZ input must contain `masks` with shape `[V,H,W]` and calibrated projection matrices `cameras` with shape `[V,3,4]`.
- Install Blender and optionally set `BLENDER_EXECUTABLE` to enable FBX/USD exports.
- Sparse tiled inference uses local tiles with the existing dense U-Net; it is not a sparse-convolution kernel.
- No pretrained texture checkpoint, new dataset, or Blender binary is included.

## Verification

`pytest tests` from `Bunyaan-Core` passes the current CPU test suite. Tests cover neural texture forward shape, sparse tile execution, multi-view carving, and adaptive refinement in addition to existing core tests.
