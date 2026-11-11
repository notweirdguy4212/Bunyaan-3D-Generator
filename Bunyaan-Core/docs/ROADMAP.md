# Bunyaan native model roadmap

## Implemented in v1.0.0-dev
- Native byte-level text encoder and native RGB image encoder.
- Conditional 3D U-Net diffusion denoiser and DDPM sampler.
- Classifier-free text dropout during training and optional classifier-free guidance at inference.
- Volumetric mesh extraction and OBJ/PLY/GLB/GLTF/STL export.
- Licensed mesh voxelization utility, training loop, checkpoint save/load.
- Local SQLite asset library, FastAPI endpoints and browser workspace.

## Required before a public production launch
1. Curate a large, diverse, documented dataset with explicit commercial training rights and provenance; implement deduplication, safety filtering and train/validation/test splits.
2. Build paired image/shape data at scale; train and evaluate image conditioning rather than assuming the encoder alone makes it useful.
3. Benchmark scaling beyond 32³. Add hierarchical sparse voxels or a validated latent representation for high-resolution detail and memory efficiency.
4. Improve geometry quality: surface losses, topology checks, watertightness, adaptive remeshing and mesh repair.
5. Implement native texture/material generation, UV unwrap/baking, PBR maps, multi-view consistency and texture export.
6. Add deterministic validation suites: Chamfer distance, IoU, F-score, normal consistency, watertight rate, diversity, prompt alignment and image reconstruction metrics.
7. Add distributed training, experiment tracking, automatic checkpoint recovery, model registry/versioning and reproducible configs.
8. Harden API: authentication, job queue, cancellation, quotas, file-size limits, safe path handling, audit logs and isolated worker processes before network exposure.
9. Build native rigging/animation and editable scene formats as separate validated model capabilities.
10. Run independent security, licensing, privacy and quality review before commercial claims.

## Model status
The code is executable and the minimal training/inference paths have been smoke-tested on CPU. No pretrained Bunyaan weights or production dataset are supplied. The included prototype is not yet a production-grade foundation model.


## v1.2 engineering additions (implemented code paths; training/deployment gated)
- **Learned UV texture synthesis:** `BunyaanTextureNet` predicts five UV-space PBR channels from text and noisy maps. Train with `training/train_texture.py` on licensed `.pt` samples containing `texture` `[5,H,W]` and `caption`. Inference is enabled only with `BUNYAAN_TEXTURE_CHECKPOINT`; procedural synthesis remains separately labelled.
- **Sparse neural inference:** `/inference/sparse` groups active `[batch,z,y,x]` coordinates into local tiles and evaluates only occupied tiles. This is sparse tiled execution over the existing dense U-Net, not a sparse-convolution architecture.
- **Multi-view reconstruction:** `/reconstruct/multiview` accepts NPZ arrays `masks[V,H,W]` and calibrated projection matrices `cameras[V,3,4]`; performs visual-hull silhouette carving. It is a calibrated geometric baseline, not learned photometric reconstruction.
- **FBX/USD bridges:** `/mesh/export-bridge` and `export_mesh` use Blender's installed background executable for actual FBX/USD serialization. Configure `BLENDER_EXECUTABLE` if Blender is not on PATH. Export support depends on the Blender build.
- **Adaptive surface refinement:** `/mesh/refine` and `refine_surface` split overlong edges with a face budget and apply Taubin smoothing; validate topology and feature preservation on production assets.

### Training and runtime requirements
- Texture training requires paired, UV-aligned PBR examples; no texture checkpoint or pretrained weights are bundled.
- Multi-view quality depends on correct camera calibration, consistent masks, and sufficient view coverage.
- Sparse tile inference is intended for localized active regions; overlapping tile halos and sparse-convolution kernels are future performance work.
- Blender is an external optional executable. It is not installed by pip requirements.
- Run `pytest Bunyaan-Core/tests` from the Bunyaan-Core directory. Production release still requires dataset governance, GPU benchmarks, regression fixtures, security hardening, and independent quality validation.


## v1.4 native architecture work
- Added token-to-spatial cross-attention in native shape and material denoisers.
- Added EMA validation/inference, deterministic DDIM shape sampling, asset-level 3D validation splits, and best checkpoints.
- Added fair-comparison and benchmark reporting protocol.

## Next major capability milestone (not yet implemented)
1. Replace the fixed low-resolution occupancy bottleneck with a hierarchical latent representation and sampled implicit SDF decoder.
2. Add mesh-surface and multi-scale geometric losses (SDF, normals, edge and topology regularizers) with stable mesh-to-field supervision.
3. Add native multi-view feature fusion with calibrated camera conditioning and view-consistency objectives.
4. Expand material prediction to separately trained normal, height and AO channels with physically correct tangent-space conventions; keep the current five-channel baseline.
5. Add tiled/high-resolution material inference, UV seam-aware losses and geometry-aware texture conditioning.
6. Run large-scale, independently reproducible comparisons against open-source baselines on matched data, checkpoints, hardware and compute budgets.


## v1.5 hierarchical implicit milestone
- Implemented: coarse-volume 3D feature encoder and arbitrary-coordinate implicit SDF/PBR decoder.
- Implemented: chunked field queries, calibrated multi-view patch fusion, SDF/normal/material losses and optional Eikonal regularization.
- Implemented: asset-level implicit-field training path with EMA checkpoints.
- Remaining: large licensed dataset, pretrained weights, adaptive sparse latent allocation, end-to-end mesh/export integration, GPU scaling and independent TRELLIS.2 comparison.
