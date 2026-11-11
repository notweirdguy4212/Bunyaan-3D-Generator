# Bunyaan v1.5 — Hierarchical Implicit Fields

## Status

v1.5 adds an independently authored research architecture and training path. It is **not a pretrained foundation model**. The code has CPU smoke-test coverage; it has not yet been trained on a large licensed asset corpus or evaluated against TRELLIS.2. The previous v1.4 dense-occupancy generator remains available.

## New native pipeline

```text
coarse generated volume ──► 3D feature encoder ──► compact feature grid
                                                        │
text tokens ────────────────────────────────────────────┤
                                                        ├─► implicit field decoder
calibrated RGB views + 3x4 cameras ─► view fusion ──────┤
                                                        │
arbitrary xyz query points ─► Fourier coordinates ─────┘
                                                        │
                          SDF + normals + base colour + roughness
                                  + metallic + opacity
                                                        │
                                chunked field sampling ─► mesh extraction
```

### Coarse-to-fine implicit geometry

`BunyaanCoarseFeatureEncoder` converts a coarse volume into a compact multiscale 3D feature grid. `BunyaanImplicitFieldDecoder` samples that grid at arbitrary normalized coordinates and predicts a continuous signed-distance field (SDF), surface normals and PBR attributes. The output resolution is determined by query density, not by a fixed output voxel tensor. `query_field_in_chunks` bounds query activation memory.

The decoder uses explicit trilinear feature sampling so the Eikonal SDF regularizer can compute higher-order derivatives on CPU backends that do not implement second derivatives for 3D grid sampling.

### Native calibrated multi-view fusion

`BunyaanMultiViewEncoder` encodes image patches from each view, embeds the corresponding 3x4 world-to-image camera matrix and fuses the view set with a transformer. A padding mask supports variable numbers of views. Camera matrices must use a consistent coordinate convention and calibration; uncalibrated images must not be treated as calibrated multi-view supervision.

### Geometry and material supervision

`implicit_field_losses` supports:

- SDF smooth-L1 supervision
- normal cosine consistency
- linear base-colour L1
- roughness, metallic and opacity L1
- optional Eikonal regularization, encouraging `||∇SDF|| = 1`

Training data must contain real aligned geometry/material targets. The loss implementation does not create ground truth, UV maps or calibrated camera data.

## Training record format

Each `.pt` file consumed by `training/train_implicit.py` contains:

```python
{
  "coarse_volume": FloatTensor[C, D, H, W],
  "query_points": FloatTensor[N, 3],  # xyz normalized to [-1, 1]
  "fields": {
    "sdf": FloatTensor[N, 1],
    "normal": FloatTensor[N, 3],
    "base_color": FloatTensor[N, 3],  # linear RGB
    "roughness": FloatTensor[N, 1],
    "metallic": FloatTensor[N, 1],
    "opacity": FloatTensor[N, 1]
  },
  "caption": "...",  # optional
  "images": FloatTensor[V, 3, H, W],  # optional, calibrated views
  "cameras": FloatTensor[V, 3, 4],     # required when images are supplied
  "provenance": {"model_id": "unique-source-asset-id"}
}
```

All tensors in a batch must have compatible shapes. Split assignment is by source asset ID to reduce train/validation leakage. Use only assets with documented training rights and retain source/transform provenance.

Example:

```bash
python training/train_implicit.py --data data/implicit_fields \
  --out checkpoints/bunyaan-implicit-v1.5.pt --epochs 100 \
  --batch-size 2 --device cuda
```

Generate a mesh from a trained checkpoint and a `.pt` input containing `coarse_volume` (and optionally `caption`, `images`, `cameras`):

```bash
python tools/generate_implicit.py --checkpoint checkpoints/bunyaan-implicit-v1.5.pt \
  --input examples/coarse_input.pt --out outputs/asset.glb --resolution 128
```

The GLB stores vertex base colour/opacity. A sibling `.pbr.npz` preserves per-vertex base colour, roughness, metallic, opacity, normals, vertices and faces. This sidecar avoids silently reducing spatially varying material channels to one scalar GLB material.

## Important distinctions from TRELLIS.2

TRELLIS.2's official release describes a 4B-parameter rectified-flow model, an O-Voxel field-free sparse representation, a sparse 3D VAE with 16x spatial downsampling, approximately 9.6K latent tokens for a 1024³ asset, training on 500K assets, and H100 timing figures. Bunyaan v1.5 does **not** currently match those demonstrated properties: it has no pretrained weights, no 500K-scale training run, no sparse-convolution/O-Voxel equivalent, no validated 1536³ end-to-end generation path, and no comparable speed or quality results.

Bunyaan's research direction is different: coarse volume conditioning followed by arbitrary-point implicit SDF/PBR decoding and calibrated multi-view fusion. This is an architectural capability, not proof of superior outputs. TRELLIS.2 remains a high bar and a benchmark target, not a dependency.

## Verification

Run:

```bash
PYTHONPATH=. python -m pytest -q
python -m py_compile bunyaan_model/*.py training/*.py tools/*.py
```

Tests cover decoder tensor contracts, PBR value ranges, chunked-query equivalence, camera-aware view masking, Eikonal gradients, and a coarse-volume-to-field forward pass. A one-epoch CPU smoke run validates the new trainer/checkpoint path only; synthetic smoke data says nothing about perceptual quality or generalization.

## Research gates before performance claims

1. Train on a large, deduplicated and legally cleared dataset with paired SDF, normals, PBR and camera data.
2. Validate at asset level on held-out sources and publish all preprocessing and checkpoints.
3. Add adaptive sparse spatial allocation and a compact learned latent autoencoder before claiming TRELLIS-like memory or throughput.
4. Compare image-only and calibrated multi-view inputs separately.
5. Run matched evaluations for geometry, topology, PBR, prompt/image alignment, runtime, peak VRAM and human preference.
6. Publish failures and uncertainty; never compare incompatible benchmark protocols.
