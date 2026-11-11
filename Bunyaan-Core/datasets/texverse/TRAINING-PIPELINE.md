# Bunyaan TexVerse → UV PBR diffusion pipeline

This workflow converts locally acquired, rights-cleared TexVerse GLB assets into Bunyaan-native five-channel tensors. It does not bundle TexVerse GLBs or pretrained weights.

## 1. Acquire and license-screen assets

Obtain the official TexVerse metadata and GLB files separately. Keep the original metadata and verify that each asset's license permits the intended ML use and redistribution. The manifest builder is a metadata filter; it is not legal advice:

```bash
python tools/prepare_texverse_manifest.py --metadata /path/to/metadata.json \
  --out datasets/texverse/bunyaan-texverse-manifest.jsonl
```

## 2. Extract GLB geometry, UVs and PBR textures

```bash
python tools/extract_texverse.py --input /path/to/glbs \
  --out datasets/bunyaan-texverse \
  --manifest datasets/bunyaan-texverse/manifest.jsonl \
  --resolution 512 --license CC-BY-4.0 \
  --license-url 'https://creativecommons.org/licenses/by/4.0/' \
  --attribution 'Original TexVerse creator attribution'
```

The extractor reads mesh geometry, per-vertex UV coordinates and glTF PBR material properties. It samples the base-color image as RGB and the glTF metallic-roughness image's G and B channels as roughness and metallic. If the packed texture is absent, scalar material factors are used. Each sample is a float32 `[5,H,W]` tensor in `[0,1]`, with a UV coverage mask, UVs, faces and per-asset provenance (source path/hash, model ID, license, attribution and channel semantics).

Missing UVs, malformed/non-finite UVs and UVs outside `[0,1]` are skipped by default. UV triangle overlap and undersized source textures are recorded as validation warnings. `--allow-invalid-uv` allows warning-bearing samples to be written for manual review; it does not repair UVs. Overlapping UVs can be intentional (mirrored/shared layouts), so review the report rather than assuming every overlap is an error. The extractor does not perform a geometry-aware bake or unwrap meshes.

## 3. Validate the corpus

```bash
python tools/validate_texverse.py --data datasets/bunyaan-texverse \
  --min-resolution 256 --report reports/bunyaan-texverse-validation.json
```

This checks tensor shape, finite normalized values, resolution and required provenance. Any nonzero exit indicates invalid samples requiring review/removal.

## 4. Train the conditional texture diffusion model

```bash
python training/train_texture.py --data datasets/bunyaan-texverse \
  --out checkpoints/bunyaan-texverse-texture.pt --epochs 100 \
  --batch-size 4 --resolution 256 --amp auto --deterministic
```

Validation splitting is deterministic and grouped by model ID to avoid leakage between meshes from one source asset. CUDA mixed precision is enabled automatically when available. JSONL logs include total and per-channel train/validation denoising losses. A latest checkpoint is saved every epoch and the best validation checkpoint is saved as `*-best.pt`.

Resume from the latest checkpoint (epochs is the final epoch number):

```bash
python training/train_texture.py --data datasets/bunyaan-texverse \
  --out checkpoints/bunyaan-texverse-texture.pt \
  --resume checkpoints/bunyaan-texverse-texture.pt --epochs 200 \
  --resolution 256 --amp auto --deterministic
```

## 5. Held-out evaluation

```bash
python training/evaluate_texture.py \
  --checkpoint checkpoints/bunyaan-texverse-texture-best.pt \
  --data datasets/bunyaan-texverse \
  --out reports/bunyaan-texture-evaluation.json
```

Denoising MSE is a diagnostic, not a perceptual material-quality metric. Preserve the manifest, validation report, split IDs, logs and checkpoints as one experiment record. Training quality depends on the diversity and rights-cleared quality of the acquired corpus.
