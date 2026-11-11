# Bunyaan v1.1 upgrade notes

## Added
- `Bunyaan-Core/bunyaan_model/voxel.py`: adaptive octree representation and dense round-trip/stats.
- `Bunyaan-Core/bunyaan_model/mesh_intelligence.py`: mesh diagnostics, conservative cleanup/repair, optional decimation and watertight gate.
- `Bunyaan-Core/bunyaan_model/materials.py`: deterministic UV-space procedural PBR maps (base color, roughness, metallic, tangent-style normal, AO) at 1K/2K/4K.
- API routes for octree stats, mesh analyze/process and PBR synthesis, plus local UI controls.
- Training resume via `--resume` and held-out evaluation via `training/evaluate.py`.

## Verification performed
- Python compileall completed.
- `python -m pytest -q`: 6 passed.
- FastAPI TestClient: `/health` and `/docs` returned HTTP 200.
- Browser JavaScript syntax checked with `node --check`.
- Manual smoke tests covered octree round-trip, mesh inspection/repair, and PBR archive creation.

## Still not implemented / important boundaries
- No pretrained Bunyaan checkpoint is included. Text/image 3D generation remains unavailable until the native model is trained.
- The octree is a CPU-side representation and is not wired into sparse neural inference; the denoiser remains dense and fixed-resolution.
- PBR synthesis is procedural, not a trained texture AI; it does not unwrap meshes or attach maps to arbitrary imported mesh materials.
- Native multi-view reconstruction is not implemented.
- FBX/USD export is not implemented in the native exporter. Use a separately installed DCC bridge where appropriate.
- Remeshing is not a full production remeshing pipeline. Decimation requires the optional `fast-simplification` package.

All upstream reference source archives and their licensing notices remain intact and isolated from the Bunyaan native runtime.
