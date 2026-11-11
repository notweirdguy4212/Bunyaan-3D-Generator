# Bunyaan v1.5 verification record

## Environment and checks

- Automated test suite: **22 passed** on the development environment.
- Python compilation: all `bunyaan_model/*.py`, `training/*.py` and `tools/*.py` compile successfully.
- Implicit trainer smoke test: one CPU epoch on synthetic field records completed and produced a checkpoint containing the native field model, text conditioner, EMA state, optimizer/scheduler state and asset-level split metadata.
- Implicit mesh extraction: analytic sphere fixture exercised marching-cubes extraction and per-vertex PBR attribute sampling.
- Checkpoint round-trip: native v1.5 EMA checkpoint loader tested.

## What this proves

The new architecture components, loss code, chunked query path, calibrated-view mask handling, mesh extraction and trainer/checkpoint paths execute on small synthetic fixtures.

## What this does not prove

- No pretrained Bunyaan v1.5 weights are distributed.
- No full-scale licensed dataset has been trained.
- Synthetic one-epoch loss is not a quality metric.
- No claim is made about photorealism, topology robustness, generalization, GPU throughput, peak VRAM or production readiness.
- No matched or independently reproduced TRELLIS.2 benchmark has been completed.
- The new implicit path predicts PBR channels, but export currently preserves per-vertex roughness/metallic/opacity in an NPZ sidecar; GLB vertex colours carry base colour/opacity. A full texture-map/material-baking pipeline remains future work.

## Release decision

v1.5 is an architecture/training-path research milestone. It must not be described as better than TRELLIS.2 until the fair comparison gates in `docs/BENCHMARK-GAP-TRELLIS.md` are completed.
