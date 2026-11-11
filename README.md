# Bunyaan

Independent open-source generative 3D and PBR model. Start with [`Bunyaan-Core/README.md`](Bunyaan-Core/README.md) for installation, training, inference, dataset governance and architecture details.

**Current source release:** v1.5 hierarchical implicit-field research upgrade. Bunyaan includes its own conditional 3D and material diffusion networks, token-to-spatial attention, EMA training, deterministic DDIM sampling, a coarse-volume-to-implicit-SDF/PBR decoder, chunked arbitrary-coordinate queries and camera-aware multi-view fusion. It does not require another generative 3D model at runtime.

**Research status:** no pretrained weights are bundled. The legacy generator still uses dense occupancy; the new implicit path is not pretrained and superiority over TRELLIS.2 has not been demonstrated. See [`Bunyaan-Core/docs/ARCHITECTURE-v1.5.md`](Bunyaan-Core/docs/ARCHITECTURE-v1.5.md) and [`Bunyaan-Core/docs/BENCHMARK-GAP-TRELLIS.md`](Bunyaan-Core/docs/BENCHMARK-GAP-TRELLIS.md).

Licensed under Apache-2.0. Dataset and asset licenses remain separate and must be checked per source and asset.
