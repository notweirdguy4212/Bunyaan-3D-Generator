# Bunyaan vs TRELLIS.2 — capability and evidence gap

This document prevents architecture aspirations from being mistaken for benchmark results. It uses official TRELLIS.2 project materials as the reference baseline.

## Publicly documented TRELLIS.2 reference

The official Microsoft project describes TRELLIS.2 as a 4B-parameter image-to-3D model using a field-free sparse O-Voxel representation, a sparse 3D VAE with 16x spatial downsampling, approximately 9.6K latent tokens for a 1024³ asset, training on 500K 3D assets, PBR base colour/metallic/roughness/opacity, arbitrary topology support, and up to 1536³ generation. Its published H100 timing is approximately 3s at 512³, 17s at 1024³ and 60s at 1536³. These are project-reported figures, not independent measurements.

Primary references:
- Project page: https://microsoft.github.io/TRELLIS.2/
- Source and training code: https://github.com/microsoft/TRELLIS.2
- Model card and weights: https://huggingface.co/microsoft/TRELLIS.2-4B

## Honest capability matrix

| Capability | TRELLIS.2 public evidence | Bunyaan v1.5 status |
|---|---|---|
| Pretrained model weights | Public 4B checkpoint | None supplied |
| Training scale | 500K assets reported | No large-scale run |
| Representation | Sparse O-Voxel + sparse 3D VAE | Coarse volume + learned feature grid + implicit SDF decoder |
| Spatial compression | 16x downsampling; ~9.6K tokens for 1024³ reported | No comparable compression figure |
| Geometry | Arbitrary topology, open/non-manifold/internal structures claimed by project | Continuous SDF path; topology performance unmeasured |
| PBR | Base colour, metallic, roughness, opacity | Predicts base colour, roughness, metallic, opacity per query; no trained weights |
| Multi-view | Primary public model is single-image; shape-conditioned texture workflow also documented | Native camera-aware multi-view encoder implemented; untrained |
| Output resolution | Up to 1536³ documented | Arbitrary query density supported in code; practical limits unmeasured |
| Speed / VRAM | H100 figures published | Not benchmarked |
| Independent quality comparison | External comparisons exist, protocols vary | None yet |

## Minimum fair head-to-head protocol

1. Select the same legally redistributable held-out input images and publish their hashes.
2. Compare image-to-3D only with image-to-3D. Compare multi-view only when both systems receive equivalent views and camera calibration.
3. Fix prompt text, image crop, background handling, output coordinate frame, material colour space, and post-processing policy.
4. Report geometry metrics: Chamfer-L1/L2, F-score at stated thresholds, normal consistency, connected components, watertight rate, self-intersections and topology failures. Report metric implementation and mesh sampling count.
5. Report PBR metrics separately: base-colour error in linear RGB, roughness/metallic error where ground truth exists, opacity error, relighting consistency and blinded human appearance preference.
6. Report runtime, peak VRAM, hardware, precision, resolution, batch size, warm-up, preprocessing and post-processing. Never compare H100 figures with consumer GPU results as if they were equivalent.
7. Publish exact checkpoints, seeds, configs, scripts, dataset licences and all failed samples. Distinguish author-reported results from independent reproduction.

## Bunyaan gates before any superiority claim

- Complete legally cleared training data with asset-level train/validation/test splits.
- Train the v1.5 implicit path at meaningful scale; publish the checkpoint and training compute.
- Add adaptive sparse spatial allocation and measure actual memory/throughput scaling.
- Verify mesh topology and PBR quality on held-out assets.
- Reproduce a matched TRELLIS.2 baseline on the same hardware and data protocol.
- Require independent reruns and uncertainty estimates before claiming Bunyaan is better.

Until these gates are passed, Bunyaan's status is **research architecture implemented; competitive quality unproven**.
