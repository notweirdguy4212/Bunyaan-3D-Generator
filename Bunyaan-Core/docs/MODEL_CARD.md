# Bunyaan Model Card

## Identity
Bunyaan is an independent research project for native generative 3D shape and UV-space PBR material synthesis. This repository contains Bunyaan-authored model implementations and training utilities.

## Intended use
Research, experimentation, education, and development of generative 3D workflows using data with explicit rights for machine-learning training.

## Out-of-scope use
The project is not represented as a production-ready foundation model, certified CAD system, safety-critical geometry tool, or substitute for professional asset review. Do not use it to create deceptive, infringing, or unsafe assets.

## Architecture
The shape path uses native text/image conditioning, multi-scale residual 3D diffusion U-Net blocks, token-to-spatial cross-attention plus factorized linear global attention at the bottleneck, and dense occupancy volumes. v1.4 adds EMA training and deterministic DDIM sampling. The material path uses a native conditional 2D denoiser with token-to-spatial cross-attention, factorized linear global attention, a deeper bottleneck and EMA inference for five UV-aligned channels (linear RGB/base-colour representation as prepared by the corpus pipeline, roughness, metallic). See the code and configs for exact tensor contracts.

## Training data
No pretrained weights are included. Users must assemble licensed datasets. Store source ID, URL, SHA-256, author/attribution, license identifier and permission evidence where available. Exclude unknown or incompatible licenses. Split by asset identity to reduce leakage.

## Evaluation
Use held-out asset-level splits. Report denoising losses and per-channel errors, and add visual/material plausibility review. For shape generation, assess occupancy metrics, mesh validity, watertightness where applicable, diversity, prompt alignment and human review. No benchmark result is claimed by this source distribution.

## Limitations
The 3D representation is currently a low-resolution dense occupancy prototype (default 32³); fine topology, thin structures and high-frequency details are limited. The image-conditioning pathway requires paired image/shape training. Material synthesis requires a substantial UV-aligned PBR corpus and trained weights. UV overlap checks are diagnostics, not UV repair. Output quality and memory use vary with hardware, data, settings and training.

## Risks and mitigations
Potential risks include memorization, training-data license violations, biased coverage, malformed meshes, misleading material values and unsafe downstream use. Mitigate with provenance, license review, asset-level splits, memorization checks, mesh validation, human review and clear output labeling.

## v1.4 changes

The native denoisers now use token-to-spatial cross-attention. Training uses asset-level validation splits and EMA weights; shape training also saves best-validation checkpoints and full resume state. Shape inference supports deterministic DDIM and DDPM. These are engineering and architecture improvements, not evidence of superiority over other generators. See `ARCHITECTURE-v1.4.md` and `BENCHMARKING.md`.

## Release status
Research preview. Source code is open under Apache-2.0. No trained model weights or dataset are bundled. The software license does not grant rights to third-party datasets or future weights.


## v1.5 research architecture status

The v1.5 code adds a coarse-volume feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, camera-aware multi-view fusion and an implicit-field training script. These are untrained architecture components unless a user trains them on suitable licensed supervision. No v1.5 pretrained weights, large-scale training run or matched TRELLIS.2 benchmark is provided. Do not infer competitive quality from unit tests or synthetic smoke training.
