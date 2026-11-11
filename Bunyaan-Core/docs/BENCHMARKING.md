# Bunyaan benchmarking protocol

Bunyaan does not currently publish a verified leaderboard position. This protocol is intended to make future comparisons reproducible and fair.

## Baselines

Compare against open-source systems only after recording the exact repository commit, checkpoint provenance, license, inference configuration, hardware, and whether the baseline uses pretrained weights. Keep pretrained-model comparisons separate from same-data-from-scratch architecture comparisons.

## Shape evaluation

Use an asset-disjoint test set with fixed prompts and, where available, paired reference views. Report Chamfer distance, precision/recall/F-score at declared thresholds, normal consistency, watertight percentage, connected-component statistics, prompt alignment, diversity, peak VRAM and latency. Inspect a fixed random sample manually for topology defects and prompt mismatch.

## Material evaluation

Use held-out source assets, not random texture crops from training models. Report RGB/linear-colour error, roughness and metallic MAE, per-channel calibration, seam discontinuity across UV island boundaries, valid UV coverage, material plausibility review, peak VRAM and latency. Report missing channels separately; do not synthesize ground truth labels for missing metallic/roughness maps.

## Required report fields

- Bunyaan git commit and architecture/config version
- Dataset source, snapshot date, license allowlist and manifest SHA-256
- Train/validation/test asset IDs or split-manifest hashes
- Checkpoint SHA-256 and training steps/epochs
- Prompt set and random seeds
- Hardware, software versions, precision and batch size
- Resolution, sampler, steps and guidance
- Quality metrics, latency, peak VRAM and failure count
- Limitations and any manual filtering

## Release gate

Do not use terms such as *state of the art*, *better than*, *beats*, or *more powerful than* other generators unless a reproducible, independently checked evaluation supports the exact claim and comparison conditions are disclosed.
