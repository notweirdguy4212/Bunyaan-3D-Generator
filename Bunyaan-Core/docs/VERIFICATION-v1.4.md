# Bunyaan v1.4 verification record

## Environment used

- Python: 3.13
- PyTorch: 2.10.0+cpu
- Device: CPU
- Dataset: synthetic smoke fixtures only (not a production training corpus)

## Automated checks

Command:

```bash
PYTHONPATH=. python -m pytest -q
python -m py_compile bunyaan_model/*.py training/*.py tools/*.py
```

Result at release packaging: **15 tests passed**; Python compilation completed without errors.

## Training smoke checks

Both native training entry points completed one CPU epoch using tiny synthetic examples with four distinct provenance model IDs:

- `training/train.py`: created latest and best Bunyaan 3D checkpoints with EMA, optimizer/scheduler/scaler state and model-level train/validation split metadata.
- `training/train_texture.py`: created latest and best Bunyaan material checkpoints with EMA, optimizer/scaler state and model-level train/validation split metadata.

## Inference smoke check

A newly trained one-epoch synthetic shape checkpoint loaded through `BunyaanService`, generated a mesh with four-step DDIM, and exported a GLB successfully.

## What this does and does not prove

These checks prove that key source paths compile, unit tests pass, minimal CPU training runs, checkpoint loading works, and DDIM mesh export works. They do **not** prove perceptual quality, convergence, generalization, production performance, GPU scaling, license compliance of any external dataset, or superiority to other 3D generators. The smoke data is synthetic and must not be presented as a trained Bunyaan model release.
