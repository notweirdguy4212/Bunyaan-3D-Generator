# Bunyaan

Independent, open-source generative 3D and PBR modeling for researchers building and training their own geometry and material pipelines.

<p align="center">
  <a href="https://github.com/notweirdguy4212/Bunyaan-3D-Generator/stargazers"><img src="https://img.shields.io/github/stars/notweirdguy4212/Bunyaan-3D-Generator?style=flat-square" alt="GitHub stars"></a>
  <a href="https://github.com/notweirdguy4212/Bunyaan-3D-Generator/commits"><img src="https://img.shields.io/github/last-commit/notweirdguy4212/Bunyaan-3D-Generator?style=flat-square" alt="Last commit"></a>
  <img src="https://img.shields.io/github/languages/top/notweirdguy4212/Bunyaan-3D-Generator?style=flat-square" alt="Top language">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/SQLite-003B57?style=flat-square&amp;logo=sqlite&amp;logoColor=white" alt="SQLite">
</p>

## Overview

Bunyaan provides its own conditional 3D and material diffusion networks for experimenting with geometry generation and physically based rendering (PBR) materials. It does not require another generative 3D model at runtime.

**Current source release: v1.5 — hierarchical implicit-field research upgrade.** It adds a coarse-volume-to-implicit-SDF/PBR decoder, chunked arbitrary-coordinate queries, and camera-aware multi-view fusion alongside token-to-spatial attention, EMA training, and deterministic DDIM sampling.

> **Research status:** No pretrained weights are bundled. The legacy generator still uses dense occupancy, and the new implicit path is not pretrained. Superiority over TRELLIS.2 has not been demonstrated.

Start with the [core README](Bunyaan-Core/README.md) for installation, training, inference, dataset governance, and architecture details. For technical scope and limitations, see the [v1.5 architecture](Bunyaan-Core/docs/ARCHITECTURE-v1.5.md), [model card](Bunyaan-Core/docs/MODEL_CARD.md), and [TRELLIS benchmark gap](Bunyaan-Core/docs/BENCHMARK-GAP-TRELLIS.md).

## Features

- **Independent generation stack:** Experiment with Bunyaan’s own conditional geometry and material diffusion networks without a separate generative 3D model at runtime.
- **Implicit geometry and materials:** Explore a coarse-volume-to-implicit-SDF/PBR path alongside the legacy dense-occupancy generator.
- **Flexible spatial queries:** Query arbitrary coordinates in chunks through the implicit path.
- **Conditioning and multi-view fusion:** Use token-to-spatial attention and camera-aware multi-view fusion.
- **Training and sampling tools:** Work with EMA training, deterministic DDIM sampling, and separate geometry, implicit-field, and texture training entry points.
- **Dataset preparation workflows:** Prepare, extract, and validate TexVerse data, with separate documentation for dataset governance and training.

## Tech stack

| Layer | Choice | Why it fits |
| --- | --- | --- |
| Model and tooling | Python | Contains model implementations, training scripts, generation tools, and tests. |
| Dependency installation | pip with `requirements.txt` | Keeps the core’s Python dependency list in one place. |
| Configuration | JSON | Provides versioned model configurations, including v1.5. |
| Library data | SQLite | Stores the bundled `bunyaan-library.sqlite3` database. |
| Web entry point | HTML | Provides the browser-facing page in `Bunyaan-Core/web/`. |
| Windows entry points | Command scripts | Provides repository-level setup and start scripts. |

See [`requirements.txt`](Bunyaan-Core/requirements.txt) for the actual Python dependencies.

## Getting started

### Prerequisites

- Git and a Python environment with `pip` and `venv`.
- Hardware and runtime dependencies appropriate to your training or inference configuration.
- Trained weights for meaningful inference; **none are bundled**.

Check the [core installation instructions](Bunyaan-Core/README.md) for supported Python versions, accelerator requirements, and model setup before installing.

### Clone and install

```bash
git clone https://github.com/notweirdguy4212/Bunyaan-3D-Generator.git
cd Bunyaan-3D-Generator/Bunyaan-Core

python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Install the declared dependencies:

```bash
python -m pip install -r requirements.txt
```

### Configuration and environment variables

Versioned configurations live in [`Bunyaan-Core/configs/`](Bunyaan-Core/configs/). Follow the core README for configuration selection, dataset locations, and checkpoint paths.

No environment-variable contract is established by this root README; do not assume API keys, service ports, or checkpoint variables. Use only variables documented by the core tools.

### Local development and service startup

From `Bunyaan-Core`, inspect the service’s supported options before starting it:

```bash
python tools/serve.py --help
```

Windows users also have repository-root entry points:

```powershell
.\Setup-Bunyaan.cmd
.\Start-Bunyaan.cmd
```

Consult the [core README](Bunyaan-Core/README.md) for their setup and startup workflow.

### Production use

This is a research source release, not a pretrained production package. There is no production deployment command documented in the supplied root README. Follow the core service instructions, supply trained checkpoints, and review the [model card](Bunyaan-Core/docs/MODEL_CARD.md) and [security policy](Bunyaan-Core/SECURITY.md) before exposing a service.

## Usage

From `Bunyaan-Core`, inspect the two generation entry points:

```bash
# Legacy generation path
python tools/generate.py --help

# Implicit-field generation path
python tools/generate_implicit.py --help
```

Use the documented options with your configuration and trained checkpoint. Installing the source alone does not provide a pretrained generator.

For training and data preparation, see the [core README](Bunyaan-Core/README.md) and [TexVerse training pipeline](Bunyaan-Core/datasets/texverse/TRAINING-PIPELINE.md).

## Testing

The repository includes tests for the model, implicit 3D functionality, and advanced features:

- `Bunyaan-Core/tests/test_model.py`
- `Bunyaan-Core/tests/test_implicit3d.py`
- `Bunyaan-Core/tests/test_advanced_features.py`

From `Bunyaan-Core`, standard-library discovery can run unittest-compatible tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

Check the [core README](Bunyaan-Core/README.md) for any runner-specific instructions or additional setup. Unit tests do not establish generation quality or superiority over another model; see the [benchmarking guide](Bunyaan-Core/docs/BENCHMARKING.md) and [v1.5 verification notes](Bunyaan-Core/docs/VERIFICATION-v1.5.md).

## Project structure

```text
.
├── Bunyaan-Core/                 # Models, tools, training, data workflows, and documentation
│   ├── assets/
│   ├── bunyaan_model/
│   ├── configs/
│   ├── datasets/
│   ├── docs/
│   ├── tests/
│   ├── tools/
│   ├── training/
│   ├── web/
│   ├── CONTRIBUTING.md
│   ├── README.md
│   ├── SECURITY.md
│   └── requirements.txt
├── Setup-Bunyaan.cmd
├── Start-Bunyaan.cmd
├── UPGRADE-NOTES-v1.1.md
├── LICENSE
├── NOTICE
└── README.md
```

## Contributing

- **Issues:** Report reproducible bugs or propose changes through [GitHub Issues](https://github.com/notweirdguy4212/Bunyaan-3D-Generator/issues); follow the [security policy](Bunyaan-Core/SECURITY.md) for vulnerabilities.
- **Branches:** Use a descriptive branch name, such as `fix/mesh-export` or `docs/setup-clarity`.
- **Pull requests:** Follow the [contribution guide](Bunyaan-Core/CONTRIBUTING.md), explain the change, include relevant tests, and update affected documentation.

## License

Bunyaan is licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for additional notices.

**Dataset and asset licenses are separate.** Check the terms for every source and asset before downloading, training, or redistributing; start with the [TexVerse data notice](Bunyaan-Core/datasets/TEXVERSE-DATA-NOTICE.md).