**Update v1.4:** Native Text-zu-Raum-Cross-Attention, EMA-Gewichte, Validierung nach Asset-Identität und deterministisches DDIM-Sampling wurden ergänzt. Die [Architektur v1.4](docs/ARCHITECTURE-v1.4.md) beschreibt Fähigkeiten und Grenzen.

# Bunyaan — Unabhängiges generatives 3D-Modell

**Bunyaan ist ein Open-Source-Forschungsprojekt für generative 3D-Inhalte und PBR-Materialien. Architektur, Konditionierung, Diffusion, Training, Inferenz und Checkpoint-Formate sind Bunyaan-eigenständig. Kein anderes generatives Modell wird eingebunden oder benötigt.**

> **Das Repository enthält trainierbare native Architekturen und reproduzierbare Werkzeuge, aber keine vortrainierten Gewichte. Die Qualität hängt von Daten, Rechenleistung und Evaluation ab.**

## النطاق / Scope

Der 3D-Pfad nutzt Text-/Bildkonditionierung und Diffusion auf Occupancy-Volumen. Der Materialpfad erzeugt fünf UV-Kanäle: RGB, Rauheit und Metallizität.

## بيانات التدريب والحقوق

Verwende nur Assets, deren Lizenz ML-Training ausdrücklich erlaubt. Bewahre Hashes, Herkunft, Attribution und Lizenzdaten auf. Eine unbekannte Lizenz ist keine Freigabe.

## البدء / Quick start

Lies `README.md` für die vollständige Anleitung. Installiere Python 3.10–3.12, eine passende PyTorch-Version und `pip install -r requirements.txt`. Nutze `python training/train.py --help`, `python training/train_texture.py --help` und `python training/evaluate_texture.py --help`.

## الترخيص

Bunyaan-Code und Dokumentation stehen unter Apache-2.0. Datensätze und Abhängigkeiten behalten ihre eigenen Lizenzen.

اقرأ ملف [`README.md`](README.md) الإنجليزي للحصول على تفاصيل البنية، والقيود، وخطوات التقييم، وخريطة الطريق، والمساهمة، والأمان.


## Bunyaan v1.5 engineering update

The native v1.5 research path adds a hierarchical 3D feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, chunked field queries, calibrated multi-view fusion, and SDF/normal/PBR/Eikonal training. This is not pretrained and has not been shown to outperform TRELLIS.2. See `docs/ARCHITECTURE-v1.5.md`, `UPGRADE-NOTES-v1.5.md`, and `training/train_implicit.py`.
