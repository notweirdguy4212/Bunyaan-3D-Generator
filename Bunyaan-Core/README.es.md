**Actualización v1.4:** se incorporan atención cruzada texto-espacio nativa, pesos EMA, validación por identidad de activo y muestreo DDIM determinista. Consulta la [arquitectura v1.4](docs/ARCHITECTURE-v1.4.md) para conocer las capacidades y limitaciones exactas.

# Bunyaan — Modelo generativo 3D independiente

**Bunyaan es un proyecto de investigación de código abierto para generación 3D y materiales PBR. Su arquitectura, condicionamiento, difusión, entrenamiento, inferencia y formatos de checkpoints son propios de Bunyaan. No integra ni necesita otro modelo generativo.**

> **El repositorio incluye modelos nativos entrenables y herramientas reproducibles; no incluye pesos preentrenados. La calidad depende de los datos, el cómputo y la evaluación.**

## النطاق / Scope

La ruta 3D utiliza condicionamiento de texto e imagen y difusión sobre volúmenes de ocupación. La ruta de materiales genera cinco canales UV: RGB, rugosidad y metalicidad.

## بيانات التدريب والحقوق

Utiliza únicamente activos cuya licencia permita expresamente el entrenamiento de aprendizaje automático. Conserva hashes, procedencia, atribución y licencia. Una licencia desconocida no debe considerarse aprobada.

## البدء / Quick start

Consulta `README.md` para la guía completa. Instala Python 3.10–3.12, PyTorch compatible con tu hardware y `pip install -r requirements.txt`. Usa `python training/train.py --help`, `python training/train_texture.py --help` y `python training/evaluate_texture.py --help`.

## الترخيص

El código y la documentación de Bunyaan se publican bajo Apache-2.0. Los conjuntos de datos y dependencias conservan sus propias licencias.

اقرأ ملف [`README.md`](README.md) الإنجليزي للحصول على تفاصيل البنية، والقيود، وخطوات التقييم، وخريطة الطريق، والمساهمة، والأمان.


## Bunyaan v1.5 engineering update

The native v1.5 research path adds a hierarchical 3D feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, chunked field queries, calibrated multi-view fusion, and SDF/normal/PBR/Eikonal training. This is not pretrained and has not been shown to outperform TRELLIS.2. See `docs/ARCHITECTURE-v1.5.md`, `UPGRADE-NOTES-v1.5.md`, and `training/train_implicit.py`.
