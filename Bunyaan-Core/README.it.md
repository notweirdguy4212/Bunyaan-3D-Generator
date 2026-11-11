**Aggiornamento v1.4:** aggiunti cross-attention nativa testo-spazio, pesi EMA, validazione per identità dell’asset e campionamento DDIM deterministico. Consulta l’[architettura v1.4](docs/ARCHITECTURE-v1.4.md) per capacità e limiti precisi.

# Bunyaan — Modello generativo 3D indipendente

**Bunyaan è un progetto di ricerca open source per la generazione 3D e i materiali PBR. Architettura, condizionamento, diffusione, addestramento, inferenza e formati dei checkpoint sono nativi di Bunyaan. Non integra né richiede altri modelli generativi.**

> **Il repository include architetture native addestrabili e strumenti riproducibili, ma non pesi preaddestrati. La qualità dipende da dati, hardware e valutazione.**

## النطاق / Scope

Il percorso 3D usa condizionamento testuale/visivo e diffusione su volumi di occupazione. Il percorso materiali genera cinque canali UV: RGB, roughness e metallic.

## بيانات التدريب والحقوق

Usa solo risorse con licenze che autorizzano esplicitamente l’addestramento ML. Conserva hash, provenienza, attribuzione e licenza. Una licenza sconosciuta non è un’autorizzazione.

## البدء / Quick start

Consulta `README.md` per la guida completa. Installa Python 3.10–3.12, PyTorch compatibile e `pip install -r requirements.txt`. Usa `python training/train.py --help`, `python training/train_texture.py --help` e `python training/evaluate_texture.py --help`.

## الترخيص

Codice e documentazione Bunyaan sono sotto Apache-2.0. Dataset e dipendenze mantengono le proprie licenze.

اقرأ ملف [`README.md`](README.md) الإنجليزي للحصول على تفاصيل البنية، والقيود، وخطوات التقييم، وخريطة الطريق، والمساهمة، والأمان.


## Bunyaan v1.5 engineering update

The native v1.5 research path adds a hierarchical 3D feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, chunked field queries, calibrated multi-view fusion, and SDF/normal/PBR/Eikonal training. This is not pretrained and has not been shown to outperform TRELLIS.2. See `docs/ARCHITECTURE-v1.5.md`, `UPGRADE-NOTES-v1.5.md`, and `training/train_implicit.py`.
