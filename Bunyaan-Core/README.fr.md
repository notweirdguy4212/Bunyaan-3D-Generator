**Mise à jour v1.4 :** ajout de l’attention croisée texte-espace native, des poids EMA, de la validation par identité d’actif et de l’échantillonnage DDIM déterministe. Consultez l’[architecture v1.4](docs/ARCHITECTURE-v1.4.md) pour connaître les capacités et limites exactes.

# Bunyaan — Modèle génératif 3D indépendant

**Bunyaan est un projet de recherche open source dédié à la génération 3D et aux matériaux PBR. Son architecture, son conditionnement, sa diffusion, son entraînement, son inférence et ses formats de checkpoints sont propres à Bunyaan. Aucun autre modèle génératif n’est intégré ni requis.**

> **Le dépôt fournit des architectures natives entraînables et des outils reproductibles ; aucun poids préentraîné n’est fourni. La qualité dépend des données, du calcul et de l’évaluation.**

## النطاق / Scope

La branche 3D utilise le conditionnement texte/image et la diffusion sur des volumes d’occupation. La branche matériau produit cinq canaux UV : RGB, rugosité et métallicité.

## بيانات التدريب والحقوق

N’utilisez que des ressources dont la licence autorise explicitement l’entraînement automatique. Conservez les empreintes, la provenance, l’attribution et la licence. Une licence inconnue n’est pas une autorisation.

## البدء / Quick start

Consultez `README.md` pour le guide complet. Installez Python 3.10–3.12, une version de PyTorch adaptée, puis `pip install -r requirements.txt`. Lancez `python training/train.py --help`, `python training/train_texture.py --help` et `python training/evaluate_texture.py --help`.

## الترخيص

Le code et la documentation Bunyaan sont sous Apache-2.0. Les jeux de données et dépendances conservent leurs licences.

اقرأ ملف [`README.md`](README.md) الإنجليزي للحصول على تفاصيل البنية، والقيود، وخطوات التقييم، وخريطة الطريق، والمساهمة، والأمان.


## Bunyaan v1.5 engineering update

The native v1.5 research path adds a hierarchical 3D feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, chunked field queries, calibrated multi-view fusion, and SDF/normal/PBR/Eikonal training. This is not pretrained and has not been shown to outperform TRELLIS.2. See `docs/ARCHITECTURE-v1.5.md`, `UPGRADE-NOTES-v1.5.md`, and `training/train_implicit.py`.
