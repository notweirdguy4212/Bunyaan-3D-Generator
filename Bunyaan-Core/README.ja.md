**v1.4 更新:** ネイティブなテキスト・空間クロスアテンション、EMA 重み、アセット ID 単位の検証、決定論的 DDIM サンプリングを追加しました。正確な機能と制限は[アーキテクチャ v1.4](docs/ARCHITECTURE-v1.4.md)を参照してください。

# Bunyaan — 独立した生成3Dモデル

**Bunyaanは、3D生成とPBRマテリアルのためのオープンソース研究プロジェクトです。アーキテクチャ、条件付け、拡散、学習、推論、チェックポイント形式はBunyaan独自です。他の生成モデルを組み込まず、依存もしません。**

> **このリポジトリには学習可能な独自アーキテクチャと再現可能なツールが含まれますが、事前学習済み重みは含まれません。品質はデータ、計算資源、評価に依存します。**

## النطاق / Scope

3D経路はテキスト/画像条件付けと占有ボリューム拡散を使用します。マテリアル経路はUV上の5チャンネル（RGB、roughness、metallic）を生成します。

## بيانات التدريب والحقوق

機械学習での利用を明示的に許可するライセンスのアセットのみ使用してください。ハッシュ、出典、帰属表示、ライセンスを保存してください。不明なライセンスは許可とみなしません。

## البدء / Quick start

完全な手順は `README.md` を参照してください。Python 3.10–3.12と対応するPyTorchを導入し、`pip install -r requirements.txt` を実行します。学習・評価スクリプトには `--help` を指定してください。

## الترخيص

Bunyaanのコードと文書はApache-2.0です。データセットと依存ライブラリには各自のライセンスが適用されます。

اقرأ ملف [`README.md`](README.md) الإنجليزي للحصول على تفاصيل البنية، والقيود، وخطوات التقييم، وخريطة الطريق، والمساهمة، والأمان.


## Bunyaan v1.5 engineering update

The native v1.5 research path adds a hierarchical 3D feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, chunked field queries, calibrated multi-view fusion, and SDF/normal/PBR/Eikonal training. This is not pretrained and has not been shown to outperform TRELLIS.2. See `docs/ARCHITECTURE-v1.5.md`, `UPGRADE-NOTES-v1.5.md`, and `training/train_implicit.py`.
