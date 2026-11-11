**v1.4 更新：**新增原生文字－空間交叉注意力、EMA 權重、依資產 ID 劃分的驗證集，以及確定性 DDIM 取樣。準確的功能與限制請參閱 [v1.4 架構說明](docs/ARCHITECTURE-v1.4.md)。

# Bunyaan — 獨立生成式 3D 模型

**Bunyaan 是專為 3D 生成與 PBR 材質打造的開源研究專案。其架構、條件模組、擴散流程、訓練、推論與檢查點格式皆為 Bunyaan 原生實作，不整合也不依賴其他生成模型。**

> **專案提供可訓練的原生架構與可重現工具，但不包含預訓練權重。生成品質取決於資料、算力與評估。**

## النطاق / Scope

3D 路徑使用文字/影像條件與佔據體擴散。材質路徑產生 UV 五通道：RGB、粗糙度與金屬度。

## بيانات التدريب والحقوق

僅使用授權明確允許機器學習訓練的資產。保留雜湊、來源、署名與授權紀錄。授權未知不代表已獲准。

## البدء / Quick start

完整指南請參閱 `README.md`。安裝 Python 3.10–3.12、相容的 PyTorch，接著執行 `pip install -r requirements.txt`。訓練與評估指令可加上 `--help` 查看參數。

## الترخيص

Bunyaan 程式碼與文件採 Apache-2.0。資料集與相依套件仍受各自授權條款約束。

اقرأ ملف [`README.md`](README.md) الإنجليزي للحصول على تفاصيل البنية، والقيود، وخطوات التقييم، وخريطة الطريق، والمساهمة، والأمان.


## Bunyaan v1.5 engineering update

The native v1.5 research path adds a hierarchical 3D feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, chunked field queries, calibrated multi-view fusion, and SDF/normal/PBR/Eikonal training. This is not pretrained and has not been shown to outperform TRELLIS.2. See `docs/ARCHITECTURE-v1.5.md`, `UPGRADE-NOTES-v1.5.md`, and `training/train_implicit.py`.
