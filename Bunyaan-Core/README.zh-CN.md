**v1.4 更新：**新增原生文本-空间交叉注意力、EMA 权重、按资产 ID 划分的验证集，以及确定性 DDIM 采样。准确的功能与限制请参阅 [v1.4 架构说明](docs/ARCHITECTURE-v1.4.md)。

# Bunyaan — 独立生成式 3D 模型

**Bunyaan 是一个面向 3D 生成和 PBR 材质的开源研究项目。其架构、条件模块、扩散过程、训练、推理和检查点格式均由 Bunyaan 原生实现。不集成也不依赖其他生成模型。**

> **仓库提供可训练的原生架构和可复现工具，但不包含预训练权重。生成质量取决于数据、算力和评估。**

## النطاق / Scope

3D 路径使用文本/图像条件和占据体扩散。材质路径生成 UV 五通道：RGB、粗糙度和金属度。

## بيانات التدريب والحقوق

仅使用许可证明确允许机器学习训练的资产。保留哈希、来源、署名和许可证记录。许可证未知不代表获得授权。

## البدء / Quick start

完整指南请参阅 `README.md`。安装 Python 3.10–3.12、兼容的 PyTorch，然后运行 `pip install -r requirements.txt`。训练和评估脚本可使用 `--help` 查看参数。

## الترخيص

Bunyaan 代码和文档采用 Apache-2.0 许可证。数据集和依赖项仍受各自许可证约束。

اقرأ ملف [`README.md`](README.md) الإنجليزي للحصول على تفاصيل البنية، والقيود، وخطوات التقييم، وخريطة الطريق، والمساهمة، والأمان.


## Bunyaan v1.5 engineering update

The native v1.5 research path adds a hierarchical 3D feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, chunked field queries, calibrated multi-view fusion, and SDF/normal/PBR/Eikonal training. This is not pretrained and has not been shown to outperform TRELLIS.2. See `docs/ARCHITECTURE-v1.5.md`, `UPGRADE-NOTES-v1.5.md`, and `training/train_implicit.py`.
