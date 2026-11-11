**Atualização v1.4:** adiciona atenção cruzada nativa entre texto e espaço, pesos EMA, validação por identidade do ativo e amostragem DDIM determinística. Consulte a [arquitetura v1.4](docs/ARCHITECTURE-v1.4.md) para capacidades e limitações exatas.

# Bunyaan — Modelo generativo 3D independente

**Bunyaan é um projeto de pesquisa de código aberto para geração 3D e materiais PBR. Arquitetura, condicionamento, difusão, treinamento, inferência e formatos de checkpoint são nativos do Bunyaan. Nenhum outro modelo generativo é incorporado ou necessário.**

> **O repositório oferece arquiteturas nativas treináveis e ferramentas reproduzíveis; não inclui pesos pré-treinados. A qualidade depende dos dados, do hardware e da avaliação.**

## النطاق / Scope

O caminho 3D usa condicionamento de texto/imagem e difusão em volumes de ocupação. O caminho de materiais gera cinco canais UV: RGB, rugosidade e metalicidade.

## بيانات التدريب والحقوق

Use somente ativos cuja licença permita explicitamente treinamento de aprendizado de máquina. Preserve hashes, procedência, atribuição e licença. Licença desconhecida não significa autorização.

## البدء / Quick start

Consulte `README.md` para o guia completo. Instale Python 3.10–3.12, PyTorch compatível e `pip install -r requirements.txt`. Execute `python training/train.py --help`, `python training/train_texture.py --help` e `python training/evaluate_texture.py --help`.

## الترخيص

O código e a documentação do Bunyaan usam Apache-2.0. Conjuntos de dados e dependências mantêm suas próprias licenças.

اقرأ ملف [`README.md`](README.md) الإنجليزي للحصول على تفاصيل البنية، والقيود، وخطوات التقييم، وخريطة الطريق، والمساهمة، والأمان.


## Bunyaan v1.5 engineering update

The native v1.5 research path adds a hierarchical 3D feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, chunked field queries, calibrated multi-view fusion, and SDF/normal/PBR/Eikonal training. This is not pretrained and has not been shown to outperform TRELLIS.2. See `docs/ARCHITECTURE-v1.5.md`, `UPGRADE-NOTES-v1.5.md`, and `training/train_implicit.py`.
