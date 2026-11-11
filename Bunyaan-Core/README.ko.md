**v1.4 업데이트:** 네이티브 텍스트-공간 교차 어텐션, EMA 가중치, 에셋 ID 기반 검증, 결정론적 DDIM 샘플링을 추가했습니다. 정확한 기능과 한계는 [v1.4 아키텍처](docs/ARCHITECTURE-v1.4.md)를 참조하세요.

# Bunyaan — 독립형 생성 3D 모델

**Bunyaan은 3D 생성과 PBR 머티리얼을 위한 오픈 소스 연구 프로젝트입니다. 아키텍처, 조건화, 확산, 학습, 추론 및 체크포인트 형식은 Bunyaan 자체 구현입니다. 다른 생성 모델을 포함하거나 필요로 하지 않습니다.**

> **저장소에는 학습 가능한 자체 아키텍처와 재현 가능한 도구가 있지만 사전 학습 가중치는 없습니다. 품질은 데이터, 연산 자원 및 평가에 따라 달라집니다.**

## النطاق / Scope

3D 경로는 텍스트/이미지 조건화와 점유 볼륨 확산을 사용합니다. 머티리얼 경로는 UV 5채널(RGB, roughness, metallic)을 생성합니다.

## بيانات التدريب والحقوق

머신러닝 학습을 명시적으로 허용하는 라이선스의 에셋만 사용하세요. 해시, 출처, 저작자 표시 및 라이선스를 보존하세요. 라이선스가 불명확하면 허용된 것으로 간주하지 마세요.

## البدء / Quick start

전체 안내는 `README.md`를 참조하세요. Python 3.10–3.12와 호환 PyTorch를 설치한 뒤 `pip install -r requirements.txt`를 실행합니다. 학습 및 평가 명령에 `--help`를 사용하세요.

## الترخيص

Bunyaan 코드와 문서는 Apache-2.0입니다. 데이터셋과 종속성은 각각의 라이선스를 따릅니다.

اقرأ ملف [`README.md`](README.md) الإنجليزي للحصول على تفاصيل البنية، والقيود، وخطوات التقييم، وخريطة الطريق، والمساهمة، والأمان.


## Bunyaan v1.5 engineering update

The native v1.5 research path adds a hierarchical 3D feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, chunked field queries, calibrated multi-view fusion, and SDF/normal/PBR/Eikonal training. This is not pretrained and has not been shown to outperform TRELLIS.2. See `docs/ARCHITECTURE-v1.5.md`, `UPGRADE-NOTES-v1.5.md`, and `training/train_implicit.py`.
