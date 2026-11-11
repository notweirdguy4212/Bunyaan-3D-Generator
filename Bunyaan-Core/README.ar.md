**تحديث v1.4:** تمت إضافة انتباه متقاطع أصلي بين النص والسمات المكانية، وأوزان EMA، والتحقق حسب هوية الأصل، وأخذ عينات DDIM حتمي. راجع [معمارية v1.4](docs/ARCHITECTURE-v1.4.md) لمعرفة القدرات والقيود الدقيقة.

# Bunyaan — نموذج ثلاثي الأبعاد توليدي مستقل

**Bunyaan مشروع بحثي مفتوح المصدر لتوليد الأصول ثلاثية الأبعاد ومواد PBR. البنية والتكييف والانتشار والتدريب والاستدلال وتنسيقات نقاط الحفظ مطوّرة داخل Bunyaan. لا يدمج المشروع نموذجاً توليدياً آخر ولا يعتمد عليه.**

> **يتضمن المستودع بنيات أصلية قابلة للتدريب وأدوات قابلة لإعادة الإنتاج، لكنه لا يتضمن أوزاناً مدرّبة مسبقاً. تعتمد الجودة على البيانات والموارد الحاسوبية والتقييم.**

## النطاق / Scope

يستخدم مسار الأشكال تكييف النص والصورة وانتشار حقل الإشغال. وينتج مسار المواد خمس قنوات UV: RGB والخشونة والمعدنية.

## بيانات التدريب والحقوق

استخدم الأصول التي تسمح تراخيصها صراحةً بالتدريب الآلي فقط. احتفظ بالبصمات ومعلومات المصدر والنسب والترخيص. الترخيص المجهول لا يعني الإذن.

## البدء / Quick start

راجع `README.md` للدليل الكامل. ثبّت Python 3.10–3.12 وإصدار PyTorch المتوافق، ثم نفّذ `pip install -r requirements.txt`. استخدم `--help` لأوامر التدريب والتقييم.

## الترخيص

شفرة Bunyaan ووثائقه مرخصة بموجب Apache-2.0. وتظل مجموعات البيانات والتبعيات خاضعة لتراخيصها الخاصة.

اقرأ ملف [`README.md`](README.md) الإنجليزي للحصول على تفاصيل البنية، والقيود، وخطوات التقييم، وخريطة الطريق، والمساهمة، والأمان.


## Bunyaan v1.5 engineering update

The native v1.5 research path adds a hierarchical 3D feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, chunked field queries, calibrated multi-view fusion, and SDF/normal/PBR/Eikonal training. This is not pretrained and has not been shown to outperform TRELLIS.2. See `docs/ARCHITECTURE-v1.5.md`, `UPGRADE-NOTES-v1.5.md`, and `training/train_implicit.py`.
