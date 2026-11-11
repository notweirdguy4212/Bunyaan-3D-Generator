**v1.4 আপডেট:** নেটিভ টেক্সট-স্পেশিয়াল ক্রস-অ্যাটেনশন, EMA ওয়েট, অ্যাসেট-আইডি ভিত্তিক ভ্যালিডেশন এবং deterministic DDIM sampling যোগ করা হয়েছে। সঠিক সক্ষমতা ও সীমাবদ্ধতার জন্য [v1.4 আর্কিটেকচার](docs/ARCHITECTURE-v1.4.md) দেখুন।

# Bunyaan — স্বাধীন জেনারেটিভ 3D মডেল

**Bunyaan হলো 3D জেনারেশন ও PBR উপকরণের জন্য একটি ওপেন-সোর্স গবেষণা প্রকল্প। এর আর্কিটেকচার, কন্ডিশনিং, ডিফিউশন, প্রশিক্ষণ, ইনফারেন্স এবং চেকপয়েন্ট ফরম্যাট Bunyaan-এর নিজস্ব। এটি অন্য কোনো জেনারেটিভ মডেল যুক্ত বা নির্ভর করে না।**

> **রিপোজিটরিতে প্রশিক্ষণযোগ্য নিজস্ব আর্কিটেকচার ও পুনরুৎপাদনযোগ্য টুল রয়েছে; প্রি-ট্রেইনড ওজন নেই। গুণমান ডেটা, কম্পিউট এবং মূল্যায়নের ওপর নির্ভর করে।**

## النطاق / Scope

3D পথ টেক্সট/ইমেজ কন্ডিশনিং ও occupancy-volume diffusion ব্যবহার করে। উপকরণ পথ পাঁচটি UV চ্যানেল তৈরি করে: RGB, roughness ও metallic।

## بيانات التدريب والحقوق

শুধু সেই অ্যাসেট ব্যবহার করুন যেগুলোর লাইসেন্স ML প্রশিক্ষণের স্পষ্ট অনুমতি দেয়। হ্যাশ, উৎস, স্বীকৃতি ও লাইসেন্স সংরক্ষণ করুন। অজানা লাইসেন্স অনুমতি নয়।

## البدء / Quick start

সম্পূর্ণ নির্দেশিকার জন্য `README.md` দেখুন। Python 3.10–3.12 ও উপযুক্ত PyTorch ইনস্টল করে `pip install -r requirements.txt` চালান। প্রশিক্ষণ ও মূল্যায়ন কমান্ডে `--help` ব্যবহার করুন।

## الترخيص

Bunyaan কোড ও ডকুমেন্টেশন Apache-2.0 লাইসেন্সে প্রকাশিত। ডেটাসেট ও নির্ভরতাগুলোর নিজস্ব লাইসেন্স প্রযোজ্য।

اقرأ ملف [`README.md`](README.md) الإنجليزي للحصول على تفاصيل البنية، والقيود، وخطوات التقييم، وخريطة الطريق، والمساهمة، والأمان.


## Bunyaan v1.5 engineering update

The native v1.5 research path adds a hierarchical 3D feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, chunked field queries, calibrated multi-view fusion, and SDF/normal/PBR/Eikonal training. This is not pretrained and has not been shown to outperform TRELLIS.2. See `docs/ARCHITECTURE-v1.5.md`, `UPGRADE-NOTES-v1.5.md`, and `training/train_implicit.py`.
