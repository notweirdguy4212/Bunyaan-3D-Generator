**v1.4 अपडेट:** नेटिव टेक्स्ट-स्पेशियल क्रॉस-अटेंशन, EMA वेट्स, एसेट-ID आधारित वैलिडेशन और deterministic DDIM sampling जोड़े गए हैं। सटीक क्षमताओं और सीमाओं के लिए [v1.4 आर्किटेक्चर](docs/ARCHITECTURE-v1.4.md) देखें।

# Bunyaan — स्वतंत्र जनरेटिव 3D मॉडल

**Bunyaan 3D निर्माण और PBR सामग्री के लिए एक ओपन-सोर्स शोध परियोजना है। इसकी आर्किटेक्चर, कंडीशनिंग, डिफ्यूजन, प्रशिक्षण, इन्फरेंस और चेकपॉइंट प्रारूप Bunyaan के अपने हैं। यह किसी अन्य जनरेटिव मॉडल को शामिल या आवश्यक नहीं करता।**

> **रिपॉज़िटरी में प्रशिक्षित की जा सकने वाली मूल आर्किटेक्चर और पुनरुत्पादित टूल हैं, लेकिन पहले से प्रशिक्षित वज़न नहीं हैं। गुणवत्ता डेटा, कंप्यूट और मूल्यांकन पर निर्भर करती है।**

## النطاق / Scope

3D पथ टेक्स्ट/इमेज कंडीशनिंग और occupancy-volume diffusion का उपयोग करता है। सामग्री पथ पाँच UV चैनल बनाता है: RGB, roughness और metallic।

## بيانات التدريب والحقوق

केवल उन्हीं एसेट का उपयोग करें जिनका लाइसेंस ML प्रशिक्षण की स्पष्ट अनुमति देता हो। हैश, स्रोत, श्रेय और लाइसेंस रिकॉर्ड सुरक्षित रखें। अज्ञात लाइसेंस अनुमति नहीं है।

## البدء / Quick start

पूरी मार्गदर्शिका के लिए `README.md` देखें। Python 3.10–3.12 और संगत PyTorch स्थापित करें, फिर `pip install -r requirements.txt` चलाएँ। प्रशिक्षण और मूल्यांकन कमांड के लिए `--help` उपयोग करें।

## الترخيص

Bunyaan कोड और दस्तावेज़ Apache-2.0 के अंतर्गत हैं। डेटासेट और निर्भरताओं पर उनके अपने लाइसेंस लागू रहते हैं।

اقرأ ملف [`README.md`](README.md) الإنجليزي للحصول على تفاصيل البنية، والقيود، وخطوات التقييم، وخريطة الطريق، والمساهمة، والأمان.


## Bunyaan v1.5 engineering update

The native v1.5 research path adds a hierarchical 3D feature encoder, arbitrary-coordinate implicit SDF/PBR decoder, chunked field queries, calibrated multi-view fusion, and SDF/normal/PBR/Eikonal training. This is not pretrained and has not been shown to outperform TRELLIS.2. See `docs/ARCHITECTURE-v1.5.md`, `UPGRADE-NOTES-v1.5.md`, and `training/train_implicit.py`.
