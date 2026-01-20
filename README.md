# Smart Bin: Akıllı Atık Yönetimi ve Doluluk Tahmini

Bu proje IoT tabanlı çöp konteynerlerinden alınan sensör verilerini analiz eder. Konteynerlerin boşaltılma gerekliliğini tahmin eder. Uçtan uca bir makine öğrenmesi hattı sunar.

Veri sızıntısını önleyen pipeline mimarisi kullanır. Otomatik model karşılaştırma süreçlerini içerir.

## Proje Amacı

Bu sistem şu hedeflere ulaşır:

* **Operasyonel Optimizasyon:** Kritik doluluk seviyesine ulaşan konteynerleri tespit eder.
* **Gürbüz Modelleme:** Eksik sensör verilerine karşı bir yapı kurar.
* **Otomatik Model Seçimi:** Farklı algoritmaları yarıştırır. En iyi modeli otomatik seçer.

## Veri Seti

Veri seti konteynerlerin fiziksel özelliklerini ve anlık sensör ölçümlerini içerir.

* **Class:** Hedef değişken. Durum tahmini yapar 
* **FL_B:** Alt sensörden okunan güncel doluluk yüzdesi
* **FL_A:** Üst sensörden okunan referans doluluk yüzdesi
* **VS:** Titreşim ve hacim sensörü verisi
* **Container Type:** Konteynerin fiziksel yapısı 
* **Doluluk_Artisi:** FL_B ve FL_A farkı ile hesaplanan anlık dolum hızı

## Teknik Mimari

### 1. Veri Sızıntısını Önleme
Geleneksel yöntemler veriyi önce doldurur sonra böler. Bu yanlıştır. Veri işleme adımları Pipeline içine gömülüdür. Test verisi eğitim sürecinden izole kalır.

### 2. Özellik Mühendisliği
Sadece ham sensör verileri kullanılmaz. İş bilgisi sürece dahil edilir. `Doluluk_Artisi` özelliği türetilmiştir. Model konteynerin dolma hızını matematiksel olarak algılar.

### 3. Gelişmiş Ön İşleme
* **Sayısal Veriler:** Medyan stratejisi kullanılır. Sensör hatalarından kaynaklanan uç değerlerin etkisi kırılır.
* **Kategorik Veriler:** OneHotEncoder kullanılır. Yeni konteyner türleri kodun çalışmasını durdurmaz.

## Model Performansı

Proje 4 farklı algoritmayı test eder. 5-Katlı Çapraz Doğrulama yöntemi kullanılır.

| Model | Doğruluk (Accuracy) | Yorum |
| :--- | :--- | :--- |
| **Random Forest** | **%96.01** | Karmaşık ilişkileri en iyi yakalar. |
| Gradient Boosting | %91.70 | Başarılıdır ancak eğitim maliyeti yüksektir. |
| KNN (k=5) | %89.41 | Gürültülü verilerde performansı düşüktür. |
| Logistic Regression | %87.15 | Doğrusal olmayan ilişkilerde yetersiz kalır. |

Random Forest en yüksek doğruluğu verir. Varyansa karşı dirençlidir. Nihai model olarak seçilmiştir.

## Analiz Sonuçları

**Pivot Tablo Analizi:**
Her konteyner tipi aynı hızda dolmaz. Diamond tipi konteynerler Mixed atık türünde %70 üzeri doluluk oranına sahiptir.

**Hata Analizi:**
Model yanlış negatifleri minimize eder. Dolu kutuya boş denmesini engeller.

**Özellik Önem Düzeyleri:**
Model karar verirken şunlara odaklanır:
1. FL_B (Anlık doluluk)
2. Doluluk_Artisi (Türetilen özellik)
3. VS (Hacim verisi)

## Grafikler
<img width="1400" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/e54ade90-8dc9-437e-8351-f10996304d9f" />
<img width="1000" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/5630e433-1b68-4f41-89f0-4644a09cd0cd" />

