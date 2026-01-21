# Akıllı Atık Yönetimi ve Doluluk Tahmini

Bu proje çöp konteynerlerı üzerindeki sensörlerden alınan verileri göz önüne alarak konteynerların ne zaman boşaltılması gerektiğini tahmin eden bir model eğitmeyi amaçlar. Veri sızıntısını önleyen pipeline mimarisi kullanır.
## Proje Amacı

Sistemin hedefleri:

* **Operasyonel Optimizasyon:** Kritik doluluk seviyesine ulaşan konteynerleri tespit etmek.
* **Veri Güvenilirliği:** Sensörlerden gelebilecek absürt verilerin modeli bozmaması.
* **Otomatik Model Seçimi:** Farklı algoritmaları karşılaştırır. En verimlisini tercih eder.

## Veri Seti

* **Class:** Hedef değişken. Durum tahmini yapar 
* **FL_B:** Alt sensörden okunan güncel doluluk yüzdesi
* **FL_A:** Üst sensörden okunan referans doluluk yüzdesi
* **VS:** Titreşim ve hacim sensörü verisi
* **Container Type:** Konteynerin fiziksel yapısı 
* **Doluluk_Artisi:** FL_B ve FL_A farkı ile hesaplanan anlık dolum hızı

## Teknik Mimari

### 1. Veri Sızıntısını Önleme
Veri sızıntısını önlemek amacıyla pipeline kullanıldı. Kategoriler sadece train setinden öğreniliyor. Test seti bilgisi eğitimde kullanılmıyor.

### 2. Özellik Türetimi
Sadece sensör verileri kullanılmadı. `Doluluk_Artisi` özelliği türetildi. Böylece model, sadece konteynırın ne kadar dolu olduğunu değil aynı zamanda ne kadar hızlı doldugunu da biliyor.

### 3. Ön İşleme
* **Sayısal Veriler:** Eksik değerler medyan verisi ile dolduruldu, böylece sensör hatalarından kaynaklanan absürt değerlerin etkisi azaldı.
* **Kategorik Veriler:** OneHotEncoder kullanılır. Bu sayede yeni konteyner türleri kodun çalışmasını durdurmaz.

## Model Performansı

Proje 4 farklı algoritmayı test eder. 

| Model | Doğruluk (Accuracy) | Yorum |
| :--- | :--- | :--- |
| **Random Forest** | **%96.01** | Karmaşık ilişkileri en iyi yakalar. |
| Gradient Boosting | %91.70 | Eğitim maliyeti yüksek. |
| KNN (k=5) | %89.41 | Kirli verilerde performansı düşüktür. |
| Logistic Regression | %87.15 | Doğrusal olmayan ilişkilerde yetersiz kalır. |

Random Forest en yüksek doğruluğu verdi. Nihai model olarak seçildi.

## Analiz Sonuçları

**Pivot Tablo Analizi:**
Her konteyner tipi aynı hızda dolmaz. Diamond tipi konteynerler Mixed atık türünde %70 üzeri doluluk oranına sahiptir.

**Hata Analizi:**
Model dolu kutuya boş denmesini engeller, böylece kritik hataların önüne geçer.

**Özellik Önem Düzeyleri:**
Model karar verirken şunlara odaklanır:
1. FL_B (Anlık doluluk)
2. Doluluk_Artisi (Türetilen özellik)
3. VS (Hacim verisi)

## Grafikler
<img width="1400" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/e54ade90-8dc9-437e-8351-f10996304d9f" />
<img width="1000" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/5630e433-1b68-4f41-89f0-4644a09cd0cd" />

