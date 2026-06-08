# Adult Veri Seti ile Random Forest Kullanarak Gelir Tahmini

Bu proje, nüfus sayımı verilerine dayanarak bir bireyin yıllık gelirinin 50K\$'ı aşıp aşmadığını tahmin etmek amacıyla **Veri Madenciliği (Data Mining)** tekniklerinin uygulanmasını içerir. Proje **Python** dilinde geliştirilmiş olup, **UCI Adult Veri Seti** kullanılarak **Random Forest (Rastgele Orman)** sınıflandırma algoritması ile modellenmiştir.

## 📌 Proje Özeti

Gelir düzeyi gibi sosyoekonomik niteliklerin tahmini, demografik analizlerin temel taşlarından biridir. Bu çalışmada, ham veriden yola çıkılarak model optimizasyonuna kadar uzanan veri bilimi süreçlerinin tamamı uçtan uca uygulanmıştır:

### Öne Çıkanlar
* **Kullanılan Algoritma:** Random Forest Sınıflandırıcı (Topluluk Öğrenmesi - Ensemble Learning)
* **Veri Seti:** UCI Adult Veri Seti (~48,842 örnek; yaş, eğitim, meslek, medeni durum, cinsiyet gibi 14 öznitelik içerir)
* **Hedef Değişken (Target):** `income` (>50K veya <=50K)
* **Elde Edilen Başarı (Accuracy):** **~%85.83**

---

## 🛠️ Veri Madenciliği İş Hattı (Pipeline)

Proje, veri kalitesini artırmak ve yüksek tahmin performansı elde etmek amacıyla şu yapılandırılmış adımları takip eder:

### 1. Keşifçi Veri Analizi (EDA)
* Veri dağılımlarının, eksik verilerin ve veri setinde `?` işareti ile belirtilen belirsiz değerlerin analiz edilmesi.
* Demografik özelliklerin (örneğin eğitim seviyesi, sermaye kazancı vb.) gelir grupları üzerindeki etkisinin görselleştirilmesi.

### 2. Veri Ön İşleme & Temizleme (Preprocessing)
* **Eksik Verilerin Yönetimi:** `workclass`, `occupation` ve `native-country` sütunlarındaki bilinmeyen (`?`) değerlerin mod (en sık tekrar eden değer) yöntemiyle doldurulması veya temizlenmesi.
* **Kategorik Verilerin Dönüştürülmesi:** Sözel/kategorik değişkenlerin, modelin işleyebileceği sayısal formata getirilmesi (One-Hot Encoding veya Label Encoding).
* **Öznitelik Ölçekleme (Scaling):** `age` ve `capital-gain` gibi sürekli değişkenlerin, ağaç bölünmelerini daha dengeli hale getirmek için standartlaştırılması (Standard Scaler).

### 3. Model Eğitimi ve Değerlendirme
* Veri setinin %80 eğitim, %20 test olacak şekilde ayrılması.
* **Random Forest** modelinin eğitilmesi; aşırı öğrenmeyi (overfitting) engellemek adına `n_estimators` (ağaç sayısı) ve `max_depth` (maksimum derinlik) gibi hiperparametrelerin optimize edilmesi.
* Model performansının Doğruluk (Accuracy), Keskinlik (Precision), Duyarlılık (Recall) ve Karışıklık Matrisi (Confusion Matrix) ile ölçülmesi.

---

## 📊 Sonuçlar

Eğitilen Random Forest modeli, test veri seti üzerinde oldukça kararlı ve başarılı bir performans göstermiştir:
* **Genel Doğruluk Oranı (Accuracy):** **%85.83**
* Model, özellikle gelir düzeyi <=50K olan çoğunluk sınıfı yüksek keskinlikle tahmin ederken; hiperparametre ince ayarlarından sonra >50K gelir grubunda da başarılı sonuçlar vermiştir.

---

## 🚀 Kurulum ve Çalıştırma

### Gerekli Kütüphaneler
Projeyi çalıştırmadan önce aşağıdaki Python kütüphanelerinin yüklü olduğundan emin olun:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn