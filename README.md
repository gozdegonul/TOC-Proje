## ✈️ Uçuş Gecikmesi Tahmin Sistemi ##

## 🎯 Proje Hakkında ## 
Bu proje, hava durumu verilerini kullanarak uçuşların varışta gecikip gecikmeyeceğini tahmin eden bir makine öğrenmesi sistemidir.

## 📊 Hedefler ## 
Mevcut Accuracy: %85-88

Hedef Accuracy: %90+

Kullanılan Model: XGBoost

Veri Kaynağı: Kaggle (Uçuş + Hava Durumu)

Tasarım: Leakage-free pipeline

## ✨ Temel Özellikler ## 

✅ Leakage-free tasarım - Test verisi hiç görülmüyor

✅ Hava durumu entegrasyonu - Kalkış ve varış havaalanları

✅ Gerçek zamanlı tahmin - API desteği

✅ Toplu işlem - Çoklu uçuş tahmini

✅ Otomatik veri yönetimi - Kaggle verisi yoksa örnek veri oluşturur


## 📁 Veri Seti ##

### 🔗 Kaggle Dataset Bağlantısı
[Historical Flight and Weather Data USA](https://www.kaggle.com/datasets)

## 🔧 Teknik Detaylar ##
Model: XGBoost

Accuracy: %85-88 (hedef %90+)

Dataset: Kaggle - Historical Flight and Weather Data

Özellik: Leakage-free tasarım

## 📁 Proje Yapısı ##

text

flight_delay/

├── data/              # Veri dosyaları

├── models/            # Eğitilmiş modeller

├── flight_delay_system.py      # Ana sistem

└── predict_flight_delay.py     # Tahmin fonksiyonu


##🚀 Nasıl Çalıştırılır? ## 
Kaggle'dan veriyi indir

python flight_delay_system.py komutunu çalıştır

Sistem otomatik modeli eğitir ve test eder

## 🎯 %90 Accuracy İçin ##
Daha fazla veri kullan

Feature engineering geliştir

Hyperparameter tuning yap

Target threshold optimize et

🔧 Sistem Nasıl Çalışır?

📥 Veri Yükleme

🧹 Veri Temizleme

🎯 Hedef Tanımı (DELAYED)

🔍 Feature Engineering

📊 Train-Test Split (%80/%20)

🤖 Model Eğitimi (XGBoost)

📈 Değerlendirme

💾 Model Kaydetme

🔮 Tahmin Sistemi

## 🎯 Ana Bileşenler ##
flight_delay_system.py - Tüm pipeline'ı çalıştırır

predict_flight_delay.py - Tahmin fonksiyonları

FlightDelayFeatureEngineer class - Feature engineering

SimpleFeatureEngineer class - Basit feature'lar

