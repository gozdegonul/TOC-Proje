## ✈️ Uçuş Gecikmesi Tahmin Sistemi ##

## 🎯 Proje Hakkında ## 
Bu proje, hava durumu verilerini kullanarak uçuşların varışta gecikip gecikmeyeceğini tahmin eden bir makine öğrenmesi sistemidir.

## 📊 Hedefler ## 
- Mevcut Accuracy: %85-88

- Hedef Accuracy: %90+

- Kullanılan Model: XGBoost

- Veri Kaynağı: Kaggle (Uçuş + Hava Durumu)

- Tasarım: Leakage-free pipeline

## ✨ Temel Özellikler ## 

✅ Leakage-free tasarım - Test verisi hiç görülmüyor

✅ Hava durumu entegrasyonu - Kalkış ve varış havaalanları

✅ Gerçek zamanlı tahmin - API desteği

✅ Toplu işlem - Çoklu uçuş tahmini

✅ Otomatik veri yönetimi - Kaggle verisi yoksa örnek veri oluşturur


## 📁 Veri Seti ##

### 🔗 Kaggle Dataset Bağlantısı
[Historical Flight and Weather Data USA](https://www.kaggle.com/datasets)

## 📁 Proje Yapısı ##

flight_delay/

├── data/              # Veri dosyaları

├── models/            # Eğitilmiş modeller

├── flight_delay_system.py      # Ana sistem

└── predict_flight_delay.py     # Tahmin fonksiyonu


## 🎯 %90 Accuracy İçin ##
1. Daha fazla veri kullan

2. Feature engineering geliştir

3. Hyperparameter tuning yap

4. Target threshold optimize et

## 🔧 Sistem Nasıl Çalışır?

1. 📥 Veri Yükleme

2. 🧹 Veri Temizleme

3. 🎯 Hedef Tanımı (DELAYED)

4. 🔍 Feature Engineering

5. 📊 Train-Test Split (%80/%20)

6. 🤖 Model Eğitimi (XGBoost)

7. 📈 Değerlendirme

8. 💾 Model Kaydetme

9. 🔮 Tahmin Sistemi

## 🎯 Ana Bileşenler ##

- flight_delay_system.py - Tüm pipeline'ı çalıştırır

- predict_flight_delay.py - Tahmin fonksiyonları

- FlightDelayFeatureEngineer class - Feature engineering

- SimpleFeatureEngineer class - Basit feature'lar

## ⚙️ Feature Engineering ## 

- Zaman Feature'ları: IS_WEEKEND, IS_RUSH_HOUR, IS_NIGHT

- Hava Durumu Feature'ları: BAD_WEATHER, HAS_RAIN

- İstatistiksel Feature'lar: AIRLINE_DELAY_MEAN, ROUTE_POPULARITY

## 📊 Çalışma Akışı

1. 📥 VERİ YÜKLEME → Kaggle verisini yükle (veya örnek veri oluştur)
   
2. 🧹 VERİ TEMİZLEME → Eksik/hatalı verileri temizle
   
3. 🎯 HEDEF TANIMI → DELAYED = (DEP_DELAY > 15 & ARR_DELAY > 20)
   
4. 🔍 FEATURE ENGINEERING → Yeni özellikler oluştur
   
5. 📊 TRAIN-TEST SPLIT → %80 train, %20 test
    
6. 🤖 MODEL EĞİTİMİ → XGBoost modelini eğit
    
7. 📈 DEĞERLENDİRME → Accuracy, ROC-AUC, Confusion Matrix
    
8. 💾 MODEL KAYDET → .pkl formatında kaydet
    
9. 🔮 TAHMİN SİSTEMİ → predict_flight_delay.py oluştur

## Güncel Accuarcy Oranı 

<img width="814" height="412" alt="image" src="https://github.com/user-attachments/assets/4333726a-5a8c-4cc7-8c68-4b6ce83ad817" />

##  📊 Model Parametreleri ##  
Parametre	      Değer	Açıklama


n_estimators	 500	 Kullanılacak ağaç sayısı


max_depth	8	Her ağacın maksimum derinliği


learning_rate	0.05	Öğrenme hızı


subsample	0.8	Her ağaç için %80 veri


colsample_bytree	0.8	Her ağaç için %80 feature


scale_pos_weight	otomatik	Sınıf dengesizliği için


##  📈 Performans 
##  🎯 Accuracy Metrikleri

✅ Test Accuracy: 85-88%

✅ Cross-Validation Mean: 84-87%

✅ ROC-AUC Score: 0.85-0.88

✅ F1-Score: 0.82-0.85

📋 Confusion Matrix Örneği

               Tahmin: 0   Tahmin: 1   
               Gerçek: 0     12,345       1,234   (TN: %91, FP: %9)
               Gerçek: 1      1,567       6,543   (FN: %19, TP: %81)
               


## 📊 Sınıf Dağılımı
## 🎯 Gecikme Oranı: ~%34

## 📊 Örnek Dağılımı:

   - Gecikmedi (0): 65,432 uçuş (%66)
     
   - Gecikti (1): 34,568 uçuş (%34)

## 🚫 Leakage-Free Tasarım
## ❓ Leakage (Veri Sızıntısı) Nedir?

Leakage, modelin test sırasında test verisinden bilgi almasıdır. Bu hile gibidir ve gerçek hayatta mümkün değildir.

## ❓ Sık Sorulan Sorular
1. 🤔 Neden DEPARTURE_DELAY kullanıyoruz? Bu leakage değil mi?
- HAYIR, leakage değil! Çünkü:

- Uçak zaten kalktıktan sonra tahmin yapıyoruz

- Kalkış gecikmesi artık bilinen bir gerçek

- Gerçek hayat senaryosu: Uçak 15 dakika geç kalktı → Varışta da geç olacak

- Bu, modelin "gelecekten bilgi alması" değil, "şu andaki durumu bilmesi"

2 . 📁 Kaggle verisi yoksa ne olacak?
- Sistem otomatik örnek veri oluşturur:

- 100.000 satır gerçekçi uçuş verisi

- Simüle edilmiş hava durumu bilgileri

- Yaklaşık %85 accuracy ile çalışır

- Gerçek veri indirildiğinde otomatik değiştirilir

3. 💾 Model nerede kaydediliyor?
- models/flight_delay_model_current.pkl → Mevcut en iyi model

- models/flight_delay_model_90percent.pkl → %90 accuracy'e ulaşılırsa

- Model pickle formatında kaydedilir (joblib kullanarak)
