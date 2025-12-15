✈️ Uçuş Gecikmesi Tahmin Sistemi
📌 Proje Hakkında
Hava durumu verilerini kullanarak uçuşların varışta gecikip gecikmeyeceğini tahmin eden makine öğrenmesi sistemi. %90 accuracy hedefliyoruz.

🔧 Teknik Detaylar
Model: XGBoost

Accuracy: %85-88 (hedef %90+)

Dataset: Kaggle - Historical Flight and Weather Data

Özellik: Leakage-free tasarım

📁 Proje Yapısı
text
flight_delay/
├── data/              # Veri dosyaları
├── models/            # Eğitilmiş modeller
├── flight_delay_system.py      # Ana sistem
└── predict_flight_delay.py     # Tahmin fonksiyonu
🚀 Nasıl Çalıştırılır?
Kaggle'dan veriyi indir

python flight_delay_system.py komutunu çalıştır

Sistem otomatik modeli eğitir ve test eder

🎯 %90 Accuracy İçin
Daha fazla veri kullan

Feature engineering geliştir

Hyperparameter tuning yap

Target threshold optimize et


