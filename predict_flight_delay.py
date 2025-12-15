def predict_flight_delay(new_data, model_path='models/flight_delay_model_current.pkl'):
    """
    Yeni uçuş verisi için gecikme tahmini yapar

    Parameters:
    -----------
    new_data : pandas DataFrame veya dict
        Yeni uçuş verisi. Aşağıdaki sütunları içermeli:
        - MONTH, DAY, DAY_OF_WEEK
        - AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT
        - DISTANCE, DEP_HOUR, DEPARTURE_DELAY
        - İsteğe bağlı hava durumu sütunları

    Returns:
    --------
    dict : Tahmin sonuçları
        - prediction: 0 (gecikme yok) veya 1 (gecikme var)
        - probability: Gecikme olasılığı
        - confidence: Güven seviyesi
    """

    import pandas as pd
    import joblib
    import os

    # Modeli yükle
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    model_data = joblib.load(model_path)
    model = model_data['model']
    feature_engineer = model_data['feature_engineer']
    threshold = model_data.get('threshold', 0.5)

    # Veriyi DataFrame'e çevir
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])

    # Feature engineering uygula
    X_new = feature_engineer.transform(new_data)

    # Tahmin yap
    probability = model.predict_proba(X_new)[:, 1][0]
    prediction = 1 if probability >= threshold else 0

    # Güven seviyesi
    confidence = probability if prediction == 1 else 1 - probability

    return {
        'prediction': prediction,
        'probability': float(probability),
        'confidence': float(confidence),
        'threshold': float(threshold)
    }

# Örnek kullanım:
example_data = {
    'MONTH': 7,
    'DAY': 15,
    'DAY_OF_WEEK': 3,
    'AIRLINE': 'AA',
    'ORIGIN_AIRPORT': 'JFK',
    'DESTINATION_AIRPORT': 'LAX',
    'DISTANCE': 2475,
    'DEP_HOUR': 14,
    'DEPARTURE_DELAY': 25,
    'ORIGIN_TEMPERATURE': 22.5,
    'ORIGIN_WIND_SPEED': 12.3,
    'ORIGIN_VISIBILITY': 8.2
}

result = predict_flight_delay(example_data)
print(f"Tahmin sonucu: {result}")
