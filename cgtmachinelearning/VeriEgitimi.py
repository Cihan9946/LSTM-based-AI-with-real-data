import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Veriyi yükleme ve gereksiz sütunları kaldırma
data = pd.read_excel("yeni_data.xlsx")
data = data.drop(["id", "temp", "w_status"], axis=1)

# 'date' sütununu datetime formatına çevirme
data['date_v'] = pd.to_datetime(data['date_v'])

# 'date' sütununu Unix zaman damgasına çevirme
data['unix_time'] = data['date_v'].apply(lambda x: int((x - datetime(1970, 1, 1)).total_seconds()))

# Gereksiz sütunu ve orijinal tarih sütununu kaldırma
data = data.drop(["date_v"], axis=1)

# Veriyi ölçeklendirme
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Zaman serisi formatına dönüştürme
def create_time_series_data(data, window_size):
    X, y = [], []

    for i in range(len(data) - window_size):
        # Pencere boyutu kadar veriyi seçme
        window = data[i:i+window_size, :]
        target = data[i + window_size, 0]  # Örneğin, "yukh" sütununu hedef olarak alıyoruz

        X.append(window)
        y.append(target)

    return np.array(X), np.array(y)

# Pencere boyutunu belirleme
window_size = 10

# Giriş ve hedef verilerini oluşturma
X, y = create_time_series_data(data_scaled, window_size)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Modeli oluşturma
model = Sequential()
model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50, activation='relu', return_sequences=True))
model.add(LSTM(units=50, activation='relu'))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Modeli eğitme
model.fit(X_train, y_train, epochs=50, batch_size=64, validation_data=(X_test, y_test))

#MODEL KAYDETME
model.save('cgt_model.h5')

# Test seti üzerinde tahmin yapma
test_predictions = model.predict(X_test)
r2_test = r2_score(y_test, test_predictions)

print(f'Modelin Test R-kare skoru: {r2_test}')
