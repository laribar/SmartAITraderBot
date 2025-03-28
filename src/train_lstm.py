import os
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def prepare_lstm_data(data, feature_col="Close", window_size=20):
    values = data[feature_col].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, window_size, 1)
    y = np.array(y)
    return X, y, scaler

def train_lstm_model(data, symbol="BTC-USD", timeframe="1h", window_size=20):
    if len(data) < window_size + 20:
        return None

    X, y, scaler = prepare_lstm_data(data, window_size=window_size)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    # ⬇️ Salvando o modelo LSTM e o scaler
    model_dir = f"models/lstm/{symbol}/{timeframe}"
    os.makedirs(model_dir, exist_ok=True)

    model_path = f"{model_dir}/lstm_model.h5"
    scaler_path = f"{model_dir}/scaler.pkl"

    model.save(model_path)
    joblib.dump({"scaler": scaler, "window_size": window_size}, scaler_path)

    print(f"📦 LSTM salvo em {model_path}")
    print(f"📦 Scaler salvo em {scaler_path}")

    # Adiciona para uso em tempo real
    model.scaler = scaler
    model.window_size = window_size
       
    print("✅ Modelo treinado com sucesso.")
    return model

def predict_with_lstm(model, data):
    values = data["Close"].values.reshape(-1, 1)
    scaled = model.scaler.transform(values)
    X_pred = np.reshape(scaled[-model.window_size:], (1, model.window_size, 1))
    pred_scaled = model.predict(X_pred, verbose=0)[0][0]
    return model.scaler.inverse_transform([[pred_scaled]])[0][0]
