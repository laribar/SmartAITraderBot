import joblib
import pandas as pd
import matplotlib.pyplot as plt
from src.data import get_stock_data
from src.indicators import calculate_indicators
from src.train_lstm import predict_with_lstm
from src.utils import get_feature_columns

ASSETS = ["BTC-USD", "ETH-USD"]
TIMEFRAMES = ["15m", "1h", "1d"]

for asset in ASSETS:
    print(f"\n🔍 Previsão para {asset}")

    for interval in TIMEFRAMES:
        print(f"\n⏱️ Timeframe: {interval}")

        try:
            # Carregar dados e indicadores
            df = calculate_indicators(get_stock_data(asset, interval=interval, period="30d"))
            latest = df.iloc[-1:]
            features = get_feature_columns()

            # Carregar modelo XGBoost
            model_path = f"models/xgb_{asset}_{interval}.pkl"
            model = joblib.load(model_path)
            prediction = model.predict(latest[features])[0]
            proba = model.predict_proba(latest[features])[0][prediction]

            print(f"🤖 XGBoost sinal: {'COMPRA' if prediction==1 else 'VENDA'} ({proba*100:.2f}% confiança)")

            # Carregar e prever com LSTM
            from tensorflow.keras.models import load_model
            lstm_path = f"models/lstm_{asset}_{interval}.h5"
            lstm_model = load_model(lstm_path)
            lstm_model.window_size = 20  # Padrão do treino

            from sklearn.preprocessing import MinMaxScaler
            _, _, scaler = df["Close"].values.reshape(-1, 1), None, MinMaxScaler()
            scaler.fit(df["Close"].values.reshape(-1, 1))
            lstm_model.scaler = scaler

            lstm_pred = predict_with_lstm(lstm_model, df)
            current_price = df["Close"].iloc[-1]

            print(f"🔮 LSTM preço atual: ${current_price:.2f} | Previsão: ${lstm_pred:.2f}")

            # Plot
            plt.figure(figsize=(10, 4))
            plt.plot(df["Close"].values[-50:], label="Real")
            plt.axhline(lstm_pred, color='orange', linestyle='--', label='LSTM Previsto')
            plt.title(f"{asset} - {interval} - Previsão")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Falha em {asset} [{interval}]: {e}")
