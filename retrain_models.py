import os
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from src.data import get_stock_data
from src.indicators import calculate_indicators
from src.train_xgb import train_ml_model
from src.train_lstm import train_lstm_model, predict_with_lstm

ASSETS = ["BTC-USD", "ETH-USD"]
TIMEFRAMES = [
    {"interval": "15m", "period": "30d"},
    {"interval": "1h", "period": "90d"},
    {"interval": "1d", "period": "1000d"}
]

# Caminho absoluto para salvar os modelos
BASE_DIR = os.getcwd()  # /content/SmartAITraderBot
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

for asset in ASSETS:
    print(f"\n🚀 Treinando modelos para {asset}...")

    for tf in TIMEFRAMES:
        interval = tf["interval"]
        period = tf["period"]

        print(f"\n⏱️ Timeframe: {interval} | Período: {period}")

        try:
            df = calculate_indicators(get_stock_data(asset, interval, period))

            # XGBoost
            model = train_ml_model(df, verbose=True)
            if model:
                xgb_path = os.path.join(MODELS_DIR, f"xgb_{asset}_{interval}.pkl")
                joblib.dump(model, xgb_path)
                print(f"✅ XGBoost salvo em {xgb_path}")
            else:
                print("⚠️ Modelo XGBoost não treinado.")

            # LSTM
            lstm_model = train_lstm_model(df)
            if lstm_model:
                lstm_path = os.path.join(MODELS_DIR, f"lstm_{asset}_{interval}.h5")
                lstm_model.save(lstm_path)
                print(f"✅ LSTM salvo em {lstm_path}")
            else:
                print("⚠️ Modelo LSTM não treinado.")

            # Visualização simples
            if lstm_model:
                last_days = 60
                df_plot = df.tail(last_days).copy()
                predicted_prices = [predict_with_lstm(lstm_model, df_plot.iloc[:i+1])
                                    if i >= 20 else None for i in range(len(df_plot))]
                df_plot["LSTM_PRED"] = predicted_prices

                plt.figure(figsize=(10, 4))
                plt.plot(df_plot["Close"].values, label="Real")
                plt.plot(df_plot["LSTM_PRED"].values, label="LSTM Previsto")
                plt.title(f"{asset} - {interval} - Previsão com LSTM")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.show()

        except Exception as e:
            print(f"❌ Erro ao processar {asset} [{interval}]: {e}")

print("\n✅ Treinamento finalizado para todos os ativos e timeframes.")
