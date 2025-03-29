import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.data import get_stock_data
from src.indicators import calculate_indicators
from src.train_lstm import predict_with_lstm
from src.utils import get_feature_columns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import subprocess

# ✅ Função utilitária para carregar o XGBoost
def load_xgb_model(symbol, timeframe):
    paths = [
        Path(f"models/{symbol}/{timeframe}/xgb_model.joblib"),
        Path(f"models/xgb_{symbol}_{timeframe}.pkl"),
        Path(f"models/{symbol}_{timeframe}_xgb_model.joblib"),
    ]

    for path in paths:
        if path.exists():
            print(f"✅ Modelo XGBoost carregado: {path}")
            return joblib.load(path)

    raise FileNotFoundError(f"❌ Modelo XGBoost não encontrado para {symbol} [{timeframe}]")

ASSETS = ["BTC-USD", "ETH-USD"]
TIMEFRAMES = ["15m", "1h", "1d"]

for asset in ASSETS:
    print(f"\n🔍 Previsão para {asset}")

    for interval in TIMEFRAMES:
        print(f"\n⏱️ Timeframe: {interval}")

        try:
            # Carregar dados e indicadores
            df = calculate_indicators(get_stock_data(asset, interval=interval, period="30d"))
            features = get_feature_columns()

            # Carregar e prever com LSTM
            lstm_path = Path(f"models/{asset}/{interval}/lstm/lstm_model.h5")
            scaler_path = Path(f"models/{asset}/{interval}/lstm/scaler.pkl")

            if not lstm_path.exists() or not scaler_path.exists():
                raise FileNotFoundError("Modelo LSTM ou scaler não encontrado.")

            lstm_model = load_model(lstm_path)
            lstm_model.window_size = 20
            lstm_model.scaler = joblib.load(scaler_path)

            # Previsão LSTM incremental
            df["LSTM_PRED"] = [
                predict_with_lstm(lstm_model, df.iloc[:i+1])
                if i >= lstm_model.window_size else None
                for i in range(len(df))
            ]

            df.dropna(inplace=True)
            latest = df.iloc[-1:]

            # Carregar modelo XGBoost
            model = load_xgb_model(asset, interval)
            prediction = model.predict(latest[features])[0]
            proba = model.predict_proba(latest[features])[0][prediction]

            print(f"🤖 XGBoost sinal: {'COMPRA' if prediction==1 else 'VENDA'} ({proba*100:.2f}% confiança)")

            # Previsão LSTM atual
            lstm_pred = latest["LSTM_PRED"].values[0]
            current_price = latest["Close"].values[0]
            print(f"🔮 LSTM preço atual: ${current_price:.2f} | Previsão: ${lstm_pred:.2f}")

            # Plot
            plt.figure(figsize=(10, 4))
            plt.plot(df["Close"].values[-50:], label="Real")
            plt.plot(df["LSTM_PRED"].values[-50:], label="LSTM Previsto", linestyle='--')
            plt.title(f"{asset} - {interval} - Previsão")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plot_path = Path(f"models/{asset}/{interval}/{asset}_{interval}_lstm_plot.png")
            plt.savefig(plot_path)
            print(f"🖼️ Plot salvo em: {plot_path}")
            plt.close()

        except Exception as e:
            print(f"❌ Falha em {asset} [{interval}]: {e}")

# ✅ Git auto-commit e push
print("\n🚀 Enviando modelos salvos para o GitHub...")
try:
    subprocess.run(["git", "add", "models"], check=True)
    subprocess.run(["git", "commit", "-m", "feat: adiciona modelos treinados automaticamente"], check=True)
    subprocess.run(["git", "push"], check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Falha ao subir para o GitHub: {e}")
