import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.data import get_stock_data
from src.indicators import calculate_indicators
from src.train_lstm import train_lstm_model, predict_with_lstm
from src.train_xgb import train_ml_model
from src.utils import get_feature_columns
from sklearn.preprocessing import MinMaxScaler
import subprocess

ASSETS = ["BTC-USD", "ETH-USD"]
TIMEFRAMES = ["15m", "1h", "1d"]

for asset in ASSETS:
    print(f"\n🚀 Treinando modelos para {asset}...")

    for interval in TIMEFRAMES:
        print(f"\n⏱️ Timeframe: {interval} | Período: 30d")

        try:
            df = calculate_indicators(get_stock_data(asset, interval, period="30d"))

            # Normalização
            scaler = MinMaxScaler()
            df["Close_scaled"] = scaler.fit_transform(df[["Close"]])

            # Treinar modelo LSTM
            lstm_model = train_lstm_model(df)

            # Caminhos LSTM
            model_dir = Path(f"models/{asset}/{interval}/lstm")
            model_dir.mkdir(parents=True, exist_ok=True)

            model_path = model_dir / "lstm_model.keras"
            scaler_path = model_dir / "scaler.pkl"

            lstm_model.save(model_path)
            joblib.dump(scaler, scaler_path)

            print(f"✅ LSTM salvo em: {model_path}")
            print(f"✅ Scaler salvo em: {scaler_path}")

            # Previsão para gráfico
            df["LSTM_PRED"] = [
                predict_with_lstm(lstm_model, df.iloc[:i+1])
                if i >= 20 else None
                for i in range(len(df))
            ]

            plt.figure(figsize=(10, 4))
            plt.plot(df["Close"].values[-50:], label="Real")
            plt.plot(df["LSTM_PRED"].values[-50:], label="LSTM Previsto", linestyle="--")
            plt.title(f"{asset} - {interval} - Previsão com LSTM")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plot_path = model_dir / f"{asset}_{interval}_lstm_plot.png"
            plt.savefig(plot_path)
            print(f"🖼️ Plot salvo em: {plot_path}")
            plt.close()

            # Treinar modelo XGBoost
            features = get_feature_columns()
            xgb_model = train_ml_model(df, asset, interval)
            xgb_path = Path(f"models/{asset}/{interval}/xgb_model.joblib")
            joblib.dump(xgb_model, xgb_path)
            print(f"📦 XGBoost salvo em: {xgb_path}")

        except Exception as e:
            print(f"❌ Falha em {asset} [{interval}]: {e}")

# ✅ Git auto-commit e push
print("\n🚀 Enviando modelos salvos para o GitHub...")
try:
    subprocess.run(["git", "add", "--all", "models"], check=True)
    subprocess.run(["git", "commit", "-m", "feat: adiciona modelos LSTM e XGBoost"], check=True)
    subprocess.run(["git", "pull", "--rebase"], check=True)
    subprocess.run(["git", "push"], check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Falha ao subir para o GitHub: {e}")
