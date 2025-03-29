import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from src.data import get_stock_data
from src.indicators import calculate_indicators
from src.train_lstm import predict_with_lstm
from src.utils import get_feature_columns
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import csv
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

# 📁 Arquivo onde os alertas serão salvos
datahora = datetime.now().strftime("%Y-%m-%d_%H-%M")
alerts_file = Path(f"alerts/alerts_{datahora}.csv")
alerts_file.parent.mkdir(exist_ok=True)

with alerts_file.open("w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["timestamp", "symbol", "timeframe", "model", "sinal", "confianca", "preco_atual", "previsao_lstm"])

    for asset in ASSETS:
        print(f"\n🔍 Previsão para {asset}")

        for interval in TIMEFRAMES:
            print(f"\n⏱️ Timeframe: {interval}")

            try:
                df = calculate_indicators(get_stock_data(asset, interval=interval, period="30d"))
                features = get_feature_columns()

                # LSTM
                lstm_path = Path(f"models/{asset}/{interval}/lstm/lstm_model.h5")
                scaler_path = Path(f"models/{asset}/{interval}/lstm/scaler.pkl")

                if not lstm_path.exists() or not scaler_path.exists():
                    raise FileNotFoundError("Modelo LSTM ou scaler não encontrado.")

                lstm_model = load_model(lstm_path)
                lstm_model.window_size = 20
                lstm_model.scaler = joblib.load(scaler_path)

                df["LSTM_PRED"] = [
                    predict_with_lstm(lstm_model, df.iloc[:i+1])
                    if i >= lstm_model.window_size else None
                    for i in range(len(df))
                ]

                df.dropna(inplace=True)
                if df.empty:
                    raise ValueError("Dados insuficientes após aplicar janela do LSTM.")

                latest = df.iloc[-1:]

                # XGBoost
                model = load_xgb_model(asset, interval)
                prediction = model.predict(latest[features])[0]
                proba = model.predict_proba(latest[features])[0][prediction]

                signal = "COMPRA" if prediction == 1 else "VENDA"
                print(f"🤖 XGBoost sinal: {signal} ({proba*100:.2f}% confiança)")

                # LSTM preço
                lstm_pred = latest["LSTM_PRED"].values[0]
                current_price = latest["Close"].values[0]
                print(f"🔮 LSTM preço atual: ${current_price:.2f} | Previsão: ${lstm_pred:.2f}")

                # Plot
                plot_path = Path(f"models/{asset}/{interval}/{asset}_{interval}_lstm_plot.png")
                plt.figure(figsize=(10, 4))
                plt.plot(df["Close"].values[-50:], label="Real")
                plt.axhline(lstm_pred, color='orange', linestyle='--', label='LSTM Previsto')
                plt.title(f"{asset} - {interval} - Previsão")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                print(f"🖼️ Plot salvo em: {plot_path}")

                # Salvando alerta
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"), asset, interval, "XGBoost", signal,
                    round(proba * 100, 2), round(current_price, 2), round(lstm_pred, 2)
                ])

            except Exception as e:
                print(f"❌ Falha em {asset} [{interval}]: {e}")

print(f"\n✅ Alertas salvos em: {alerts_file}")

# 🚀 Subir para o GitHub
try:
    subprocess.run(["git", "pull"], check=True)
    subprocess.run(["git", "add", str(alerts_file)], check=True)
    subprocess.run(["git", "commit", "-m", "feat: adiciona novo alerta gerado automaticamente"], check=True)
    subprocess.run(["git", "push"], check=True)
except subprocess.CalledProcessError as e:
    print(f"❌ Falha ao subir para o GitHub: {e}")

# ✅ Subir também o relatório de avaliação, se existir
eval_file = Path("alerts/evaluation_result.csv")
if eval_file.exists():
    try:
        subprocess.run(["git", "add", str(eval_file)], check=True)
        subprocess.run(["git", "commit", "-m", "chore: adiciona avaliação de performance dos alertas"], check=True)
        subprocess.run(["git", "push"], check=True)
        print("📈 Avaliação de performance enviada para o GitHub.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Falha ao subir avaliação para o GitHub: {e}")
