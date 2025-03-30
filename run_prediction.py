import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.data import get_stock_data
from src.indicators import calculate_indicators
from src.train_lstm import predict_with_lstm
from src.utils import get_feature_columns
from tensorflow.keras.models import load_model
from datetime import datetime
from telegram import Bot
import json

# === CONFIGURAÇÕES ===
ASSETS = ["BTC-USD", "ETH-USD"]
TIMEFRAMES = ["15m", "1h", "1d"]
TELEGRAM_TOKEN = "COLE_SEU_TOKEN_AQUI"
TELEGRAM_CHAT_ID = "COLE_SEU_CHAT_ID_AQUI"
ALERTS_PATH = Path("data/alerts_log.csv")

# === Funções utilitárias ===
def load_models(symbol, timeframe):
    xgb_path = Path(f"models/{symbol}/{timeframe}/xgb_model.joblib")
    lstm_path = Path(f"models/{symbol}/{timeframe}/lstm/lstm_model.keras")
    scaler_path = Path(f"models/{symbol}/{timeframe}/lstm/scaler.pkl")

    xgb_model = joblib.load(xgb_path)
    lstm_model = load_model(lstm_path)
    lstm_model.window_size = 20
    lstm_model.scaler = joblib.load(scaler_path)

    return xgb_model, lstm_model

def enviar_telegram(msg):
    try:
        bot = Bot(token=TELEGRAM_TOKEN)
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        print("✅ Alerta enviado para o Telegram!")
    except Exception as e:
        print(f"❌ Erro ao enviar alerta: {e}")

def salvar_alerta(data):
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([data])
    if ALERTS_PATH.exists():
        df.to_csv(ALERTS_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(ALERTS_PATH, index=False)

# === Execução ===
for asset in ASSETS:
    print(f"\n🔍 Previsão para {asset}")

    for interval in TIMEFRAMES:
        print(f"\n⏱️ Timeframe: {interval}")

        try:
            df = calculate_indicators(get_stock_data(asset, interval, period="30d"))
            features = get_feature_columns()
            latest = df.iloc[-1:]

            xgb_model, lstm_model = load_models(asset, interval)

            # XGBoost
            xgb_pred = xgb_model.predict(latest[features])[0]
            xgb_proba = xgb_model.predict_proba(latest[features])[0][xgb_pred]
            sinal = "COMPRA" if xgb_pred == 1 else "VENDA"

            # LSTM
            lstm_pred = predict_with_lstm(lstm_model, df)
            current_price = df["Close"].iloc[-1]

            msg = (
                f"📈 [{asset} | {interval}]\n"
                f"🤖 XGBoost: {sinal} ({xgb_proba*100:.2f}% confiança)\n"
                f"🔮 LSTM: Atual ${current_price:.2f} → Previsto ${lstm_pred:.2f}"
            )
            print(msg)

            salvar_alerta({
                "timestamp": datetime.utcnow().isoformat(),
                "asset": asset,
                "timeframe": interval,
                "xgb_signal": sinal,
                "xgb_proba": round(xgb_proba, 4),
                "lstm_pred": round(lstm_pred, 2),
                "price_now": round(current_price, 2)
            })

            enviar_telegram(msg)

        except Exception as e:
            print(f"❌ Falha em {asset} [{interval}]: {e}")
