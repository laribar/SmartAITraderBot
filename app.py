import streamlit as st
import pandas as pd
from pathlib import Path
from src.data import get_stock_data
from src.indicators import calculate_indicators
from src.train_lstm import predict_with_lstm
from src.utils import get_feature_columns
from tensorflow.keras.models import load_model
import joblib

st.set_page_config(page_title="Smart AI Trader", layout="wide")
st.title("📈 Smart AI Trader - Previsões com IA")

ASSETS = ["BTC-USD", "ETH-USD"]
TIMEFRAMES = ["15m", "1h", "1d"]

@st.cache_data
def load_models(symbol, timeframe):
    xgb_path = Path(f"models/{symbol}/{timeframe}/xgb_model.joblib")
    lstm_path = Path(f"models/{symbol}/{timeframe}/lstm/lstm_model.keras")
    scaler_path = Path(f"models/{symbol}/{timeframe}/lstm/scaler.pkl")

    xgb_model = joblib.load(xgb_path)
    lstm_model = load_model(lstm_path)
    lstm_model.window_size = 20
    lstm_model.scaler = joblib.load(scaler_path)

    return xgb_model, lstm_model

asset = st.sidebar.selectbox("Escolha o ativo", ASSETS)
timeframe = st.sidebar.selectbox("Escolha o timeframe", TIMEFRAMES)

st.subheader(f"Previsão para {asset} ({timeframe})")

with st.spinner("Carregando modelos e dados..."):
    try:
        df = calculate_indicators(get_stock_data(asset, interval=timeframe, period="30d"))
        xgb_model, lstm_model = load_models(asset, timeframe)

        latest = df.iloc[-1:]
        features = get_feature_columns()

        # Previsão XGBoost
        pred = xgb_model.predict(latest[features])[0]
        proba = xgb_model.predict_proba(latest[features])[0][pred]
        sinal = "COMPRA" if pred == 1 else "VENDA"

        # Previsão LSTM
        lstm_pred = predict_with_lstm(lstm_model, df)
        current_price = df["Close"].iloc[-1]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("🔍 XGBoost Sinal", sinal, f"Confiança: {proba*100:.2f}%")
        with col2:
            st.metric("🔮 LSTM Preço Previsto", f"${lstm_pred:.2f}", f"Atual: ${current_price:.2f}")

        # Gráfico
        st.line_chart(df["Close"].tail(50), use_container_width=True)

    except Exception as e:
        st.error(f"❌ Erro ao carregar dados/modelo: {e}")
