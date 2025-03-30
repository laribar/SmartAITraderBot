import streamlit as st
from run_prediction import run_predictions  # ajustado para sua função principal

st.set_page_config(page_title="SmartAITraderBot", layout="wide")
st.title("🤖 Smart AI Trader Bot")
st.markdown("Visualize os sinais gerados pelos modelos XGBoost e LSTM.")

if st.button("🔮 Executar Previsões Agora"):
    with st.spinner("Executando previsões..."):
        run_predictions()
    st.success("✅ Previsões concluídas!")
