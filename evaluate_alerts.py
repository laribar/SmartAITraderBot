# evaluate_alerts.py

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Carregar alertas mais recentes
alerts_folder = Path("alerts")
alerts_files = sorted(alerts_folder.glob("alerts_*.csv"), reverse=True)

if not alerts_files:
    print("❌ Nenhum arquivo de alertas encontrado.")
    exit()

latest_alerts_file = alerts_files[0]
print(f"📄 Avaliando alertas do arquivo: {latest_alerts_file.name}")

alerts = pd.read_csv(latest_alerts_file)
alerts["timestamp"] = pd.to_datetime(alerts["timestamp"])

# Observar retorno 5 candles após o alerta
results = []

for _, row in alerts.iterrows():
    symbol = row["symbol"]
    timeframe = row["timeframe"]
    timestamp = row["timestamp"]
    sinal = row["sinal"]
    preco_alerta = row["preco_atual"]

    # Obter candles futuros (até 5 candles após o alerta)
    from src.data import get_stock_data

    future_data = get_stock_data(symbol, interval=timeframe, period="7d")
    future_data["Datetime"] = pd.to_datetime(future_data["Datetime"])

    future_prices = future_data[future_data["Datetime"] > timestamp].head(5)
    if future_prices.empty:
        resultado = "Indefinido"
        retorno = None
    else:
        preco_futuro = future_prices["Close"].iloc[-1]
        retorno = (preco_futuro / preco_alerta) - 1
        resultado = "Correto" if (
            (sinal == "COMPRA" and preco_futuro > preco_alerta) or
            (sinal == "VENDA" and preco_futuro < preco_alerta)
        ) else "Errado"

    results.append({
        "timestamp": timestamp,
        "symbol": symbol,
        "timeframe": timeframe,
        "sinal": sinal,
        "preco_alerta": preco_alerta,
        "retorno_apos_5_candles": retorno,
        "resultado": resultado
    })

# Salvar resultado
results_df = pd.DataFrame(results)
eval_path = Path("alerts/evaluation_result.csv")
results_df.to_csv(eval_path, index=False)
print(f"✅ Avaliação salva em: {eval_path}")
