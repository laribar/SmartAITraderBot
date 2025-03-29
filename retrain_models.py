import os
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
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

# Caminho absoluto para o repositório clonado no GitHub (dentro do Colab)
GITHUB_REPO_DIR = Path("/content/SmartAITraderBot")
MODELS_DIR = GITHUB_REPO_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

for asset in ASSETS:
    print(f"\n🚀 Treinando modelos para {asset}...")

    for tf in TIMEFRAMES:
        interval = tf["interval"]
        period = tf["period"]

        print(f"\n⏱️ Timeframe: {interval} | Período: {period}")

        try:
            # Coleta e indicadores
            df = calculate_indicators(get_stock_data(asset, interval, period))

            # Treinamento XGBoost
            model = train_ml_model(df, symbol=asset, timeframe=interval, verbose=True)
            if model:
                model_dir = MODELS_DIR / asset / interval
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, model_dir / "xgb_model.joblib")
                print(f"✅ XGBoost salvo em {model_dir / 'xgb_model.joblib'}")
            else:
                print("⚠️ Modelo XGBoost não treinado.")

            # Treinamento LSTM
            lstm_model = train_lstm_model(df)
            if lstm_model:
                lstm_dir = MODELS_DIR / asset / interval / "lstm"
                lstm_dir.mkdir(parents=True, exist_ok=True)
                lstm_model.save(lstm_dir / "lstm_model.h5")
                joblib.dump(lstm_model.scaler, lstm_dir / "scaler.pkl")
                print(f"✅ LSTM salvo em {lstm_dir / 'lstm_model.h5'}")
                print(f"✅ Scaler salvo em {lstm_dir / 'scaler.pkl'}")
            else:
                print("⚠️ Modelo LSTM não treinado.")

            # Plot previsão LSTM (visual)
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
                plt.savefig(model_dir / f"{asset}_{interval}_lstm_plot.png")
                plt.close()

                # Avaliação simples
                real = df_plot["Close"].values[20:]
                pred = df_plot["LSTM_PRED"].dropna().values
                if len(pred) == len(real):
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    mae = mean_absolute_error(real, pred)
                    mse = mean_squared_error(real, pred)
                    print(f"📊 Avaliação LSTM: MAE = {mae:.2f}, MSE = {mse:.2f}")

        except Exception as e:
            print(f"❌ Erro ao processar {asset} [{interval}]: {e}")

print("\n✅ Treinamento finalizado para todos os ativos e timeframes.")

# 🚀 Commit automático para o GitHub
print("\n🚀 Enviando modelos salvos para o GitHub...")
os.system("git config --global user.email 'barbarotolarissa@gmail.com'")
os.system("git config --global user.name 'Larissa Barbaroto'")
os.chdir(GITHUB_REPO_DIR)
os.system("git add models/")
os.system("git commit -m 'feat: adiciona modelos treinados automaticamente'")
os.system("git push origin main")
