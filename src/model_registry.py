# src/model_registry.py
import os
import joblib
from datetime import datetime


def save_model_with_version(model, asset, interval, model_type="xgb"):
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"models/{model_type}_{asset}_{interval}_{timestamp}.pkl"
    joblib.dump(model, filename)
    print(f"✅ Modelo salvo com versão: {filename}")
    return filename


def save_lstm_with_version(model, asset, interval):
    from tensorflow.keras.models import save_model

    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"models/lstm_{asset}_{interval}_{timestamp}.h5"
    save_model(model, filename)
    print(f"✅ LSTM salvo com versão: {filename}")
    return filename


def list_model_versions(asset=None, interval=None, model_type=None):
    models = os.listdir("models")
    filtered = [f for f in models if f.endswith(('.pkl', '.h5'))]
    if asset:
        filtered = [f for f in filtered if asset in f]
    if interval:
        filtered = [f for f in filtered if interval in f]
    if model_type:
        filtered = [f for f in filtered if f.startswith(model_type)]
    return sorted(filtered, reverse=True)


def load_latest_model(asset, interval, model_type="xgb"):
    from tensorflow.keras.models import load_model as keras_load_model
    import joblib

    versions = list_model_versions(asset=asset, interval=interval, model_type=model_type)
    if not versions:
        raise FileNotFoundError(f"Nenhum modelo encontrado para {asset} {interval} ({model_type})")

    latest_path = os.path.join("models", versions[0])
    print(f"📦 Carregando modelo mais recente: {latest_path}")

    if model_type == "lstm":
        return keras_load_model(latest_path)
    else:
        return joblib.load(latest_path)
