import numpy as np

def get_feature_columns():
    return [
        "RSI", "MACD", "MACD_Signal", "SMA_50", "SMA_200", "Bollinger_Upper",
        "Bollinger_Lower", "ADX", "Stoch_K", "Stoch_D", "VWAP",
        "Doji", "Engulfing", "Hammer", "LSTM_PRED"
    ]
