import pandas as pd
import numpy as np
import ta

def calculate_indicators(data):
    data = data.copy()
    data.reset_index(drop=True, inplace=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        data[col] = data[col].astype(float)

    data["RSI"] = ta.momentum.RSIIndicator(close=data["Close"], window=14).rsi()
    data["SMA_50"] = ta.trend.SMAIndicator(close=data["Close"], window=50).sma_indicator()
    data["SMA_200"] = ta.trend.SMAIndicator(close=data["Close"], window=200).sma_indicator()

    macd = ta.trend.MACD(close=data["Close"])
    data["MACD"] = macd.macd()
    data["MACD_Signal"] = macd.macd_signal()

    bb = ta.volatility.BollingerBands(close=data["Close"], window=20)
    data["Bollinger_Upper"] = bb.bollinger_hband()
    data["Bollinger_Lower"] = bb.bollinger_lband()

    adx = ta.trend.ADXIndicator(high=data["High"], low=data["Low"], close=data["Close"], window=14)
    data["ADX"] = adx.adx()

    stoch = ta.momentum.StochasticOscillator(high=data["High"], low=data["Low"], close=data["Close"], window=14)
    data["Stoch_K"] = stoch.stoch()
    data["Stoch_D"] = stoch.stoch_signal()

    data["TP"] = (data["High"] + data["Low"] + data["Close"]) / 3
    data["VWAP"] = (data["TP"] * data["Volume"]).cumsum() / (data["Volume"].replace(0, np.nan).cumsum())
    data.drop("TP", axis=1, inplace=True)

    data["Doji"] = ((abs(data["Close"] - data["Open"]) / (data["High"] - data["Low"] + 1e-9)) < 0.1).astype(int)
    data["Engulfing"] = ((data["Open"].shift(1) > data["Close"].shift(1)) & (data["Open"] < data["Close"]) &
                          (data["Close"] > data["Open"].shift(1)) & (data["Open"] < data["Close"].shift(1))).astype(int)
    data["Hammer"] = (((data["High"] - data["Low"]) > 3 * abs(data["Open"] - data["Close"])) &
                       ((data["Close"] - data["Low"]) / (data["High"] - data["Low"] + 1e-9) > 0.6) &
                       ((data["Open"] - data["Low"]) / (data["High"] - data["Low"] + 1e-9) > 0.6)).astype(int)

    data.dropna(inplace=True)
    return data
