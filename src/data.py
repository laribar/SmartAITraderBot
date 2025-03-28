import yfinance as yf
import pandas as pd

def get_stock_data(asset, interval="15m", period="700d"):
    data = yf.download(asset, period=period, interval=interval, progress=False, auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [col.split()[-1] if " " in col else col for col in data.columns]
    data = data.loc[:, ~data.columns.duplicated()]
    col_map = {}
    for col in data.columns:
        for std_col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if std_col.lower() in col.lower():
                col_map[col] = std_col
    data = data.rename(columns=col_map)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in data.columns for col in required):
        raise ValueError(f"Dados de {asset} não possuem todas as colunas necessárias.")
    return data
