import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from src.train_lstm import train_lstm_model, predict_with_lstm
from src.utils import get_feature_columns

def train_ml_model(data, verbose=False):
    df = data.copy()
    df["Future_Close"] = df["Close"].shift(-5)
    df["Future_Return"] = df["Future_Close"] / df["Close"] - 1
    df = df[(df["Future_Return"] > 0.015) | (df["Future_Return"] < -0.015)].copy()
    df["Signal"] = np.where(df["Future_Return"] > 0.015, 1, 0)

    try:
        lstm_model = train_lstm_model(df)
        lstm_preds = [np.nan if i < 20 else predict_with_lstm(lstm_model, df.iloc[:i+1]) for i in range(len(df))]
        df["LSTM_PRED"] = lstm_preds
    except:
        df["LSTM_PRED"] = np.nan

    df.dropna(inplace=True)
    X = df[get_feature_columns()]
    y = df["Signal"]

    if len(np.unique(y)) < 2:
        return None

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    scale_pos_weight = len(y_train[y_train == 0]) / max(1, len(y_train[y_train == 1]))

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        learning_rate=0.05,
        early_stopping_rounds=10,
        eval_metric="logloss",
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    model.validation_score = report
    return model
