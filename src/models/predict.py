"""Loading of the saved LSTM and prediction of each engine's current RUL.

Used by the evaluation (src.models.evaluate) and the dashboard (src.dashboard.app).
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src.data.windows import last_windows
from src.models.lstm import LSTMRegressor

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def load_model(models_dir: Path = MODELS_DIR) -> tuple[LSTMRegressor, MinMaxScaler, dict]:
    """Reload the LSTM (in eval mode), the scaler and the config saved by training."""
    ckpt = torch.load(models_dir / "lstm_fd001.pt", weights_only=True)
    cfg = {**ckpt["config"], "epoch": ckpt["epoch"]}
    model = LSTMRegressor(cfg["n_features"], cfg["hidden_size"], cfg["num_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()  # disables dropout
    scaler = joblib.load(models_dir / "scaler_fd001.joblib")
    return model, scaler, cfg


def predict_rul(
    df: pd.DataFrame, model: LSTMRegressor, scaler: MinMaxScaler, cfg: dict
) -> pd.Series:
    """Predicted RUL (in cycles, clipped to [0, cap]) from the last window of each engine.

    df holds RAW sensor values: they are normalized here with the train scaler
    (transform only, never fit).
    """
    sensors = cfg["sensors"]
    df = df.copy()
    df[sensors] = scaler.transform(df[sensors])

    X, units = last_windows(df, cfg["window_size"], sensors)
    with torch.no_grad():
        y_pred = model(torch.from_numpy(X)).numpy() * cfg["cap"]  # output in [0, 1] -> cycles
    y_pred = np.clip(y_pred, 0, cfg["cap"])
    return pd.Series(y_pred, index=pd.Index(units, name="unit_number"), name="RUL_pred")
