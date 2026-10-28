"""LSTM training on FD001, with an engine-level validation split.

Usage: python -m src.models.train
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.labeling import add_rul
from src.data.load import load_cmapss
from src.data.preprocess import USEFUL_SENSORS, normalize
from src.data.windows import make_windows
from src.models.lstm import LSTMRegressor

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
SEED = 42


def split_units(
    df: pd.DataFrame, val_frac: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the engines (not the rows) into training / validation.

    Splitting by row would put cycles of the same engine on both sides:
    the model would be validated on engines it has already seen (data leakage).
    """
    units = df["unit_number"].unique()
    rng = np.random.default_rng(seed)
    val_units = rng.choice(units, size=round(len(units) * val_frac), replace=False)
    is_val = df["unit_number"].isin(val_units)
    return df[~is_val], df[is_val]


@torch.no_grad()
def rmse(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, cap: int, batch_size: int = 1024
) -> float:
    """RMSE in cycles over a whole set of windows (eval mode, no dropout).

    The model predicts RUL divided by cap: multiply back by cap to get cycles.
    """
    model.eval()
    preds = torch.cat([model(X[i : i + batch_size]) for i in range(0, len(X), batch_size)]) * cap
    return torch.sqrt(torch.mean((preds - y) ** 2)).item()


def train(
    epochs: int = 20,
    batch_size: int = 256,
    lr: float = 1e-3,
    window_size: int = 30,
    cap: int = 125,
    hidden_size: int = 64,
    num_layers: int = 2,
) -> float:
    """Train the LSTM and save the best model (validation RMSE). Returns that RMSE."""
    torch.manual_seed(SEED)

    # 1. Data: capped RUL, split BY ENGINE, then normalization and windowing.
    #    The scaler is fitted on the 80 training engines only: the validation set
    #    plays the role of unseen data, like the test set later on.
    full = add_rul(load_cmapss("FD001", "train"), cap=cap)
    train_df, val_df = split_units(full)
    train_df, val_df, scaler = normalize(train_df, val_df)

    X_train, y_train = map(torch.from_numpy, make_windows(train_df, window_size))
    X_val, y_val = map(torch.from_numpy, make_windows(val_df, window_size))
    print(
        f"Engines: {train_df['unit_number'].nunique()} training / "
        f"{val_df['unit_number'].nunique()} validation"
    )
    print(f"Windows: X_train {tuple(X_train.shape)} | X_val {tuple(X_val.shape)}\n")

    # The network learns RUL divided by cap (between 0 and 1): otherwise its output starts
    # near 0 and has to climb to ~125, which slows learning down dramatically.
    loader = DataLoader(TensorDataset(X_train, y_train / cap), batch_size=batch_size, shuffle=True)

    # 2. Model, loss, optimizer
    model = LSTMRegressor(len(USEFUL_SENSORS), hidden_size, num_layers)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_path = MODELS_DIR / "lstm_fd001.pt"
    best_val = float("inf")

    # 3. Training loop
    print(f"{'epoch':>5} | {'RMSE train':>10} | {'RMSE val':>8}")
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        # RMSE recomputed on the full set at the end of each epoch (more accurate than a batch average)
        train_rmse = rmse(model, X_train, y_train, cap)
        val_rmse = rmse(model, X_val, y_val, cap)

        # Keep only the best model according to validation
        marker = ""
        if val_rmse < best_val:
            best_val = val_rmse
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": {
                        "n_features": len(USEFUL_SENSORS),
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "window_size": window_size,
                        "cap": cap,  # model output × cap = RUL in cycles
                        "sensors": USEFUL_SENSORS,
                    },
                    "epoch": epoch,
                    "val_rmse": val_rmse,
                },
                ckpt_path,
            )
            marker = "  <- best, saved"
        print(f"{epoch:>5} | {train_rmse:>10.2f} | {val_rmse:>8.2f}{marker}")

    # The scaler is saved alongside: the same one is needed to normalize the test set
    joblib.dump(scaler, MODELS_DIR / "scaler_fd001.joblib")
    print(f"\nBest validation RMSE: {best_val:.2f} -> {ckpt_path.name}")
    return best_val


if __name__ == "__main__":
    train()
