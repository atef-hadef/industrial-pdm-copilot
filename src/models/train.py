"""Entraînement du LSTM sur FD001, avec validation par moteur.

Usage : python -m src.models.train
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
    """Sépare les moteurs (et non les lignes) en entraînement / validation.

    Découper par ligne mettrait des cycles d'un même moteur des deux côtés :
    le modèle serait validé sur des moteurs qu'il a déjà vus (fuite de données).
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
    """RMSE en cycles sur tout un jeu de fenêtres (mode évaluation, sans dropout).

    Le modèle prédit la RUL divisée par cap : on remultiplie par cap pour revenir en cycles.
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
    """Entraîne le LSTM et sauvegarde le meilleur modèle (RMSE validation). Renvoie cette RMSE."""
    torch.manual_seed(SEED)

    # 1. Données : RUL plafonnée, découpage PAR MOTEUR, puis normalisation et fenêtres.
    #    Le scaler est appris sur les 80 moteurs d'entraînement seulement : la validation
    #    joue le rôle de données jamais vues, comme le test plus tard.
    full = add_rul(load_cmapss("FD001", "train"), cap=cap)
    train_df, val_df = split_units(full)
    train_df, val_df, scaler = normalize(train_df, val_df)

    X_train, y_train = map(torch.from_numpy, make_windows(train_df, window_size))
    X_val, y_val = map(torch.from_numpy, make_windows(val_df, window_size))
    print(
        f"Moteurs : {train_df['unit_number'].nunique()} entraînement / "
        f"{val_df['unit_number'].nunique()} validation"
    )
    print(f"Fenêtres : X_train {tuple(X_train.shape)} | X_val {tuple(X_val.shape)}\n")

    # Le réseau apprend la RUL divisée par cap (entre 0 et 1) : sans cela, la sortie part
    # de ~0 et doit monter jusqu'à ~125, ce qui ralentit énormément l'apprentissage.
    loader = DataLoader(TensorDataset(X_train, y_train / cap), batch_size=batch_size, shuffle=True)

    # 2. Modèle, perte, optimiseur
    model = LSTMRegressor(len(USEFUL_SENSORS), hidden_size, num_layers)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    MODELS_DIR.mkdir(exist_ok=True)
    ckpt_path = MODELS_DIR / "lstm_fd001.pt"
    best_val = float("inf")

    # 3. Boucle d'entraînement
    print(f"{'epoch':>5} | {'RMSE train':>10} | {'RMSE val':>8}")
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        # RMSE recalculée sur tout le jeu en fin d'epoch (plus juste qu'une moyenne des batchs)
        train_rmse = rmse(model, X_train, y_train, cap)
        val_rmse = rmse(model, X_val, y_val, cap)

        # On ne garde que le meilleur modèle selon la validation
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
                        "cap": cap,  # sortie du modèle × cap = RUL en cycles
                        "sensors": USEFUL_SENSORS,
                    },
                    "epoch": epoch,
                    "val_rmse": val_rmse,
                },
                ckpt_path,
            )
            marker = "  <- meilleur, sauvegardé"
        print(f"{epoch:>5} | {train_rmse:>10.2f} | {val_rmse:>8.2f}{marker}")

    # Le scaler est sauvegardé à côté : il faudra le même pour normaliser le test
    joblib.dump(scaler, MODELS_DIR / "scaler_fd001.joblib")
    print(f"\nMeilleure RMSE validation : {best_val:.2f} -> {ckpt_path.name}")
    return best_val


if __name__ == "__main__":
    train()
