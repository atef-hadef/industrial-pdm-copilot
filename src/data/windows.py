"""Fenêtres glissantes pour les modèles séquentiels (LSTM, CNN...).

Usage : python -m src.data.windows
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from src.data.labeling import add_rul
from src.data.load import load_cmapss
from src.data.preprocess import USEFUL_SENSORS, normalize


def make_windows(
    df: pd.DataFrame, window_size: int = 30, features: list[str] = USEFUL_SENSORS
) -> tuple[np.ndarray, np.ndarray]:
    """Découpe chaque moteur en fenêtres glissantes de `window_size` cycles (pas de 1).

    Renvoie :
    - X de forme (n_fenetres, window_size, n_features) ;
    - y de forme (n_fenetres,) = la RUL au DERNIER cycle de chaque fenêtre.

    Les fenêtres ne mélangent jamais deux moteurs : le découpage est fait moteur par moteur.
    Un moteur avec moins de `window_size` cycles ne produit aucune fenêtre.
    """
    if "RUL" not in df:
        raise ValueError("df doit contenir une colonne 'RUL' (voir add_rul)")

    X_parts, y_parts = [], []
    for _, unit in df.groupby("unit_number"):
        # Cycles dans l'ordre chronologique
        unit = unit.sort_values("time_in_cycles")
        values = unit[features].to_numpy(dtype=np.float32)
        if len(values) < window_size:
            continue

        # sliding_window_view donne (n_fenetres, n_features, window_size) :
        # on remet l'axe du temps avant celui des capteurs -> (n_fenetres, window_size, n_features)
        X_parts.append(sliding_window_view(values, window_size, axis=0).transpose(0, 2, 1))

        # Fenêtre i = cycles [i, i + window_size - 1] -> cible = RUL du dernier cycle
        y_parts.append(unit["RUL"].to_numpy(dtype=np.float32)[window_size - 1 :])

    return np.concatenate(X_parts), np.concatenate(y_parts)


if __name__ == "__main__":
    # Pipeline FD001 : chargement -> RUL plafonnée -> normalisation -> fenêtres
    train = add_rul(load_cmapss("FD001", "train"), cap=125)
    test = load_cmapss("FD001", "test")
    train, test, _ = normalize(train, test)

    X, y = make_windows(train, window_size=30)
    print(f"X train : {X.shape}  (n_fenetres, window_size, n_capteurs)")
    print(f"y train : {y.shape}")
    print(f"Capteurs train dans [{train[USEFUL_SENSORS].min().min():.2f}, "
          f"{train[USEFUL_SENSORS].max().max():.2f}] | test dans "
          f"[{test[USEFUL_SENSORS].min().min():.2f}, {test[USEFUL_SENSORS].max().max():.2f}]")
