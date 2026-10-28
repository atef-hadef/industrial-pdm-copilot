"""Sélection des capteurs utiles et normalisation min-max (FD001)."""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Les 14 capteurs informatifs repérés dans notebooks/01_eda.ipynb
# (sensor_1, 5, 6, 10, 16, 18, 19 sont constants ou quasi constants en FD001)
USEFUL_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9", "sensor_11",
    "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]


def normalize(
    train: pd.DataFrame, test: pd.DataFrame, sensors: list[str] = USEFUL_SENSORS
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Garde les capteurs utiles et les ramène dans [0, 1].

    Le min/max est appris UNIQUEMENT sur le train, puis appliqué tel quel au test :
    le test simule des données futures qu'on ne connaît pas au moment de l'entraînement.
    Utiliser ses min/max ferait fuiter de l'information du test vers le modèle (data leakage)
    et rendrait l'évaluation trop optimiste. Conséquence normale : sur le test, quelques
    valeurs peuvent sortir légèrement de [0, 1].

    Renvoie aussi le scaler, à réutiliser plus tard pour les nouvelles données (API).
    """
    # Colonnes conservées : identifiants + RUL si elle existe + capteurs utiles
    def select(df: pd.DataFrame) -> pd.DataFrame:
        keep = ["unit_number", "time_in_cycles"] + (["RUL"] if "RUL" in df else []) + sensors
        return df[keep].copy()

    train, test = select(train), select(test)

    # fit sur le train seulement, transform sur les deux
    scaler = MinMaxScaler()
    train[sensors] = scaler.fit_transform(train[sensors])
    test[sensors] = scaler.transform(test[sensors])
    return train, test, scaler
