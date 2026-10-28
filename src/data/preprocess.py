"""Selection of informative sensors and min-max normalization (FD001)."""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# The 14 informative sensors identified in notebooks/01_eda.ipynb
# (sensor_1, 5, 6, 10, 16, 18, 19 are constant or near-constant in FD001)
USEFUL_SENSORS = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_8", "sensor_9", "sensor_11",
    "sensor_12", "sensor_13", "sensor_14", "sensor_15", "sensor_17", "sensor_20", "sensor_21",
]


def normalize(
    train: pd.DataFrame, test: pd.DataFrame, sensors: list[str] = USEFUL_SENSORS
) -> tuple[pd.DataFrame, pd.DataFrame, MinMaxScaler]:
    """Keep the informative sensors and scale them to [0, 1].

    The min/max are learned ONLY on the train set, then applied as-is to the test set:
    the test set simulates future data that is unknown at training time.
    Using its min/max would leak test information into the model (data leakage)
    and make the evaluation overly optimistic. Expected side effect: on the test set,
    a few values may fall slightly outside [0, 1].

    Also returns the scaler, to be reused later on new data (API).
    """
    # Columns kept: identifiers + RUL if present + informative sensors
    def select(df: pd.DataFrame) -> pd.DataFrame:
        keep = ["unit_number", "time_in_cycles"] + (["RUL"] if "RUL" in df else []) + sensors
        return df[keep].copy()

    train, test = select(train), select(test)

    # fit on the train set only, transform both
    scaler = MinMaxScaler()
    train[sensors] = scaler.fit_transform(train[sensors])
    test[sensors] = scaler.transform(test[sensors])
    return train, test, scaler
