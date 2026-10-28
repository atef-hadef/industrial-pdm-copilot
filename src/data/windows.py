"""Sliding windows for sequence models (LSTM, CNN...).

Usage: python -m src.data.windows
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
    """Split each engine into sliding windows of `window_size` cycles (stride 1).

    Returns:
    - X of shape (n_windows, window_size, n_features);
    - y of shape (n_windows,) = the RUL at the LAST cycle of each window.

    Windows never mix two engines: the split is done engine by engine.
    An engine with fewer than `window_size` cycles produces no window.
    """
    if "RUL" not in df:
        raise ValueError("df must contain a 'RUL' column (see add_rul)")

    X_parts, y_parts = [], []
    for _, unit in df.groupby("unit_number"):
        # Cycles in chronological order
        unit = unit.sort_values("time_in_cycles")
        values = unit[features].to_numpy(dtype=np.float32)
        if len(values) < window_size:
            continue

        # sliding_window_view returns (n_windows, n_features, window_size):
        # move the time axis before the sensor axis -> (n_windows, window_size, n_features)
        X_parts.append(sliding_window_view(values, window_size, axis=0).transpose(0, 2, 1))

        # Window i = cycles [i, i + window_size - 1] -> target = RUL of the last cycle
        y_parts.append(unit["RUL"].to_numpy(dtype=np.float32)[window_size - 1 :])

    return np.concatenate(X_parts), np.concatenate(y_parts)


def last_windows(
    df: pd.DataFrame, window_size: int = 30, features: list[str] = USEFUL_SENSORS
) -> tuple[np.ndarray, np.ndarray]:
    """Return the LAST window of each engine (test set case, truncated before failure).

    Returns:
    - X of shape (n_engines, window_size, n_features);
    - the matching engine numbers, in the same order.
    """
    X_parts, units = [], []
    for unit_number, unit in df.groupby("unit_number"):
        values = unit.sort_values("time_in_cycles")[features].to_numpy(dtype=np.float32)
        if len(values) < window_size:
            raise ValueError(f"engine {unit_number}: {len(values)} cycles < window_size={window_size}")
        # The last window_size observed cycles
        X_parts.append(values[-window_size:])
        units.append(unit_number)
    return np.stack(X_parts), np.array(units)


if __name__ == "__main__":
    # FD001 pipeline: loading -> capped RUL -> normalization -> windows
    train = add_rul(load_cmapss("FD001", "train"), cap=125)
    test = load_cmapss("FD001", "test")
    train, test, _ = normalize(train, test)

    X, y = make_windows(train, window_size=30)
    print(f"X train: {X.shape}  (n_windows, window_size, n_sensors)")
    print(f"y train: {y.shape}")
    print(f"Train sensors in [{train[USEFUL_SENSORS].min().min():.2f}, "
          f"{train[USEFUL_SENSORS].max().max():.2f}] | test in "
          f"[{test[USEFUL_SENSORS].min().min():.2f}, {test[USEFUL_SENSORS].max().max():.2f}]")
