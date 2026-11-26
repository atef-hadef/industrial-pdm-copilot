"""Unit tests for the data pipeline: RUL labeling, scaling and sliding windows."""

import numpy as np
import pandas as pd
import pytest

from src.data.labeling import add_rul
from src.data.preprocess import USEFUL_SENSORS, normalize
from src.data.windows import last_windows, make_windows


def make_df(cycles_per_engine: dict[int, int], seed: int = 0) -> pd.DataFrame:
    """Synthetic run-to-failure data with the C-MAPSS column names."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit, n_cycles in cycles_per_engine.items():
        for cycle in range(1, n_cycles + 1):
            row = {"unit_number": unit, "time_in_cycles": cycle}
            row.update({s: rng.random() for s in USEFUL_SENSORS})
            rows.append(row)
    return pd.DataFrame(rows)


class TestAddRul:
    def test_rul_is_zero_at_last_cycle_of_each_engine(self):
        df = add_rul(make_df({1: 10, 2: 7}), cap=None)
        last = df.sort_values("time_in_cycles").groupby("unit_number").tail(1)
        assert (last["RUL"] == 0).all()

    def test_rul_decreases_by_one_each_cycle(self):
        df = add_rul(make_df({1: 10}), cap=None)
        assert df["RUL"].tolist() == list(range(9, -1, -1))

    def test_cap_limits_rul_and_keeps_end_of_life_values(self):
        df = add_rul(make_df({1: 200}), cap=125)
        assert df["RUL"].max() == 125
        assert df["RUL"].iloc[-1] == 0

    def test_input_dataframe_is_not_modified(self):
        raw = make_df({1: 5})
        add_rul(raw)
        assert "RUL" not in raw.columns


class TestNormalize:
    def test_scaler_is_fitted_on_train_only(self):
        train = make_df({1: 20}, seed=1)
        test = make_df({2: 20}, seed=2)
        test[USEFUL_SENSORS] = test[USEFUL_SENSORS] * 10  # far outside the train range
        train_n, test_n, scaler = normalize(train, test)
        # train is scaled to [0, 1]
        assert train_n[USEFUL_SENSORS].min().min() == pytest.approx(0.0)
        assert train_n[USEFUL_SENSORS].max().max() == pytest.approx(1.0)
        # test is NOT rescaled to [0, 1]: its out-of-range values prove no leakage
        assert test_n[USEFUL_SENSORS].max().max() > 1.0
        assert np.allclose(scaler.data_min_, train[USEFUL_SENSORS].min().to_numpy())

    def test_keeps_only_ids_rul_and_informative_sensors(self):
        train = add_rul(make_df({1: 5}))
        test = make_df({2: 5})
        train_n, test_n, _ = normalize(train, test)
        assert list(train_n.columns) == ["unit_number", "time_in_cycles", "RUL"] + USEFUL_SENSORS
        assert "RUL" not in test_n.columns


class TestWindows:
    def test_shapes_and_target_is_rul_of_last_cycle(self):
        df = add_rul(make_df({1: 40}), cap=None)
        X, y = make_windows(df, window_size=30)
        assert X.shape == (11, 30, len(USEFUL_SENSORS))
        assert y.shape == (11,)
        assert y[0] == df["RUL"].iloc[29]
        assert y[-1] == 0

    def test_windows_never_span_two_engines(self):
        df = add_rul(make_df({1: 35, 2: 32}), cap=None)
        X, y = make_windows(df, window_size=30)
        assert len(X) == (35 - 29) + (32 - 29)

    def test_short_engine_produces_no_window(self):
        df = add_rul(make_df({1: 10, 2: 31}), cap=None)
        X, _ = make_windows(df, window_size=30)
        assert len(X) == 2

    def test_missing_rul_column_raises(self):
        with pytest.raises(ValueError, match="RUL"):
            make_windows(make_df({1: 40}))

    def test_last_windows_returns_one_window_per_engine(self):
        X, units = last_windows(make_df({3: 31, 5: 45}), window_size=30)
        assert X.shape == (2, 30, len(USEFUL_SENSORS))
        assert list(units) == [3, 5]

    def test_last_windows_rejects_too_short_engine(self):
        with pytest.raises(ValueError):
            last_windows(make_df({1: 10}), window_size=30)
