"""RUL (Remaining Useful Life) labeling for C-MAPSS.

Usage: python -m src.data.labeling
"""

import pandas as pd

from src.data.load import load_cmapss


def add_rul(df: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    """Add a "RUL" column = engine's max cycle - current cycle, capped at `cap`.

    Cap (piecewise-linear RUL): early in life the engine is healthy and the sensors
    show no degradation, so RUL=300 cannot be told apart from RUL=150.
    Any RUL > cap is set to cap: the target is flat, then decreases linearly.
    With cap=None, no cap is applied (raw linear RUL).

    Use on the TRAIN set only: it is run-to-failure, so the last cycle of each
    engine is the failure time (RUL = 0 on that row).
    On the test set, series are truncated before failure: this computation would be wrong.
    """
    # Work on a copy so the caller's DataFrame is not modified
    df = df.copy()

    # Max cycle of each engine, broadcast to all of its rows
    # (transform keeps the same length as df, unlike a plain groupby().max())
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")

    # RUL = number of cycles left before failure
    df["RUL"] = max_cycle - df["time_in_cycles"]

    # Cap: any RUL above cap is set to cap
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


if __name__ == "__main__":
    cols = ["unit_number", "time_in_cycles", "RUL"]
    train = load_cmapss("FD001", "train")
    brute = add_rul(train, cap=None)
    plafonnee = add_rul(train, cap=125)

    # Check 1: the max RUL must equal the cap
    print(f"Max RUL without cap: {brute['RUL'].max()}")
    print(f"Max RUL with cap=125: {plafonnee['RUL'].max()}")

    # Check 2: engine 1, before / after capping
    print("\nEngine 1 - first 3 rows, no cap:")
    print(brute.loc[brute["unit_number"] == 1, cols].head(3).to_string(index=False))
    print("\nEngine 1 - first 3 rows, cap=125:")
    print(plafonnee.loc[plafonnee["unit_number"] == 1, cols].head(3).to_string(index=False))

    # Check 3: number of rows at the cap
    n_cap = (plafonnee["RUL"] == 125).sum()
    print(f"\nRows at the cap (RUL == 125): {n_cap} / {len(plafonnee)} ({n_cap / len(plafonnee):.1%})")
