"""Chargement des fichiers bruts NASA C-MAPSS.

Usage : python -m src.data.load
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SPLITS = ("train", "test")

ID_COLUMNS = ["unit_number", "time_in_cycles"]
SETTING_COLUMNS = [f"op_setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS


def _check_subset(subset: str) -> None:
    if subset not in SUBSETS:
        raise ValueError(f"subset inconnu : {subset!r} (attendu : {SUBSETS})")


def load_cmapss(
    subset: str = "FD001", split: str = "train", raw_dir: Path = RAW_DIR
) -> pd.DataFrame:
    """Lit train_<subset>.txt ou test_<subset>.txt et renvoie un DataFrame propre.

    Les fichiers n'ont pas d'en-tête et finissent par des espaces en trop,
    qui produisent des colonnes vides : on les supprime avant de nommer les 26 colonnes.
    """
    _check_subset(subset)
    if split not in SPLITS:
        raise ValueError(f"split inconnu : {split!r} (attendu : {SPLITS})")

    path = Path(raw_dir) / f"{split}_{subset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.dropna(axis=1, how="all")

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{path.name} : {df.shape[1]} colonnes lues, {len(COLUMNS)} attendues"
        )
    df.columns = COLUMNS
    df[ID_COLUMNS] = df[ID_COLUMNS].astype(int)
    return df


def load_rul(subset: str = "FD001", raw_dir: Path = RAW_DIR) -> pd.Series:
    """Lit RUL_<subset>.txt : la RUL réelle de chaque moteur du jeu de test, en fin de série."""
    _check_subset(subset)
    path = Path(raw_dir) / f"RUL_{subset}.txt"
    rul = pd.read_csv(path, sep=r"\s+", header=None).dropna(axis=1, how="all")
    rul = rul.iloc[:, 0].astype(int)
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="unit_number")
    return rul.rename("RUL")


def main(subset: str = "FD001") -> None:
    train = load_cmapss(subset, "train")
    test = load_cmapss(subset, "test")
    rul = load_rul(subset)

    print(f"=== C-MAPSS {subset} ===")
    print(f"train : {train.shape}  ({train['unit_number'].nunique()} moteurs)")
    print(f"test  : {test.shape}  ({test['unit_number'].nunique()} moteurs)")
    print(f"RUL   : {rul.shape}")
    print("\nAperçu train :")
    print(train.head())


if __name__ == "__main__":
    main()
