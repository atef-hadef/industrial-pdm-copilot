"""Calcul des labels RUL (Remaining Useful Life) pour C-MAPSS.

Usage : python -m src.data.labeling
"""

import pandas as pd

from src.data.load import load_cmapss


def add_rul(df: pd.DataFrame, cap: int | None = 125) -> pd.DataFrame:
    """Ajoute une colonne "RUL" = cycle max du moteur - cycle courant, plafonnée à `cap`.

    Plafond (RUL piecewise-linear) : en début de vie le moteur est sain et les capteurs
    ne montrent aucune dégradation, on ne peut donc pas distinguer RUL=300 de RUL=150.
    On ramène toute RUL > cap à cap : la courbe est plate puis décroît linéairement.
    Avec cap=None, pas de plafond (RUL linéaire brute).

    À utiliser sur le jeu TRAIN uniquement : il est run-to-failure, donc le dernier
    cycle de chaque moteur est l'instant de la panne (RUL = 0 à cette ligne).
    Sur le jeu test, les séries sont tronquées avant la panne : ce calcul serait faux.
    """
    # On travaille sur une copie pour ne pas modifier le DataFrame de l'appelant
    df = df.copy()

    # Cycle maximum de chaque moteur, recopié sur toutes ses lignes
    # (transform garde la même longueur que df, contrairement à un simple groupby().max())
    max_cycle = df.groupby("unit_number")["time_in_cycles"].transform("max")

    # RUL = nombre de cycles restants avant la panne
    df["RUL"] = max_cycle - df["time_in_cycles"]

    # Plafond : toute RUL supérieure à cap est ramenée à cap
    if cap is not None:
        df["RUL"] = df["RUL"].clip(upper=cap)
    return df


if __name__ == "__main__":
    cols = ["unit_number", "time_in_cycles", "RUL"]
    train = load_cmapss("FD001", "train")
    brute = add_rul(train, cap=None)
    plafonnee = add_rul(train, cap=125)

    # Vérification 1 : la RUL maximale doit valoir le plafond
    print(f"RUL max sans plafond : {brute['RUL'].max()}")
    print(f"RUL max avec cap=125 : {plafonnee['RUL'].max()}")

    # Vérification 2 : moteur 1, avant / après plafond
    print("\nMoteur 1 - 3 premières lignes, sans plafond :")
    print(brute.loc[brute["unit_number"] == 1, cols].head(3).to_string(index=False))
    print("\nMoteur 1 - 3 premières lignes, cap=125 :")
    print(plafonnee.loc[plafonnee["unit_number"] == 1, cols].head(3).to_string(index=False))

    # Vérification 3 : nombre de lignes au plafond
    n_cap = (plafonnee["RUL"] == 125).sum()
    print(f"\nLignes au plafond (RUL == 125) : {n_cap} / {len(plafonnee)} ({n_cap / len(plafonnee):.1%})")
