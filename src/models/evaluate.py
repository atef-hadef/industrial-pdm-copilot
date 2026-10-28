"""Évaluation du LSTM sauvegardé sur le jeu TEST FD001 (RMSE + score NASA).

Usage : python -m src.models.evaluate
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre : le graphique est enregistré en PNG
import matplotlib.pyplot as plt
import numpy as np

from src.data.load import load_cmapss, load_rul
from src.models.predict import load_model, predict_rul

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Racine de l'erreur quadratique moyenne, en cycles."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Score NASA (PHM08) : somme des pénalités exponentielles, asymétrique.

    d = RUL prédite - RUL vraie.
    - d < 0 (prédiction précoce, on intervient trop tôt) : exp(-d/13) - 1
    - d >= 0 (prédiction tardive, le moteur peut casser avant l'intervention) : exp(d/10) - 1
    Le retard est plus pénalisé (division par 10 au lieu de 13). Plus le score est bas, mieux c'est.
    """
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate(subset: str = "FD001") -> tuple[float, float]:
    """Prédit la RUL de chaque moteur de test et renvoie (RMSE, score NASA)."""
    # 1. Modèle et scaler sauvegardés par src.models.train
    model, scaler, cfg = load_model()

    # 2. Dernière fenêtre de chaque moteur de test -> RUL prédite
    #    (normalisation avec le scaler du train : transform seulement, JAMAIS fit)
    pred = predict_rul(load_cmapss(subset, "test"), model, scaler, cfg)
    units, y_pred = pred.index.to_numpy(), pred.to_numpy()

    # 3. Vraie RUL, plafonnée comme les labels d'entraînement (pratique de la littérature)
    y_true = load_rul(subset).loc[units].to_numpy().astype(np.float32)
    y_true = np.minimum(y_true, cfg["cap"])

    score_rmse = rmse(y_pred, y_true)
    score_nasa = nasa_score(y_pred, y_true)
    print(f"=== Test {subset} : {len(units)} moteurs (modèle de l'epoch {cfg['epoch']}) ===")
    print(f"RMSE       : {score_rmse:.2f} cycles")
    print(f"Score NASA : {score_nasa:.1f}")

    d = y_pred - y_true
    print(f"Prédictions tardives (d > 0) : {(d > 0).sum()} | précoces (d < 0) : {(d < 0).sum()}")
    print(f"Plus grosse erreur : {d[np.abs(d).argmax()]:+.1f} cycles (moteur {units[np.abs(d).argmax()]})")

    plot_predictions(y_pred, y_true, subset, score_rmse, score_nasa)
    return score_rmse, score_nasa


def plot_predictions(
    y_pred: np.ndarray, y_true: np.ndarray, subset: str, score_rmse: float, score_nasa: float
) -> Path:
    """RUL prédite vs RUL vraie, moteurs triés par RUL vraie croissante. Enregistre un PNG."""
    order = np.argsort(y_true, kind="stable")
    x = np.arange(1, len(order) + 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)

    # Vraie RUL en trait, prédiction en points (palette catégorielle, ordre fixe)
    ax.plot(x, y_true[order], color="#2a78d6", linewidth=2, label="RUL vraie (plafonnée à 125)")
    ax.scatter(x, y_pred[order], color="#eb6834", s=22, edgecolors="white", linewidths=0.8,
               zorder=3, label="RUL prédite (LSTM)")

    ax.set_title(f"Test {subset} : RUL prédite vs vraie  (RMSE {score_rmse:.2f}, score NASA {score_nasa:.0f})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Moteurs de test, triés par RUL vraie croissante")
    ax.set_ylabel("RUL (cycles)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"eval_{subset.lower()}.png"
    fig.savefig(path, dpi=120)
    print(f"Graphique : {path.relative_to(ROOT)}")
    return path


if __name__ == "__main__":
    evaluate()
