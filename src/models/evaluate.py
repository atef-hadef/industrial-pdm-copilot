"""Evaluation of the saved LSTM on the FD001 TEST set (RMSE + NASA score).

Usage: python -m src.models.evaluate
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window: the plot is saved as a PNG
import matplotlib.pyplot as plt
import numpy as np

from src.data.load import load_cmapss, load_rul
from src.models.predict import load_model, predict_rul

ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT / "reports"


def rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Root mean squared error, in cycles."""
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """NASA (PHM08) score: sum of asymmetric exponential penalties.

    d = predicted RUL - true RUL.
    - d < 0 (early prediction, maintenance happens too soon): exp(-d/13) - 1
    - d >= 0 (late prediction, the engine may fail before maintenance): exp(d/10) - 1
    Late predictions are penalized more (divided by 10 instead of 13). Lower is better.
    """
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def evaluate(subset: str = "FD001") -> tuple[float, float]:
    """Predict the RUL of each test engine and return (RMSE, NASA score)."""
    # 1. Model and scaler saved by src.models.train
    model, scaler, cfg = load_model()

    # 2. Last window of each test engine -> predicted RUL
    #    (normalized with the train scaler: transform only, NEVER fit)
    pred = predict_rul(load_cmapss(subset, "test"), model, scaler, cfg)
    units, y_pred = pred.index.to_numpy(), pred.to_numpy()

    # 3. True RUL, capped like the training labels (standard practice in the literature)
    y_true = load_rul(subset).loc[units].to_numpy().astype(np.float32)
    y_true = np.minimum(y_true, cfg["cap"])

    score_rmse = rmse(y_pred, y_true)
    score_nasa = nasa_score(y_pred, y_true)
    print(f"=== {subset} test set: {len(units)} engines (model from epoch {cfg['epoch']}) ===")
    print(f"RMSE:       {score_rmse:.2f} cycles")
    print(f"NASA score: {score_nasa:.1f}")

    d = y_pred - y_true
    print(f"Late predictions (d > 0): {(d > 0).sum()} | early (d < 0): {(d < 0).sum()}")
    print(f"Largest error: {d[np.abs(d).argmax()]:+.1f} cycles (engine {units[np.abs(d).argmax()]})")

    plot_predictions(y_pred, y_true, subset, score_rmse, score_nasa)
    return score_rmse, score_nasa


def plot_predictions(
    y_pred: np.ndarray, y_true: np.ndarray, subset: str, score_rmse: float, score_nasa: float
) -> Path:
    """Predicted vs. true RUL, engines sorted by increasing true RUL. Saves a PNG."""
    order = np.argsort(y_true, kind="stable")
    x = np.arange(1, len(order) + 1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)

    # True RUL as a line, predictions as dots (categorical palette, fixed order)
    ax.plot(x, y_true[order], color="#2a78d6", linewidth=2, label="True RUL (capped at 125)")
    ax.scatter(x, y_pred[order], color="#eb6834", s=22, edgecolors="white", linewidths=0.8,
               zorder=3, label="Predicted RUL (LSTM)")

    ax.set_title(f"{subset} test set: predicted vs. true RUL  (RMSE {score_rmse:.2f}, NASA score {score_nasa:.0f})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Test engines, sorted by increasing true RUL")
    ax.set_ylabel("RUL (cycles)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()

    REPORTS_DIR.mkdir(exist_ok=True)
    path = REPORTS_DIR / f"eval_{subset.lower()}.png"
    fig.savefig(path, dpi=120)
    print(f"Plot: {path.relative_to(ROOT)}")
    return path


if __name__ == "__main__":
    evaluate()
