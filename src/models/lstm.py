"""Modèle LSTM de prédiction de RUL."""

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM qui lit une fenêtre de capteurs et prédit la RUL au dernier cycle.

    Entrée : (batch, window_size, n_features) - Sortie : (batch,)
    """

    def __init__(
        self, n_features: int = 14, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2
    ):
        super().__init__()
        # batch_first=True : les tenseurs sont (batch, temps, capteurs), comme nos fenêtres
        # Le dropout s'applique entre les couches LSTM (inactif s'il n'y a qu'une couche)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Tête de régression : état caché final -> 1 valeur de RUL
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # On ne garde que la sortie du dernier pas de temps : elle résume toute la fenêtre
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)
