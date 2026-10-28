"""LSTM model for RUL prediction."""

import torch
from torch import nn


class LSTMRegressor(nn.Module):
    """LSTM that reads a window of sensor readings and predicts the RUL at its last cycle.

    Input: (batch, window_size, n_features) - Output: (batch,)
    """

    def __init__(
        self, n_features: int = 14, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2
    ):
        super().__init__()
        # batch_first=True: tensors are (batch, time, sensors), like our windows
        # Dropout is applied between LSTM layers (inactive with a single layer)
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Regression head: final hidden state -> 1 RUL value
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Keep only the output of the last time step: it summarizes the whole window
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)
