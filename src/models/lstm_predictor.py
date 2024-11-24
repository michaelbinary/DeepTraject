# src/models/lstm_predictor.py
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from utils.types import VehicleState
from utils.constants import DEFAULT_LSTM_HIDDEN_SIZE, DEFAULT_LSTM_NUM_LAYERS, DEFAULT_LSTM_DROPOUT

class LSTMPredictor(nn.Module):
    """LSTM-based prediction model with attention mechanism"""

    def __init__(self,
                 input_size=2,
                 hidden_size=DEFAULT_LSTM_HIDDEN_SIZE,
                 num_layers=DEFAULT_LSTM_NUM_LAYERS,
                 dropout=DEFAULT_LSTM_DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.history: List[VehicleState] = []

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)

        lstm_out, _ = self.lstm(x, (h0, c0))
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)

        return self.fc(context)

    def update(self, state: VehicleState):
        """Update model with new state"""
        self.history.append(state)
        if len(self.history) > 10:
            sequence = self._prepare_sequence()
            if sequence is not None:
                self.train()
                optimizer = torch.optim.Adam(self.parameters())
                optimizer.zero_grad()

                pred = self(sequence)
                target = torch.tensor([self.history[-1].position]).float()

                loss = nn.MSELoss()(pred, target)
                loss.backward()
                optimizer.step()

    def predict(self) -> Tuple[float, float]:
        """Make a prediction"""
        self.eval()
        sequence = self._prepare_sequence()
        if sequence is None:
            return self.history[-1].position if self.history else (0, 0)

        with torch.no_grad():
            pred = self(sequence)
            return tuple(pred[0].numpy())

    def _prepare_sequence(self, sequence_length: int = 10) -> Optional[torch.Tensor]:
        if len(self.history) < sequence_length:
            return None

        positions = [(state.position[0], state.position[1])
                    for state in self.history[-sequence_length:]]
        return torch.tensor(positions).unsqueeze(0).float()
