# src/utils/constants.py
from typing import Dict

# Model configuration
DEFAULT_GRID_SIZE = (50, 50)
DEFAULT_SEQUENCE_LENGTH = 10
DEFAULT_LSTM_HIDDEN_SIZE = 64
DEFAULT_LSTM_NUM_LAYERS = 2
DEFAULT_LSTM_DROPOUT = 0.1

# Visualization settings
COLORS: Dict[str, str] = {
    'kalman': '#FF4B4B',
    'lstm': '#4B4BFF',
    'ensemble': '#4BFF4B'
}

# Model weights
DEFAULT_ENSEMBLE_WEIGHTS = {
    'kalman': 0.6,
    'lstm': 0.4
}

# Simulation parameters
DEFAULT_SIMULATION_STEPS = 100
DEFAULT_ANIMATION_INTERVAL = 50