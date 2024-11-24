# src/models/ensemble_predictor.py
from typing import Dict, Any, Tuple, Optional

from utils.types import VehicleState
from utils.constants import DEFAULT_ENSEMBLE_WEIGHTS

class EnsemblePredictor:
    """Ensemble predictor combining multiple models"""

    def __init__(self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None):
        self.models = models
        self.weights = weights or DEFAULT_ENSEMBLE_WEIGHTS
        self.history = []

    def update(self, state: VehicleState):
        """Update all models with new state"""
        self.history.append(state)
        for model in self.models.values():
            if hasattr(model, 'update'):
                model.update(state)

    def predict(self) -> Tuple[float, float]:
        """Make ensemble prediction"""
        predictions = {}
        for name, model in self.models.items():
            pred = model.predict()
            predictions[name] = pred

        x = sum(pred[0] * self.weights[name] for name, pred in predictions.items())
        y = sum(pred[1] * self.weights[name] for name, pred in predictions.items())

        return (x, y)