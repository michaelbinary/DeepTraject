# src/utils/metrics.py
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BenchmarkMetrics:
    """Calculates and stores benchmark metrics for prediction models"""

    def __init__(self):
        """Initialize metrics storage"""
        self.metrics = defaultdict(lambda: defaultdict(list))

    def calculate_metrics(self,
                          predicted: Tuple[float, float],
                          actual: Tuple[float, float],
                          model_name: str,
                          timestamp: float):
        """
        Calculate metrics for a single prediction.

        Args:
            predicted: Tuple of (x, y) predicted position
            actual: Tuple of (x, y) actual position
            model_name: Name of the prediction model
            timestamp: Time of prediction
        """
        error = np.sqrt(
            (predicted[0] - actual[0]) ** 2 +
            (predicted[1] - actual[1]) ** 2
        )

        self.metrics[model_name]['errors'].append(error)
        self.metrics[model_name]['timestamps'].append(timestamp)
        self.metrics[model_name]['predicted'].append(predicted)
        self.metrics[model_name]['actual'].append(actual)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all models.

        Returns:
            Dictionary containing performance metrics for each model
        """
        summary = {}

        for model_name, data in self.metrics.items():
            errors = np.array(data['errors'])
            predicted = np.array(data['predicted'])
            actual = np.array(data['actual'])

            summary[model_name] = {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'max_error': np.max(errors),
                'min_error': np.min(errors),
                'mse': mean_squared_error(actual, predicted),
                'mae': mean_absolute_error(actual, predicted),
                'r2': r2_score(actual.flatten(), predicted.flatten())
            }

        return summary

    def get_model_metrics(self, model_name: str) -> Dict[str, List]:
        """
        Get detailed metrics for a specific model.

        Args:
            model_name: Name of the model to get metrics for

        Returns:
            Dictionary containing all recorded metrics for the model
        """
        if model_name not in self.metrics:
            raise KeyError(f"No metrics found for model: {model_name}")

        return dict(self.metrics[model_name])

    def get_error_statistics(self, model_name: str) -> Dict[str, float]:
        """
        Get detailed error statistics for a specific model.

        Args:
            model_name: Name of the model to analyze

        Returns:
            Dictionary containing error statistics
        """
        if model_name not in self.metrics:
            raise KeyError(f"No metrics found for model: {model_name}")

        errors = np.array(self.metrics[model_name]['errors'])
        return {
            'mean': np.mean(errors),
            'median': np.median(errors),
            'std': np.std(errors),
            'min': np.min(errors),
            'max': np.max(errors),
            'percentile_25': np.percentile(errors, 25),
            'percentile_75': np.percentile(errors, 75)
        }

    def clear(self):
        """Clear all stored metrics"""
        self.metrics = defaultdict(lambda: defaultdict(list))