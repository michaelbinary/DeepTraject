# tests/test_utils.py
import numpy as np
from utils.metrics import BenchmarkMetrics


class TestBenchmarkMetrics:
    def test_metric_calculation(self):
        metrics = BenchmarkMetrics()

        # Test single prediction
        predicted = (1.0, 1.0)
        actual = (2.0, 2.0)
        metrics.calculate_metrics(predicted, actual, "test_model", 0.0)

        summary = metrics.get_summary()
        assert "test_model" in summary
        assert abs(summary["test_model"]["mean_error"] - np.sqrt(2)) < 1e-6

    def test_multiple_models(self):
        metrics = BenchmarkMetrics()

        # Add data for two models
        models = ["model1", "model2"]
        for model in models:
            metrics.calculate_metrics((1.0, 1.0), (2.0, 2.0), model, 0.0)

        summary = metrics.get_summary()
        assert all(model in summary for model in models)

    def test_error_statistics(self):
        metrics = BenchmarkMetrics()

        # Add multiple predictions
        predicted_positions = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        actual_positions = [(1.5, 1.5), (2.5, 2.5), (3.5, 3.5)]

        for pred, actual in zip(predicted_positions, actual_positions):
            metrics.calculate_metrics(pred, actual, "test_model", 0.0)

        stats = metrics.get_error_statistics("test_model")
        assert all(key in stats for key in [
            'mean', 'median', 'std', 'min', 'max',
            'percentile_25', 'percentile_75'
        ])

    def test_clear_metrics(self):
        metrics = BenchmarkMetrics()
        metrics.calculate_metrics((1.0, 1.0), (2.0, 2.0), "test_model", 0.0)
        metrics.clear()

        with pytest.raises(KeyError):
            metrics.get_error_statistics("test_model")
