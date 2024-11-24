# Models API Reference

## KalmanFilter

Classical Kalman filter implementation for state estimation.

```python
class KalmanFilter:
    def __init__(self, dim_x=4, dim_z=2, dt=0.1):
        """
        Initialize Kalman filter.
        
        Args:
            dim_x (int): State vector dimension (default: 4 for x,y,vx,vy)
            dim_z (int): Measurement vector dimension (default: 2 for x,y)
            dt (float): Time step (default: 0.1)
        """

    def predict(self) -> Tuple[float, float]:
        """
        Predict next state.
        
        Returns:
            Tuple[float, float]: Predicted (x, y) position
        """

    def update(self, state: VehicleState):
        """
        Update state with new measurement.
        
        Args:
            state (VehicleState): New state measurement
        """
```

## LSTMPredictor

LSTM-based prediction model with attention mechanism.

```python
class LSTMPredictor(nn.Module):
    def __init__(self,
                 input_size=2,
                 hidden_size=64,
                 num_layers=2,
                 dropout=0.1):
        """
        Initialize LSTM predictor.
        
        Args:
            input_size (int): Input feature dimension
            hidden_size (int): LSTM hidden size
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
        """

    def predict(self) -> Tuple[float, float]:
        """
        Make prediction based on historical data.
        
        Returns:
            Tuple[float, float]: Predicted (x, y) position
        """

    def update(self, state: VehicleState):
        """
        Update model with new state.
        
        Args:
            state (VehicleState): New state observation
        """
```

## EnsemblePredictor

Combines multiple prediction models.

```python
class EnsemblePredictor:
    def __init__(self,
                 models: Dict[str, Any],
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize ensemble predictor.
        
        Args:
            models: Dictionary of prediction models
            weights: Optional weights for each model
        """

    def predict(self) -> Tuple[float, float]:
        """
        Make ensemble prediction.
        
        Returns:
            Tuple[float, float]: Predicted (x, y) position
        """

    def update(self, state: VehicleState):
        """
        Update all models with new state.
        
        Args:
            state (VehicleState): New state observation
        """
```