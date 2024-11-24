# Basic Trajectory Prediction Example

This example demonstrates how to set up a basic prediction pipeline using DeepTraject.

## Simple Circular Trajectory

```python
from deeptraject.simulation import SimulationRunner
from deeptraject.utils.types import TrajectoryType
import asyncio

async def main():
    # Initialize simulation environment
    runner = SimulationRunner(grid_size=(50, 50))
    
    # Run simulation with circular trajectory
    await runner.run_simulation(
        trajectory_type=TrajectoryType.CIRCULAR,
        steps=100,
        radius=10.0
    )
    
    # Results are automatically saved to experiments/run_{timestamp}/
    print(f"Results saved to: {runner.experiment_path}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Using Individual Components

For more control, you can use components separately:

```python
from deeptraject.generators import TrajectoryGenerator
from deeptraject.models import KalmanFilter, LSTMPredictor
from deeptraject.visualization import Visualizer

# Initialize components
generator = TrajectoryGenerator(grid_size=(50, 50))
kalman = KalmanFilter()
lstm = LSTMPredictor()
viz = Visualizer(grid_size=(50, 50))

# Generate trajectory
trajectory = generator.generate_trajectory(
    TrajectoryType.CIRCULAR,
    steps=100,
    radius=10.0
)

# Make predictions
predictions = []
for state in trajectory:
    # Update models
    kalman.update(state)
    lstm.update(state)
    
    # Get predictions
    kalman_pred = kalman.predict()
    lstm_pred = lstm.predict()
    
    predictions.append({
        'actual': state.position,
        'kalman': kalman_pred,
        'lstm': lstm_pred
    })

# Visualize results
viz.create_dashboard(predictions, save_path='results/dashboard')
```

## Analyzing Results

```python
from deeptraject.utils.metrics import BenchmarkMetrics

# Calculate metrics
metrics = BenchmarkMetrics()

for pred in predictions:
    metrics.calculate_metrics(
        predicted=pred['kalman'],
        actual=pred['actual'],
        model_name='kalman',
        timestamp=0.1  # assumes 0.1s timestep
    )

# Get performance summary
summary = metrics.get_summary()
print("Kalman Filter Performance:")
print(f"Mean Error: {summary['kalman']['mean_error']:.3f}")
print(f"MSE: {summary['kalman']['mse']:.3f}")
```

## Customizing Visualization

```python
# Create custom dashboard
viz.create_dashboard(
    metrics=metrics,
    save_path='results/custom_dashboard',
    show_uncertainty=True  # Show prediction uncertainty
)

# Create animation
viz.create_animation(
    history=predictions,
    save_path='results/animation',
    interval=50  # 50ms between frames
)
```