# Quick Start Guide

## Basic Usage

```python
from deeptraject.simulation import SimulationRunner
from deeptraject.utils.types import TrajectoryType

# Initialize simulation
runner = SimulationRunner()

# Generate and predict a circular trajectory
async def run_simulation():
    await runner.run_simulation(
        trajectory_type=TrajectoryType.CIRCULAR,
        steps=100,
        radius=10.0
    )

# Run simulation
import asyncio
asyncio.run(run_simulation())
```

## Common Patterns

### 1. Trajectory Generation

```python
from deeptraject.generators import TrajectoryGenerator

# Initialize generator
generator = TrajectoryGenerator(grid_size=(50, 50))

# Generate different trajectories
circular = generator.generate_trajectory(
    TrajectoryType.CIRCULAR,
    steps=100,
    radius=10.0
)

figure_eight = generator.generate_trajectory(
    TrajectoryType.FIGURE_EIGHT,
    steps=100,
    size=10.0
)
```

### 2. Model Usage

```python
from deeptraject.models import KalmanFilter, LSTMPredictor

# Initialize models
kalman = KalmanFilter()
lstm = LSTMPredictor()

# Update with new state
kalman.update(state)
lstm.update(state)

# Get predictions
kalman_pred = kalman.predict()
lstm_pred = lstm.predict()
```

### 3. Visualization

```python
from deeptraject.visualization import Visualizer

# Create visualizer
viz = Visualizer(grid_size=(50, 50))

# Create dashboard
viz.create_dashboard(
    metrics=runner.metrics,
    save_path="results/dashboard"
)

# Create animation
viz.create_animation(
    history=runner.history,
    save_path="results/animation"
)
```

