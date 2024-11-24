# Custom Trajectory Generation

Learn how to create and use custom trajectory patterns.

## Creating Figure-Eight Pattern

```python
import numpy as np
from deeptraject.utils.types import VehicleState
from deeptraject.generators import TrajectoryGenerator

class CustomTrajectoryGenerator(TrajectoryGenerator):
    def generate_lemniscate(self, steps: int, size: float = 10.0) -> List[VehicleState]:
        """
        Generate a lemniscate (figure-eight) trajectory.
        
        Args:
            steps: Number of timesteps
            size: Size of the pattern
            
        Returns:
            List[VehicleState]: Trajectory states
        """
        states = []
        for t in range(steps):
            # Parametric equations for lemniscate
            angle = 2 * np.pi * t / steps
            x = size * np.cos(angle) / (1 + np.sin(angle)**2)
            y = size * np.sin(angle) * np.cos(angle) / (1 + np.sin(angle)**2)
            
            # Calculate velocity (derivative of position)
            vx = -size * np.sin(angle) / (1 + np.sin(angle)**2)
            vy = size * (np.cos(angle)**2 - np.sin(angle)**2) / (1 + np.sin(angle)**2)
            
            # Create state
            state = VehicleState(
                position=(x + self.grid_size[0]/2, y + self.grid_size[1]/2),
                velocity=(vx, vy),
                acceleration=(0.0, 0.0),  # simplified
                heading=np.arctan2(vy, vx),
                timestamp=t * 0.1
            )
            states.append(state)
            
        return states

# Usage
generator = CustomTrajectoryGenerator(grid_size=(50, 50))
trajectory = generator.generate_lemniscate(steps=100, size=10.0)
```

## Complex Multi-Pattern Trajectory

```python
def generate_complex_pattern(self, steps: int) -> List[VehicleState]:
    """Generate a trajectory combining multiple patterns."""
    
    # Generate sub-trajectories
    circular = self._circular_trajectory(steps=steps//3, radius=5.0)
    figure_eight = self._figure_eight_trajectory(steps=steps//3, size=8.0)
    lane_change = self._lane_change_trajectory(steps=steps//3, lane_width=3.0)
    
    # Combine trajectories with smooth transitions
    combined = []
    for i, traj in enumerate([circular, figure_eight, lane_change]):
        if i > 0:
            # Create smooth transition
            last_state = combined[-1]
            first_state = traj[0]
            
            # Generate transition states
            transition = self._generate_transition(
                start=last_state,
                end=first_state,
                steps=10
            )
            combined.extend(transition)
            
        combined.extend(traj)
    
    return combined

def _generate_transition(self, 
                        start: VehicleState,
                        end: VehicleState,
                        steps: int) -> List[VehicleState]:
    """Generate smooth transition between states."""
    transition = []
    
    for i in range(steps):
        t = i / (steps - 1)
        # Linear interpolation
        x = start.position[0] + t * (end.position[0] - start.position[0])
        y = start.position[1] + t * (end.position[1] - start.position[1])
        vx = start.velocity[0] + t * (end.velocity[0] - start.velocity[0])
        vy = start.velocity[1] + t * (end.velocity[1] - start.velocity[1])
        
        state = VehicleState(
            position=(x, y),
            velocity=(vx, vy),
            acceleration=(0.0, 0.0),
            heading=np.arctan2(vy, vx),
            timestamp=start.timestamp + t * 0.1
        )
        transition.append(state)
    
    return transition
```

## Using Custom Trajectories with Prediction

```python
# Initialize components
generator = CustomTrajectoryGenerator(grid_size=(50, 50))
kalman = KalmanFilter()
lstm = LSTMPredictor()
viz = Visualizer(grid_size=(50, 50))

# Generate complex trajectory
trajectory = generator.generate_complex_pattern(steps=300)

# Make and visualize predictions
history = []
for state in trajectory:
    kalman.update(state)
    lstm.update(state)
    
    frame = {
        'state': state,
        'predictions': {
            'kalman': kalman.predict(),
            'lstm': lstm.predict()
        },
        'timestamp': state.timestamp
    }
    history.append(frame)

# Create visualization
viz.create_animation(
    history=history,
    save_path='results/complex_trajectory',
    interval=50
)
```