# Generators API Reference

## TrajectoryGenerator

Generates various types of test trajectories.

```python
class TrajectoryGenerator:
    def __init__(self, grid_size: Tuple[int, int]):
        """
        Initialize trajectory generator.
        
        Args:
            grid_size: Tuple of (width, height) for simulation grid
        """

    def generate_trajectory(self,
                          trajectory_type: TrajectoryType,
                          steps: int,
                          **kwargs) -> List[VehicleState]:
        """
        Generate a trajectory of specified type.
        
        Args:
            trajectory_type: Type of trajectory to generate
            steps: Number of steps in trajectory
            **kwargs: Additional parameters for specific trajectories
                     - radius: for CIRCULAR
                     - size: for FIGURE_EIGHT
                     - lane_width: for LANE_CHANGE
                     - smoothness: for RANDOM
        
        Returns:
            List of VehicleState objects representing trajectory
        """

    def _circular_trajectory(self,
                           steps: int,
                           radius: float = 10.0,
                           center: Optional[Tuple[float, float]] = None
                           ) -> List[VehicleState]:
        """Generate circular trajectory"""

    def _figure_eight_trajectory(self,
                               steps: int,
                               size: float = 10.0,
                               center: Optional[Tuple[float, float]] = None
                               ) -> List[VehicleState]:
        """Generate figure-eight trajectory"""

    def _lane_change_trajectory(self,
                              steps: int,
                              lane_width: float = 5.0,
                              center: Optional[Tuple[float, float]] = None
                              ) -> List[VehicleState]:
        """Generate lane change trajectory"""

    def _random_trajectory(self,
                         steps: int,
                         smoothness: float = 0.1
                         ) -> List[VehicleState]:
        """Generate random but smooth trajectory"""
```