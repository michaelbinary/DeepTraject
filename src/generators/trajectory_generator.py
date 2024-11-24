# src/generators/trajectory_generator.py
import numpy as np
from typing import List, Tuple, Optional
from utils.types import TrajectoryType, VehicleState

class TrajectoryGenerator:
    """Generates various types of test trajectories for vehicle path prediction"""

    def __init__(self, grid_size: Tuple[int, int]):
        self.grid_size = grid_size

    def generate_trajectory(self,
                            trajectory_type: TrajectoryType,
                            steps: int,
                            **kwargs) -> List[VehicleState]:
        """
        Generate a trajectory of the specified type.

        Args:
            trajectory_type: Type of trajectory to generate
            steps: Number of steps in the trajectory
            **kwargs: Additional parameters specific to trajectory type

        Returns:
            List of VehicleState objects representing the trajectory
        """
        generators = {
            TrajectoryType.CIRCULAR: self._circular_trajectory,
            TrajectoryType.FIGURE_EIGHT: self._figure_eight_trajectory,
            TrajectoryType.LANE_CHANGE: self._lane_change_trajectory,
            TrajectoryType.RANDOM: self._random_trajectory
        }

        if trajectory_type not in generators:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}")

        return generators[trajectory_type](steps, **kwargs)

    def _circular_trajectory(self,
                             steps: int,
                             radius: float = 10.0,
                             center: Optional[Tuple[float, float]] = None) -> List[VehicleState]:
        """Generate a circular trajectory"""
        if center is None:
            center = (self.grid_size[0] / 2, self.grid_size[1] / 2)

        states = []
        for t in range(steps):
            angle = 2 * np.pi * t / steps
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)

            vx = -radius * (2 * np.pi / steps) * np.sin(angle)
            vy = radius * (2 * np.pi / steps) * np.cos(angle)

            ax = -radius * (2 * np.pi / steps) ** 2 * np.cos(angle)
            ay = -radius * (2 * np.pi / steps) ** 2 * np.sin(angle)

            states.append(VehicleState(
                position=(x, y),
                velocity=(vx, vy),
                acceleration=(ax, ay),
                heading=angle,
                timestamp=t * 0.1
            ))
        return states

    def _figure_eight_trajectory(self,
                                 steps: int,
                                 size: float = 10.0,
                                 center: Optional[Tuple[float, float]] = None) -> List[VehicleState]:
        """Generate a figure-eight trajectory"""
        if center is None:
            center = (self.grid_size[0] / 2, self.grid_size[1] / 2)

        states = []
        for t in range(steps):
            angle = 2 * np.pi * t / steps
            x = center[0] + size * np.sin(angle)
            y = center[1] + size * np.sin(angle) * np.cos(angle)

            vx = size * (2 * np.pi / steps) * np.cos(angle)
            vy = size * (2 * np.pi / steps) * (np.cos(angle) ** 2 - np.sin(angle) ** 2)

            ax = -size * (2 * np.pi / steps) ** 2 * np.sin(angle)
            ay = -2 * size * (2 * np.pi / steps) ** 2 * np.sin(angle) * np.cos(angle)

            states.append(VehicleState(
                position=(x, y),
                velocity=(vx, vy),
                acceleration=(ax, ay),
                heading=np.arctan2(vy, vx),
                timestamp=t * 0.1
            ))
        return states

    def _lane_change_trajectory(self,
                                steps: int,
                                lane_width: float = 5.0,
                                center: Optional[Tuple[float, float]] = None) -> List[VehicleState]:
        """Generate a lane change trajectory"""
        if center is None:
            center = (self.grid_size[0] / 4, self.grid_size[1] / 2)

        states = []
        for t in range(steps):
            progress = t / steps
            x = center[0] + 40 * progress
            y = center[1] + lane_width * (1 - np.cos(np.pi * progress)) if progress < 0.5 else center[1] + lane_width

            vx = 40 / steps
            vy = lane_width * np.pi * np.sin(np.pi * progress) / steps if progress < 0.5 else 0

            ax = 0
            ay = lane_width * np.pi ** 2 * np.cos(np.pi * progress) / steps ** 2 if progress < 0.5 else 0

            states.append(VehicleState(
                position=(x, y),
                velocity=(vx, vy),
                acceleration=(ax, ay),
                heading=np.arctan2(vy, vx),
                timestamp=t * 0.1
            ))
        return states

    def _random_trajectory(self,
                           steps: int,
                           smoothness: float = 0.1) -> List[VehicleState]:
        """Generate a random but smooth trajectory"""
        states = []
        x, y = (self.grid_size[0] / 2, self.grid_size[1] / 2)
        vx, vy = (0, 0)

        for t in range(steps):
            ax = (np.random.random() - 0.5) * smoothness
            ay = (np.random.random() - 0.5) * smoothness

            vx += ax
            vy += ay

            speed = np.sqrt(vx * vx + vy * vy)
            if speed > 1.0:
                vx /= speed
                vy /= speed

            x += vx
            y += vy

            x = np.clip(x, 0, self.grid_size[0])
            y = np.clip(y, 0, self.grid_size[1])

            states.append(VehicleState(
                position=(x, y),
                velocity=(vx, vy),
                acceleration=(ax, ay),
                heading=np.arctan2(vy, vx),
                timestamp=t * 0.1
            ))
        return states