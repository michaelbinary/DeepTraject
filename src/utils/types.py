# src/utils/types.py
from dataclasses import dataclass
from typing import Tuple, Dict
from enum import Enum

class TrajectoryType(Enum):
    """Types of predefined trajectories available for testing"""
    CIRCULAR = "circular"
    FIGURE_EIGHT = "figure_eight"
    LANE_CHANGE = "lane_change"
    RANDOM = "random"
    CUSTOM = "custom"

class PredictionModel(Enum):
    """Available prediction models"""
    KALMAN = "kalman"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"

@dataclass
class VehicleState:
    """Represents the state of a vehicle at a point in time"""
    position: Tuple[float, float]
    velocity: Tuple[float, float]
    acceleration: Tuple[float, float]
    heading: float
    timestamp: float

    def to_dict(self) -> dict:
        """Convert state to dictionary for serialization"""
        return {
            'position_x': self.position[0],
            'position_y': self.position[1],
            'velocity_x': self.velocity[0],
            'velocity_y': self.velocity[1],
            'acceleration_x': self.acceleration[0],
            'acceleration_y': self.acceleration[1],
            'heading': self.heading,
            'timestamp': self.timestamp
        }