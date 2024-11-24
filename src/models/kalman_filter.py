# src/models/kalman_filter.py
import numpy as np
from typing import Tuple
from utils.types import VehicleState

class KalmanFilter:
    """Kalman filter for state estimation and prediction"""

    def __init__(self, dim_x=4, dim_z=2, dt=0.1):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt

        # State vector [x, y, vx, vy]
        self.x = np.zeros((dim_x, 1))

        # State transition matrix F for constant velocity model
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Measurement matrix H
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        self.P = np.eye(dim_x) * 100
        self.R = np.eye(dim_z) * 1

        # Process noise matrix
        q = 0.1
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * q

    def predict(self) -> Tuple[float, float]:
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return tuple(self.x[:2].flatten())

    def update(self, state: VehicleState):
        """Update state with new measurement"""
        z = np.array([state.position[0], state.position[1]]).reshape(-1, 1)
        self.x[2:] = np.array([state.velocity[0], state.velocity[1]]).reshape(-1, 1)

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(self.dim_x) - K @ self.H) @ self.P