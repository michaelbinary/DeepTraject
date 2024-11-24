import pytest
import numpy as np
from utils.types import TrajectoryType


class TestTrajectoryGenerator:
    def test_circular_trajectory_properties(self, trajectory_generator):
        steps = 100
        radius = 10.0
        trajectory = trajectory_generator.generate_trajectory(
            TrajectoryType.CIRCULAR,
            steps=steps,
            radius=radius
        )

        # Test length
        assert len(trajectory) == steps

        # Test if points roughly form a circle
        center_x = trajectory_generator.grid_size[0] / 2
        center_y = trajectory_generator.grid_size[1] / 2

        for state in trajectory:
            x, y = state.position
            distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
            assert abs(distance - radius) < 0.5  # Allow small deviation

    def test_figure_eight_trajectory_crossover(self, trajectory_generator):
        steps = 100
        size = 10.0
        trajectory = trajectory_generator.generate_trajectory(
            TrajectoryType.FIGURE_EIGHT,
            steps=steps,
            size=size
        )

        # Test if trajectory crosses itself in the middle
        middle_points = [state.position for state in trajectory[45:55]]
        center_x = trajectory_generator.grid_size[0] / 2
        center_y = trajectory_generator.grid_size[1] / 2

        # Should find some points near the center
        close_to_center = any(
            abs(x - center_x) < 1.0 and abs(y - center_y) < 1.0
            for x, y in middle_points
        )
        assert close_to_center

    def test_lane_change_trajectory(self, trajectory_generator):
        steps = 100
        lane_width = 5.0
        trajectory = trajectory_generator.generate_trajectory(
            TrajectoryType.LANE_CHANGE,
            steps=steps,
            lane_width=lane_width
        )

        # Test start and end y-positions differ by lane_width
        start_y = trajectory[0].position[1]
        end_y = trajectory[-1].position[1]
        assert abs(end_y - start_y - lane_width) < 0.5

    def test_random_trajectory_bounds(self, trajectory_generator):
        steps = 100
        trajectory = trajectory_generator.generate_trajectory(
            TrajectoryType.RANDOM,
            steps=steps,
            smoothness=0.1
        )

        # Test if all points are within grid bounds
        for state in trajectory:
            x, y = state.position
            assert 0 <= x <= trajectory_generator.grid_size[0]
            assert 0 <= y <= trajectory_generator.grid_size[1]

    def test_invalid_trajectory_type(self, trajectory_generator):
        with pytest.raises(ValueError):
            trajectory_generator.generate_trajectory("invalid_type", steps=100)