# tests/conftest.py
import pytest
import numpy as np
from utils.types import VehicleState, TrajectoryType
from models.kalman_filter import KalmanFilter
from models.lstm_predictor import LSTMPredictor
from models.ensemble_predictor import EnsemblePredictor
from generators.trajectory_generator import TrajectoryGenerator

@pytest.fixture
def grid_size():
    return (50, 50)

@pytest.fixture
def sample_state():
    return VehicleState(
        position=(25.0, 25.0),
        velocity=(1.0, 0.0),
        acceleration=(0.0, 0.0),
        heading=0.0,
        timestamp=0.0
    )

@pytest.fixture
def trajectory_generator(grid_size):
    return TrajectoryGenerator(grid_size)

@pytest.fixture
def kalman_filter():
    return KalmanFilter()

@pytest.fixture
def lstm_predictor():
    return LSTMPredictor()

@pytest.fixture
def ensemble_predictor(kalman_filter, lstm_predictor):
    models = {
        'kalman': kalman_filter,
        'lstm': lstm_predictor
    }
    return EnsemblePredictor(models)

# tests/test_models.py
def test_kalman_filter_prediction(kalman_filter, sample_state):
    kalman_filter.update(sample_state)
    prediction = kalman_filter.predict()
    assert len(prediction) == 2
    assert isinstance(prediction[0], float)
    assert isinstance(prediction[1], float)

def test_lstm_predictor_prediction(lstm_predictor, sample_state):
    lstm_predictor.update(sample_state)
    prediction = lstm_predictor.predict()
    assert len(prediction) == 2
    assert isinstance(prediction[0], float)
    assert isinstance(prediction[1], float)

def test_ensemble_predictor(ensemble_predictor, sample_state):
    ensemble_predictor.update(sample_state)
    prediction = ensemble_predictor.predict()
    assert len(prediction) == 2
    assert isinstance(prediction[0], float)
    assert isinstance(prediction[1], float)

# tests/test_generators.py
def test_circular_trajectory(trajectory_generator):
    steps = 100
    trajectory = trajectory_generator.generate_trajectory(
        TrajectoryType.CIRCULAR,
        steps=steps,
        radius=10.0
    )
    assert len(trajectory) == steps
    assert all(isinstance(state, VehicleState) for state in trajectory)

def test_figure_eight_trajectory(trajectory_generator):
    steps = 100
    trajectory = trajectory_generator.generate_trajectory(
        TrajectoryType.FIGURE_EIGHT,
        steps=steps,
        size=10.0
    )
    assert len(trajectory) == steps
    assert all(isinstance(state, VehicleState) for state in trajectory)

# tests/test_simulation.py
import pytest
from simulation.runner import SimulationRunner

@pytest.mark.asyncio
async def test_simulation_runner():
    runner = SimulationRunner()
    try:
        await runner.run_benchmarks()
    except Exception as e:
        pytest.fail(f"Simulation failed with error: {str(e)}")