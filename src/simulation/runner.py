# src/simulation/runner.py
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import pandas as pd
from rich.console import Console
from rich.progress import track

from models.kalman_filter import KalmanFilter
from models.lstm_predictor import LSTMPredictor
from models.ensemble_predictor import EnsemblePredictor
from generators.trajectory_generator import TrajectoryGenerator
from visualization.visualizer import Visualizer, BenchmarkMetrics
from utils.types import TrajectoryType, VehicleState
from utils.constants import (
    DEFAULT_GRID_SIZE,
    DEFAULT_ENSEMBLE_WEIGHTS,
    DEFAULT_SIMULATION_STEPS
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deeptraject.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SimulationRunner:
    """Main simulation runner with benchmarking capabilities"""

    def __init__(self, grid_size: tuple = DEFAULT_GRID_SIZE):
        """
        Initialize the simulation environment.

        Args:
            grid_size: Tuple of (width, height) for simulation grid
        """
        self.grid_size = grid_size
        self.trajectory_generator = TrajectoryGenerator(grid_size)
        self.metrics = BenchmarkMetrics()
        self.visualizer = Visualizer(grid_size)

        # Create experiment directory
        self.experiment_path = self._create_experiment_directory()

        # Initialize models
        self.models = self._initialize_models()
        logger.info("Simulation environment initialized successfully")

    def _create_experiment_directory(self) -> Path:
        """Create and return the experiment directory path"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = Path("experiments") / f"run_{timestamp}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _initialize_models(self) -> Dict:
        """Initialize prediction models"""
        models = {
            'kalman': KalmanFilter(),
            'lstm': LSTMPredictor()
        }

        # Create ensemble after other models
        models['ensemble'] = EnsemblePredictor(
            models={'kalman': models['kalman'], 'lstm': models['lstm']},
            weights=DEFAULT_ENSEMBLE_WEIGHTS
        )

        return models

    async def run_benchmarks(self):
        """Run comprehensive benchmarks on different trajectories"""
        trajectories = {
            TrajectoryType.CIRCULAR: {'steps': DEFAULT_SIMULATION_STEPS, 'radius': 10.0},
            TrajectoryType.FIGURE_EIGHT: {'steps': DEFAULT_SIMULATION_STEPS, 'size': 10.0},
            TrajectoryType.LANE_CHANGE: {'steps': DEFAULT_SIMULATION_STEPS, 'lane_width': 5.0}
        }

        console = Console()
        console.print("[bold blue]Starting DeepTraject Benchmark Suite[/bold blue]")

        for traj_type, params in trajectories.items():
            console.print(f"\n[yellow]Running {traj_type.value} trajectory test...[/yellow]")

            states = self.trajectory_generator.generate_trajectory(
                traj_type,
                **params
            )

            await self.run_simulation(states, traj_type.value)
            self.save_results(traj_type.value)

            console.print(f"[green]Completed {traj_type.value} trajectory test[/green]")

        console.print("\n[bold green]Benchmark suite completed successfully![/bold green]")
        self._generate_final_report()

    async def run_simulation(self, states: List[VehicleState], test_name: str):
        """Run simulation with given trajectory"""
        self.history = []

        for i, state in enumerate(track(states, description="Simulating...")):
            frame = {
                'state': state,
                'predictions': {},
                'timestamp': state.timestamp
            }

            # Update models and get predictions
            for name, model in self.models.items():
                model.update(state)

                if i > 0:  # Only predict after first state
                    prediction = model.predict()
                    frame['predictions'][name] = prediction

                    self.metrics.calculate_metrics(
                        prediction,
                        state.position,
                        name,
                        state.timestamp
                    )

            self.history.append(frame)

            # Allow other async operations
            if i % 10 == 0:
                await asyncio.sleep(0)

        # Generate visualizations
        save_path = self.experiment_path / test_name
        save_path.mkdir(exist_ok=True)

        self.visualizer.create_dashboard(self.metrics, save_path)
        self.visualizer.create_animation(self.history, save_path)

        logger.info(f"Simulation completed for {test_name} trajectory")

    def save_results(self, test_name: str):
        """Save simulation results and metrics"""
        save_path = self.experiment_path / test_name
        save_path.mkdir(exist_ok=True)

        # Save metrics summary
        summary = self.metrics.get_summary()
        with open(save_path / 'metrics_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Save trajectory data
        trajectory_data = [frame['state'].to_dict() for frame in self.history]
        pd.DataFrame(trajectory_data).to_csv(save_path / 'trajectory.csv', index=False)

        # Save predictions data
        predictions_data = []
        for frame in self.history:
            if frame['predictions']:
                pred_data = {
                    'timestamp': frame['timestamp'],
                    'actual_x': frame['state'].position[0],
                    'actual_y': frame['state'].position[1]
                }
                for model_name, pred in frame['predictions'].items():
                    pred_data[f'{model_name}_x'] = pred[0]
                    pred_data[f'{model_name}_y'] = pred[1]
                predictions_data.append(pred_data)

        if predictions_data:
            pd.DataFrame(predictions_data).to_csv(
                save_path / 'predictions.csv',
                index=False
            )

        logger.info(f"Results saved to {save_path}")

    def _generate_final_report(self):
        """Generate comprehensive final report"""
        report_path = self.experiment_path / "final_report.md"

        report_content = [
            "# DeepTraject Simulation Report\n",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
            "## Summary of Results\n",
            "### Model Performance Comparison\n",
            "| Model | Mean Error | MSE | R² Score |",
            "|-------|------------|-----|----------|"
        ]

        # Add overall metrics for each model
        for model_name, metrics in self.metrics.get_summary().items():
            report_content.append(
                f"| {model_name} | "
                f"{metrics['mean_error']:.3f} | "
                f"{metrics['mse']:.3f} | "
                f"{metrics['r2']:.3f} |"
            )

        with open(report_path, 'w') as f:
            f.write('\n'.join(report_content))