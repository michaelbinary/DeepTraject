# src/visualization/visualizer.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import List, Dict, Tuple
from utils.types import VehicleState
from utils.metrics import BenchmarkMetrics
from utils.constants import COLORS


class Visualizer:
    """Enhanced visualization system with interactive plots and animations"""

    def __init__(self, grid_size: Tuple[int, int]):
        """
        Initialize visualizer with grid dimensions.

        Args:
            grid_size: Tuple of (width, height) for plotting area
        """
        self.grid_size = grid_size
        plt.ion()  # Enable interactive mode

    def create_dashboard(self,
                         metrics: BenchmarkMetrics,
                         save_path: Path,
                         show_uncertainty: bool = True):
        """
        Create interactive dashboard with predictions and metrics.

        Args:
            metrics: BenchmarkMetrics object containing performance data
            save_path: Path to save the dashboard
            show_uncertainty: Whether to show uncertainty plots
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Trajectory and Predictions',
                'Prediction Errors Over Time',
                'Error Distribution',
                'Model Performance',
                'State Uncertainty',
                'Prediction Heatmap'
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'histogram'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'heatmap'}]]
        )

        self._add_trajectory_plot(fig, metrics)
        self._add_error_plot(fig, metrics)
        self._add_error_distribution(fig, metrics)
        self._add_performance_bars(fig, metrics)
        if show_uncertainty:
            self._add_uncertainty_plot(fig, metrics)
        self._add_prediction_heatmap(fig, metrics)

        fig.update_layout(
            height=1200,
            width=1600,
            showlegend=True,
            title_text="Path Prediction Analysis Dashboard",
            template="plotly_white",
            font=dict(size=12)
        )

        # Save dashboard
        save_path.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(save_path / 'dashboard.html'))
        fig.write_image(str(save_path / 'dashboard.png'))

    def _add_trajectory_plot(self, fig: go.Figure, metrics: BenchmarkMetrics):
        """Add trajectory and predictions plot"""
        for model_name, data in metrics.metrics.items():
            actual = np.array(data['actual'])
            predicted = np.array(data['predicted'])

            # Plot actual trajectory
            fig.add_trace(
                go.Scatter(
                    x=actual[:, 0],
                    y=actual[:, 1],
                    mode='lines+markers',
                    name='Actual Path',
                    marker=dict(size=8),
                    line=dict(width=3, color='blue')
                ),
                row=1, col=1
            )

            # Plot predictions
            fig.add_trace(
                go.Scatter(
                    x=predicted[:, 0],
                    y=predicted[:, 1],
                    mode='markers',
                    name=f'{model_name} Predictions',
                    marker=dict(
                        size=8,
                        color=COLORS[model_name],
                        symbol='x'
                    )
                ),
                row=1, col=1
            )

        fig.update_xaxes(title_text="X Position", row=1, col=1)
        fig.update_yaxes(title_text="Y Position", row=1, col=1)

    def _add_error_plot(self, fig: go.Figure, metrics: BenchmarkMetrics):
        """Add error over time plot"""
        for model_name, data in metrics.metrics.items():
            errors = np.array(data['errors'])
            timestamps = np.array(data['timestamps'])

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=errors,
                    name=f'{model_name} Error',
                    line=dict(color=COLORS[model_name])
                ),
                row=1, col=2
            )

        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Error (m)", row=1, col=2)

    def _add_error_distribution(self, fig: go.Figure, metrics: BenchmarkMetrics):
        """Add error distribution histogram"""
        for model_name, data in metrics.metrics.items():
            errors = np.array(data['errors'])

            fig.add_trace(
                go.Histogram(
                    x=errors,
                    name=f'{model_name} Distribution',
                    opacity=0.7,
                    nbinsx=30,
                    marker_color=COLORS[model_name]
                ),
                row=2, col=1
            )

        fig.update_xaxes(title_text="Error Magnitude (m)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)

    def _add_performance_bars(self, fig: go.Figure, metrics: BenchmarkMetrics):
        """Add performance comparison bars"""
        summary = metrics.get_summary()
        models = list(summary.keys())
        metrics_to_plot = ['mean_error', 'mse', 'r2']

        for i, metric in enumerate(metrics_to_plot):
            values = [summary[model][metric] for model in models]

            fig.add_trace(
                go.Bar(
                    name=metric,
                    x=models,
                    y=values,
                    marker_color=[COLORS[model] for model in models]
                ),
                row=2, col=2
            )

        fig.update_xaxes(title_text="Model", row=2, col=2)
        fig.update_yaxes(title_text="Score", row=2, col=2)

    def _add_uncertainty_plot(self, fig: go.Figure, metrics: BenchmarkMetrics):
        """Add uncertainty visualization"""
        for model_name, data in metrics.metrics.items():
            actual = np.array(data['actual'])
            predicted = np.array(data['predicted'])
            timestamps = np.array(data['timestamps'])

            uncertainty = np.sqrt(
                (actual[:, 0] - predicted[:, 0]) ** 2 +
                (actual[:, 1] - predicted[:, 1]) ** 2
            )

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=uncertainty,
                    name=f'{model_name} Uncertainty',
                    line=dict(color=COLORS[model_name])
                ),
                row=3, col=1
            )

        fig.update_xaxes(title_text="Time (s)", row=3, col=1)
        fig.update_yaxes(title_text="Uncertainty (m)", row=3, col=1)

    def _add_prediction_heatmap(self, fig: go.Figure, metrics: BenchmarkMetrics):
        """Add prediction accuracy heatmap"""
        model_names = list(metrics.metrics.keys())
        timestamps = metrics.metrics[model_names[0]]['timestamps']

        accuracy_matrix = np.zeros((len(model_names), len(timestamps)))
        for i, model in enumerate(model_names):
            errors = np.array(metrics.metrics[model]['errors'])
            max_error = np.max(errors)
            accuracy_matrix[i] = 1 - (errors / max_error if max_error > 0 else errors)

        fig.add_trace(
            go.Heatmap(
                z=accuracy_matrix,
                x=timestamps,
                y=model_names,
                colorscale='Viridis',
                name='Prediction Accuracy'
            ),
            row=3, col=2
        )

        fig.update_xaxes(title_text="Time (s)", row=3, col=2)
        fig.update_yaxes(title_text="Model", row=3, col=2)

    def create_animation(self,
                         history: List[dict],
                         save_path: Path,
                         interval: int = 50):
        """
        Create animation of the prediction process.

        Args:
            history: List of simulation history frames
            save_path: Path to save the animation
            interval: Time between frames in milliseconds
        """
        fig, ax = plt.subplots(figsize=(12, 12))

        def init():
            ax.set_xlim(0, self.grid_size[0])
            ax.set_ylim(0, self.grid_size[1])
            return []

        def update(frame_number):
            ax.clear()
            frame = history[frame_number]

            # Plot trajectory
            positions = [(f['state'].position) for f in history[:frame_number + 1]]
            if positions:
                xs, ys = zip(*positions)
                ax.plot(xs, ys, 'b-', label='Actual Path', linewidth=2)

            self._plot_frame(ax, frame)

            ax.set_xlim(0, self.grid_size[0])
            ax.set_ylim(0, self.grid_size[1])
            ax.grid(True)
            ax.legend()
            ax.set_title(f'Time: {frame["timestamp"]:.1f}s')

            return ax.get_children()

        anim = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=len(history),
            interval=interval,
            blit=True,
            repeat=True
        )

        writer = PillowWriter(fps=20)
        save_path.mkdir(parents=True, exist_ok=True)
        anim.save(str(save_path / 'prediction_animation.gif'), writer=writer)
        plt.close()

    def _plot_frame(self, ax: plt.Axes, frame: dict):
        """Plot a single frame of the animation"""
        state = frame['state']
        predictions = frame['predictions']

        # Plot predictions
        for model_name, pred in predictions.items():
            ax.scatter(
                pred[0], pred[1],
                c=COLORS[model_name],
                label=f'{model_name} Prediction',
                alpha=0.8,
                s=100
            )

        # Plot current position
        ax.scatter(
            state.position[0],
            state.position[1],
            c='blue',
            s=150,
            label='Current Position'
        )

        # Add velocity vector
        if state.velocity[0] != 0 or state.velocity[1] != 0:
            ax.arrow(
                state.position[0],
                state.position[1],
                state.velocity[0],
                state.velocity[1],
                head_width=0.5,
                head_length=0.8,
                fc='blue',
                ec='blue',
                alpha=0.7
            )


class DashboardBuilder:
    """Helper class for building custom dashboards"""

    def __init__(self, metrics: BenchmarkMetrics):
        self.metrics = metrics
        self.fig = make_subplots(rows=1, cols=1)

    def add_subplot(self, row: int, col: int, title: str):
        """Add a new subplot to the dashboard"""
        self.fig.add_subplot(row, col, title)
        return self

    def save(self, path: Path):
        """Save the dashboard"""
        path.mkdir(parents=True, exist_ok=True)
        self.fig.write_html(str(path / 'custom_dashboard.html'))
        self.fig.write_image(str(path / 'custom_dashboard.png'))