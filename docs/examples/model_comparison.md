# Model Comparison and Performance Analysis

This example shows how to compare different prediction models and analyze their performance.

## Setting Up Benchmark

```python
from deeptraject.simulation import SimulationRunner
from deeptraject.utils.types import TrajectoryType
from deeptraject.utils.metrics import BenchmarkMetrics
import numpy as np
import pandas as pd
import plotly.express as px

class ModelBenchmark:
    def __init__(self, grid_size=(50, 50)):
        self.runner = SimulationRunner(grid_size=grid_size)
        self.metrics = BenchmarkMetrics()
        self.results = {}

    async def run_benchmark(self):
        # Test different trajectory types
        trajectories = {
            TrajectoryType.CIRCULAR: {'radius': 10.0},
            TrajectoryType.FIGURE_EIGHT: {'size': 10.0},
            TrajectoryType.LANE_CHANGE: {'lane_width': 5.0},
            TrajectoryType.RANDOM: {'smoothness': 0.1}
        }

        for traj_type, params in trajectories.items():
            # Run simulation
            await self.runner.run_simulation(
                trajectory_type=traj_type,
                steps=100,
                **params
            )

            # Store results
            self.results[traj_type.value] = {
                'metrics': self.runner.metrics.get_summary(),
                'history': self.runner.history
            }

    def generate_report(self):
        # Create performance comparison dataframe
        data = []
        for traj_type, result in self.results.items():
            for model, metrics in result['metrics'].items():
                data.append({
                    'trajectory': traj_type,
                    'model': model,
                    'mean_error': metrics['mean_error'],
                    'mse': metrics['mse'],
                    'r2': metrics['r2']
                })

        df = pd.DataFrame(data)
        return df

    def plot_comparisons(self, df):
        # Error comparison plot
        fig_error = px.bar(
            df,
            x='trajectory',
            y='mean_error',
            color='model',
            barmode='group',
            title='Mean Error by Model and Trajectory Type'
        )
        fig_error.write_html('results/error_comparison.html')

        # R² score comparison
        fig_r2 = px.bar(
            df,
            x='trajectory',
            y='r2',
            color='model',
            barmode='group',
            title='R² Score by Model and Trajectory Type'
        )
        fig_r2.write_html('results/r2_comparison.html')
```

## Running Comparison

```python
import asyncio

async def main():
    # Run benchmark
    benchmark = ModelBenchmark()
    await benchmark.run_benchmark()

    # Generate report
    df = benchmark.generate_report()
    print("\nPerformance Summary:")
    print(df.groupby('model').mean())

    # Create visualizations
    benchmark.plot_comparisons(df)

    # Detailed analysis for each trajectory type
    for traj_type, result in benchmark.results.items():
        print(f"\nDetailed Analysis - {traj_type}")
        for model, metrics in result['metrics'].items():
            print(f"\n{model}:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Advanced Analysis

```python
def analyze_prediction_patterns(history):
    """Analyze prediction patterns and biases."""
    
    patterns = {
        'kalman': {
            'over_predictions': 0,
            'under_predictions': 0,
            'error_distribution': []
        },
        'lstm': {
            'over_predictions': 0,
            'under_predictions': 0,
            'error_distribution': []
        }
    }
    
    for frame in history:
        actual = np.array(frame['state'].position)
        for model, pred in frame['predictions'].items():
            pred = np.array(pred)
            error = np.linalg.norm(pred - actual)
            
            # Analyze prediction bias
            if np.all(pred > actual):
                patterns[model]['over_predictions'] += 1
            elif np.all(pred < actual):
                patterns[model]['under_predictions'] += 1
                
            patterns[model]['error_distribution'].append(error)
    
    return patterns

# Usage
benchmark = ModelBenchmark()
await benchmark.run_benchmark()

for traj_type, result in benchmark.results.items():
    patterns = analyze_prediction_patterns(result['history'])
    
    print(f"\nPrediction Patterns - {traj_type}")
    for model, stats in patterns.items():
        print(f"\n{model}:")
        print(f"Over-predictions: {stats['over_predictions']}")
        print(f"Under-predictions: {stats['under_predictions']}")
        print(f"Mean error: {np.mean(stats['error_distribution']):.4f}")
        print(f"Error std: {np.std(stats['error_distribution']):.4f}")
```

