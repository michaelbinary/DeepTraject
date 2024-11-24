# Visualization API Reference

## Visualizer

Creates interactive visualizations and dashboards.

```python
class Visualizer:
    def __init__(self, grid_size: Tuple[int, int]):
        """
        Initialize visualizer.
        
        Args:
            grid_size: Tuple of (width, height) for plotting area
        """

    def create_dashboard(self,
                        metrics: BenchmarkMetrics,
                        save_path: Path,
                        show_uncertainty: bool = True):
        """
        Create interactive dashboard.
        
        Args:
            metrics: BenchmarkMetrics object with performance data
            save_path: Path to save dashboard
            show_uncertainty: Whether to show uncertainty plots
        """

    def create_animation(self,
                        history: List[dict],
                        save_path: Path,
                        interval: int = 50):
        """
        Create animation of prediction process.
        
        Args:
            history: List of simulation history frames
            save_path: Path to save animation
            interval: Time between frames (ms)
        """
```