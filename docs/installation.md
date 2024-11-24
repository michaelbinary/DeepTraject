# Installation Guide

## Prerequisites

DeepTraject requires:
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
- Plotly

## Basic Installation

```bash
# Clone the repository
cd deeptraject

# Install dependencies
pip install -r requirements.txt

# Optional: Install development dependencies
pip install -r requirements-dev.txt
```

## Development Setup

For development work:

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Verification

Verify your installation:

```python
from deeptraject.simulation import SimulationRunner
from deeptraject.utils.types import TrajectoryType

# Initialize simulation
runner = SimulationRunner()

# Run a simple test
trajectory = runner.generate_trajectory(
    trajectory_type=TrajectoryType.CIRCULAR,
    steps=100
)
```

## Common Issues

1. **PyTorch Installation**
   If you encounter issues with PyTorch, install it directly from the official website:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

2. **CUDA Support**
   For GPU support, ensure you have compatible CUDA drivers installed.