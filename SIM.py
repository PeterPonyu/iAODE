"""Backward-compatible entry point for simulation classes.

The concrete implementations now live in the following modules:

- `latent_space_simulator.LatentSpaceSimulator`
- `metric_validator.MetricValidator`
- `latent_space_visualizer.LatentSpaceVisualizer`

This module simply re-exports these names so existing imports such as
`from SIM import LatentSpaceSimulator` keep working unchanged.
"""

from latent_space_simulator import LatentSpaceSimulator
from metric_validator import MetricValidator
from latent_space_visualizer import LatentSpaceVisualizer

__all__ = [
    "LatentSpaceSimulator",
    "MetricValidator",
    "LatentSpaceVisualizer",
]
