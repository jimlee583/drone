"""
Quad Step 1: Quadrotor Dynamics & Nonlinear SE(3) Control

A modular quadrotor simulation with geometric tracking control.
"""

from quad.types import State, Control, TrajPoint
from quad.params import Params, default_params
from quad.sim import run_sim

__version__ = "0.1.0"

__all__ = [
    "State",
    "Control",
    "TrajPoint",
    "Params",
    "default_params",
    "run_sim",
]
