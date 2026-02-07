"""
Gymnasium environments for quadrotor RL.

Requires the optional ``gymnasium`` dependency::

    pip install gymnasium
    # or
    uv pip install gymnasium
"""

from quad.envs.quad_racing_env import QuadRacingEnv
from quad.envs.track import (
    WaypointTrack,
    circle_track,
    figure8_track,
    line_track,
)

__all__ = [
    "QuadRacingEnv",
    "WaypointTrack",
    "circle_track",
    "figure8_track",
    "line_track",
]
