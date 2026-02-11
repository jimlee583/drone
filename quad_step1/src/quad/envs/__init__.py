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
from quad.envs.gates import Gate, gate_from_waypoint, waypoints_to_gates
from quad.envs.gate_track import GateTrack

__all__ = [
    "QuadRacingEnv",
    "WaypointTrack",
    "circle_track",
    "figure8_track",
    "line_track",
    "Gate",
    "GateTrack",
    "gate_from_waypoint",
    "waypoints_to_gates",
]
