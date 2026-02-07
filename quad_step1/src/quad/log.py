"""
Simulation logging utilities.

Provides pre-allocated logging for efficient data collection
during simulation.
"""

import numpy as np
from numpy.typing import NDArray

from quad.types import State, Control, TrajPoint, SimLog


def allocate_log(n_steps: int) -> SimLog:
    """
    Allocate a SimLog with pre-allocated arrays.

    This is a convenience wrapper around SimLog.allocate().

    Args:
        n_steps: Number of timesteps to allocate

    Returns:
        Pre-allocated SimLog
    """
    return SimLog.allocate(n_steps)


def record_step(
    log: SimLog,
    t: float,
    state: State,
    control: Control,
    traj: TrajPoint,
    e_pos: NDArray[np.float64],
    e_vel: NDArray[np.float64],
    e_att: NDArray[np.float64],
    e_rate: NDArray[np.float64],
) -> None:
    """
    Record one timestep of simulation data.

    This is a convenience wrapper around SimLog.record().

    Args:
        log: SimLog instance to record into
        t: Current time [s]
        state: Current quadrotor state
        control: Current control inputs
        traj: Current desired trajectory point
        e_pos: Position error
        e_vel: Velocity error
        e_att: Attitude error
        e_rate: Rate error
    """
    log.record(t, state, control, traj, e_pos, e_vel, e_att, e_rate)


def compute_statistics(log: SimLog) -> dict:
    """
    Compute summary statistics from simulation log.

    Args:
        log: Completed simulation log

    Returns:
        Dictionary with statistics:
        - pos_rmse: RMS position error [m]
        - vel_rmse: RMS velocity error [m/s]
        - max_pos_error: Maximum position error [m]
        - max_thrust: Maximum thrust [N]
        - mean_thrust: Mean thrust [N]
    """
    # Position error magnitude
    pos_err_mag = np.linalg.norm(log.e_pos, axis=1)
    vel_err_mag = np.linalg.norm(log.e_vel, axis=1)

    stats = {
        "pos_rmse": np.sqrt(np.mean(pos_err_mag**2)),
        "vel_rmse": np.sqrt(np.mean(vel_err_mag**2)),
        "max_pos_error": np.max(pos_err_mag),
        "max_thrust": np.max(log.thrust),
        "mean_thrust": np.mean(log.thrust),
        "simulation_time": log.t[-1] if len(log.t) > 0 else 0.0,
    }

    return stats


def print_statistics(log: SimLog, name: str = "Simulation") -> None:
    """
    Print summary statistics to console.

    Args:
        log: Completed simulation log
        name: Name of simulation for display
    """
    stats = compute_statistics(log)

    print(f"\n{name} Statistics:")
    print(f"  Duration:        {stats['simulation_time']:.2f} s")
    print(f"  Position RMSE:   {stats['pos_rmse']*1000:.2f} mm")
    print(f"  Velocity RMSE:   {stats['vel_rmse']*1000:.2f} mm/s")
    print(f"  Max pos error:   {stats['max_pos_error']*1000:.2f} mm")
    print(f"  Mean thrust:     {stats['mean_thrust']:.3f} N")
    print(f"  Max thrust:      {stats['max_thrust']:.3f} N")


if __name__ == "__main__":
    """Quick test for log module."""
    print("Running log tests...")

    # Create and fill a small log
    log = allocate_log(100)

    state = State.zeros()
    control = Control.zeros()
    control.thrust_N = 5.0
    traj = TrajPoint.hover(np.array([0.0, 0.0, 1.0]))

    for i in range(100):
        t = i * 0.01
        state.p = np.array([0.0, 0.0, 1.0 + 0.001 * i])
        e_pos = state.p - traj.p
        record_step(log, t, state, control, traj, e_pos, np.zeros(3), np.zeros(3), np.zeros(3))

    log_trimmed = log.trim()
    assert len(log_trimmed.t) == 100, "Log should have 100 entries"
    assert log_trimmed.t[-1] == 0.99, f"Last time should be 0.99, got {log_trimmed.t[-1]}"

    stats = compute_statistics(log_trimmed)
    print(f"  Stats computed: RMSE = {stats['pos_rmse']*1000:.2f} mm")

    print("  [PASS] Log allocation and recording")
    print("\nAll log tests passed!")
