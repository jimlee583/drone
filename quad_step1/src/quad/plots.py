"""
Visualization functions for simulation results.

Provides various plots for analyzing quadrotor tracking performance.
"""

from typing import List
import numpy as np
from numpy.typing import NDArray

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from quad.types import SimLog
from quad.math3d import quat_to_euler, quat_to_R


def plot_xy_path(
    log: SimLog,
    title: str = "XY Path",
    show: bool = False,
) -> Figure:
    """
    Plot XY path: actual vs desired trajectory.

    Args:
        log: Simulation log
        title: Plot title
        show: If True, call plt.show()

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot desired path
    ax.plot(log.p_des[:, 0], log.p_des[:, 1], 'b--', 
            label='Desired', linewidth=2, alpha=0.7)
    
    # Plot actual path
    ax.plot(log.p[:, 0], log.p[:, 1], 'r-', 
            label='Actual', linewidth=1.5)

    # Mark start and end
    ax.plot(log.p[0, 0], log.p[0, 1], 'go', markersize=10, label='Start')
    ax.plot(log.p[-1, 0], log.p[-1, 1], 'rx', markersize=10, label='End')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_3d_path(
    log: SimLog,
    title: str = "3D Path",
    show: bool = False,
) -> Figure:
    """
    Plot 3D path: actual vs desired trajectory.

    Args:
        log: Simulation log
        title: Plot title
        show: If True, call plt.show()

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot desired path
    ax.plot(log.p_des[:, 0], log.p_des[:, 1], log.p_des[:, 2],
            'b--', label='Desired', linewidth=2, alpha=0.7)
    
    # Plot actual path
    ax.plot(log.p[:, 0], log.p[:, 1], log.p[:, 2],
            'r-', label='Actual', linewidth=1.5)

    # Mark start and end
    ax.scatter(*log.p[0], c='g', s=100, label='Start', marker='o')
    ax.scatter(*log.p[-1], c='r', s=100, label='End', marker='x')

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_pos_time(
    log: SimLog,
    title: str = "Position vs Time",
    show: bool = False,
) -> Figure:
    """
    Plot position components over time.

    Args:
        log: Simulation log
        title: Plot title
        show: If True, call plt.show()

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(log.t, log.p_des[:, i], '--', color=color, 
                label=f'{label} desired', alpha=0.7, linewidth=2)
        ax.plot(log.t, log.p[:, i], '-', color=color,
                label=f'{label} actual', linewidth=1.5)
        ax.set_ylabel(f'{label} [m]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    axes[0].set_title(title)

    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_velocity_time(
    log: SimLog,
    title: str = "Velocity vs Time",
    show: bool = False,
) -> Figure:
    """
    Plot velocity components over time.

    Args:
        log: Simulation log
        title: Plot title
        show: If True, call plt.show()

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    labels = ['Vx', 'Vy', 'Vz']
    colors = ['r', 'g', 'b']

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(log.t, log.v_des[:, i], '--', color=color,
                label=f'{label} desired', alpha=0.7, linewidth=2)
        ax.plot(log.t, log.v[:, i], '-', color=color,
                label=f'{label} actual', linewidth=1.5)
        ax.set_ylabel(f'{label} [m/s]')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    axes[0].set_title(title)

    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_errors(
    log: SimLog,
    title: str = "Tracking Errors",
    show: bool = False,
) -> Figure:
    """
    Plot position and velocity tracking errors over time.

    Args:
        log: Simulation log
        title: Plot title
        show: If True, call plt.show()

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Position error magnitude
    pos_err = np.linalg.norm(log.e_pos, axis=1) * 1000  # Convert to mm
    axes[0].plot(log.t, pos_err, 'b-', linewidth=1.5)
    axes[0].set_ylabel('Position Error [mm]')
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    # Velocity error magnitude
    vel_err = np.linalg.norm(log.e_vel, axis=1) * 1000  # Convert to mm/s
    axes[1].plot(log.t, vel_err, 'r-', linewidth=1.5)
    axes[1].set_ylabel('Velocity Error [mm/s]')
    axes[1].set_xlabel('Time [s]')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_attitude_errors(
    log: SimLog,
    title: str = "Attitude Errors",
    show: bool = False,
) -> Figure:
    """
    Plot attitude and rate tracking errors over time.

    Args:
        log: Simulation log
        title: Plot title
        show: If True, call plt.show()

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Attitude error (convert to degrees)
    att_err_deg = np.rad2deg(log.e_att)
    for i, (label, color) in enumerate(zip(['x (roll)', 'y (pitch)', 'z (yaw)'], ['r', 'g', 'b'])):
        axes[0].plot(log.t, att_err_deg[:, i], color=color, 
                     label=f'{label} error', linewidth=1.5)
    axes[0].set_ylabel('Attitude Error [deg]')
    axes[0].set_title(title)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Rate error (convert to deg/s)
    rate_err_deg = np.rad2deg(log.e_rate)
    for i, (label, color) in enumerate(zip(['p', 'q', 'r'], ['r', 'g', 'b'])):
        axes[1].plot(log.t, rate_err_deg[:, i], color=color,
                     label=f'{label} error', linewidth=1.5)
    axes[1].set_ylabel('Rate Error [deg/s]')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_controls(
    log: SimLog,
    title: str = "Control Inputs",
    show: bool = False,
) -> Figure:
    """
    Plot control inputs (thrust and moments) over time.

    Args:
        log: Simulation log
        title: Plot title
        show: If True, call plt.show()

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Thrust
    axes[0].plot(log.t, log.thrust, 'k-', linewidth=1.5)
    axes[0].set_ylabel('Thrust [N]')
    axes[0].set_title(title)
    axes[0].grid(True, alpha=0.3)

    # Moments
    labels = ['τ_x (roll)', 'τ_y (pitch)', 'τ_z (yaw)']
    colors = ['r', 'g', 'b']
    for i, (label, color) in enumerate(zip(labels, colors)):
        axes[1].plot(log.t, log.moments[:, i] * 1000, color=color,
                     label=label, linewidth=1.5)  # Convert to mN·m
    axes[1].set_ylabel('Moments [mN·m]')
    axes[1].set_xlabel('Time [s]')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_tilt_angle(
    log: SimLog,
    title: str = "Tilt Angle",
    show: bool = False,
) -> Figure:
    """
    Plot the tilt angle of the quadrotor over time.

    Tilt is the angle between body z-axis and world z-axis.

    Args:
        log: Simulation log
        title: Plot title
        show: If True, call plt.show()

    Returns:
        matplotlib Figure
    """
    # Compute tilt angle from quaternions
    n_steps = len(log.t)
    tilt = np.zeros(n_steps)
    
    e3 = np.array([0.0, 0.0, 1.0])  # World z-axis
    
    for i in range(n_steps):
        R = quat_to_R(log.q[i])
        body_z = R @ e3  # Body z-axis in world frame
        
        # Tilt is angle between body_z and e3
        cos_tilt = np.clip(np.dot(body_z, e3), -1.0, 1.0)
        tilt[i] = np.arccos(cos_tilt)
    
    tilt_deg = np.rad2deg(tilt)

    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(log.t, tilt_deg, 'b-', linewidth=1.5)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Tilt Angle [deg]')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_euler_angles(
    log: SimLog,
    title: str = "Euler Angles",
    show: bool = False,
) -> Figure:
    """
    Plot Euler angles (roll, pitch, yaw) over time.

    Note: Euler angles are only for visualization, not used in control.

    Args:
        log: Simulation log
        title: Plot title
        show: If True, call plt.show()

    Returns:
        matplotlib Figure
    """
    # Convert quaternions to Euler angles
    n_steps = len(log.t)
    euler = np.zeros((n_steps, 3))
    
    for i in range(n_steps):
        euler[i] = quat_to_euler(log.q[i])
    
    euler_deg = np.rad2deg(euler)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    labels = ['Roll (φ)', 'Pitch (θ)', 'Yaw (ψ)']
    colors = ['r', 'g', 'b']
    
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(log.t, euler_deg[:, i], color=color, linewidth=1.5)
        
        # Also plot desired yaw for comparison
        if i == 2:  # Yaw
            ax.plot(log.t, np.rad2deg(log.yaw_des), '--', color='gray',
                    label='Desired yaw', alpha=0.7)
            ax.legend(loc='upper right')
        
        ax.set_ylabel(f'{label} [deg]')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time [s]')
    axes[0].set_title(title)

    fig.tight_layout()
    
    if show:
        plt.show()
    
    return fig


def plot_thrust_comparison(
    log: SimLog,
    title: str = "Commanded vs Applied Thrust",
    show: bool = False,
) -> Figure:
    """
    Plot commanded vs applied thrust and moments over time.

    Overlays the commanded signal (from the controller) with the applied
    signal (after actuator dynamics).  Both are stored in the same SimLog.

    Args:
        log: Simulation log containing both applied and commanded controls.
        title: Plot title.
        show: If True, call plt.show().

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # --- Thrust subplot ---
    ax = axes[0]
    ax.plot(log.t, log.thrust, 'b-', linewidth=1.5, label='Applied thrust')
    ax.plot(log.t, log.thrust_cmd, 'r--', linewidth=1.0,
            alpha=0.7, label='Commanded thrust')
    ax.set_ylabel('Thrust [N]')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Moments subplot ---
    ax = axes[1]
    labels = ['τ_x (roll)', 'τ_y (pitch)', 'τ_z (yaw)']
    colors = ['r', 'g', 'b']
    for i, (label, color) in enumerate(zip(labels, colors)):
        ax.plot(log.t, log.moments[:, i] * 1000, color=color,
                label=f'{label} applied', linewidth=1.5)
        ax.plot(log.t, log.moments_cmd[:, i] * 1000, '--', color=color,
                alpha=0.7, label=f'{label} commanded', linewidth=1.0)
    ax.set_ylabel('Moments [mN·m]')
    ax.set_xlabel('Time [s]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if show:
        plt.show()

    return fig


def plot_all(
    log: SimLog,
    name: str = "Simulation",
    show: bool = True,
) -> List[Figure]:
    """
    Generate all standard plots for a simulation.

    Args:
        log: Simulation log
        name: Name prefix for plot titles
        show: If True, display all plots

    Returns:
        List of matplotlib Figures
    """
    figures = [
        plot_xy_path(log, f"{name}: XY Path"),
        plot_3d_path(log, f"{name}: 3D Path"),
        plot_pos_time(log, f"{name}: Position vs Time"),
        plot_errors(log, f"{name}: Tracking Errors"),
        plot_controls(log, f"{name}: Control Inputs"),
        plot_tilt_angle(log, f"{name}: Tilt Angle"),
    ]
    
    if show:
        plt.show()
    
    return figures


if __name__ == "__main__":
    """Quick test of plotting functions."""
    print("Running plots test...")

    # Create a simple mock log for testing
    n = 100
    t = np.linspace(0, 2, n)
    
    # Mock circular trajectory
    log = SimLog(
        t=t,
        p=np.column_stack([np.cos(t), np.sin(t), np.ones(n)]),
        v=np.column_stack([-np.sin(t), np.cos(t), np.zeros(n)]),
        q=np.tile([1, 0, 0, 0], (n, 1)),
        w_body=np.zeros((n, 3)),
        thrust=5.0 * np.ones(n),
        moments=np.zeros((n, 3)),
        thrust_cmd=5.0 * np.ones(n),
        moments_cmd=np.zeros((n, 3)),
        p_des=np.column_stack([np.cos(t), np.sin(t), np.ones(n)]),
        v_des=np.column_stack([-np.sin(t), np.cos(t), np.zeros(n)]),
        a_des=np.column_stack([-np.cos(t), -np.sin(t), np.zeros(n)]),
        yaw_des=np.zeros(n),
        e_pos=0.01 * np.random.randn(n, 3),
        e_vel=0.01 * np.random.randn(n, 3),
        e_att=0.01 * np.random.randn(n, 3),
        e_rate=0.01 * np.random.randn(n, 3),
        _idx=n,
    )

    # Test that plots can be created without error
    fig = plot_xy_path(log, show=False)
    plt.close(fig)
    print("  [PASS] plot_xy_path")

    fig = plot_pos_time(log, show=False)
    plt.close(fig)
    print("  [PASS] plot_pos_time")

    fig = plot_errors(log, show=False)
    plt.close(fig)
    print("  [PASS] plot_errors")

    fig = plot_controls(log, show=False)
    plt.close(fig)
    print("  [PASS] plot_controls")

    fig = plot_tilt_angle(log, show=False)
    plt.close(fig)
    print("  [PASS] plot_tilt_angle")

    print("\nAll plots tests passed!")
