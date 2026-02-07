"""
Reference trajectory generators.

Provides analytic trajectories (position, velocity, acceleration)
for testing and validating the controller.
"""

from typing import Callable, Literal
import numpy as np
from numpy.typing import NDArray

from quad.types import TrajPoint


# Type alias for trajectory functions
TrajectoryFn = Callable[[float], TrajPoint]


def hover(
    z: float = 1.0,
    yaw: float = 0.0,
    position: NDArray[np.float64] | None = None,
) -> TrajectoryFn:
    """
    Create a hover trajectory at fixed position.

    Args:
        z: Hover altitude [m]
        yaw: Fixed yaw angle [rad]
        position: Optional full 3D position (overrides z if given)

    Returns:
        Trajectory function t -> TrajPoint
    """
    if position is not None:
        p_hover = np.array(position, dtype=np.float64)
    else:
        p_hover = np.array([0.0, 0.0, z])

    def traj_fn(t: float) -> TrajPoint:
        return TrajPoint(
            p=p_hover.copy(),
            v=np.zeros(3),
            a=np.zeros(3),
            yaw=yaw,
            yaw_rate=0.0,
        )

    return traj_fn


def step_to(
    target_p: NDArray[np.float64],
    t_step: float = 1.0,
    start_p: NDArray[np.float64] | None = None,
    yaw: float = 0.0,
) -> TrajectoryFn:
    """
    Create a step trajectory: hold at start, then move to target.

    Uses smooth 5th-order polynomial interpolation for the transition
    to provide continuous acceleration.

    Args:
        target_p: Target position [m]
        t_step: Time to start transition [s]
        start_p: Starting position (default: origin at z=0)
        yaw: Fixed yaw angle [rad]

    Returns:
        Trajectory function t -> TrajPoint
    """
    if start_p is None:
        start_p = np.zeros(3)
    
    start = np.array(start_p, dtype=np.float64)
    target = np.array(target_p, dtype=np.float64)
    
    # Transition duration (smooth over 2 seconds)
    t_duration = 2.0

    def smooth_step(s: float) -> tuple[float, float, float]:
        """
        5th-order polynomial interpolation.
        s goes from 0 to 1 over transition.
        Returns (position_interp, velocity_interp, accel_interp) coefficients.
        """
        if s <= 0:
            return 0.0, 0.0, 0.0
        if s >= 1:
            return 1.0, 0.0, 0.0
        
        # 5th order polynomial: p(s) = 10s³ - 15s⁴ + 6s⁵
        # Ensures p(0)=0, p(1)=1, p'(0)=p'(1)=0, p''(0)=p''(1)=0
        s2 = s * s
        s3 = s2 * s
        s4 = s3 * s
        s5 = s4 * s
        
        pos = 10*s3 - 15*s4 + 6*s5
        vel = 30*s2 - 60*s3 + 30*s4  # Derivative of position
        acc = 60*s - 180*s2 + 120*s3  # Second derivative
        
        return pos, vel, acc

    def traj_fn(t: float) -> TrajPoint:
        if t < t_step:
            # Before step: hold at start
            return TrajPoint(
                p=start.copy(),
                v=np.zeros(3),
                a=np.zeros(3),
                yaw=yaw,
                yaw_rate=0.0,
            )
        else:
            # During/after transition
            s = (t - t_step) / t_duration
            p_interp, v_interp, a_interp = smooth_step(s)
            
            delta = target - start
            
            p = start + p_interp * delta
            v = (v_interp / t_duration) * delta
            a = (a_interp / (t_duration * t_duration)) * delta
            
            return TrajPoint(
                p=p,
                v=v,
                a=a,
                yaw=yaw,
                yaw_rate=0.0,
            )

    return traj_fn


def circle(
    radius: float = 1.0,
    speed: float = 0.5,
    z: float = 1.0,
    center: NDArray[np.float64] | None = None,
    yaw_mode: Literal["tangent", "fixed"] = "fixed",
) -> TrajectoryFn:
    """
    Create a circular trajectory.

    The circle is in the XY plane at fixed altitude.

    Args:
        radius: Circle radius [m]
        speed: Tangential speed [m/s]
        z: Altitude [m]
        center: Center of circle [x, y] (default: origin)
        yaw_mode: "tangent" to point along velocity, "fixed" for yaw=0

    Returns:
        Trajectory function t -> TrajPoint
    """
    if center is None:
        cx, cy = 0.0, 0.0
    else:
        cx, cy = center[0], center[1]

    # Angular velocity
    omega = speed / radius

    def traj_fn(t: float) -> TrajPoint:
        # Phase angle
        theta = omega * t

        # Position: start at (cx + radius, cy, z) and go counter-clockwise
        p = np.array([
            cx + radius * np.cos(theta),
            cy + radius * np.sin(theta),
            z,
        ])

        # Velocity: tangent to circle
        v = np.array([
            -radius * omega * np.sin(theta),
            radius * omega * np.cos(theta),
            0.0,
        ])

        # Acceleration: centripetal (points to center)
        a = np.array([
            -radius * omega**2 * np.cos(theta),
            -radius * omega**2 * np.sin(theta),
            0.0,
        ])

        # Yaw
        if yaw_mode == "tangent":
            # Point along velocity direction
            yaw = np.arctan2(v[1], v[0])
            yaw_rate = omega
        else:
            yaw = 0.0
            yaw_rate = 0.0

        return TrajPoint(p=p, v=v, a=a, yaw=yaw, yaw_rate=yaw_rate)

    return traj_fn


def figure8(
    a: float = 1.0,
    b: float = 0.5,
    speed: float = 0.5,
    z: float = 1.0,
    yaw_mode: Literal["tangent", "fixed"] = "fixed",
) -> TrajectoryFn:
    """
    Create a figure-8 (lemniscate) trajectory.

    Parametric form:
        x(t) = a * sin(omega*t)
        y(t) = b * sin(2*omega*t)

    This creates a figure-8 with:
    - Width (x-extent): 2a
    - Height (y-extent): 2b

    Args:
        a: X amplitude [m]
        b: Y amplitude [m]
        speed: Approximate average speed [m/s]
        z: Altitude [m]
        yaw_mode: "tangent" to point along velocity, "fixed" for yaw=0

    Returns:
        Trajectory function t -> TrajPoint
    """
    # Compute angular frequency from desired speed
    # The path length of a figure-8 is approximately 4*sqrt(a² + 4b²)
    # for the parametric form used here
    path_length = 4.0 * np.sqrt(a**2 + 4*b**2)
    period = path_length / speed
    omega = 2 * np.pi / period

    def traj_fn(t: float) -> TrajPoint:
        # Phase
        theta = omega * t

        # Position
        p = np.array([
            a * np.sin(theta),
            b * np.sin(2 * theta),
            z,
        ])

        # Velocity (derivative of position)
        v = np.array([
            a * omega * np.cos(theta),
            2 * b * omega * np.cos(2 * theta),
            0.0,
        ])

        # Acceleration (derivative of velocity)
        a_vec = np.array([
            -a * omega**2 * np.sin(theta),
            -4 * b * omega**2 * np.sin(2 * theta),
            0.0,
        ])

        # Yaw
        if yaw_mode == "tangent":
            yaw = np.arctan2(v[1], v[0])
            # Yaw rate from derivative of arctan
            v_norm_sq = v[0]**2 + v[1]**2
            if v_norm_sq > 1e-6:
                yaw_rate = (v[0] * a_vec[1] - v[1] * a_vec[0]) / v_norm_sq
            else:
                yaw_rate = 0.0
        else:
            yaw = 0.0
            yaw_rate = 0.0

        return TrajPoint(p=p, v=v, a=a_vec, yaw=yaw, yaw_rate=yaw_rate)

    return traj_fn


def takeoff_and_hover(
    target_z: float = 1.0,
    takeoff_time: float = 2.0,
    yaw: float = 0.0,
) -> TrajectoryFn:
    """
    Create a takeoff trajectory: smoothly rise from ground to hover altitude.

    Args:
        target_z: Target hover altitude [m]
        takeoff_time: Time to reach target altitude [s]
        yaw: Fixed yaw angle [rad]

    Returns:
        Trajectory function t -> TrajPoint
    """
    return step_to(
        target_p=np.array([0.0, 0.0, target_z]),
        t_step=0.0,
        start_p=np.zeros(3),
        yaw=yaw,
    )


if __name__ == "__main__":
    """Quick tests for trajectory module."""
    print("Running trajectory tests...")

    # Test 1: Hover trajectory
    hover_fn = hover(z=1.5, yaw=0.5)
    traj = hover_fn(10.0)
    assert np.allclose(traj.p, [0, 0, 1.5]), f"Hover position wrong: {traj.p}"
    assert np.allclose(traj.v, 0.0), f"Hover velocity wrong: {traj.v}"
    assert np.allclose(traj.a, 0.0), f"Hover acceleration wrong: {traj.a}"
    assert traj.yaw == 0.5, f"Hover yaw wrong: {traj.yaw}"
    print("  [PASS] Hover trajectory")

    # Test 2: Circle trajectory
    circle_fn = circle(radius=1.0, speed=1.0, z=2.0)
    
    # At t=0, should be at (1, 0, 2)
    traj0 = circle_fn(0.0)
    assert np.allclose(traj0.p, [1, 0, 2], atol=1e-6), f"Circle t=0 position: {traj0.p}"
    
    # Velocity should be tangent (along +y initially)
    assert np.abs(traj0.v[0]) < 1e-6, f"Circle t=0 vx should be ~0: {traj0.v[0]}"
    assert traj0.v[1] > 0, f"Circle t=0 vy should be positive: {traj0.v[1]}"
    
    # Acceleration should point to center (along -x initially)
    assert traj0.a[0] < 0, f"Circle t=0 ax should be negative: {traj0.a[0]}"
    assert np.abs(traj0.a[1]) < 1e-6, f"Circle t=0 ay should be ~0: {traj0.a[1]}"
    print("  [PASS] Circle trajectory")

    # Test 3: Figure-8 trajectory
    fig8_fn = figure8(a=1.0, b=0.5, speed=0.5)
    
    # At t=0, should be at origin (x and y)
    traj0 = fig8_fn(0.0)
    assert np.abs(traj0.p[0]) < 1e-6, f"Figure-8 t=0 x should be 0: {traj0.p[0]}"
    assert np.abs(traj0.p[1]) < 1e-6, f"Figure-8 t=0 y should be 0: {traj0.p[1]}"
    print("  [PASS] Figure-8 trajectory")

    # Test 4: Step trajectory
    step_fn = step_to(target_p=np.array([1.0, 1.0, 1.0]), t_step=1.0)
    
    # Before step
    traj_before = step_fn(0.5)
    assert np.allclose(traj_before.p, [0, 0, 0]), f"Before step position: {traj_before.p}"
    
    # After step (t=3 is well after transition)
    traj_after = step_fn(5.0)
    assert np.allclose(traj_after.p, [1, 1, 1]), f"After step position: {traj_after.p}"
    assert np.allclose(traj_after.v, 0.0, atol=1e-6), f"After step velocity: {traj_after.v}"
    print("  [PASS] Step trajectory")

    # Test 5: Tangent yaw mode
    circle_tangent = circle(radius=1.0, speed=1.0, z=1.0, yaw_mode="tangent")
    traj = circle_tangent(0.0)
    # At t=0, velocity is in +y direction, so yaw should be pi/2
    assert np.abs(traj.yaw - np.pi/2) < 1e-6, f"Tangent yaw at t=0: {traj.yaw}"
    print("  [PASS] Tangent yaw mode")

    print("\nAll trajectory tests passed!")
