"""
Geometric SE(3) tracking controller for quadrotor.

Implements the controller from:
Lee, T., Leok, M., & McClamroch, N. H. (2010).
"Geometric Tracking Control of a Quadrotor UAV on SE(3)"

The controller computes thrust and body moments to track a
desired trajectory (position, velocity, acceleration, yaw).
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from quad.types import State, Control, TrajPoint
from quad.params import Params
from quad.math3d import quat_to_R, hat, vee, safe_normalize


@dataclass
class ControllerState:
    """
    Internal controller state and computed errors.

    Used for logging and debugging.
    """

    e_pos: NDArray[np.float64]  # Position error
    e_vel: NDArray[np.float64]  # Velocity error
    e_att: NDArray[np.float64]  # Attitude error (vee of skew)
    e_rate: NDArray[np.float64]  # Angular rate error
    R_des: NDArray[np.float64]  # Desired rotation matrix
    a_cmd: NDArray[np.float64]  # Commanded acceleration


def compute_desired_rotation(
    a_cmd: NDArray[np.float64],
    yaw: float,
) -> NDArray[np.float64]:
    """
    Compute desired rotation matrix from commanded acceleration and yaw.

    The desired body z-axis (b3d) points along the commanded acceleration.
    The desired body x-axis (b1d) is constructed from yaw angle while
    remaining orthogonal to b3d.

    Args:
        a_cmd: Commanded acceleration vector (unnormalized), shape (3,)
        yaw: Desired yaw angle [rad]

    Returns:
        Desired rotation matrix R_des, shape (3, 3)
    """
    # Desired body z-axis (thrust direction)
    # Handle near-zero case: if a_cmd is too small, default to hover orientation
    e3 = np.array([0.0, 0.0, 1.0])
    b3d = safe_normalize(a_cmd, fallback=e3)

    # Desired heading direction from yaw
    b1c = np.array([np.cos(yaw), np.sin(yaw), 0.0])

    # Construct orthonormal frame
    # b2d = b3d × b1c (perpendicular to thrust and heading)
    b2d_raw = np.cross(b3d, b1c)
    
    # Handle degenerate case when b3d is parallel to b1c
    if np.linalg.norm(b2d_raw) < 1e-6:
        # b3d is nearly vertical and aligned with desired heading
        # Use a perpendicular vector
        b1c_alt = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        b2d_raw = np.cross(b3d, b1c_alt)
    
    b2d = safe_normalize(b2d_raw, fallback=np.array([0.0, 1.0, 0.0]))

    # b1d = b2d × b3d (complete the right-handed frame)
    b1d = np.cross(b2d, b3d)

    # Construct rotation matrix [b1d | b2d | b3d]
    R_des = np.column_stack([b1d, b2d, b3d])

    return R_des


def attitude_error(
    R: NDArray[np.float64],
    R_des: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute attitude error on SO(3).

    The error is defined as:
        e_R = 0.5 * vee(R_des^T @ R - R^T @ R_des)

    This error is zero when R = R_des and has nice properties
    for stability analysis.

    Args:
        R: Current rotation matrix, shape (3, 3)
        R_des: Desired rotation matrix, shape (3, 3)

    Returns:
        Attitude error vector, shape (3,)
    """
    error_matrix = R_des.T @ R - R.T @ R_des
    e_R = 0.5 * vee(error_matrix)
    return e_R


def angular_rate_error(
    w: NDArray[np.float64],
    w_des: NDArray[np.float64],
    R: NDArray[np.float64],
    R_des: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Compute angular rate error.

    The desired body rates are transformed to current body frame:
        e_w = w - R^T @ R_des @ w_des

    For simplicity in trajectory tracking, we often set w_des = 0
    unless explicit body rate commands are given.

    Args:
        w: Current body angular velocity, shape (3,)
        w_des: Desired body angular velocity, shape (3,)
        R: Current rotation matrix, shape (3, 3)
        R_des: Desired rotation matrix, shape (3, 3)

    Returns:
        Angular rate error in body frame, shape (3,)
    """
    # Transform desired rates to current body frame
    w_des_in_current = R.T @ R_des @ w_des
    e_w = w - w_des_in_current
    return e_w


def se3_controller(
    state: State,
    traj: TrajPoint,
    params: Params,
) -> Tuple[Control, ControllerState]:
    """
    Compute SE(3) geometric tracking control.

    The controller works in two stages:
    1. Position control: compute desired thrust direction and magnitude
    2. Attitude control: compute body moments to track desired orientation

    Args:
        state: Current quadrotor state
        traj: Desired trajectory point
        params: Controller and system parameters

    Returns:
        Tuple of (Control, ControllerState)
    """
    # Current state
    p = state.p
    v = state.v
    q = state.q
    w = state.w_body

    # Desired trajectory
    p_des = traj.p
    v_des = traj.v
    a_des = traj.a
    yaw_des = traj.yaw
    yaw_rate_des = traj.yaw_rate

    # Current rotation matrix
    R = quat_to_R(q)

    # =========================================================================
    # Position Control
    # =========================================================================

    # Position and velocity errors
    e_pos = p - p_des
    e_vel = v - v_des

    # Desired acceleration command
    # a_cmd = a_des - Kp * e_pos - Kd * e_vel + g * e3
    # The +g term compensates for gravity
    e3 = np.array([0.0, 0.0, 1.0])
    a_cmd = (
        a_des
        - params.kp_pos * e_pos
        - params.kd_pos * e_vel
        + params.g * e3
    )

    # Compute thrust magnitude
    # T = m * a_cmd · (R @ e3)
    # This is the projection of desired force onto current body z-axis
    body_z_world = R @ e3
    thrust = params.m * np.dot(a_cmd, body_z_world)

    # =========================================================================
    # Attitude Control
    # =========================================================================

    # Compute desired rotation from commanded acceleration and yaw
    R_des = compute_desired_rotation(a_cmd, yaw_des)

    # Attitude error
    e_att = attitude_error(R, R_des)

    # Desired body angular velocity
    # For trajectory tracking with yaw rate, we set:
    #   w_des = [0, 0, yaw_rate] in world frame, need to map to body
    # Simplified: assume w_des = 0 for position tracking, add yaw rate term
    # The yaw rate only affects the z-component in a hover-like attitude
    w_des_world = np.array([0.0, 0.0, yaw_rate_des])
    w_des_body = R_des.T @ w_des_world

    # Angular rate error
    e_rate = angular_rate_error(w, w_des_body, R, R_des)

    # Compute moments
    # tau = -Kr * e_att - Kw * e_rate + w × (J @ w)
    # The gyroscopic term (w × Jw) provides feedforward compensation
    Jw = params.J @ w
    gyroscopic = np.cross(w, Jw)

    moments = (
        -params.kr_att * e_att
        - params.kw_rate * e_rate
        + gyroscopic
    )

    # =========================================================================
    # Saturation
    # =========================================================================

    # Apply thrust limits
    thrust_sat = np.clip(thrust, params.T_min, params.T_max)

    # Apply moment limits
    moments_sat = np.clip(moments, -params.tau_max, params.tau_max)

    # Create output
    control = Control(thrust_N=thrust_sat, moments_Nm=moments_sat)

    controller_state = ControllerState(
        e_pos=e_pos,
        e_vel=e_vel,
        e_att=e_att,
        e_rate=e_rate,
        R_des=R_des,
        a_cmd=a_cmd,
    )

    return control, controller_state


if __name__ == "__main__":
    """Quick tests for controller module."""
    print("Running controller tests...")

    from quad.math3d import R_to_quat

    params = Params()

    # Test 1: Hover equilibrium
    # At hover, thrust should equal weight and moments should be zero
    state = State.zeros()
    state.p = np.array([0.0, 0.0, 1.0])

    traj = TrajPoint(
        p=np.array([0.0, 0.0, 1.0]),
        v=np.zeros(3),
        a=np.zeros(3),
        yaw=0.0,
        yaw_rate=0.0,
    )

    control, ctrl_state = se3_controller(state, traj, params)

    # At hover with no error, thrust should be mg
    expected_thrust = params.m * params.g
    assert np.abs(control.thrust_N - expected_thrust) < 1e-6, \
        f"Hover thrust should be {expected_thrust}, got {control.thrust_N}"
    print(f"  [PASS] Hover thrust: {control.thrust_N:.3f} N (expected {expected_thrust:.3f} N)")

    # Moments should be zero at hover equilibrium
    assert np.allclose(control.moments_Nm, 0.0, atol=1e-6), \
        f"Hover moments should be zero, got {control.moments_Nm}"
    print("  [PASS] Hover moments: zero")

    # Test 2: Position error creates correct thrust direction
    state = State.zeros()
    state.p = np.array([0.0, 0.0, 0.5])  # Below target

    traj = TrajPoint(
        p=np.array([0.0, 0.0, 1.0]),
        v=np.zeros(3),
        a=np.zeros(3),
        yaw=0.0,
        yaw_rate=0.0,
    )

    control, ctrl_state = se3_controller(state, traj, params)

    # Should have higher thrust to climb
    assert control.thrust_N > expected_thrust, \
        f"Thrust should be above hover to climb, got {control.thrust_N}"
    print(f"  [PASS] Climbing thrust: {control.thrust_N:.3f} N > {expected_thrust:.3f} N")

    # Test 3: Desired rotation construction
    a_cmd = np.array([0.0, 0.0, 10.0])  # Vertical thrust
    yaw = 0.0
    R_des = compute_desired_rotation(a_cmd, yaw)

    # Should be close to identity for vertical thrust with zero yaw
    assert np.allclose(R_des, np.eye(3), atol=1e-6), \
        f"R_des for vertical thrust should be identity, got\n{R_des}"
    print("  [PASS] Desired rotation for hover")

    # Test 4: Yaw affects desired rotation
    a_cmd = np.array([0.0, 0.0, 10.0])
    yaw = np.pi / 2  # 90 degrees
    R_des = compute_desired_rotation(a_cmd, yaw)

    # Body x-axis should point along world y (after 90 deg yaw)
    b1 = R_des[:, 0]
    assert np.allclose(b1, [0, 1, 0], atol=1e-6), \
        f"Body x should point along world y for 90 deg yaw, got {b1}"
    print("  [PASS] Desired rotation with yaw")

    # Test 5: Attitude error is zero when R = R_des
    R = np.eye(3)
    R_des = np.eye(3)
    e_R = attitude_error(R, R_des)
    assert np.allclose(e_R, 0.0, atol=1e-10), f"Attitude error should be zero, got {e_R}"
    print("  [PASS] Zero attitude error at equilibrium")

    # Test 6: Saturation works
    state = State.zeros()
    state.p = np.array([0.0, 0.0, -10.0])  # Way below target

    traj = TrajPoint(
        p=np.array([0.0, 0.0, 1.0]),
        v=np.zeros(3),
        a=np.zeros(3),
        yaw=0.0,
        yaw_rate=0.0,
    )

    control, _ = se3_controller(state, traj, params)
    assert control.thrust_N == params.T_max, \
        f"Thrust should be saturated at max, got {control.thrust_N}"
    print(f"  [PASS] Thrust saturation: {control.thrust_N:.3f} N")

    print("\nAll controller tests passed!")
