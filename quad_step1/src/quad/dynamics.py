"""
Quadrotor rigid body dynamics.

Implements continuous-time dynamics and RK4 integration.
Uses quaternion attitude representation throughout.
"""

import numpy as np
from numpy.typing import NDArray

from quad.types import State, Control
from quad.params import Params
from quad.math3d import quat_normalize, quat_to_R


def omega_matrix(w: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Construct the quaternion derivative matrix Omega(w).

    For quaternion kinematics: q_dot = 0.5 * Omega(w) @ q

    Args:
        w: Angular velocity in body frame [rad/s], shape (3,)

    Returns:
        Omega matrix, shape (4, 4)
    """
    wx, wy, wz = w
    return np.array([
        [0.0, -wx, -wy, -wz],
        [wx,  0.0,  wz, -wy],
        [wy, -wz,  0.0,  wx],
        [wz,  wy, -wx,  0.0],
    ])


def state_derivative(
    state: State,
    control: Control,
    params: Params,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute time derivatives of all state components.

    Dynamics:
        p_dot = v
        v_dot = [0, 0, -g] + (1/m) * R(q) @ [0, 0, T] - drag * v
        q_dot = 0.5 * Omega(w_body) @ q
        w_dot = J^{-1} * (tau - w × (J @ w))

    Args:
        state: Current state
        control: Control inputs
        params: System parameters

    Returns:
        Tuple of (p_dot, v_dot, q_dot, w_dot)
    """
    # Unpack state
    p = state.p
    v = state.v
    q = state.q
    w = state.w_body

    # Unpack control
    T = control.thrust_N
    tau = control.moments_Nm

    # Rotation matrix from body to world
    R = quat_to_R(q)

    # Thrust vector in world frame
    # Body z-axis points up, thrust acts along body z
    thrust_body = np.array([0.0, 0.0, T])
    thrust_world = R @ thrust_body

    # Position derivative
    p_dot = v

    # Velocity derivative
    # Gravity points down in world frame (negative z)
    gravity = np.array([0.0, 0.0, -params.g])
    
    # Optional linear drag
    drag = -params.drag_coeff * v if params.drag_coeff > 0 else np.zeros(3)
    
    v_dot = gravity + (1.0 / params.m) * thrust_world + drag

    # Quaternion derivative (quaternion kinematics)
    Omega = omega_matrix(w)
    q_dot = 0.5 * Omega @ q

    # Angular velocity derivative (Euler's equation)
    # tau = J @ w_dot + w × (J @ w)
    # w_dot = J^{-1} @ (tau - w × (J @ w))
    Jw = params.J @ w
    gyroscopic = np.cross(w, Jw)
    w_dot = params.J_inv @ (tau - gyroscopic)

    return p_dot, v_dot, q_dot, w_dot


def step_euler(
    state: State,
    control: Control,
    params: Params,
    dt: float,
) -> State:
    """
    Simple forward Euler integration step.

    Less accurate than RK4, but useful for comparison/debugging.

    Args:
        state: Current state
        control: Control inputs
        params: System parameters
        dt: Time step [s]

    Returns:
        Next state after dt
    """
    p_dot, v_dot, q_dot, w_dot = state_derivative(state, control, params)

    # Integrate
    new_p = state.p + dt * p_dot
    new_v = state.v + dt * v_dot
    new_q = state.q + dt * q_dot
    new_w = state.w_body + dt * w_dot

    # Normalize quaternion to prevent drift
    new_q = quat_normalize(new_q)

    return State(p=new_p, v=new_v, q=new_q, w_body=new_w)


def step_rk4(
    state: State,
    control: Control,
    params: Params,
    dt: float,
) -> State:
    """
    4th-order Runge-Kutta integration step.

    Provides good accuracy for the quadrotor dynamics.
    Control is assumed constant over the timestep.

    Args:
        state: Current state
        control: Control inputs
        params: System parameters
        dt: Time step [s]

    Returns:
        Next state after dt
    """
    # Helper to pack/unpack state as single array
    def pack(s: State) -> NDArray[np.float64]:
        return np.concatenate([s.p, s.v, s.q, s.w_body])

    def unpack(x: NDArray[np.float64]) -> State:
        return State(
            p=x[0:3],
            v=x[3:6],
            q=x[6:10],
            w_body=x[10:13],
        )

    def f(x: NDArray[np.float64]) -> NDArray[np.float64]:
        s = unpack(x)
        p_dot, v_dot, q_dot, w_dot = state_derivative(s, control, params)
        return np.concatenate([p_dot, v_dot, q_dot, w_dot])

    # RK4 integration
    x = pack(state)
    k1 = f(x)
    k2 = f(x + 0.5 * dt * k1)
    k3 = f(x + 0.5 * dt * k2)
    k4 = f(x + dt * k3)

    x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # Unpack and normalize quaternion
    new_state = unpack(x_next)
    new_state.q = quat_normalize(new_state.q)

    return new_state


def saturate_control(control: Control, params: Params) -> Control:
    """
    Apply actuator saturation limits to control inputs.

    Args:
        control: Unsaturated control
        params: System parameters with limits

    Returns:
        Saturated control
    """
    # Saturate thrust
    thrust_sat = np.clip(control.thrust_N, params.T_min, params.T_max)

    # Saturate moments element-wise
    moments_sat = np.clip(control.moments_Nm, -params.tau_max, params.tau_max)

    return Control(thrust_N=thrust_sat, moments_Nm=moments_sat)


if __name__ == "__main__":
    """Quick tests for dynamics module."""
    print("Running dynamics tests...")

    # Test 1: Hover equilibrium
    params = Params()
    state = State.zeros()
    state.p = np.array([0.0, 0.0, 1.0])
    
    # Hover thrust should keep velocity constant
    control = Control(thrust_N=params.hover_thrust, moments_Nm=np.zeros(3))
    
    _, v_dot, _, w_dot = state_derivative(state, control, params)
    assert np.allclose(v_dot, 0.0, atol=1e-10), f"Hover v_dot should be zero, got {v_dot}"
    assert np.allclose(w_dot, 0.0, atol=1e-10), f"Hover w_dot should be zero, got {w_dot}"
    print("  [PASS] Hover equilibrium")

    # Test 2: Free fall (zero thrust)
    control_zero = Control(thrust_N=0.0, moments_Nm=np.zeros(3))
    _, v_dot, _, _ = state_derivative(state, control_zero, params)
    assert np.allclose(v_dot, [0, 0, -params.g], atol=1e-10), "Zero thrust should give free fall"
    print("  [PASS] Free fall")

    # Test 3: RK4 conserves quaternion norm
    state = State.zeros()
    control = Control(thrust_N=5.0, moments_Nm=np.array([0.01, 0.01, 0.0]))
    
    for _ in range(1000):
        state = step_rk4(state, control, params, dt=0.002)
    
    q_norm = np.linalg.norm(state.q)
    assert np.abs(q_norm - 1.0) < 1e-10, f"Quaternion norm should be 1, got {q_norm}"
    print("  [PASS] RK4 quaternion norm preservation")

    # Test 4: Omega matrix is skew-symmetric in upper 3x3
    w = np.array([1.0, 2.0, 3.0])
    Omega = omega_matrix(w)
    assert np.allclose(Omega, -Omega.T), "Omega should be skew-symmetric"
    print("  [PASS] Omega skew-symmetry")

    # Test 5: Saturation
    control_big = Control(thrust_N=100.0, moments_Nm=np.array([1.0, 1.0, 1.0]))
    control_sat = saturate_control(control_big, params)
    assert control_sat.thrust_N == params.T_max, "Thrust should be saturated"
    assert np.all(np.abs(control_sat.moments_Nm) <= params.tau_max), "Moments should be saturated"
    print("  [PASS] Control saturation")

    print("\nAll dynamics tests passed!")
