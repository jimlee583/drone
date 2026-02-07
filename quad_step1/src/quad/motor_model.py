"""
First-order actuator dynamics with saturation and rate limiting.

Models the lag between commanded and actual thrust/moments due to
motor inertia, ESC response time, and propeller aerodynamics.

Physical motivation:
- Real motors cannot change thrust instantaneously.
- ESCs and motor windings introduce a first-order lag.
- Propeller aerodynamics add further delay.
- The combined effect is well-approximated by a first-order system
  with time constant tau ~ 20-50 ms for small racing quads.

Continuous-time model:
    dT/dt = (T_cmd - T_actual) / tau_T          (thrust lag)
    dτ/dt = (τ_cmd - τ_actual) / tau_tau         (moment lag)

Discrete-time (exact ZOH):
    T[k+1] = T[k] + alpha_T * (T_cmd - T[k])
    where alpha_T = 1 - exp(-dt / tau_T)

Rate limiting is applied to the *output delta* of the first-order lag
to prevent physically impossible slew rates.

Hard saturation is applied AFTER integration to enforce actuator limits.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from quad.types import Control


@dataclass
class ActuatorParams:
    """
    Parameters for the first-order actuator model.

    Time Constants:
        tau_T: Thrust time constant [s].  Smaller = faster response.
               Typical: 0.02-0.05 s for small racing quads.
        tau_tau: Moment time constant [s].  Usually similar to tau_T.

    Rate Limits (slew rate):
        T_rate_max: Maximum thrust rate of change [N/s].
        tau_rate_max: Maximum moment rate of change per axis [N·m/s], shape (3,).

    Hard Saturation:
        T_min: Minimum achievable thrust [N] (motors can't reverse).
        T_max: Maximum achievable thrust [N].
        tau_max: Maximum moment per axis [N·m], shape (3,).

    Enable Flag:
        enabled: If False, actuator is a perfect pass-through (no lag).
    """

    # First-order time constants
    tau_T: float = 0.02        # 20 ms thrust lag
    tau_tau: float = 0.015     # 15 ms moment lag

    # Rate limits  (generous defaults — only clip extreme transients)
    T_rate_max: float = 200.0  # N/s — can slew full range in ~75 ms
    tau_rate_max: NDArray[np.float64] = field(
        default_factory=lambda: np.array([5.0, 5.0, 2.5])
    )  # N·m/s

    # Hard saturation (should match or be tighter than params.py limits)
    T_min: float = 0.0
    T_max: float = 15.0
    tau_max: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.1, 0.1, 0.05])
    )

    # Master enable
    enabled: bool = True


@dataclass
class ActuatorState:
    """
    Internal state of the actuator model.

    Tracks the *actual* (lagged) thrust and moments being delivered
    to the airframe at the current timestep.

    Attributes:
        thrust: Current actual thrust [N].
        moments: Current actual moments [N·m], shape (3,).
    """

    thrust: float = 0.0
    moments: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )

    def copy(self) -> "ActuatorState":
        return ActuatorState(
            thrust=self.thrust,
            moments=self.moments.copy(),
        )

    @staticmethod
    def from_hover(hover_thrust: float) -> "ActuatorState":
        """Initialize actuator state at hover equilibrium."""
        return ActuatorState(
            thrust=hover_thrust,
            moments=np.zeros(3),
        )


def step_actuator(
    act_state: ActuatorState,
    cmd: Control,
    act_params: ActuatorParams,
    dt: float,
) -> tuple[ActuatorState, Control]:
    """
    Advance the actuator model by one timestep.

    Pipeline per axis:
        1. Compute raw desired rate:  rate = (cmd - actual) / tau
        2. Clamp rate to slew-rate limit.
        3. Integrate:  actual += rate * dt   (Euler step, stable for dt << tau)
        4. Clamp to hard saturation limits.

    For better numerical behaviour when dt is not much smaller than tau,
    we use exact ZOH discretisation for the first-order lag and then
    apply rate limiting as a post-clamp on the effective rate.

    Args:
        act_state: Current actuator state (actual outputs).
        cmd: Commanded control from the controller.
        act_params: Actuator parameters.
        dt: Simulation timestep [s].

    Returns:
        (new_act_state, applied_control):
            new_act_state — updated actuator internal state.
            applied_control — the Control actually applied to the dynamics.
    """
    if not act_params.enabled:
        # Pass-through: no lag, no extra saturation beyond controller's own.
        return act_state.copy(), cmd

    # --- Thrust channel ---------------------------------------------------
    new_thrust = _step_first_order_rate_limited(
        actual=act_state.thrust,
        commanded=cmd.thrust_N,
        tau=act_params.tau_T,
        rate_max=act_params.T_rate_max,
        lo=act_params.T_min,
        hi=act_params.T_max,
        dt=dt,
    )

    # --- Moment channels (per-axis) ----------------------------------------
    new_moments = np.empty(3)
    for i in range(3):
        new_moments[i] = _step_first_order_rate_limited(
            actual=act_state.moments[i],
            commanded=cmd.moments_Nm[i],
            tau=act_params.tau_tau,
            rate_max=act_params.tau_rate_max[i],
            lo=-act_params.tau_max[i],
            hi=act_params.tau_max[i],
            dt=dt,
        )

    new_state = ActuatorState(thrust=new_thrust, moments=new_moments)
    applied = Control(thrust_N=new_thrust, moments_Nm=new_moments)
    return new_state, applied


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _step_first_order_rate_limited(
    actual: float,
    commanded: float,
    tau: float,
    rate_max: float,
    lo: float,
    hi: float,
    dt: float,
) -> float:
    """
    Advance a single first-order channel with rate limiting and saturation.

    Uses exact ZOH discretisation for the lag, then clips the effective
    rate so it never exceeds the physical slew-rate limit.

    Args:
        actual: Current actual value.
        commanded: Desired (commanded) value.
        tau: First-order time constant [s].
        rate_max: Maximum absolute rate of change [unit/s].
        lo, hi: Hard saturation bounds.
        dt: Timestep [s].

    Returns:
        New actual value after one timestep.
    """
    # Exact ZOH step:  y[k+1] = y[k] + alpha * (cmd - y[k])
    alpha = 1.0 - np.exp(-dt / tau)
    delta_lag = alpha * (commanded - actual)

    # Enforce slew-rate limit on the *effective* change
    max_delta = rate_max * dt
    delta_clamped = np.clip(delta_lag, -max_delta, max_delta)

    new_val = actual + delta_clamped

    # Hard saturation
    new_val = np.clip(new_val, lo, hi)

    return float(new_val)


if __name__ == "__main__":
    """Quick self-test for actuator model."""
    print("Running actuator model tests...")

    ap = ActuatorParams()

    # Test 1: Step response converges
    state = ActuatorState(thrust=0.0, moments=np.zeros(3))
    cmd = Control(thrust_N=5.0, moments_Nm=np.array([0.01, -0.01, 0.005]))
    dt = 0.002
    for _ in range(500):  # 1 second
        state, applied = step_actuator(state, cmd, ap, dt)
    assert abs(state.thrust - 5.0) < 0.01, f"Thrust should converge to 5.0, got {state.thrust}"
    print(f"  [PASS] Thrust converges: {state.thrust:.4f} N")

    # Test 2: Saturation
    state = ActuatorState(thrust=14.0, moments=np.zeros(3))
    cmd_big = Control(thrust_N=100.0, moments_Nm=np.array([1.0, 1.0, 1.0]))
    for _ in range(1000):
        state, applied = step_actuator(state, cmd_big, ap, dt)
    assert applied.thrust_N <= ap.T_max, f"Thrust should saturate at {ap.T_max}"
    assert np.all(np.abs(applied.moments_Nm) <= ap.tau_max + 1e-10), "Moments should saturate"
    print(f"  [PASS] Saturation: T={applied.thrust_N:.2f}, tau={applied.moments_Nm}")

    # Test 3: Disabled pass-through
    ap_off = ActuatorParams(enabled=False)
    state0 = ActuatorState(thrust=0.0, moments=np.zeros(3))
    cmd2 = Control(thrust_N=7.0, moments_Nm=np.array([0.05, 0.0, 0.0]))
    _, applied2 = step_actuator(state0, cmd2, ap_off, dt)
    assert applied2.thrust_N == 7.0, "Disabled actuator should pass through"
    print("  [PASS] Disabled pass-through")

    # Test 4: Rate limiting clips extreme transients
    state_rl = ActuatorState(thrust=0.0, moments=np.zeros(3))
    cmd_jump = Control(thrust_N=15.0, moments_Nm=np.zeros(3))
    state_rl, _ = step_actuator(state_rl, cmd_jump, ap, dt)
    effective_rate = state_rl.thrust / dt
    assert effective_rate <= ap.T_rate_max + 1e-6, (
        f"Rate should be limited to {ap.T_rate_max}, got {effective_rate}"
    )
    print(f"  [PASS] Rate limit: effective rate = {effective_rate:.1f} N/s")

    print("\nAll actuator model tests passed!")
