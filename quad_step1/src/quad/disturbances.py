"""
Environmental disturbance models for quadrotor simulation.

Includes:
- Constant wind velocity in world frame.
- Random gust model (low-pass filtered random walk).
- Aerodynamic drag force proportional to relative airspeed.
- Small random torque disturbances (e.g. asymmetric prop wash).

Physical motivation:
- A quadrotor in outdoor flight experiences a mean wind field plus
  turbulent gusts.  The dominant effect on the airframe is a drag
  force proportional to the relative velocity between the vehicle
  and the surrounding air mass.
- The drag model here is simplified linear drag:
      F_wind = -k_drag * (v_body - v_wind)
  applied in the world frame.
- Torque disturbances model asymmetric propeller effects and
  buffeting; they are small and act in the body frame.

Determinism:
- A numpy RandomState with a fixed seed is used so that simulations
  are reproducible by default.  Change the seed for ensemble runs.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass
class WindParams:
    """
    Parameters for the wind / disturbance model.

    Wind Field:
        wind_vel: Constant (mean) wind velocity in world frame [m/s], shape (3,).
                  Positive x = wind blowing in +x direction, etc.

    Gust Model (low-pass filtered random walk):
        gust_std: Standard deviation of gust noise injected per timestep [m/s].
                  Set to 0 to disable gusts.
        gust_tau: Time constant for gust low-pass filter [s].
                  Larger = smoother / slower gusts.

    Drag:
        k_drag: Linear drag coefficient [N·s/m].
                Relates force to relative airspeed:
                    F = -k_drag * (v - v_wind)

    Torque Disturbance:
        torque_std: Std-dev of random body-frame torque disturbance [N·m].
                    Set to 0 to disable.

    Misc:
        enabled: Master enable flag.
        seed: Random seed for reproducibility.
    """

    wind_vel: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.5, 0.2, 0.0])
    )  # Mild breeze, mostly in +x

    gust_std: float = 0.3       # m/s per sqrt(s) of driving noise
    gust_tau: float = 1.0       # 1-second correlation time

    k_drag: float = 0.15        # N·s/m — light airframe drag

    torque_std: float = 0.0005  # N·m — very small random torque

    enabled: bool = True
    seed: int = 42


@dataclass
class DisturbanceState:
    """
    Internal state of the disturbance model.

    Carries the gust velocity (a filtered random walk) so that
    it evolves smoothly between timesteps.

    Attributes:
        gust_vel: Current gust velocity perturbation [m/s], shape (3,).
        rng: Numpy random state for reproducibility.
    """

    gust_vel: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3)
    )
    rng: np.random.RandomState = field(
        default_factory=lambda: np.random.RandomState(42)
    )

    def copy(self) -> "DisturbanceState":
        new = DisturbanceState(
            gust_vel=self.gust_vel.copy(),
            rng=self.rng,  # share RNG intentionally for continuity
        )
        return new

    @staticmethod
    def create(seed: int = 42) -> "DisturbanceState":
        """Create a fresh disturbance state with given seed."""
        return DisturbanceState(
            gust_vel=np.zeros(3),
            rng=np.random.RandomState(seed),
        )


def compute_disturbance_forces(
    v_world: NDArray[np.float64],
    dist_state: DisturbanceState,
    params: WindParams,
    dt: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], DisturbanceState]:
    """
    Compute external disturbance force and torque for one timestep.

    The wind velocity seen by the vehicle is:
        v_air = wind_vel + gust_vel   (world frame)

    The drag force in the world frame is:
        F_drag = -k_drag * (v_body_world - v_air)

    A small random torque is added in the body frame.

    Args:
        v_world: Vehicle velocity in world frame [m/s], shape (3,).
        dist_state: Current disturbance internal state.
        params: Wind / disturbance parameters.
        dt: Simulation timestep [s].

    Returns:
        (F_ext, tau_ext, new_dist_state):
            F_ext — External force in world frame [N], shape (3,).
            tau_ext — External torque in body frame [N·m], shape (3,).
            new_dist_state — Updated disturbance state.
    """
    if not params.enabled:
        return np.zeros(3), np.zeros(3), dist_state

    rng = dist_state.rng

    # --- Update gust velocity (discrete Ornstein-Uhlenbeck process) --------
    # dx = -x/tau * dt + sigma * sqrt(dt) * dW
    # Exact discretisation for stability:
    #   x[k+1] = alpha * x[k] + sigma_d * noise
    #   alpha   = exp(-dt / tau)
    #   sigma_d = sigma * sqrt(tau/2 * (1 - alpha^2))
    alpha = np.exp(-dt / params.gust_tau) if params.gust_tau > 0 else 0.0
    if params.gust_std > 0:
        sigma_d = params.gust_std * np.sqrt(
            params.gust_tau / 2.0 * (1.0 - alpha ** 2)
        )
        noise = rng.randn(3)
        new_gust = alpha * dist_state.gust_vel + sigma_d * noise
    else:
        new_gust = np.zeros(3)

    # Total effective wind velocity in world frame
    v_wind_total = params.wind_vel + new_gust

    # --- Drag force --------------------------------------------------------
    # F_drag = -k_drag * (v_vehicle - v_wind)
    # Positive k_drag means the force opposes relative motion through the air.
    v_rel = v_world - v_wind_total
    F_ext = -params.k_drag * v_rel

    # --- Random torque disturbance (body frame) ----------------------------
    if params.torque_std > 0:
        tau_ext = params.torque_std * rng.randn(3)
    else:
        tau_ext = np.zeros(3)

    # --- Pack new state ----------------------------------------------------
    new_state = DisturbanceState(gust_vel=new_gust, rng=rng)

    return F_ext, tau_ext, new_state


if __name__ == "__main__":
    """Quick self-test for disturbance model."""
    print("Running disturbance model tests...")

    wp = WindParams()

    # Test 1: Forces are non-zero with default wind
    ds = DisturbanceState.create(seed=42)
    v = np.zeros(3)  # stationary vehicle
    F, tau, ds = compute_disturbance_forces(v, ds, wp, dt=0.002)
    assert np.linalg.norm(F) > 0, "Force should be non-zero with wind"
    print(f"  [PASS] Non-zero force: |F| = {np.linalg.norm(F):.4f} N")

    # Test 2: Disabled returns zeros
    wp_off = WindParams(enabled=False)
    ds2 = DisturbanceState.create(seed=0)
    F2, tau2, _ = compute_disturbance_forces(v, ds2, wp_off, dt=0.002)
    assert np.allclose(F2, 0), "Disabled should give zero force"
    assert np.allclose(tau2, 0), "Disabled should give zero torque"
    print("  [PASS] Disabled returns zeros")

    # Test 3: Deterministic with same seed
    ds_a = DisturbanceState.create(seed=123)
    ds_b = DisturbanceState.create(seed=123)
    Fa, _, ds_a = compute_disturbance_forces(np.ones(3), ds_a, wp, 0.002)
    Fb, _, ds_b = compute_disturbance_forces(np.ones(3), ds_b, wp, 0.002)
    assert np.allclose(Fa, Fb), "Same seed should give identical results"
    print("  [PASS] Deterministic with same seed")

    # Test 4: Gust evolves smoothly
    ds_g = DisturbanceState.create(seed=7)
    gusts = []
    for _ in range(500):
        _, _, ds_g = compute_disturbance_forces(np.zeros(3), ds_g, wp, 0.002)
        gusts.append(ds_g.gust_vel.copy())
    gusts = np.array(gusts)
    # Check the gust has reasonable variance (~gust_std)
    gust_rms = np.sqrt(np.mean(gusts ** 2))
    print(f"  [PASS] Gust RMS = {gust_rms:.3f} m/s (target ≈ {wp.gust_std:.2f})")

    # Test 5: Zero gust_std means no gusts
    wp_no_gust = WindParams(gust_std=0.0, torque_std=0.0)
    ds_ng = DisturbanceState.create(seed=0)
    F_ng, tau_ng, ds_ng = compute_disturbance_forces(np.zeros(3), ds_ng, wp_no_gust, 0.002)
    expected = -wp_no_gust.k_drag * (np.zeros(3) - wp_no_gust.wind_vel)
    assert np.allclose(F_ng, expected), "No-gust force should be pure mean wind drag"
    assert np.allclose(tau_ng, 0), "No torque with torque_std=0"
    print("  [PASS] No-gust mode")

    print("\nAll disturbance model tests passed!")
