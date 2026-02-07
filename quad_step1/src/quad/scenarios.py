"""
Evaluation scenario definitions and parameter randomization.

Provides a registry of named scenarios (trajectory + duration) and a
function that randomizes physical parameters for Monte Carlo evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from quad.params import Params
from quad.trajectory import TrajectoryFn, hover, step_to, circle, figure8
from quad.motor_model import ActuatorParams
from quad.disturbances import WindParams


@dataclass(frozen=True)
class ScenarioSpec:
    """Specification for an evaluation scenario."""

    name: str
    t_final: float
    traj_fn: Callable[[], TrajectoryFn]  # factory — call to get traj fn
    dt: float = 0.002
    params_overrides: dict | None = None  # reserved for future use


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

_SCENARIOS: dict[str, ScenarioSpec] = {}


def _register(spec: ScenarioSpec) -> None:
    _SCENARIOS[spec.name] = spec


_register(ScenarioSpec(
    name="hover",
    t_final=10.0,
    traj_fn=lambda: hover(z=1.0),
))

_register(ScenarioSpec(
    name="step",
    t_final=10.0,
    traj_fn=lambda: step_to(np.array([1.0, 0.0, 1.0]), t_step=1.0),
))

_register(ScenarioSpec(
    name="circle",
    t_final=30.0,
    traj_fn=lambda: circle(radius=3, speed=2, z=1.0),
))

_register(ScenarioSpec(
    name="figure8",
    t_final=40.0,
    traj_fn=lambda: figure8(a=2, b=1, speed=1.5, z=1.0),
))


def get_scenario(name: str) -> ScenarioSpec:
    """Return a scenario by name. Raises ``KeyError`` if unknown."""
    return _SCENARIOS[name]


def list_scenarios() -> list[str]:
    """Return sorted list of registered scenario names."""
    return sorted(_SCENARIOS)


# ---------------------------------------------------------------------------
# Parameter randomization
# ---------------------------------------------------------------------------

def randomize_params(base_params: Params, rng: np.random.Generator) -> Params:
    """Create a new ``Params`` with randomized physical properties.

    Randomization ranges:
      - mass: uniform +/-10 %
      - inertia diagonal: uniform +/-15 % per axis
      - motor time constants (tau_T, tau_tau): uniform +/-30 %
      - wind velocity: random direction, magnitude uniform [0, 3] m/s
      - gust seed: random integer

    All other fields are copied unchanged.  The original ``base_params``
    is never mutated.
    """
    # --- mass ---------------------------------------------------------------
    m = base_params.m * rng.uniform(0.9, 1.1)

    # --- inertia ------------------------------------------------------------
    J_diag = np.diag(base_params.J) * rng.uniform(0.85, 1.15, size=3)
    J = np.diag(J_diag)

    # --- actuator params (guarded) ------------------------------------------
    act_kw: dict = {}
    if hasattr(base_params, "actuator"):
        act = base_params.actuator
        act_kw["actuator"] = ActuatorParams(
            tau_T=act.tau_T * rng.uniform(0.7, 1.3),
            tau_tau=act.tau_tau * rng.uniform(0.7, 1.3),
            T_rate_max=act.T_rate_max,
            tau_rate_max=act.tau_rate_max.copy(),
            T_min=act.T_min,
            T_max=act.T_max,
            tau_max=act.tau_max.copy(),
            enabled=act.enabled,
        )

    # --- wind params (guarded) ----------------------------------------------
    wind_kw: dict = {}
    if hasattr(base_params, "wind"):
        wp = base_params.wind
        # Random direction on sphere (uniform)
        direction = rng.standard_normal(3)
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            direction = np.array([1.0, 0.0, 0.0])
        else:
            direction = direction / norm
        magnitude = rng.uniform(0.0, 3.0)
        wind_vel = direction * magnitude

        gust_seed = int(rng.integers(0, 2**31))

        wind_kw["wind"] = WindParams(
            wind_vel=wind_vel,
            gust_std=wp.gust_std,
            gust_tau=wp.gust_tau,
            k_drag=wp.k_drag,
            torque_std=wp.torque_std,
            enabled=wp.enabled,
            seed=gust_seed,
        )

    # --- estimator params (guarded) ------------------------------------------
    est_kw: dict = {}
    if getattr(base_params, "use_estimator", False):
        est_kw["use_estimator"] = True
        if hasattr(base_params, "estimator_params"):
            est_kw["estimator_params"] = base_params.estimator_params
        # Randomize sensor seed per trial
        from quad.sensors import SensorParams as _SP
        sp = getattr(base_params, "sensor_params", _SP())
        est_kw["sensor_params"] = _SP(
            gyro_noise_std=sp.gyro_noise_std,
            accel_noise_std=sp.accel_noise_std,
            gyro_bias_rw_std=sp.gyro_bias_rw_std,
            accel_bias_rw_std=sp.accel_bias_rw_std,
            alt_enabled=sp.alt_enabled,
            alt_noise_std=sp.alt_noise_std,
            alt_rate_hz=sp.alt_rate_hz,
            posfix_enabled=sp.posfix_enabled,
            posfix_noise_std=sp.posfix_noise_std,
            posfix_rate_hz=sp.posfix_rate_hz,
            seed=int(rng.integers(0, 2**31)),
        )

    return Params(
        m=m,
        J=J,
        g=base_params.g,
        T_min=base_params.T_min,
        T_max=base_params.T_max,
        tau_max=base_params.tau_max.copy(),
        kp_pos=base_params.kp_pos.copy(),
        kd_pos=base_params.kd_pos.copy(),
        kr_att=base_params.kr_att.copy(),
        kw_rate=base_params.kw_rate.copy(),
        drag_coeff=base_params.drag_coeff,
        **act_kw,
        **wind_kw,
        **est_kw,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from quad.params import default_params

    print("Running scenarios self-test...")

    # 1. list / get
    names = list_scenarios()
    assert names == ["circle", "figure8", "hover", "step"], f"Unexpected names: {names}"
    print(f"  [PASS] list_scenarios -> {names}")

    for name in names:
        s = get_scenario(name)
        assert s.name == name
        fn = s.traj_fn()
        pt = fn(0.0)
        assert pt.p.shape == (3,)
    print("  [PASS] get_scenario returns valid specs")

    # 2. invalid name raises KeyError
    try:
        get_scenario("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass
    print("  [PASS] invalid name raises KeyError")

    # 3. randomization bounds
    base = default_params()
    rng = np.random.default_rng(123)
    for _ in range(100):
        p = randomize_params(base, rng)
        assert 0.9 * base.m <= p.m <= 1.1 * base.m, f"mass out of range: {p.m}"
        for i in range(3):
            lo = 0.85 * np.diag(base.J)[i]
            hi = 1.15 * np.diag(base.J)[i]
            assert lo <= np.diag(p.J)[i] <= hi, f"J[{i}] out of range"
        if hasattr(p, "actuator"):
            assert 0.7 * base.actuator.tau_T <= p.actuator.tau_T <= 1.3 * base.actuator.tau_T
            assert 0.7 * base.actuator.tau_tau <= p.actuator.tau_tau <= 1.3 * base.actuator.tau_tau
        if hasattr(p, "wind"):
            assert np.linalg.norm(p.wind.wind_vel) <= 3.0 + 1e-9
    print("  [PASS] randomization bounds (100 samples)")

    # 4. determinism with same seed
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    p1 = randomize_params(base, rng1)
    p2 = randomize_params(base, rng2)
    assert p1.m == p2.m
    assert np.array_equal(p1.J, p2.J)
    print("  [PASS] determinism with same seed")

    print("\nAll scenarios self-tests passed!")
