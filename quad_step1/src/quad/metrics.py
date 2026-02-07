"""
Evaluation metrics for quadrotor simulation logs.

All functions take a ``SimLog`` (and ``Params`` where limits are needed)
and return scalar or dict values suitable for tabulation.
"""

from __future__ import annotations

import numpy as np

from quad.types import SimLog
from quad.params import Params


def rms_position_error(log: SimLog) -> float:
    """RMS position tracking error [m]."""
    norms = np.linalg.norm(log.e_pos, axis=1)
    return float(np.sqrt(np.mean(norms**2)))


def max_position_error(log: SimLog) -> float:
    """Maximum position tracking error [m]."""
    norms = np.linalg.norm(log.e_pos, axis=1)
    return float(np.max(norms))


def rms_velocity_error(log: SimLog) -> float:
    """RMS velocity tracking error [m/s]."""
    norms = np.linalg.norm(log.e_vel, axis=1)
    return float(np.sqrt(np.mean(norms**2)))


def control_effort(log: SimLog) -> float:
    """Integrated squared control effort (thrust^2 + ||moments||^2)."""
    integrand = log.thrust**2 + np.sum(log.moments**2, axis=1)
    return float(np.trapezoid(integrand, log.t))


def thrust_saturation_pct(log: SimLog, params: Params) -> float:
    """Percentage of timesteps where thrust is within 1 % of T_min or T_max."""
    margin = 0.01 * (params.T_max - params.T_min)
    near_min = log.thrust <= params.T_min + margin
    near_max = log.thrust >= params.T_max - margin
    saturated = near_min | near_max
    return float(100.0 * np.mean(saturated))


def moment_saturation_pct(log: SimLog, params: Params) -> dict[str, float]:
    """Percentage of timesteps where moments are >= 99 % of tau_max.

    Returns dict with keys: roll, pitch, yaw, overall.
    """
    labels = ["roll", "pitch", "yaw"]
    result: dict[str, float] = {}
    any_sat = np.zeros(len(log.t), dtype=bool)
    for i, label in enumerate(labels):
        threshold = 0.99 * params.tau_max[i]
        sat = np.abs(log.moments[:, i]) >= threshold
        result[label] = float(100.0 * np.mean(sat))
        any_sat |= sat
    result["overall"] = float(100.0 * np.mean(any_sat))
    return result


def detect_crash(log: SimLog) -> bool:
    """Detect if the simulation crashed.

    Checks for:
      - ||p|| > 50 m  (diverged)
      - NaN in any state array
      - quaternion norm deviation > 0.01
    """
    # Position divergence
    if np.any(np.linalg.norm(log.p, axis=1) > 50.0):
        return True

    # NaN in state
    for arr in (log.p, log.v, log.q, log.w_body):
        if np.any(np.isnan(arr)):
            return True
        if np.any(np.isinf(arr)):
            return True

    # Quaternion norm check
    q_norms = np.linalg.norm(log.q, axis=1)
    if np.any(np.abs(q_norms - 1.0) > 0.01):
        return True

    return False


def compute_all_metrics(log: SimLog, params: Params) -> dict[str, float]:
    """Compute all metrics and return a flat dict.

    The ``crashed`` key is 1.0 if a crash was detected, 0.0 otherwise.
    Moment saturation keys are prefixed with ``moment_sat_``.
    """
    crashed = detect_crash(log)

    result: dict[str, float] = {
        "rms_pos_err": rms_position_error(log),
        "max_pos_err": max_position_error(log),
        "rms_vel_err": rms_velocity_error(log),
        "control_effort": control_effort(log),
        "thrust_sat_pct": thrust_saturation_pct(log, params),
    }

    moment_sat = moment_saturation_pct(log, params)
    for key, val in moment_sat.items():
        result[f"moment_sat_{key}"] = val

    result["crashed"] = 1.0 if crashed else 0.0

    return result


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from quad.params import default_params
    from quad.sim import run_sim
    from quad.trajectory import hover

    print("Running metrics self-test...")

    params = default_params()
    traj_fn = hover(z=1.0)
    log = run_sim(params, traj_fn, t_final=2.0, dt=0.002)

    rms_p = rms_position_error(log)
    max_p = max_position_error(log)
    rms_v = rms_velocity_error(log)
    effort = control_effort(log)
    t_sat = thrust_saturation_pct(log, params)
    m_sat = moment_saturation_pct(log, params)
    crashed = detect_crash(log)
    all_m = compute_all_metrics(log, params)

    print(f"  rms_pos_err  = {rms_p*1000:.2f} mm")
    print(f"  max_pos_err  = {max_p*1000:.2f} mm")
    print(f"  rms_vel_err  = {rms_v*1000:.2f} mm/s")
    print(f"  effort       = {effort:.2f}")
    print(f"  thrust_sat   = {t_sat:.1f} %")
    print(f"  moment_sat   = {m_sat}")
    print(f"  crashed      = {crashed}")

    # Sanity checks
    assert rms_p > 0, f"rms_pos should be > 0, got {rms_p}"
    assert rms_p < 1.0, f"rms_pos should be < 1 m for hover, got {rms_p}"
    print("  [PASS] rms_pos_err in (0, 1 m)")

    assert max_p >= rms_p, f"max_pos ({max_p}) should >= rms_pos ({rms_p})"
    print("  [PASS] max_pos_err >= rms_pos_err")

    assert effort > 0, f"effort should be > 0, got {effort}"
    print("  [PASS] effort > 0")

    assert 0 <= t_sat <= 100, f"thrust_sat out of range: {t_sat}"
    print("  [PASS] thrust_sat in [0, 100]")

    for k, v in m_sat.items():
        assert 0 <= v <= 100, f"moment_sat[{k}] out of range: {v}"
    print("  [PASS] moment_sat values in [0, 100]")

    assert not crashed, "hover should not crash"
    print("  [PASS] no crash detected")

    assert "rms_pos_err" in all_m
    assert "crashed" in all_m
    assert all_m["crashed"] == 0.0
    print("  [PASS] compute_all_metrics keys present")

    print("\nAll metrics self-tests passed!")
