"""
Error-state Extended Kalman Filter (ES-EKF) for quadrotor state estimation.

Estimates position (3), velocity (3), attitude quaternion (4 stored,
3 in the error state), gyro bias (3), and accel bias (3).

Error-state vector (15-D):
    [dp (3), dv (3), dtheta (3), db_g (3), db_a (3)]

Prediction uses bias-corrected IMU (gyroscope + accelerometer).
Update steps for altimeter (1-D z) and position fix (3-D) are provided.

Coordinate conventions (consistent with existing codebase):
  - World frame: Z-up.  Gravity vector g_world = [0, 0, -g].
  - Body frame: Z-up through top of quad.
  - R rotates body -> world.
  - Quaternion: [w, x, y, z] scalar-first (Hamilton).
  - Quaternion error injection: q <- q * [1, 0.5*dtheta]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from quad.types import State
from quad.math3d import quat_to_R, quat_mul, quat_normalize, hat


# Dimension of the error-state vector.
DIM = 15

# Index slices for readability.
_P = slice(0, 3)      # dp
_V = slice(3, 6)      # dv
_TH = slice(6, 9)     # dtheta (attitude error)
_BG = slice(9, 12)    # db_g
_BA = slice(12, 15)   # db_a


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EstimatorParams:
    """Tuning knobs for the EKF.

    Initial covariance (diagonal entries):
        P0_pos, P0_vel, P0_att, P0_bg, P0_ba

    Process-noise PSD (continuous-time; discretised internally):
        Q_gyro, Q_accel, Q_bg, Q_ba

    Feature flags:
        use_alt_update:    Enable altimeter measurement update.
        use_posfix_update: Enable position-fix measurement update.

    Physics:
        g: Gravitational acceleration [m/s^2].  Must match ``Params.g``.
    """

    # Initial covariance
    P0_pos: float = 0.01       # m^2
    P0_vel: float = 0.01       # (m/s)^2
    P0_att: float = 0.01       # rad^2
    P0_bg: float = 1e-6        # (rad/s)^2
    P0_ba: float = 1e-4        # (m/s^2)^2

    # Process noise PSD
    Q_gyro: float = 1e-4       # rad^2/s
    Q_accel: float = 0.01      # (m/s^2)^2 * s
    Q_bg: float = 1e-8         # (rad/s)^2 * s
    Q_ba: float = 1e-6         # (m/s^2)^2 * s

    # Update enable flags
    use_alt_update: bool = True
    use_posfix_update: bool = True

    # Gravity (must match Params.g)
    g: float = 9.80665


@dataclass
class EstimatorState:
    """Nominal state + covariance carried by the EKF.

    Attributes:
        p:       Position estimate [m], (3,).
        v:       Velocity estimate [m/s], (3,).
        q:       Attitude quaternion [w,x,y,z], (4,).
        b_gyro:  Gyro bias estimate [rad/s], (3,).
        b_accel: Accel bias estimate [m/s^2], (3,).
        P:       Error-state covariance, (15, 15).
    """

    p: NDArray[np.float64]         # (3,)
    v: NDArray[np.float64]         # (3,)
    q: NDArray[np.float64]         # (4,) [w,x,y,z]
    b_gyro: NDArray[np.float64]    # (3,)
    b_accel: NDArray[np.float64]   # (3,)
    P: NDArray[np.float64]         # (15,15)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def init_estimator(
    params: EstimatorParams,
    x0_truth: Optional[State] = None,
) -> EstimatorState:
    """Create an initial estimator state.

    If *x0_truth* is given the nominal state is seeded from it (e.g.
    from an initial GPS fix or known launch pad).  Biases always start
    at zero.
    """
    if x0_truth is not None:
        p = x0_truth.p.copy()
        v = x0_truth.v.copy()
        q = quat_normalize(x0_truth.q.copy())
    else:
        p = np.zeros(3)
        v = np.zeros(3)
        q = np.array([1.0, 0.0, 0.0, 0.0])

    P = np.diag([
        params.P0_pos, params.P0_pos, params.P0_pos,
        params.P0_vel, params.P0_vel, params.P0_vel,
        params.P0_att, params.P0_att, params.P0_att,
        params.P0_bg,  params.P0_bg,  params.P0_bg,
        params.P0_ba,  params.P0_ba,  params.P0_ba,
    ])

    return EstimatorState(
        p=p, v=v, q=q,
        b_gyro=np.zeros(3), b_accel=np.zeros(3),
        P=P,
    )


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def ekf_predict(
    est: EstimatorState,
    gyro: NDArray[np.float64],
    accel: NDArray[np.float64],
    params: EstimatorParams,
    dt: float,
) -> EstimatorState:
    """IMU-driven prediction (propagate nominal state + covariance).

    Args:
        est:    Current estimator state.
        gyro:   Raw gyroscope measurement [rad/s], (3,).
        accel:  Raw accelerometer measurement [m/s^2], (3,).
        params: EKF tuning parameters.
        dt:     Timestep [s].

    Returns:
        Predicted estimator state.
    """
    # --- Bias-corrected IMU ---
    w_hat = gyro - est.b_gyro              # corrected angular rate
    a_hat = accel - est.b_accel            # corrected specific force (body)

    R = quat_to_R(est.q)                   # body -> world

    # --- Nominal state propagation ---

    # Quaternion: q <- q * [1, 0.5*w_hat*dt]
    half_w_dt = 0.5 * w_hat * dt
    dq = np.array([1.0, half_w_dt[0], half_w_dt[1], half_w_dt[2]])
    q_pred = quat_normalize(quat_mul(est.q, dq))

    # World-frame acceleration (using pre-update R)
    a_world = R @ a_hat + np.array([0.0, 0.0, -params.g])

    p_pred = est.p + est.v * dt + 0.5 * a_world * dt ** 2
    v_pred = est.v + a_world * dt

    # --- Error-state covariance propagation ---

    # Continuous-time Jacobian of the error dynamics (15x15)
    F = np.zeros((DIM, DIM))
    F[_P, _V] = np.eye(3)                        # dp_dot = dv
    F[_V, _TH] = -R @ hat(a_hat)                 # dv_dot ~ -R [a_hat]x dtheta
    F[_V, _BA] = -R                               # dv_dot ~ -R db_a
    F[_TH, _TH] = -hat(w_hat)                    # dtheta_dot ~ -[w_hat]x dtheta
    F[_TH, _BG] = -np.eye(3)                     # dtheta_dot ~ -db_g

    # First-order discrete transition: Phi ~ I + F*dt
    Phi = np.eye(DIM) + F * dt

    # Discrete process noise
    Q_d = np.zeros((DIM, DIM))
    Q_d[_V, _V] = params.Q_accel * dt * np.eye(3)
    Q_d[_TH, _TH] = params.Q_gyro * dt * np.eye(3)
    Q_d[_BG, _BG] = params.Q_bg * dt * np.eye(3)
    Q_d[_BA, _BA] = params.Q_ba * dt * np.eye(3)

    P_pred = Phi @ est.P @ Phi.T + Q_d
    P_pred = _symmetrise(P_pred)

    return EstimatorState(
        p=p_pred, v=v_pred, q=q_pred,
        b_gyro=est.b_gyro.copy(), b_accel=est.b_accel.copy(),
        P=P_pred,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _symmetrise(P: NDArray[np.float64]) -> NDArray[np.float64]:
    """Force symmetry and clamp tiny negative diagonal entries."""
    P = 0.5 * (P + P.T)
    diag = np.diag(P)
    np.fill_diagonal(P, np.maximum(diag, 1e-15))
    return P


def _ekf_update(
    est: EstimatorState,
    z: NDArray[np.float64],
    z_pred: NDArray[np.float64],
    H: NDArray[np.float64],
    R_cov: NDArray[np.float64],
) -> EstimatorState:
    """Generic EKF measurement update with error-state reset.

    Uses the Joseph form for the covariance update to improve
    numerical stability.
    """
    y = z - z_pred                          # innovation

    S = H @ est.P @ H.T + R_cov            # innovation covariance
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return est                           # singular -> skip update

    K = est.P @ H.T @ S_inv                 # Kalman gain
    dx = K @ y                              # error-state correction

    # Joseph-form covariance update
    I_KH = np.eye(DIM) - K @ H
    P_upd = I_KH @ est.P @ I_KH.T + K @ R_cov @ K.T
    P_upd = _symmetrise(P_upd)

    # --- Inject correction into nominal state ---
    p_upd = est.p + dx[_P]
    v_upd = est.v + dx[_V]

    # Attitude: q <- q * [1, 0.5*dtheta]
    dtheta = dx[_TH]
    dq = np.array([1.0, 0.5 * dtheta[0], 0.5 * dtheta[1], 0.5 * dtheta[2]])
    q_upd = quat_normalize(quat_mul(est.q, dq))

    b_gyro_upd = est.b_gyro + dx[_BG]
    b_accel_upd = est.b_accel + dx[_BA]

    return EstimatorState(
        p=p_upd, v=v_upd, q=q_upd,
        b_gyro=b_gyro_upd, b_accel=b_accel_upd,
        P=P_upd,
    )


# ---------------------------------------------------------------------------
# Public measurement updates
# ---------------------------------------------------------------------------

def ekf_update_altimeter(
    est: EstimatorState,
    z_meas: float,
    R_var: float,
) -> EstimatorState:
    """Altimeter update (scalar z-position).

    Args:
        est:    Current estimator state.
        z_meas: Measured altitude [m].
        R_var:  Measurement noise variance [m^2].
    """
    z = np.array([z_meas])
    z_pred = np.array([est.p[2]])
    H = np.zeros((1, DIM))
    H[0, 2] = 1.0                         # observes dp_z
    R_cov = np.array([[R_var]])
    return _ekf_update(est, z, z_pred, H, R_cov)


def ekf_update_posfix(
    est: EstimatorState,
    p_meas: NDArray[np.float64],
    R_cov: NDArray[np.float64],
) -> EstimatorState:
    """Position-fix update (3-D position).

    Args:
        est:    Current estimator state.
        p_meas: Measured position [m], (3,).
        R_cov:  Measurement noise covariance (3, 3).
    """
    z = p_meas
    z_pred = est.p
    H = np.zeros((3, DIM))
    H[0:3, 0:3] = np.eye(3)               # observes dp
    return _ekf_update(est, z, z_pred, H, R_cov)


# ---------------------------------------------------------------------------
# State extraction
# ---------------------------------------------------------------------------

def get_estimated_state(
    est: EstimatorState,
    gyro_meas: Optional[NDArray[np.float64]] = None,
) -> State:
    """Build a ``State`` from the estimator for the controller.

    Angular velocity is not directly part of the EKF state; instead
    the bias-corrected gyroscope measurement is used as the body-rate
    estimate.

    Args:
        est:       Current estimator state.
        gyro_meas: Latest raw gyro measurement (optional).  If given,
                   ``w_body`` is set to ``gyro - b_gyro``.
    """
    if gyro_meas is not None:
        w_hat = gyro_meas - est.b_gyro
    else:
        w_hat = np.zeros(3)

    return State(
        p=est.p.copy(),
        v=est.v.copy(),
        q=quat_normalize(est.q.copy()),
        w_body=w_hat.copy(),
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running EKF self-test...")

    from quad.sensors import SensorParams, init_sensor_state, sample_measurements

    ep = EstimatorParams()

    # Test 1: Initialisation from truth
    x0 = State.zeros()
    x0.p = np.array([0.0, 0.0, 1.0])
    est = init_estimator(ep, x0_truth=x0)
    assert np.allclose(est.p, x0.p), f"Init p mismatch: {est.p}"
    assert est.P.shape == (DIM, DIM), f"P shape: {est.P.shape}"
    print("  [PASS] Initialisation from truth")

    # Test 2: Predict with hover IMU (accel = [0,0,g], gyro = 0)
    gyro = np.zeros(3)
    accel = np.array([0.0, 0.0, 9.80665])
    est_pred = ekf_predict(est, gyro, accel, ep, dt=0.002)
    # At hover, position should barely change
    assert np.allclose(est_pred.p, x0.p, atol=1e-4), \
        f"Hover predict p: {est_pred.p}"
    assert np.allclose(est_pred.v, np.zeros(3), atol=1e-3), \
        f"Hover predict v: {est_pred.v}"
    print("  [PASS] Hover predict stable")

    # Test 3: Altimeter update corrects z
    est_off = init_estimator(ep, x0_truth=x0)
    est_off.p[2] = 0.9  # intentionally wrong
    est_corr = ekf_update_altimeter(est_off, 1.0, 0.05**2)
    assert abs(est_corr.p[2] - 1.0) < abs(est_off.p[2] - 1.0), \
        f"Altimeter update should move z closer to 1.0, got {est_corr.p[2]}"
    print(f"  [PASS] Altimeter correction: {est_off.p[2]:.3f} -> {est_corr.p[2]:.3f}")

    # Test 4: Position fix update corrects all axes
    est_off2 = init_estimator(ep, x0_truth=x0)
    est_off2.p = np.array([0.1, 0.1, 0.9])
    R_cov = 0.02**2 * np.eye(3)
    est_corr2 = ekf_update_posfix(est_off2, np.array([0.0, 0.0, 1.0]), R_cov)
    err_before = np.linalg.norm(est_off2.p - np.array([0.0, 0.0, 1.0]))
    err_after = np.linalg.norm(est_corr2.p - np.array([0.0, 0.0, 1.0]))
    assert err_after < err_before, "Position fix should reduce error"
    print(f"  [PASS] Position fix correction: err {err_before:.4f} -> {err_after:.4f}")

    # Test 5: get_estimated_state returns valid State
    s = get_estimated_state(est, gyro_meas=np.array([0.01, -0.01, 0.0]))
    assert s.p.shape == (3,)
    assert s.q.shape == (4,)
    assert abs(np.linalg.norm(s.q) - 1.0) < 1e-10
    print("  [PASS] get_estimated_state valid")

    # Test 6: P remains symmetric and positive after many steps
    sp = SensorParams()
    ss = init_sensor_state(seed=7, params=sp)
    truth = State.zeros()
    truth.p = np.array([0.0, 0.0, 1.0])
    v_dot = np.zeros(3)
    est_run = init_estimator(ep, x0_truth=truth)
    for i in range(2000):
        t = i * 0.002
        meas, ss = sample_measurements(t, 0.002, truth, v_dot, sp, ss)
        est_run = ekf_predict(est_run, meas.gyro, meas.accel, ep, 0.002)
        if meas.alt is not None:
            est_run = ekf_update_altimeter(est_run, meas.alt, sp.alt_noise_std**2)
        if meas.pos_fix is not None:
            est_run = ekf_update_posfix(
                est_run, meas.pos_fix, sp.posfix_noise_std**2 * np.eye(3),
            )
    assert np.allclose(est_run.P, est_run.P.T, atol=1e-12), "P not symmetric"
    assert np.all(np.diag(est_run.P) > 0), "P diagonal not positive"
    assert not np.any(np.isnan(est_run.P)), "P contains NaN"
    pos_err = np.linalg.norm(est_run.p - truth.p)
    assert pos_err < 0.1, f"Position error too large after 4s hover: {pos_err:.4f}"
    print(f"  [PASS] Numerical stability (2000 steps, pos_err={pos_err*1000:.1f} mm)")

    print("\nAll EKF self-tests passed!")
