"""
Sensor simulation for quadrotor state estimation.

Models IMU (gyroscope + accelerometer), optional barometric altimeter,
and optional low-rate position fix (stand-in for vision / GPS).

All measurements are generated from the TRUTH state with configurable
white noise and bias random-walk.  Deterministic by default via a
seeded numpy ``Generator``.

Coordinate conventions (consistent with existing codebase):
  - World frame: Z-up.  Gravity = [0, 0, -g].
  - Body frame: Z-up through top of quad.
  - Rotation matrix R rotates body -> world: v_world = R @ v_body.
  - Quaternion: [w, x, y, z] scalar-first (Hamilton).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from quad.types import State
from quad.math3d import quat_to_R


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SensorParams:
    """Parameters for the sensor noise model.

    IMU (always sampled every sim step):
        gyro_noise_std:      White-noise std on gyroscope [rad/s].
        accel_noise_std:     White-noise std on accelerometer [m/s^2].
        gyro_bias_rw_std:    Gyro bias random-walk driving std [rad/s / sqrt(s)].
        accel_bias_rw_std:   Accel bias random-walk driving std [m/s^2 / sqrt(s)].

    Altimeter (barometer stand-in):
        alt_enabled:    Enable altimeter measurements.
        alt_noise_std:  White-noise std on altitude [m].
        alt_rate_hz:    Altimeter update rate [Hz].

    Position fix (vision / GPS stand-in):
        posfix_enabled:    Enable low-rate position fix.
        posfix_noise_std:  White-noise std on position [m].
        posfix_rate_hz:    Update rate [Hz].
    """

    # --- IMU ---
    gyro_noise_std: float = 0.01        # rad/s
    accel_noise_std: float = 0.1        # m/s^2
    gyro_bias_rw_std: float = 0.0001    # rad/s per sqrt(s)
    accel_bias_rw_std: float = 0.001    # m/s^2 per sqrt(s)

    # --- Altimeter ---
    alt_enabled: bool = True
    alt_noise_std: float = 0.05         # m
    alt_rate_hz: float = 50.0           # Hz

    # --- Position fix ---
    posfix_enabled: bool = True
    posfix_noise_std: float = 0.02      # m
    posfix_rate_hz: float = 20.0        # Hz

    # --- Seed (for RNG in SensorState) ---
    seed: int = 0


@dataclass
class SensorState:
    """Mutable internal state of the sensor model.

    Attributes:
        b_gyro:        Current gyro bias [rad/s], shape (3,).
        b_accel:       Current accel bias [m/s^2], shape (3,).
        t_last_alt:    Time of last altimeter sample [s].
        t_last_posfix: Time of last position-fix sample [s].
        rng:           numpy Generator for deterministic noise.
    """

    b_gyro: NDArray[np.float64]         # (3,)
    b_accel: NDArray[np.float64]        # (3,)
    t_last_alt: float
    t_last_posfix: float
    rng: np.random.Generator


@dataclass
class Measurement:
    """Packet of sensor measurements at a single timestep.

    ``gyro`` and ``accel`` are always present (IMU runs every step).
    ``alt`` and ``pos_fix`` are *None* when the respective sensor did
    not fire this timestep.
    """

    gyro: NDArray[np.float64]                        # (3,) always
    accel: NDArray[np.float64]                       # (3,) always
    alt: Optional[float] = None                      # scalar altitude
    pos_fix: Optional[NDArray[np.float64]] = None    # (3,) position


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_sensor_state(seed: int, params: SensorParams) -> SensorState:
    """Create a fresh sensor state with zero biases and seeded RNG."""
    return SensorState(
        b_gyro=np.zeros(3),
        b_accel=np.zeros(3),
        t_last_alt=-np.inf,
        t_last_posfix=-np.inf,
        rng=np.random.default_rng(seed),
    )


def sample_measurements(
    t: float,
    dt: float,
    truth: State,
    truth_v_dot: NDArray[np.float64],
    params: SensorParams,
    sensor_state: SensorState,
    g: float = 9.80665,
) -> tuple[Measurement, SensorState]:
    """Generate one timestep of sensor measurements from the truth state.

    Args:
        t:            Current simulation time [s].
        dt:           Simulation timestep [s].
        truth:        Current truth state of the quadrotor.
        truth_v_dot:  World-frame linear acceleration [m/s^2] of the truth
                      state (used for the accelerometer model).
        params:       Sensor noise / rate parameters.
        sensor_state: Current internal sensor state (biases, RNG, ...).
        g:            Gravitational acceleration [m/s^2].  Must match the
                      value used in ``Params.g``.

    Returns:
        ``(measurement, new_sensor_state)``
    """
    rng = sensor_state.rng
    sqrt_dt = np.sqrt(dt)

    # ------------------------------------------------------------------
    # 1. Evolve bias random walks:  b_{k+1} = b_k + sigma_rw * sqrt(dt) * N(0, I)
    # ------------------------------------------------------------------
    b_gyro_new = (
        sensor_state.b_gyro
        + params.gyro_bias_rw_std * sqrt_dt * rng.standard_normal(3)
    )
    b_accel_new = (
        sensor_state.b_accel
        + params.accel_bias_rw_std * sqrt_dt * rng.standard_normal(3)
    )

    # ------------------------------------------------------------------
    # 2. Rotation matrix (body -> world)
    # ------------------------------------------------------------------
    R = quat_to_R(truth.q)

    # ------------------------------------------------------------------
    # 3. Gyroscope (body frame)
    #    w_meas = w_true + b_g + n_g
    # ------------------------------------------------------------------
    gyro = (
        truth.w_body
        + b_gyro_new
        + params.gyro_noise_std * rng.standard_normal(3)
    )

    # ------------------------------------------------------------------
    # 4. Accelerometer (specific force in body frame)
    #    specific_force_world = v_dot - g_world = v_dot - [0,0,-g]
    #                        = v_dot + [0,0,g]
    #    a_meas = R^T @ specific_force_world + b_a + n_a
    #
    #    At hover (v_dot=0):  a_meas ~ R^T @ [0,0,g] = [0,0,g]  (correct)
    # ------------------------------------------------------------------
    specific_force_world = truth_v_dot + np.array([0.0, 0.0, g])
    specific_force_body = R.T @ specific_force_world
    accel = (
        specific_force_body
        + b_accel_new
        + params.accel_noise_std * rng.standard_normal(3)
    )

    # ------------------------------------------------------------------
    # 5. Altimeter (optional, rate-limited)
    # ------------------------------------------------------------------
    alt: Optional[float] = None
    t_last_alt = sensor_state.t_last_alt
    if params.alt_enabled:
        period = 1.0 / params.alt_rate_hz
        if t - t_last_alt >= period - 1e-9:
            alt = float(truth.p[2]) + params.alt_noise_std * float(
                rng.standard_normal()
            )
            t_last_alt = t

    # ------------------------------------------------------------------
    # 6. Position fix (optional, rate-limited)
    # ------------------------------------------------------------------
    pos_fix: Optional[NDArray[np.float64]] = None
    t_last_posfix = sensor_state.t_last_posfix
    if params.posfix_enabled:
        period = 1.0 / params.posfix_rate_hz
        if t - t_last_posfix >= period - 1e-9:
            pos_fix = truth.p.copy() + params.posfix_noise_std * rng.standard_normal(3)
            t_last_posfix = t

    # ------------------------------------------------------------------
    # 7. Pack and return
    # ------------------------------------------------------------------
    meas = Measurement(gyro=gyro, accel=accel, alt=alt, pos_fix=pos_fix)

    new_state = SensorState(
        b_gyro=b_gyro_new,
        b_accel=b_accel_new,
        t_last_alt=t_last_alt,
        t_last_posfix=t_last_posfix,
        rng=rng,
    )

    return meas, new_state


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running sensor model self-test...")

    sp = SensorParams()

    # Test 1: init + sample produces valid shapes
    ss = init_sensor_state(seed=42, params=sp)
    truth = State.zeros()
    truth.p = np.array([0.0, 0.0, 1.0])
    v_dot = np.zeros(3)

    meas, ss = sample_measurements(0.0, 0.002, truth, v_dot, sp, ss)
    assert meas.gyro.shape == (3,), f"gyro shape: {meas.gyro.shape}"
    assert meas.accel.shape == (3,), f"accel shape: {meas.accel.shape}"
    print("  [PASS] Valid measurement shapes")

    # Test 2: At hover the accel should read ~[0, 0, g]
    accel_z = meas.accel[2]
    assert abs(accel_z - 9.80665) < 1.0, f"hover accel_z should be ~g, got {accel_z}"
    print(f"  [PASS] Hover accel_z = {accel_z:.2f} (expected ~9.81)")

    # Test 3: Altimeter fires on first sample (t_last = -inf)
    assert meas.alt is not None, "Altimeter should fire on first sample"
    assert abs(meas.alt - 1.0) < 0.5, f"Altimeter should read ~1.0, got {meas.alt}"
    print(f"  [PASS] Altimeter = {meas.alt:.3f}")

    # Test 4: Position fix fires on first sample
    assert meas.pos_fix is not None, "Position fix should fire on first sample"
    assert np.allclose(meas.pos_fix, [0, 0, 1], atol=0.5)
    print(f"  [PASS] Position fix = {meas.pos_fix}")

    # Test 5: Determinism
    ss_a = init_sensor_state(seed=123, params=sp)
    ss_b = init_sensor_state(seed=123, params=sp)
    m_a, _ = sample_measurements(0.0, 0.002, truth, v_dot, sp, ss_a)
    m_b, _ = sample_measurements(0.0, 0.002, truth, v_dot, sp, ss_b)
    assert np.allclose(m_a.gyro, m_b.gyro), "Same seed should give identical gyro"
    assert np.allclose(m_a.accel, m_b.accel), "Same seed should give identical accel"
    print("  [PASS] Deterministic with same seed")

    # Test 6: Rate limiting — alt should NOT fire on next immediate step
    meas2, ss = sample_measurements(0.002, 0.002, truth, v_dot, sp, ss)
    # 50 Hz = 20 ms period, 0.002 s after first sample is too soon
    assert meas2.alt is None, "Altimeter should not fire 2ms after last sample"
    print("  [PASS] Altimeter rate-limited")

    # Test 7: Disabled sensors
    sp_off = SensorParams(alt_enabled=False, posfix_enabled=False)
    ss_off = init_sensor_state(seed=0, params=sp_off)
    m_off, _ = sample_measurements(0.0, 0.002, truth, v_dot, sp_off, ss_off)
    assert m_off.alt is None, "Disabled altimeter should return None"
    assert m_off.pos_fix is None, "Disabled posfix should return None"
    print("  [PASS] Disabled sensors return None")

    print("\nAll sensor model self-tests passed!")
