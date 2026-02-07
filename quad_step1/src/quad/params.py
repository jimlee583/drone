"""
Quadrotor parameters and controller gains.

Default values are for a ~500g racing/hobby quadrotor.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from quad.motor_model import ActuatorParams
from quad.disturbances import WindParams
from quad.sensors import SensorParams
from quad.estimator_ekf import EstimatorParams


@dataclass
class Params:
    """
    Complete parameter set for quadrotor simulation and control.

    Physical Parameters:
        m: Mass [kg]
        J: Inertia matrix [kg·m²], shape (3, 3)
        g: Gravitational acceleration [m/s²]

    Actuator Limits:
        T_min: Minimum thrust [N]
        T_max: Maximum thrust [N]
        tau_max: Maximum moment per axis [N·m], shape (3,)

    Controller Gains (SE(3) geometric controller):
        kp_pos: Position proportional gains, shape (3,)
        kd_pos: Position derivative gains, shape (3,)
        kr_att: Attitude proportional gains, shape (3,)
        kw_rate: Angular rate derivative gains, shape (3,)

    Optional:
        drag_coeff: Linear velocity drag coefficient (0 = off)

    Actuator Dynamics (Step 2):
        actuator: Parameters for first-order motor lag / saturation model.

    Wind / Disturbances (Step 2):
        wind: Parameters for wind, gusts, and drag disturbances.
    """

    # Physical parameters
    m: float = 0.5  # kg, typical 250-500g quad
    J: NDArray[np.float64] = field(
        default_factory=lambda: np.diag([0.0023, 0.0023, 0.004])
    )  # kg·m²
    g: float = 9.80665  # m/s²

    # Actuator limits
    T_min: float = 0.0  # N, motors can't produce negative thrust
    T_max: float = 15.0  # N, ~3g thrust-to-weight ratio
    tau_max: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.1, 0.1, 0.05])
    )  # N·m

    # Position controller gains
    # Tuned for critically damped response
    kp_pos: NDArray[np.float64] = field(
        default_factory=lambda: np.array([6.0, 6.0, 8.0])
    )
    kd_pos: NDArray[np.float64] = field(
        default_factory=lambda: np.array([4.0, 4.0, 5.0])
    )

    # Attitude controller gains
    # Lower gains for attitude since inner loop should be faster
    kr_att: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.1, 0.1, 0.05])
    )
    # Rate gains — increased for zeta ~ 0.65 (from 0.33) to reduce overshoot
    kw_rate: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.02, 0.02, 0.01])
    )

    # Optional aerodynamic drag (linear drag model)
    drag_coeff: float = 0.0  # Set > 0 to enable, e.g., 0.1

    # --- Step-2 realism features -------------------------------------------

    # First-order actuator dynamics (motor lag, rate limits, saturation).
    # Enabled by default with conservative settings that keep hover stable.
    actuator: ActuatorParams = field(default_factory=ActuatorParams)

    # Wind / environmental disturbances.
    # Enabled by default with mild breeze + light gusts.
    wind: WindParams = field(default_factory=WindParams)

    # --- Step-3 state estimation ----------------------------------------

    # If True, the controller receives EKF-estimated state instead of truth.
    use_estimator: bool = False
    sensor_params: SensorParams = field(default_factory=SensorParams)
    estimator_params: EstimatorParams = field(default_factory=EstimatorParams)

    def __post_init__(self) -> None:
        """Validate and convert parameters after initialization."""
        # Pre-compute and cache the inertia inverse
        self._J_inv: NDArray[np.float64] | None = None

        # Ensure arrays are numpy arrays
        if not isinstance(self.J, np.ndarray):
            self.J = np.array(self.J)
        if not isinstance(self.tau_max, np.ndarray):
            self.tau_max = np.array(self.tau_max)
        if not isinstance(self.kp_pos, np.ndarray):
            self.kp_pos = np.array(self.kp_pos)
        if not isinstance(self.kd_pos, np.ndarray):
            self.kd_pos = np.array(self.kd_pos)
        if not isinstance(self.kr_att, np.ndarray):
            self.kr_att = np.array(self.kr_att)
        if not isinstance(self.kw_rate, np.ndarray):
            self.kw_rate = np.array(self.kw_rate)

    @property
    def J_inv(self) -> NDArray[np.float64]:
        """Inverse of inertia matrix (cached)."""
        if self._J_inv is None:
            self._J_inv = np.linalg.inv(self.J)
        return self._J_inv

    @property
    def hover_thrust(self) -> float:
        """Thrust required for hover."""
        return self.m * self.g


def default_params() -> Params:
    """
    Create default parameters for a typical quadrotor.

    Returns a Params instance with reasonable values for a
    ~500g racing/hobby quadrotor.
    """
    return Params()


def aggressive_params() -> Params:
    """
    Create parameters for more aggressive flight.

    Higher gains for faster response, suitable for
    aggressive trajectories.
    """
    return Params(
        kp_pos=np.array([10.0, 10.0, 12.0]),
        kd_pos=np.array([6.0, 6.0, 7.0]),
        kr_att=np.array([0.15, 0.15, 0.08]),
        kw_rate=np.array([0.03, 0.03, 0.016]),
    )


if __name__ == "__main__":
    # Quick sanity check
    p = default_params()
    print("Default Parameters:")
    print(f"  Mass: {p.m} kg")
    print(f"  Inertia diagonal: {np.diag(p.J)}")
    print(f"  Hover thrust: {p.hover_thrust:.2f} N")
    print(f"  Thrust limits: [{p.T_min}, {p.T_max}] N")
    print(f"  Position gains Kp: {p.kp_pos}")
    print(f"  Position gains Kd: {p.kd_pos}")
    print(f"  Attitude gains Kr: {p.kr_att}")
    print(f"  Rate gains Kw: {p.kw_rate}")
