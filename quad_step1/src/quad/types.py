"""
Core data types for quadrotor simulation.

All arrays use numpy with explicit shapes noted in comments.
Quaternion convention: [w, x, y, z] (scalar-first).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


@dataclass
class State:
    """
    Complete quadrotor state.

    Attributes:
        p: Position in world frame [m], shape (3,)
        v: Velocity in world frame [m/s], shape (3,)
        q: Attitude quaternion [w, x, y, z], shape (4,)
        w_body: Angular velocity in body frame [rad/s], shape (3,)
    """

    p: NDArray[np.float64]  # (3,)
    v: NDArray[np.float64]  # (3,)
    q: NDArray[np.float64]  # (4,) [w, x, y, z]
    w_body: NDArray[np.float64]  # (3,)

    def copy(self) -> "State":
        """Create a deep copy of this state."""
        return State(
            p=self.p.copy(),
            v=self.v.copy(),
            q=self.q.copy(),
            w_body=self.w_body.copy(),
        )

    @staticmethod
    def zeros() -> "State":
        """Create a zero state with identity quaternion."""
        return State(
            p=np.zeros(3),
            v=np.zeros(3),
            q=np.array([1.0, 0.0, 0.0, 0.0]),  # Identity quaternion
            w_body=np.zeros(3),
        )


@dataclass
class Control:
    """
    Control inputs to the quadrotor.

    Attributes:
        thrust_N: Total thrust magnitude [N]
        moments_Nm: Body-frame torques [N·m], shape (3,)
    """

    thrust_N: float
    moments_Nm: NDArray[np.float64]  # (3,)

    @staticmethod
    def zeros() -> "Control":
        """Create zero control input."""
        return Control(thrust_N=0.0, moments_Nm=np.zeros(3))


@dataclass
class TrajPoint:
    """
    Desired trajectory point at a given time.

    Attributes:
        p: Desired position [m], shape (3,)
        v: Desired velocity [m/s], shape (3,)
        a: Desired acceleration [m/s²], shape (3,)
        yaw: Desired yaw angle [rad]
        yaw_rate: Desired yaw rate [rad/s]
    """

    p: NDArray[np.float64]  # (3,)
    v: NDArray[np.float64]  # (3,)
    a: NDArray[np.float64]  # (3,)
    yaw: float = 0.0
    yaw_rate: float = 0.0

    @staticmethod
    def hover(position: NDArray[np.float64], yaw: float = 0.0) -> "TrajPoint":
        """Create a hover trajectory point at given position."""
        return TrajPoint(
            p=position.copy(),
            v=np.zeros(3),
            a=np.zeros(3),
            yaw=yaw,
            yaw_rate=0.0,
        )


@dataclass
class SimLog:
    """
    Simulation log storing time histories of all relevant quantities.

    All arrays have shape (N,) or (N, 3) or (N, 4) where N is number of timesteps.
    """

    # Time
    t: NDArray[np.float64]  # (N,)

    # State histories
    p: NDArray[np.float64]  # (N, 3)
    v: NDArray[np.float64]  # (N, 3)
    q: NDArray[np.float64]  # (N, 4)
    w_body: NDArray[np.float64]  # (N, 3)

    # Control histories
    thrust: NDArray[np.float64]  # (N,)
    moments: NDArray[np.float64]  # (N, 3)

    # Desired trajectory histories
    p_des: NDArray[np.float64]  # (N, 3)
    v_des: NDArray[np.float64]  # (N, 3)
    a_des: NDArray[np.float64]  # (N, 3)
    yaw_des: NDArray[np.float64]  # (N,)

    # Error histories (computed during control)
    e_pos: NDArray[np.float64]  # (N, 3)
    e_vel: NDArray[np.float64]  # (N, 3)
    e_att: NDArray[np.float64]  # (N, 3) attitude error (vee of skew)
    e_rate: NDArray[np.float64]  # (N, 3)

    # Current write index
    _idx: int = field(default=0, repr=False)

    @staticmethod
    def allocate(n_steps: int) -> "SimLog":
        """Pre-allocate arrays for n_steps timesteps."""
        return SimLog(
            t=np.zeros(n_steps),
            p=np.zeros((n_steps, 3)),
            v=np.zeros((n_steps, 3)),
            q=np.zeros((n_steps, 4)),
            w_body=np.zeros((n_steps, 3)),
            thrust=np.zeros(n_steps),
            moments=np.zeros((n_steps, 3)),
            p_des=np.zeros((n_steps, 3)),
            v_des=np.zeros((n_steps, 3)),
            a_des=np.zeros((n_steps, 3)),
            yaw_des=np.zeros(n_steps),
            e_pos=np.zeros((n_steps, 3)),
            e_vel=np.zeros((n_steps, 3)),
            e_att=np.zeros((n_steps, 3)),
            e_rate=np.zeros((n_steps, 3)),
            _idx=0,
        )

    def record(
        self,
        t: float,
        state: State,
        control: Control,
        traj: TrajPoint,
        e_pos: NDArray[np.float64],
        e_vel: NDArray[np.float64],
        e_att: NDArray[np.float64],
        e_rate: NDArray[np.float64],
    ) -> None:
        """Record one timestep of data."""
        i = self._idx
        self.t[i] = t
        self.p[i] = state.p
        self.v[i] = state.v
        self.q[i] = state.q
        self.w_body[i] = state.w_body
        self.thrust[i] = control.thrust_N
        self.moments[i] = control.moments_Nm
        self.p_des[i] = traj.p
        self.v_des[i] = traj.v
        self.a_des[i] = traj.a
        self.yaw_des[i] = traj.yaw
        self.e_pos[i] = e_pos
        self.e_vel[i] = e_vel
        self.e_att[i] = e_att
        self.e_rate[i] = e_rate
        self._idx += 1

    def trim(self) -> "SimLog":
        """Trim arrays to actual recorded length."""
        n = self._idx
        return SimLog(
            t=self.t[:n],
            p=self.p[:n],
            v=self.v[:n],
            q=self.q[:n],
            w_body=self.w_body[:n],
            thrust=self.thrust[:n],
            moments=self.moments[:n],
            p_des=self.p_des[:n],
            v_des=self.v_des[:n],
            a_des=self.a_des[:n],
            yaw_des=self.yaw_des[:n],
            e_pos=self.e_pos[:n],
            e_vel=self.e_vel[:n],
            e_att=self.e_att[:n],
            e_rate=self.e_rate[:n],
            _idx=n,
        )
