"""
Gymnasium environment wrapping the existing quadrotor simulation.

The policy outputs *residual* desired-acceleration (world frame) and
*residual* yaw-rate which are **added** to a baseline setpoint derived
from the waypoint track.  The inner SE(3) controller, actuator model,
disturbances and (optionally) the EKF all run untouched underneath.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import spaces

# Existing sim components --------------------------------------------------
from quad.types import State, Control, TrajPoint
from quad.params import Params, default_params
from quad.dynamics import step_rk4
from quad.controller_se3 import se3_controller
from quad.motor_model import ActuatorState, step_actuator
from quad.disturbances import DisturbanceState, compute_disturbance_forces
from quad.math3d import quat_to_R

# Track --------------------------------------------------------------------
from quad.envs.track import WaypointTrack, circle_track


# ---------------------------------------------------------------------------
# Environment config (plain dataclass so it stays JSON-friendly later)
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """Configuration for :class:`QuadRacingEnv`.

    Timing
    ------
    dt_sim : float
        Physics / actuator / disturbance timestep [s].
    control_decimation : int
        Number of sim steps per env step (action held constant).
    max_steps : int
        Episode length in env steps.

    Action scaling
    --------------
    a_residual_max : float
        Maximum residual acceleration per axis [m/s²].
    yaw_rate_max : float
        Maximum residual yaw rate [rad/s].

    Baseline outer-loop PD (generates a_des from waypoint)
    -------------------------------------------------------
    kp_outer : float
        Position proportional gain for the simple PD that creates
        the *baseline* desired acceleration toward the waypoint.
    kd_outer : float
        Velocity damping gain.
    v_max : float
        Maximum desired velocity magnitude toward waypoint [m/s].

    Reward coefficients
    -------------------
    k_progress : float
        Reward per metre of distance reduction.
    k_time : float
        Per-step time penalty.
    k_control : float
        Coefficient for action-norm penalty.
    R_success : float
        Bonus for completing the track.
    R_crash : float
        Penalty for crash / divergence.

    Crash thresholds
    ----------------
    pos_limit : float
        Maximum position norm [m].
    tilt_limit_deg : float
        Maximum tilt angle from vertical [deg].

    Misc
    ----
    use_estimator : bool
        Run controller on EKF-estimated state (more realistic).
    render_every : int
        Print a status line every N env steps when render_mode="human".
    """

    # Timing
    dt_sim: float = 0.005
    control_decimation: int = 10        # dt_control = 0.05 s = 20 Hz
    max_steps: int = 2000               # 100 s at 20 Hz

    # Action scaling
    a_residual_max: float = 5.0         # m/s²
    yaw_rate_max: float = 2.0           # rad/s

    # Baseline outer-loop PD
    kp_outer: float = 2.0
    kd_outer: float = 1.5
    v_max: float = 3.0

    # Reward
    k_progress: float = 1.0
    k_time: float = 0.01
    k_control: float = 0.002
    R_success: float = 100.0
    R_crash: float = 100.0

    # Crash thresholds
    pos_limit: float = 100.0
    tilt_limit_deg: float = 80.0

    # Misc
    use_estimator: bool = False
    render_every: int = 50


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

OBS_DIM = 15
# Layout:
#   [0:3]   relative pos to current WP  (wp - p)
#   [3:6]   velocity (world frame)
#   [6:9]   body z-axis in world frame  (R @ e3)
#   [9:12]  body angular rates
#   [12:15] relative pos to *next* WP   (wp_next - p)


def _build_obs(
    state: State,
    track: WaypointTrack,
) -> NDArray[np.float32]:
    """Construct a flat float32 observation vector."""
    R = quat_to_R(state.q)
    e3 = np.array([0.0, 0.0, 1.0])
    b3 = R @ e3  # body z in world frame

    wp_rel = track.current_waypoint - state.p
    wp_next_rel = track.next_waypoint - state.p

    obs = np.concatenate([
        wp_rel,             # 3
        state.v,            # 3
        b3,                 # 3
        state.w_body,       # 3
        wp_next_rel,        # 3
    ]).astype(np.float32)

    return obs


def _obs_space() -> spaces.Box:
    hi = np.full(OBS_DIM, 200.0, dtype=np.float32)  # generous bounds
    return spaces.Box(low=-hi, high=hi, dtype=np.float32)


# ---------------------------------------------------------------------------
# QuadRacingEnv
# ---------------------------------------------------------------------------

class QuadRacingEnv(gym.Env):
    """Gymnasium environment for quadrotor waypoint racing.

    The policy produces a 4-D action each env step:
      action[0:3] → residual world-frame acceleration  (scaled by a_residual_max)
      action[3]   → residual yaw rate                  (scaled by yaw_rate_max)

    These residuals are added to a simple PD-derived baseline setpoint and
    then passed through the existing SE(3) controller → actuator → dynamics
    pipeline.

    Parameters
    ----------
    track : WaypointTrack, optional
        Waypoint track to follow.  Defaults to a circle.
    params : Params, optional
        Quadrotor / controller / actuator / wind parameters.
    config : EnvConfig, optional
        Environment configuration.
    render_mode : str, optional
        ``"human"`` prints periodic status lines; ``"ansi"`` returns a string.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 20}

    def __init__(
        self,
        track: Optional[WaypointTrack] = None,
        params: Optional[Params] = None,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.cfg = config or EnvConfig()
        self.params = params or default_params()
        self.params.use_estimator = self.cfg.use_estimator
        self._track_template = track or circle_track()
        self.render_mode = render_mode

        # Gymnasium spaces
        self.observation_space = _obs_space()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32,
        )

        # Internal state (populated in reset)
        self._state: Optional[State] = None
        self._act_state: Optional[ActuatorState] = None
        self._dist_state: Optional[DisturbanceState] = None
        self._track: Optional[WaypointTrack] = None
        self._step_count: int = 0
        self._sim_time: float = 0.0
        self._total_reward: float = 0.0
        self._wps_reached: int = 0
        self._np_random: Optional[np.random.Generator] = None

        # Estimator state (lazy-initialised if needed)
        self._est_state: Any = None
        self._sensor_st: Any = None
        self._v_prev: Optional[NDArray] = None
        self._last_gyro: Optional[NDArray] = None

    # ------------------------------------------------------------------
    # Seeding (Gymnasium >=0.26)
    # ------------------------------------------------------------------

    def _get_np_random(self) -> np.random.Generator:
        if self._np_random is None:
            self._np_random = np.random.default_rng()
        return self._np_random

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray[np.float32], Dict[str, Any]]:
        super().reset(seed=seed)
        self._np_random = self.np_random  # set by super().reset(seed=)

        # Spawn position near the first waypoint (small random offset)
        self._track = self._track_template.copy()
        wp0 = self._track.waypoints[0]
        rng = self._get_np_random()
        offset = rng.uniform(-0.3, 0.3, size=3)
        offset[2] = 0.0  # don't randomise altitude much

        self._state = State.zeros()
        self._state.p = wp0.copy() + offset

        self._act_state = ActuatorState.from_hover(self.params.hover_thrust)
        self._dist_state = DisturbanceState.create(
            seed=int(rng.integers(0, 2**31)),
        )

        self._track.reset(self._state.p)
        self._step_count = 0
        self._sim_time = 0.0
        self._total_reward = 0.0
        self._wps_reached = 0

        # Estimator
        if self.cfg.use_estimator:
            self._init_estimator()

        obs = self._get_obs()
        info = self._info()
        return obs, info

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self, action: NDArray[np.float32],
    ) -> Tuple[NDArray[np.float32], float, bool, bool, Dict[str, Any]]:
        assert self._state is not None, "Call reset() before step()"
        action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)

        # --- Map action to residuals ---
        da_world = action[0:3] * self.cfg.a_residual_max
        dyaw_rate = float(action[3]) * self.cfg.yaw_rate_max

        # --- Baseline setpoint from waypoint PD ---
        p_d, v_d, a_d_base, yaw_d, yaw_rate_base = self._baseline_setpoint()

        # --- Add residuals ---
        a_d = a_d_base + da_world
        yaw_rate_d = yaw_rate_base + dyaw_rate

        traj = TrajPoint(
            p=p_d,
            v=v_d,
            a=a_d,
            yaw=yaw_d,
            yaw_rate=yaw_rate_d,
        )

        # --- Run sim for control_decimation steps ---
        dt = self.cfg.dt_sim
        for _ in range(self.cfg.control_decimation):
            self._sim_step(traj, dt)

        # --- Track progress ---
        progress, wp_reached = self._track.update(self._state.p)
        if wp_reached:
            self._wps_reached += 1

        self._step_count += 1

        # --- Reward ---
        reward = self._compute_reward(progress, action, wp_reached)
        self._total_reward += reward

        # --- Termination / truncation ---
        terminated = False
        truncated = False
        term_reason = ""

        if self._is_crashed():
            terminated = True
            reward -= self.cfg.R_crash
            self._total_reward -= self.cfg.R_crash
            term_reason = "crash"
        elif self._track.done:
            terminated = True
            reward += self.cfg.R_success
            self._total_reward += self.cfg.R_success
            term_reason = "success"

        if not terminated and self._step_count >= self.cfg.max_steps:
            truncated = True
            term_reason = "max_steps"

        obs = self._get_obs()
        info = self._info()
        info["term_reason"] = term_reason

        # Render if human mode
        if self.render_mode == "human" and (
            self._step_count % self.cfg.render_every == 0
            or terminated
            or truncated
        ):
            self._render_human()

        return obs, float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # render / close
    # ------------------------------------------------------------------

    def render(self) -> Optional[str]:
        if self.render_mode == "ansi":
            return self._render_ansi()
        elif self.render_mode == "human":
            self._render_human()
        return None

    def close(self) -> None:
        pass  # no resources to clean up

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _get_obs(self) -> NDArray[np.float32]:
        """Return the observation from the state visible to the policy."""
        if self.cfg.use_estimator and self._est_state is not None:
            from quad.estimator_ekf import get_estimated_state
            x_obs = get_estimated_state(self._est_state, self._last_gyro)
        else:
            x_obs = self._state
        return _build_obs(x_obs, self._track)

    # ------------------------------------------------------------------

    def _baseline_setpoint(
        self,
    ) -> Tuple[NDArray, NDArray, NDArray, float, float]:
        """Simple PD to compute desired (p, v, a, yaw, yaw_rate) from WP."""
        wp = self._track.current_waypoint
        if self.cfg.use_estimator and self._est_state is not None:
            from quad.estimator_ekf import get_estimated_state
            x_base = get_estimated_state(self._est_state, self._last_gyro)
        else:
            x_base = self._state
        p = x_base.p
        v = x_base.v

        # Vector toward waypoint
        dp = wp - p
        dist = np.linalg.norm(dp)

        # Desired velocity: limited magnitude toward waypoint
        if dist > 1e-3:
            direction = dp / dist
        else:
            direction = np.zeros(3)
        v_d = direction * min(dist * self.cfg.kp_outer, self.cfg.v_max)

        # Simple PD acceleration
        a_d = self.cfg.kp_outer * dp + self.cfg.kd_outer * (v_d - v)

        # Desired yaw: point toward waypoint
        yaw_d = float(np.arctan2(dp[1], dp[0])) if dist > 0.1 else 0.0

        return wp.copy(), v_d, a_d, yaw_d, 0.0

    # ------------------------------------------------------------------

    def _sim_step(self, traj: TrajPoint, dt: float) -> None:
        """One inner sim step: controller → actuator → disturbances → RK4."""
        state = self._state

        # Estimator path
        if self.cfg.use_estimator and self._est_state is not None:
            x_ctrl = self._estimator_step(state, dt)
        else:
            x_ctrl = state

        # Controller
        cmd_control, _ = se3_controller(x_ctrl, traj, self.params)

        # Actuator
        self._act_state, applied_control = step_actuator(
            self._act_state, cmd_control, self.params.actuator, dt,
        )

        # Disturbances
        F_ext, tau_ext, self._dist_state = compute_disturbance_forces(
            state.v, self._dist_state, self.params.wind, dt,
        )

        # Dynamics
        self._state = step_rk4(
            state, applied_control, self.params, dt,
            F_ext=F_ext, tau_ext=tau_ext,
        )

        self._sim_time += dt

    # ------------------------------------------------------------------
    # Estimator integration (lazy import to avoid hard dep)
    # ------------------------------------------------------------------

    def _init_estimator(self) -> None:
        from quad.sensors import SensorParams, init_sensor_state
        from quad.estimator_ekf import EstimatorParams, init_estimator

        sp = getattr(self.params, "sensor_params", SensorParams())
        ep = getattr(self.params, "estimator_params", EstimatorParams())

        sensor_seed = int(self._get_np_random().integers(0, 2**31))
        self._sensor_st = init_sensor_state(sensor_seed, sp)
        self._est_state = init_estimator(ep, x0_truth=self._state)
        self._v_prev = self._state.v.copy()
        self._last_gyro = np.zeros(3)

    def _estimator_step(self, truth: State, dt: float) -> State:
        from quad.sensors import sample_measurements
        from quad.estimator_ekf import (
            ekf_predict,
            ekf_update_altimeter,
            ekf_update_posfix,
            get_estimated_state,
        )

        sp = getattr(self.params, "sensor_params", None)
        ep = getattr(self.params, "estimator_params", None)
        if sp is None or ep is None or self._est_state is None:
            return truth

        if self._v_prev is None:
            truth_v_dot = np.zeros(3)
        else:
            truth_v_dot = (truth.v - self._v_prev) / dt
        self._v_prev = truth.v.copy()

        meas, self._sensor_st = sample_measurements(
            self._sim_time, dt, truth, truth_v_dot, sp, self._sensor_st,
            g=self.params.g,
        )
        self._est_state = ekf_predict(
            self._est_state, meas.gyro, meas.accel, ep, dt,
        )
        if meas.alt is not None and ep.use_alt_update:
            from quad.estimator_ekf import ekf_update_altimeter as _upd_alt
            self._est_state = _upd_alt(
                self._est_state, meas.alt, sp.alt_noise_std**2,
            )
        if meas.pos_fix is not None and ep.use_posfix_update:
            from quad.estimator_ekf import ekf_update_posfix as _upd_pos
            self._est_state = _upd_pos(
                self._est_state, meas.pos_fix,
                sp.posfix_noise_std**2 * np.eye(3),
            )
        self._last_gyro = meas.gyro
        return get_estimated_state(self._est_state, meas.gyro)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        progress: float,
        action: NDArray,
        wp_reached: bool,
    ) -> float:
        r = 0.0
        r += self.cfg.k_progress * progress
        r -= self.cfg.k_time
        r -= self.cfg.k_control * (
            float(np.sum(action[0:3] ** 2)) + float(action[3] ** 2)
        )
        return r

    # ------------------------------------------------------------------
    # Crash detection
    # ------------------------------------------------------------------

    def _is_crashed(self) -> bool:
        s = self._state
        if s is None:
            return True

        # NaN / inf check
        for arr in (s.p, s.v, s.q, s.w_body):
            if not np.all(np.isfinite(arr)):
                return True

        # Position bound
        if np.linalg.norm(s.p) > self.cfg.pos_limit:
            return True

        # Tilt check: angle between body-z and world-z
        R = quat_to_R(s.q)
        b3 = R @ np.array([0.0, 0.0, 1.0])
        cos_tilt = np.clip(b3[2], -1.0, 1.0)
        tilt_deg = np.degrees(np.arccos(cos_tilt))
        if tilt_deg > self.cfg.tilt_limit_deg:
            return True

        return False

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _info(self) -> Dict[str, Any]:
        p = self._state.p if self._state is not None else np.zeros(3)
        dist = float(np.linalg.norm(self._track.current_waypoint - p)) if self._track else 0.0
        return {
            "sim_time": self._sim_time,
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "wps_reached": self._wps_reached,
            "laps_done": self._track.laps_done if self._track else 0,
            "dist_to_wp": dist,
            "position": p.copy(),
        }

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_human(self) -> None:
        info = self._info()
        print(
            f"[step {info['step_count']:5d}]  "
            f"t={info['sim_time']:6.2f}s  "
            f"pos=({info['position'][0]:+6.2f}, {info['position'][1]:+6.2f}, {info['position'][2]:+6.2f})  "
            f"dist_wp={info['dist_to_wp']:.2f}m  "
            f"wp_idx={self._track.current_wp}/{self._track.n_waypoints}  "
            f"wps={info['wps_reached']}  "
            f"R={info['total_reward']:+8.2f}"
        )

    def _render_ansi(self) -> str:
        info = self._info()
        return (
            f"step={info['step_count']} "
            f"t={info['sim_time']:.2f} "
            f"pos={info['position']} "
            f"dist={info['dist_to_wp']:.2f} "
            f"wps={info['wps_reached']} "
            f"R={info['total_reward']:.2f}"
        )
