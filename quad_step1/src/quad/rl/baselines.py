"""
Baseline (non-learned) policy evaluation for QuadRacingEnv.

Provides two baselines:
  * **zero** — action = 0 (pure SE(3) controller follows waypoints)
  * **random** — uniform random actions in [-1, 1]

Usage::

    python -m quad.rl.baselines --policy zero --episodes 10 --seed 1
    python -m quad.rl.baselines --policy random --episodes 10 --seed 1
"""

from __future__ import annotations

import collections
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from quad.envs.quad_racing_env import QuadRacingEnv, EnvConfig as _BaseEnvConfig
from quad.envs.track import circle_track, figure8_track, line_track
from quad.rl.config import EnvConfig, FullConfig, load_config_from_args, save_config


# ---------------------------------------------------------------------------
# Track builder (from RL EnvConfig)
# ---------------------------------------------------------------------------

def _make_track(ec: EnvConfig):
    """Construct a WaypointTrack from the RL EnvConfig."""
    builders = {"circle": circle_track, "figure8": figure8_track, "line": line_track}
    builder = builders.get(ec.track)
    if builder is None:
        raise ValueError(f"Unknown track type '{ec.track}'. Choose from {list(builders)}")

    kwargs: Dict[str, Any] = {
        "z": ec.track_z,
        "wp_radius": ec.wp_radius,
        "n_laps": ec.n_laps,
    }
    if ec.track == "circle":
        kwargs["radius"] = ec.track_radius
        kwargs["n_pts"] = ec.track_n_pts
    elif ec.track == "figure8":
        kwargs["n_pts"] = ec.track_n_pts
    elif ec.track == "line":
        kwargs["n_pts"] = ec.track_n_pts

    return builder(**kwargs)


def _make_base_env_config(ec: EnvConfig) -> _BaseEnvConfig:
    """Map RL EnvConfig → QuadRacingEnv's native EnvConfig."""
    return _BaseEnvConfig(
        dt_sim=ec.dt_sim,
        control_decimation=ec.control_decimation,
        max_steps=ec.max_steps,
        a_residual_max=ec.a_residual_max,
        yaw_rate_max=ec.yaw_rate_max,
        kp_outer=ec.kp_outer,
        kd_outer=ec.kd_outer,
        v_max=ec.v_max,
        k_progress=ec.k_progress,
        k_time=ec.k_time,
        k_control=ec.k_control,
        R_success=ec.R_success,
        R_crash=ec.R_crash,
        pos_limit=ec.pos_limit,
        tilt_limit_deg=ec.tilt_limit_deg,
        use_estimator=ec.use_estimator,
        render_every=ec.render_every,
        # Gate mode
        use_gates=ec.use_gates,
        gate_radius_m=ec.gate_radius_m,
        gate_half_thickness_m=ec.gate_half_thickness_m,
        R_gate=ec.R_gate,
        terminate_on_wrong_direction=ec.terminate_on_wrong_direction,
        terminate_on_gate_miss=ec.terminate_on_gate_miss,
    )


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------

_TRACE_TAIL_LEN = 50  # max env steps kept in the tail trace


def _rl(arr, n: int | None = None) -> list:
    """Convert ndarray to a rounded JSON-friendly list of floats."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    if n is not None:
        a = a[:n]
    return [round(float(x), 6) for x in a]


# ---------------------------------------------------------------------------
# Episode config snapshot
# ---------------------------------------------------------------------------

def _try_git_commit() -> str | None:
    """Best-effort short git SHA (returns *None* on any failure)."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def _gains_id(params) -> str:
    """Deterministic short hash of SE(3) controller gains (Kp, Kd, Kr, Kw)."""
    vals = np.concatenate([
        params.kp_pos.ravel(),
        params.kd_pos.ravel(),
        params.kr_att.ravel(),
        params.kw_rate.ravel(),
    ])
    token = ",".join(f"{float(x):.8f}" for x in vals)
    return "se3_" + hashlib.sha1(token.encode()).hexdigest()[:8]


def _build_episode_config(
    policy: str,
    seed: int,
    episodes: int,
    ec: EnvConfig,
    env: QuadRacingEnv,
    run_name: str = "",
) -> Dict[str, Any]:
    """Build a compact, JSON-serializable config snapshot for one episode."""
    cfg: Dict[str, Any] = {}

    # 1) Core run settings
    cfg["policy"] = policy
    cfg["seed"] = seed
    cfg["episodes"] = episodes
    if run_name:
        cfg["run_name"] = run_name
    git_sha = _try_git_commit()
    if git_sha:
        cfg["git_commit"] = git_sha

    # 2) Track / task settings
    cfg["track"] = ec.track
    cfg["track_radius"] = float(ec.track_radius)
    cfg["track_z"] = float(ec.track_z)
    cfg["track_n_pts"] = int(ec.track_n_pts)
    cfg["wp_radius"] = float(ec.wp_radius)
    cfg["n_laps"] = int(ec.n_laps)
    cfg["use_gates"] = bool(ec.use_gates)
    cfg["gate_radius_m"] = float(ec.gate_radius_m)
    cfg["gate_half_thickness_m"] = float(ec.gate_half_thickness_m)

    # 3) Timing settings
    cfg["dt_sim"] = float(ec.dt_sim)
    cfg["control_decimation"] = int(ec.control_decimation)
    cfg["max_steps"] = int(ec.max_steps)

    # 4) Action space limits
    cfg["a_residual_max"] = float(ec.a_residual_max)
    cfg["yaw_rate_max"] = float(ec.yaw_rate_max)

    # 5) Estimation toggle
    cfg["use_estimator"] = bool(ec.use_estimator)

    # 6) Termination thresholds
    cfg["tilt_limit_deg"] = float(ec.tilt_limit_deg)
    cfg["pos_limit_m"] = float(ec.pos_limit)

    # 7) Disturbance toggles (from the env's physics Params)
    wp = env.params.wind
    cfg["wind_enabled"] = bool(wp.enabled)
    if wp.enabled:
        cfg["wind_mean_mps"] = [round(float(v), 6) for v in wp.wind_vel]
        cfg["gust_enabled"] = bool(wp.gust_std > 0)
        # WindParams.seed is the configured default; the actual per-episode
        # disturbance RNG seed is derived from the episode seed inside
        # env.reset() and is therefore implicitly reproducible via "seed".
        cfg["wind_params_seed"] = int(wp.seed)
        cfg["gust_std"] = float(wp.gust_std)
        cfg["gust_tau"] = float(wp.gust_tau)

    # 8) Controller identity
    cfg["controller_name"] = "se3"
    cfg["gains_id"] = _gains_id(env.params)
    cfg["se3_gains"] = {
        "kp_pos": [float(x) for x in env.params.kp_pos.ravel()],
        "kd_pos": [float(x) for x in env.params.kd_pos.ravel()],
        "kr_att": [float(x) for x in env.params.kr_att.ravel()],
        "kw_rate": [float(x) for x in env.params.kw_rate.ravel()],
    }

    return cfg


# ---------------------------------------------------------------------------
# Run a single episode
# ---------------------------------------------------------------------------

def _run_episode(
    env: QuadRacingEnv,
    policy: str,
    seed: int,
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    term_reason = ""
    last_action = np.zeros(env.action_space.shape, dtype=np.float32)

    # Ring buffer for compact tail trace (last N env steps)
    trace: collections.deque[Dict[str, Any]] = collections.deque(
        maxlen=_TRACE_TAIL_LEN,
    )

    while True:
        if policy == "zero":
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        elif policy == "random":
            action = env.action_space.sample()
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # --- Build trace entry ---
        entry: Dict[str, Any] = {
            "k": steps,
            "t": round(info.get("sim_time", 0.0), 6),
            "reward": round(float(reward), 6),
            "terminated": terminated,
            "truncated": truncated,
            "pos": _rl(info["position"]),
            "action": _rl(action),
        }
        vel = info.get("vel")
        if vel is not None:
            entry["vel"] = _rl(vel)
        # Gate context
        if "current_gate_idx" in info:
            entry["gate_idx"] = info["current_gate_idx"]
            if "d_plane" in info:
                entry["d_plane"] = round(float(info["d_plane"]), 6)
            if "lateral_m" in info:
                entry["lateral_m"] = round(float(info["lateral_m"]), 6)
            for flag in ("crossed", "wrong_dir", "lateral_miss"):
                if info.get(flag):
                    entry[flag] = True
        elif "wp_idx" in info:
            entry["wp_idx"] = info["wp_idx"]
            entry["dist_wp"] = round(float(info.get("dist_to_wp", 0.0)), 6)
        # Term code on the terminal step only
        if terminated or truncated:
            tc = info.get("term_code")
            if tc:
                entry["term_code"] = tc
        trace.append(entry)

        if terminated or truncated:
            term_reason = info.get("term_reason", "unknown")
            last_action = action
            break

    # === Build per-episode record (backward-compatible base) ===
    rec: Dict[str, Any] = {
        "seed": seed,
        "steps": steps,
        "total_reward": float(total_reward),
        "wps_reached": int(info.get("wps_reached", 0)),
        "laps_done": int(info.get("laps_done", 0)),
        "terminated": terminated,
        "truncated": truncated,
        "term_reason": term_reason,
        "sim_time": float(info.get("sim_time", 0.0)),
    }

    # --- Termination detail ---
    for key in ("term_code", "term_detail", "crash_type"):
        val = info.get(key)
        if val is not None:
            rec[key] = val
    rec["term_step"] = steps
    rec["term_time_s"] = round(info.get("sim_time", 0.0), 6)

    # --- Final state snapshot ---
    pos = info.get("position")
    if pos is not None:
        rec["final_pos"] = _rl(pos)
    vel = info.get("vel")
    if vel is not None:
        rec["final_vel"] = _rl(vel)
    quat = info.get("quat")
    if quat is not None:
        rec["final_quat"] = _rl(quat, 4)
    omega = info.get("omega")
    if omega is not None:
        rec["final_omega"] = _rl(omega)
    rec["final_action"] = _rl(last_action)
    for rec_key, info_key in [
        ("final_thrust_cmd", "thrust_cmd"),
        ("final_thrust_applied", "thrust_applied"),
    ]:
        val = info.get(info_key)
        if val is not None:
            rec[rec_key] = round(float(val), 6)
    for rec_key, info_key in [
        ("final_moments_cmd", "moments_cmd"),
        ("final_moments_applied", "moments_applied"),
    ]:
        val = info.get(info_key)
        if val is not None:
            rec[rec_key] = _rl(val)

    # --- Gate / track context at termination ---
    if "current_gate_idx" in info:
        rec["gate_idx"] = info["current_gate_idx"]
        rec["gates_passed"] = int(info.get("gates_passed", 0))
        if "d_plane" in info:
            rec["d_plane"] = round(float(info["d_plane"]), 6)
        if "lateral_m" in info:
            rec["lateral_m"] = round(float(info["lateral_m"]), 6)
        if "gate_radius_m" in info:
            rec["gate_radius_m"] = float(info["gate_radius_m"])
        if "gate_half_thickness_m" in info:
            rec["gate_half_thickness_m"] = float(info["gate_half_thickness_m"])
        crossing = {}
        for flag in ("crossed", "wrong_dir", "lateral_miss"):
            val = info.get(flag)
            if val is not None:
                crossing[flag] = val
        if crossing:
            rec["last_crossing_flags"] = crossing
    elif "wp_idx" in info:
        rec["wp_idx"] = info["wp_idx"]
        rec["dist_wp"] = round(float(info.get("dist_to_wp", 0.0)), 6)

    # --- Tail trace (last N steps) ---
    rec["trace_tail"] = list(trace)

    return rec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_baseline(
    policy_name: str = "zero",
    episodes: int = 10,
    seed: int = 0,
    env_config: EnvConfig | None = None,
    results_dir: str = "results_rl",
    verbose: bool = True,
    run_name: str = "",
    wind_enabled: bool = True,
) -> Dict[str, Any]:
    """Run *episodes* with the given baseline policy and report statistics.

    Parameters
    ----------
    policy_name : str
        ``"zero"`` or ``"random"``.
    episodes : int
        Number of evaluation episodes.
    seed : int
        Base seed (each episode gets ``seed + i``).
    env_config : EnvConfig, optional
        RL-level environment configuration. Uses defaults if *None*.
    results_dir : str
        Where to save the JSON summary.
    verbose : bool
        Print per-episode + aggregate info.
    run_name : str
        Optional human-readable run identifier.
    wind_enabled : bool
        Enable wind/gust disturbances. ``False`` sets
        ``Params.wind.enabled = False`` (zero external forces).

    Returns
    -------
    dict
        Aggregate statistics and per-episode results.
    """
    from quad.params import default_params

    ec = env_config or EnvConfig()
    track = _make_track(ec)
    base_cfg = _make_base_env_config(ec)

    params = default_params()
    params.wind.enabled = wind_enabled

    env = QuadRacingEnv(track=track, config=base_cfg, params=params)

    # Build the config snapshot once (shared across all episodes).
    ep_config_template = _build_episode_config(
        policy=policy_name,
        seed=seed,
        episodes=episodes,
        ec=ec,
        env=env,
        run_name=run_name,
    )

    results: List[Dict[str, Any]] = []
    t0 = time.monotonic()

    for i in range(episodes):
        ep_seed = seed + i
        res = _run_episode(env, policy_name, ep_seed)
        # Stamp per-episode seed into the snapshot copy
        ep_cfg = {**ep_config_template, "seed": ep_seed}
        res["episode_config"] = ep_cfg
        results.append(res)
        if verbose:
            print(
                f"  Episode {i + 1:3d}/{episodes}  "
                f"seed={ep_seed}  steps={res['steps']:5d}  "
                f"R={res['total_reward']:+8.2f}  "
                f"wps={res['wps_reached']}  "
                f"reason={res['term_reason']}"
            )

    env.close()
    wall_time = time.monotonic() - t0

    # --- Aggregate ---
    returns = [r["total_reward"] for r in results]
    lengths = [r["steps"] for r in results]
    success_count = sum(1 for r in results if r["term_reason"] == "success")
    crash_count = sum(1 for r in results if r["term_reason"] == "crash")

    summary: Dict[str, Any] = {
        "policy": policy_name,
        "episodes": episodes,
        "seed": seed,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": success_count / max(episodes, 1),
        "crash_rate": crash_count / max(episodes, 1),
        "mean_episode_length": float(np.mean(lengths)),
        "wall_time_s": wall_time,
        "per_episode": results,
    }

    if verbose:
        print(f"\n{'=' * 55}")
        print(f"  Baseline: {policy_name}  ({episodes} episodes)")
        print(f"{'=' * 55}")
        print(f"  Return        : {summary['return_mean']:+.2f} ± {summary['return_std']:.2f}")
        print(f"  Success rate  : {summary['success_rate']:.1%}")
        print(f"  Crash rate    : {summary['crash_rate']:.1%}")
        print(f"  Mean ep length: {summary['mean_episode_length']:.0f} steps")
        print(f"  Wall time     : {wall_time:.2f} s")
        print(f"{'=' * 55}")

    # --- Save ---
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"baseline_{policy_name}_seed{seed}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"  Results saved to {out_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _add_baseline_args(parser):
    parser.add_argument("--policy", type=str, default="zero", choices=["zero", "random"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--wind", dest="wind_enabled", action="store_true", default=None,
                        help="Enable wind/gust disturbances (default)")
    parser.add_argument("--no-wind", dest="wind_enabled", action="store_false",
                        help="Disable all wind/gust disturbances")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Baseline policy evaluation")
    _add_baseline_args(parser)

    # Reuse the shared config args (env + run only, no PPO)
    from quad.rl.config import (
        _add_env_args,
        _add_run_args,
        _apply_overrides,
        EnvConfig as _EC,
        RunConfig as _RC,
    )

    _add_env_args(parser)
    # Override wp-radius default for baseline evaluation (EnvConfig default is 0.5)
    for _act in parser._actions:
        if '--wp-radius' in getattr(_act, 'option_strings', []):
            _act.default = 1.5
            _act.help = (
                "Waypoint capture radius in meters (default 1.5 for stable "
                "baseline evaluation). Smaller values such as 0.5 increase "
                "difficulty and may cause aggressive turning."
            )
            break
    _add_run_args(parser)
    args = parser.parse_args()

    ec = _EC()
    env_keys = [
        "track", "track_radius", "track_z", "track_n_pts", "wp_radius",
        "n_laps", "dt_sim", "control_decimation", "max_steps",
        "a_residual_max", "yaw_rate_max", "use_estimator",
        "use_gates", "gate_radius_m", "gate_half_thickness_m",
    ]
    _apply_overrides(ec, args, env_keys)

    rc = _RC()
    run_keys = [
        "seed", "log_dir", "model_dir", "results_dir", "device",
        "num_envs", "run_name", "verbose",
    ]
    _apply_overrides(rc, args, run_keys)

    seed = rc.seed

    # --no-wind → False, --wind → True, neither → True (default)
    wind = args.wind_enabled if args.wind_enabled is not None else True

    run_baseline(
        policy_name=args.policy,
        episodes=args.episodes,
        seed=seed,
        env_config=ec,
        results_dir=rc.results_dir,
        verbose=True,
        run_name=rc.run_name,
        wind_enabled=wind,
    )


if __name__ == "__main__":
    main()
