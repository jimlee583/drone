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

import json
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
    )


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

        if terminated or truncated:
            term_reason = info.get("term_reason", "unknown")
            break

    return {
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

    Returns
    -------
    dict
        Aggregate statistics and per-episode results.
    """
    ec = env_config or EnvConfig()
    track = _make_track(ec)
    base_cfg = _make_base_env_config(ec)
    env = QuadRacingEnv(track=track, config=base_cfg)

    results: List[Dict[str, Any]] = []
    t0 = time.monotonic()

    for i in range(episodes):
        ep_seed = seed + i
        res = _run_episode(env, policy_name, ep_seed)
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
    _add_run_args(parser)
    args = parser.parse_args()

    ec = _EC()
    env_keys = [
        "track", "track_radius", "track_z", "track_n_pts", "wp_radius",
        "n_laps", "dt_sim", "control_decimation", "max_steps",
        "a_residual_max", "yaw_rate_max", "use_estimator",
    ]
    _apply_overrides(ec, args, env_keys)

    rc = _RC()
    run_keys = [
        "seed", "log_dir", "model_dir", "results_dir", "device",
        "num_envs", "run_name", "verbose",
    ]
    _apply_overrides(rc, args, run_keys)

    seed = rc.seed if rc.seed != 0 else (args.seed if args.seed is not None else 0)

    run_baseline(
        policy_name=args.policy,
        episodes=args.episodes,
        seed=seed,
        env_config=ec,
        results_dir=rc.results_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
