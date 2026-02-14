"""
Evaluate a saved Stable-Baselines3 model on QuadRacingEnv.

Usage::

    python -m quad.rl.eval_sb3 --model-path models/<run>/best_model.zip --episodes 50 --seed 123
    python -m quad.rl.eval_sb3 --model-path models/<run>/final_model.zip --use-estimator

Results are saved to ``results_rl/<run_name>_eval.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import gymnasium as gym

from quad.envs.quad_racing_env import QuadRacingEnv, EnvConfig as _BaseEnvConfig
from quad.envs.track import circle_track, figure8_track, line_track
from quad.rl.config import (
    EnvConfig,
    FullConfig,
    RunConfig,
    _add_env_args,
    _add_run_args,
    _apply_overrides,
)
from quad.rl.wrappers import wrap_env


# ---------------------------------------------------------------------------
# Evaluation presets (local copy — keep in sync with baselines.py if needed)
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "debug": {
        "track": "circle",
        "track_radius": 3.0,
        "track_z": 1.5,
        "track_n_pts": 72,
        "wp_radius": 2.0,
        "n_laps": 1,
        "dt_sim": 0.005,
        "control_decimation": 2,
        "max_steps": 800,
        "use_estimator": False,
        "use_gates": False,
        "a_residual_max": 5.0,
        "yaw_rate_max": 2.0,
        "verbose": 1,
        "episodes": 10,
    },
    "baseline": {
        "track": "circle",
        "track_radius": 3.0,
        "track_z": 1.5,
        "track_n_pts": 12,
        "wp_radius": 1.5,
        "n_laps": 1,
        "dt_sim": 0.005,
        "control_decimation": 10,
        "max_steps": 2000,
        "use_estimator": False,
        "use_gates": False,
        "a_residual_max": 5.0,
        "yaw_rate_max": 2.0,
    },
    "hard": {
        "track": "circle",
        "track_radius": 3.0,
        "track_z": 1.5,
        "track_n_pts": 12,
        "wp_radius": 0.5,
        "n_laps": 1,
        "dt_sim": 0.005,
        "control_decimation": 10,
        "max_steps": 2000,
        "use_estimator": True,
        "use_gates": True,
        "gate_radius_m": 0.5,
        "gate_half_thickness_m": 0.2,
        "a_residual_max": 5.0,
        "yaw_rate_max": 2.0,
    },
}


# ---------------------------------------------------------------------------
# Helpers (same as train / baselines — kept local to avoid cross-imports)
# ---------------------------------------------------------------------------

def _make_track(ec: EnvConfig):
    builders = {"circle": circle_track, "figure8": figure8_track, "line": line_track}
    builder = builders.get(ec.track)
    if builder is None:
        raise ValueError(f"Unknown track '{ec.track}'. Choose from {list(builders)}")
    kwargs: Dict[str, Any] = {"z": ec.track_z, "wp_radius": ec.wp_radius, "n_laps": ec.n_laps}
    if ec.track == "circle":
        kwargs["radius"] = ec.track_radius
        kwargs["n_pts"] = ec.track_n_pts
    elif ec.track == "figure8":
        kwargs["n_pts"] = ec.track_n_pts
    elif ec.track == "line":
        kwargs["n_pts"] = ec.track_n_pts
    return builder(**kwargs)


def _make_base_env_config(ec: EnvConfig) -> _BaseEnvConfig:
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
# Single-episode evaluation
# ---------------------------------------------------------------------------

def _eval_episode(
    model,
    env: gym.Env,
    seed: int,
    deterministic: bool = True,
) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False
    term_reason = ""

    while True:
        action, _state = model.predict(obs, deterministic=deterministic)
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

def evaluate_model(
    model_path: str | Path,
    episodes: int = 50,
    seed: int = 0,
    env_config: EnvConfig | None = None,
    results_dir: str = "results_rl",
    deterministic: bool = True,
    verbose: bool = True,
    preset: str = "",
) -> Dict[str, Any]:
    """Load a saved SB3 model and evaluate it.

    Parameters
    ----------
    model_path : str | Path
        Path to a ``.zip`` model file.
    episodes : int
        Number of evaluation episodes.
    seed : int
        Base seed (each episode gets ``seed + i``).
    env_config : EnvConfig, optional
        Environment config overrides.
    results_dir : str
        Where to save the JSON results.
    deterministic : bool
        Use deterministic (greedy) actions.
    verbose : bool
        Print progress and aggregate stats.
    preset : str
        Name of the evaluation preset used (informational, stored in results).

    Returns
    -------
    dict
        Aggregate statistics and per-episode results.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for evaluation. Install with:\n"
            "  uv sync --extra rl\n"
            "  # or\n"
            '  pip install "stable-baselines3>=2.3.0"'
        ) from exc

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    ec = env_config or EnvConfig()
    track = _make_track(ec)
    base_cfg = _make_base_env_config(ec)
    env = QuadRacingEnv(track=track, config=base_cfg)
    env = wrap_env(env, clip_actions=True, normalize_obs=False)

    model = PPO.load(str(model_path), device="auto")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  SB3 Evaluation — {model_path.name}")
        print(f"{'=' * 60}")
        if preset:
            print(f"  Preset      : {preset}")
        print(f"  Track       : {ec.track}")
        print(f"  Estimator   : {ec.use_estimator}")
        print(f"  Episodes    : {episodes}")
        print(f"  Seed        : {seed}")
        print(f"  Deterministic: {deterministic}")
        print(f"{'=' * 60}")

    results: List[Dict[str, Any]] = []
    t0 = time.monotonic()

    for i in range(episodes):
        ep_seed = seed + i
        res = _eval_episode(model, env, ep_seed, deterministic=deterministic)
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
    sim_times = [r["sim_time"] for r in results]
    success_count = sum(1 for r in results if r["term_reason"] == "success")
    crash_count = sum(1 for r in results if r["term_reason"] == "crash")

    summary: Dict[str, Any] = {
        "preset": preset,
        "model_path": str(model_path),
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": success_count / max(episodes, 1),
        "crash_rate": crash_count / max(episodes, 1),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_sim_time": float(np.mean(sim_times)),
        "wall_time_s": wall_time,
        "per_episode": results,
    }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Evaluation Results")
        print(f"{'=' * 60}")
        print(f"  Return         : {summary['return_mean']:+.2f} ± {summary['return_std']:.2f}")
        print(f"  Success rate   : {summary['success_rate']:.1%}")
        print(f"  Crash rate     : {summary['crash_rate']:.1%}")
        print(f"  Mean ep length : {summary['mean_episode_length']:.0f} steps")
        print(f"  Mean sim time  : {summary['mean_sim_time']:.1f} s")
        print(f"  Wall time      : {wall_time:.2f} s")
        print(f"{'=' * 60}")

    # --- Save ---
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = model_path.parent.name
    out_path = out_dir / f"{run_name}_eval.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"  Results saved to {out_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved SB3 model")
    parser.add_argument(
        "--preset", type=str, default="baseline",
        choices=list(PRESETS.keys()),
        help="Evaluation preset (default: baseline)",
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to saved SB3 model (.zip)",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic (greedy) actions (default: True)",
    )
    parser.add_argument(
        "--stochastic", dest="deterministic", action="store_false",
        help="Use stochastic policy sampling",
    )

    _add_env_args(parser)
    _add_run_args(parser)

    # --- Two-pass parse: resolve preset first, then apply its values as
    #     argparse defaults so that explicit CLI args always win. ---
    pre_args, _ = parser.parse_known_args()
    preset_name = pre_args.preset
    parser.set_defaults(**PRESETS[preset_name])
    args = parser.parse_args()

    print(f"Preset: {preset_name}")

    ec = EnvConfig()
    env_keys = [
        "track", "track_radius", "track_z", "track_n_pts", "wp_radius",
        "n_laps", "dt_sim", "control_decimation", "max_steps",
        "a_residual_max", "yaw_rate_max", "use_estimator",
        "use_gates", "gate_radius_m", "gate_half_thickness_m",
    ]
    _apply_overrides(ec, args, env_keys)

    rc = RunConfig()
    run_keys = [
        "seed", "log_dir", "model_dir", "results_dir", "device",
        "num_envs", "run_name", "verbose",
    ]
    _apply_overrides(rc, args, run_keys)

    seed = rc.seed

    evaluate_model(
        model_path=args.model_path,
        episodes=args.episodes,
        seed=seed,
        env_config=ec,
        results_dir=rc.results_dir,
        deterministic=args.deterministic,
        verbose=True,
        preset=preset_name,
    )


if __name__ == "__main__":
    main()
