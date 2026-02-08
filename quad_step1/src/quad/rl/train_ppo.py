"""
PPO training for QuadRacingEnv using Stable-Baselines3.

Usage::

    python -m quad.rl.train_ppo --track circle --total-timesteps 300000 --seed 1
    python -m quad.rl.train_ppo --track figure8 --use-estimator --num-envs 4

Logs are saved to ``runs/<run_name>/`` (TensorBoard) and model checkpoints
to ``models/<run_name>/``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

import gymnasium as gym

from quad.envs.quad_racing_env import QuadRacingEnv, EnvConfig as _BaseEnvConfig
from quad.envs.track import circle_track, figure8_track, line_track
from quad.rl.config import (
    EnvConfig,
    FullConfig,
    PPOConfig,
    RunConfig,
    config_to_dict,
    load_config_from_args,
    save_config,
)
from quad.rl.wrappers import wrap_env


# ---------------------------------------------------------------------------
# Track + env helpers (shared with baselines)
# ---------------------------------------------------------------------------

def _make_track(ec: EnvConfig):
    builders = {"circle": circle_track, "figure8": figure8_track, "line": line_track}
    builder = builders.get(ec.track)
    if builder is None:
        raise ValueError(f"Unknown track '{ec.track}'. Choose from {list(builders)}")
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
# Vectorised environment factory
# ---------------------------------------------------------------------------

def make_env(
    rank: int,
    seed: int,
    env_config: EnvConfig,
) -> Callable[[], gym.Env]:
    """Return a *thunk* that creates and seeds a single QuadRacingEnv.

    Each env gets a unique seed ``seed + rank`` for determinism.
    """

    def _init() -> gym.Env:
        track = _make_track(env_config)
        base_cfg = _make_base_env_config(env_config)
        env = QuadRacingEnv(track=track, config=base_cfg)
        env = wrap_env(env, clip_actions=True, normalize_obs=False)
        env.reset(seed=seed + rank)
        return env

    return _init


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train(cfg: FullConfig) -> Path:
    """Run PPO training and return the path to the saved model.

    Parameters
    ----------
    cfg : FullConfig
        Complete merged configuration.

    Returns
    -------
    Path
        Path to the final saved model (``.zip``).
    """
    # Lazy-import SB3 so the rest of the package doesn't need it
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except ImportError as exc:
        raise ImportError(
            "stable-baselines3 is required for training. Install with:\n"
            "  uv sync --extra rl\n"
            "  # or\n"
            '  pip install "stable-baselines3>=2.3.0"'
        ) from exc

    ec = cfg.env
    pc = cfg.ppo
    rc = cfg.run

    # --- Run name ---
    if not rc.run_name:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        rc.run_name = f"ppo_{ec.track}_seed{rc.seed}_{ts}"

    log_path = Path(rc.log_dir) / rc.run_name
    model_path = Path(rc.model_dir) / rc.run_name
    log_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    # --- Save config ---
    save_config(cfg, model_path / "config.json")

    # --- Vectorised training envs ---
    train_envs = DummyVecEnv(
        [make_env(i, rc.seed, ec) for i in range(rc.num_envs)]
    )
    train_envs = VecMonitor(train_envs)

    # --- Eval env (single, separate seed range) ---
    eval_envs = DummyVecEnv(
        [make_env(0, rc.seed + 1000, ec)]
    )
    eval_envs = VecMonitor(eval_envs)

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=max(pc.checkpoint_freq // rc.num_envs, 1),
        save_path=str(model_path),
        name_prefix="checkpoint",
        verbose=rc.verbose,
    )

    eval_cb = EvalCallback(
        eval_envs,
        best_model_save_path=str(model_path),
        log_path=str(log_path),
        eval_freq=max(pc.eval_freq // rc.num_envs, 1),
        n_eval_episodes=pc.n_eval_episodes,
        deterministic=True,
        verbose=rc.verbose,
    )

    # --- PPO model ---
    model = PPO(
        policy=pc.policy,
        env=train_envs,
        learning_rate=pc.learning_rate,
        n_steps=pc.n_steps,
        batch_size=pc.batch_size,
        n_epochs=pc.n_epochs,
        gamma=pc.gamma,
        gae_lambda=pc.gae_lambda,
        clip_range=pc.clip_range,
        ent_coef=pc.ent_coef,
        vf_coef=pc.vf_coef,
        max_grad_norm=pc.max_grad_norm,
        policy_kwargs=pc.policy_kwargs,
        tensorboard_log=str(log_path),
        seed=rc.seed,
        device=rc.device,
        verbose=rc.verbose,
    )

    # --- Quick zero-policy baseline (gives a reference return) ---
    if rc.verbose >= 1:
        _n_baseline = 5
        _baseline_returns = []
        _bl_env = QuadRacingEnv(
            track=_make_track(ec), config=_make_base_env_config(ec),
        )
        for _i in range(_n_baseline):
            _obs, _ = _bl_env.reset(seed=rc.seed + _i)
            _ep_ret = 0.0
            while True:
                _obs, _r, _term, _trunc, _ = _bl_env.step(
                    np.zeros(_bl_env.action_space.shape, dtype=np.float32)
                )
                _ep_ret += _r
                if _term or _trunc:
                    break
            _baseline_returns.append(_ep_ret)
        _bl_env.close()
        _bl_mean = float(np.mean(_baseline_returns))
        _bl_std = float(np.std(_baseline_returns))

    if rc.verbose >= 1:
        print(f"\n{'=' * 60}")
        print(f"  PPO Training — {rc.run_name}")
        print(f"{'=' * 60}")
        print(f"  Track          : {ec.track}")
        print(f"  Estimator      : {ec.use_estimator}")
        print(f"  Seed           : {rc.seed}")
        print(f"  Num envs       : {rc.num_envs}")
        print(f"  Total timesteps: {pc.total_timesteps:,}")
        print(f"  Log dir        : {log_path}")
        print(f"  Model dir      : {model_path}")
        print(f"  Zero-policy R  : {_bl_mean:+.2f} ± {_bl_std:.2f}  ({_n_baseline} eps)")
        print(f"{'=' * 60}\n")

    # --- Train ---
    t0 = time.monotonic()
    model.learn(
        total_timesteps=pc.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=False,
    )
    wall_time = time.monotonic() - t0

    # --- Save final model ---
    final_model_path = model_path / "final_model"
    model.save(str(final_model_path))

    # --- Training summary ---
    # Read best mean reward from eval log if available
    best_mean_reward = None
    eval_log = log_path / "evaluations.npz"
    if eval_log.exists():
        data = np.load(str(eval_log))
        if "results" in data:
            mean_rewards = data["results"].mean(axis=1)
            best_mean_reward = float(np.max(mean_rewards))

    summary = {
        "run_name": rc.run_name,
        "total_timesteps": pc.total_timesteps,
        "wall_time_s": wall_time,
        "best_mean_reward": best_mean_reward,
        "final_model_path": str(final_model_path) + ".zip",
        "seed": rc.seed,
    }
    summary_path = model_path / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    if rc.verbose >= 1:
        print(f"\n{'=' * 60}")
        print(f"  Training complete!")
        print(f"  Wall time      : {wall_time:.1f} s")
        if best_mean_reward is not None:
            print(f"  Best mean R    : {best_mean_reward:.2f}")
        print(f"  Final model    : {final_model_path}.zip")
        print(f"  Summary        : {summary_path}")
        print(f"{'=' * 60}")

    # --- Cleanup ---
    train_envs.close()
    eval_envs.close()

    return Path(str(final_model_path) + ".zip")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    cfg, _args = load_config_from_args(
        description="PPO training for QuadRacingEnv",
        include_ppo=True,
    )
    train(cfg)


if __name__ == "__main__":
    main()
