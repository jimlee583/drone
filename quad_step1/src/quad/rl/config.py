"""
Reproducible configuration for RL training and evaluation.

Provides three dataclass containers and an ``argparse``-based loader so that
every run can be reconstructed from a single JSON file.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

@dataclass
class EnvConfig:
    """Mirror of the options we expose when constructing ``QuadRacingEnv``."""

    # Track
    track: str = "circle"           # "circle" | "figure8" | "line"
    track_radius: float = 3.0
    track_z: float = 1.5
    track_n_pts: int = 12
    wp_radius: float = 0.5
    n_laps: int = 1

    # Timing
    dt_sim: float = 0.005
    control_decimation: int = 10
    max_steps: int = 2000

    # Action scaling
    a_residual_max: float = 5.0
    yaw_rate_max: float = 2.0

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
# PPO hyper-parameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """Stable-Baselines3 PPO hyper-parameters."""

    policy: str = "MlpPolicy"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    policy_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"net_arch": [dict(pi=[64, 64], vf=[64, 64])]}
    )

    # Callbacks
    eval_freq: int = 10_000          # steps between eval rounds
    n_eval_episodes: int = 5
    checkpoint_freq: int = 50_000    # steps between checkpoint saves


# ---------------------------------------------------------------------------
# Run meta-configuration
# ---------------------------------------------------------------------------

@dataclass
class RunConfig:
    """Paths, device, parallelism, seed."""

    seed: int = 0
    log_dir: str = "runs"
    model_dir: str = "models"
    results_dir: str = "results_rl"
    device: str = "auto"
    num_envs: int = 1
    run_name: str = ""               # auto-generated if empty
    verbose: int = 1


# ---------------------------------------------------------------------------
# Composite config
# ---------------------------------------------------------------------------

@dataclass
class FullConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    run: RunConfig = field(default_factory=RunConfig)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def config_to_dict(cfg: FullConfig) -> Dict[str, Any]:
    return asdict(cfg)


def save_config(cfg: FullConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config_to_dict(cfg), f, indent=2)


# ---------------------------------------------------------------------------
# Argparse loader
# ---------------------------------------------------------------------------

def _add_env_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Environment")
    g.add_argument("--track", type=str, default=None, help="circle | figure8 | line")
    g.add_argument("--track-radius", type=float, default=None)
    g.add_argument("--track-z", type=float, default=None)
    g.add_argument("--track-n-pts", type=int, default=None)
    g.add_argument("--wp-radius", type=float, default=None)
    g.add_argument("--n-laps", type=int, default=None)
    g.add_argument("--dt-sim", type=float, default=None)
    g.add_argument("--control-decimation", type=int, default=None)
    g.add_argument("--max-steps", type=int, default=None)
    g.add_argument("--a-residual-max", type=float, default=None)
    g.add_argument("--yaw-rate-max", type=float, default=None)
    g.add_argument("--use-estimator", action="store_true", default=None)
    g.add_argument("--no-estimator", dest="use_estimator", action="store_false")


def _add_ppo_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("PPO")
    g.add_argument("--total-timesteps", type=int, default=None)
    g.add_argument("--learning-rate", type=float, default=None)
    g.add_argument("--n-steps", type=int, default=None)
    g.add_argument("--batch-size", type=int, default=None)
    g.add_argument("--n-epochs", type=int, default=None)
    g.add_argument("--gamma", type=float, default=None)
    g.add_argument("--gae-lambda", type=float, default=None)
    g.add_argument("--clip-range", type=float, default=None)
    g.add_argument("--ent-coef", type=float, default=None)
    g.add_argument("--vf-coef", type=float, default=None)
    g.add_argument("--max-grad-norm", type=float, default=None)
    g.add_argument("--eval-freq", type=int, default=None)
    g.add_argument("--n-eval-episodes", type=int, default=None)
    g.add_argument("--checkpoint-freq", type=int, default=None)


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("Run")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--log-dir", type=str, default=None)
    g.add_argument("--model-dir", type=str, default=None)
    g.add_argument("--results-dir", type=str, default=None)
    g.add_argument("--device", type=str, default=None)
    g.add_argument("--num-envs", type=int, default=None)
    g.add_argument("--run-name", type=str, default=None)
    g.add_argument("--verbose", type=int, default=None)


def _apply_overrides(dc: object, ns: argparse.Namespace, keys: List[str]) -> None:
    """Apply non-None argparse values to the dataclass."""
    for key in keys:
        arg_key = key.replace("-", "_")
        val = getattr(ns, arg_key, None)
        if val is not None:
            setattr(dc, arg_key, val)


def load_config_from_args(
    description: str = "Quad RL",
    include_ppo: bool = True,
) -> Tuple[FullConfig, argparse.Namespace]:
    """Build a :class:`FullConfig` from defaults + CLI overrides.

    Returns
    -------
    cfg : FullConfig
    args : argparse.Namespace  (raw, for any extra flags the caller added)
    """
    parser = argparse.ArgumentParser(description=description)
    _add_env_args(parser)
    if include_ppo:
        _add_ppo_args(parser)
    _add_run_args(parser)
    args = parser.parse_args()

    cfg = FullConfig()

    # --- env overrides ---
    env_keys = [
        "track", "track_radius", "track_z", "track_n_pts", "wp_radius",
        "n_laps", "dt_sim", "control_decimation", "max_steps",
        "a_residual_max", "yaw_rate_max", "use_estimator",
    ]
    _apply_overrides(cfg.env, args, env_keys)

    # --- ppo overrides ---
    if include_ppo:
        ppo_keys = [
            "total_timesteps", "learning_rate", "n_steps", "batch_size",
            "n_epochs", "gamma", "gae_lambda", "clip_range", "ent_coef",
            "vf_coef", "max_grad_norm", "eval_freq", "n_eval_episodes",
            "checkpoint_freq",
        ]
        _apply_overrides(cfg.ppo, args, ppo_keys)

    # --- run overrides ---
    run_keys = [
        "seed", "log_dir", "model_dir", "results_dir", "device",
        "num_envs", "run_name", "verbose",
    ]
    _apply_overrides(cfg.run, args, run_keys)

    return cfg, args
