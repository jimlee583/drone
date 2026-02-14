"""
One-command baseline evaluation suite.

Runs multiple baseline policies across seeds and collects results into a
single timestamped folder.

Usage::

    python -m quad.rl.run_suite --suite baseline --seeds 1 2 3 --episodes 20
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Any, Dict, List

from quad.rl.baselines import PRESETS, run_baseline
from quad.rl.config import EnvConfig


# ---------------------------------------------------------------------------
# Suites: mapping of suite name → list of policy names to run
# ---------------------------------------------------------------------------

SUITES: Dict[str, List[str]] = {
    "baseline": ["zero", "random"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_env_config_from_preset(preset_name: str) -> EnvConfig:
    """Create an EnvConfig with preset values applied on top of defaults."""
    ec = EnvConfig()
    preset_vals = PRESETS[preset_name]
    env_keys = [
        "track", "track_radius", "track_z", "track_n_pts", "wp_radius",
        "n_laps", "dt_sim", "control_decimation", "max_steps",
        "a_residual_max", "yaw_rate_max", "use_estimator",
        "use_gates", "gate_radius_m", "gate_half_thickness_m",
    ]
    for k in env_keys:
        if k in preset_vals:
            setattr(ec, k, preset_vals[k])
    return ec


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a baseline evaluation suite",
    )
    parser.add_argument(
        "--suite", type=str, default="baseline",
        choices=list(SUITES.keys()),
        help="Suite to run (default: baseline)",
    )
    parser.add_argument(
        "--preset", type=str, default="baseline",
        choices=list(PRESETS.keys()),
        help="Evaluation preset (default: baseline)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1, 2, 3],
        help="Seeds to evaluate (default: 1 2 3)",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Episodes per seed (default: 10)",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results_rl",
        help="Base results directory (default: results_rl)",
    )
    args = parser.parse_args()

    policies = SUITES[args.suite]
    preset_vals = PRESETS[args.preset]
    wind_enabled: bool = preset_vals.get("wind_enabled", True)

    # Timestamped subfolder
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.results_dir) / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    ec = _build_env_config_from_preset(args.preset)

    written: List[str] = []

    for policy in policies:
        for seed in args.seeds:
            print(
                f"\n>>> policy={policy}  seed={seed}  "
                f"preset={args.preset}  episodes={args.episodes}"
            )
            run_baseline(
                policy_name=policy,
                episodes=args.episodes,
                seed=seed,
                env_config=ec,
                results_dir=str(out_dir),
                verbose=True,
                wind_enabled=wind_enabled,
                preset=args.preset,
            )
            path = out_dir / f"baseline_{policy}_seed{seed}.json"
            written.append(str(path))

    print(f"\n{'=' * 55}")
    print(f"Suite '{args.suite}' complete.  Files written:")
    for p in written:
        print(f"  {p}")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
