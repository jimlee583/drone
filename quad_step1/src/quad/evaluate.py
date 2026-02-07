"""
Monte Carlo evaluation of controller robustness.

CLI entrypoint: ``python -m quad.evaluate``

Example usage::

    python -m quad.evaluate --scenario hover --trials 50 --seed 42 --verbose
    python -m quad.evaluate --list-scenarios
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from quad.params import default_params
from quad.sim import run_sim
from quad.scenarios import get_scenario, list_scenarios, randomize_params
from quad.metrics import compute_all_metrics


def run_evaluation(
    scenario_name: str,
    n_trials: int = 50,
    seed: int = 42,
    output_dir: str = "results",
    verbose: bool = False,
    use_estimator: bool = False,
) -> list[dict]:
    """Run Monte Carlo evaluation for a given scenario.

    Args:
        scenario_name: Registered scenario name.
        n_trials: Number of randomized trials.
        seed: Master RNG seed for reproducibility.
        output_dir: Directory for output CSV / JSON files.
        verbose: Print per-trial progress.
        use_estimator: If True, run controller from EKF-estimated state.

    Returns:
        List of per-trial metric dicts.
    """
    scenario = get_scenario(scenario_name)
    base_params = default_params()
    if use_estimator:
        base_params.use_estimator = True
    rng = np.random.default_rng(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    if verbose:
        print(f"Evaluating scenario '{scenario_name}' | "
              f"{n_trials} trials | seed={seed}")
        print(f"  t_final={scenario.t_final}s  dt={scenario.dt}s")
        print("-" * 60)

    for i in range(n_trials):
        params = randomize_params(base_params, rng)
        traj_fn = scenario.traj_fn()

        try:
            log = run_sim(params, traj_fn, scenario.t_final, scenario.dt)
            metrics = compute_all_metrics(log, params)
        except Exception:
            if verbose:
                traceback.print_exc()
            metrics = {"crashed": 1.0}

        row = {"trial": i, **metrics}
        results.append(row)

        if verbose:
            crash_tag = " [CRASH]" if metrics.get("crashed", 0) else ""
            rms_mm = metrics.get("rms_pos_err", float("nan")) * 1000
            print(f"  trial {i:3d}/{n_trials}  "
                  f"rms_pos={rms_mm:7.1f} mm{crash_tag}")

    # --- Summary -------------------------------------------------------
    _print_summary(results, scenario_name)

    # --- Write output files --------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{scenario_name}_{timestamp}_seed{seed}"

    _write_csv(results, out_path / f"{stem}.csv")
    _write_json(results, scenario_name, n_trials, seed, scenario,
                out_path / f"{stem}.json")

    if verbose:
        print(f"\nResults written to {out_path / stem}.[csv|json]")

    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict], scenario_name: str) -> None:
    """Print a summary table to stdout."""
    crashes = sum(1 for r in results if r.get("crashed", 0))
    ok = [r for r in results if not r.get("crashed", 0)]

    print(f"\n{'=' * 60}")
    print(f"Scenario: {scenario_name}  |  "
          f"{len(results)} trials  |  {crashes} crashes")
    print(f"{'=' * 60}")

    if not ok:
        print("All trials crashed!")
        return

    keys = [
        ("rms_pos_err", 1000, "mm"),
        ("max_pos_err", 1000, "mm"),
        ("rms_vel_err", 1000, "mm/s"),
        ("control_effort", 1, ""),
        ("thrust_sat_pct", 1, "%"),
        ("moment_sat_overall", 1, "%"),
    ]

    header = f"{'metric':<22s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}"
    print(header)
    print("-" * len(header))

    for key, scale, unit in keys:
        vals = [r[key] * scale for r in ok if key in r]
        if not vals:
            continue
        arr = np.array(vals)
        label = f"{key} [{unit}]" if unit else key
        print(f"{label:<22s} {np.mean(arr):10.2f} {np.std(arr):10.2f} "
              f"{np.min(arr):10.2f} {np.max(arr):10.2f}")

    print()


def _write_csv(results: list[dict], path: Path) -> None:
    """Write per-trial results to CSV using stdlib csv."""
    if not results:
        return

    # Collect all keys across rows (some crashed rows may be sparse)
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in results:
        for k in r:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def _write_json(
    results: list[dict],
    scenario_name: str,
    n_trials: int,
    seed: int,
    scenario,
    path: Path,
) -> None:
    """Write full config + per-trial metrics + aggregate stats to JSON."""
    ok = [r for r in results if not r.get("crashed", 0)]
    crashes = sum(1 for r in results if r.get("crashed", 0))

    # Aggregate stats for non-crashed trials
    aggregate: dict[str, dict[str, float]] = {}
    if ok:
        metric_keys = [k for k in ok[0] if k != "trial"]
        for key in metric_keys:
            vals = [r[key] for r in ok if key in r]
            if vals:
                arr = np.array(vals)
                aggregate[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }

    doc = {
        "config": {
            "scenario": scenario_name,
            "t_final": scenario.t_final,
            "dt": scenario.dt,
            "n_trials": n_trials,
            "seed": seed,
        },
        "summary": {
            "total_trials": len(results),
            "crashes": crashes,
        },
        "aggregate": aggregate,
        "trials": results,
    }

    with open(path, "w") as f:
        json.dump(doc, f, indent=2, default=_json_default)


def _json_default(obj):
    """Fallback serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo evaluation of quadrotor controller robustness.",
    )
    parser.add_argument(
        "--scenario", type=str, default=None,
        help="Scenario name (use --list-scenarios to see options).",
    )
    parser.add_argument(
        "--trials", type=int, default=50,
        help="Number of randomized trials (default: 50).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed (default: 42).",
    )
    parser.add_argument(
        "--out", type=str, default="results",
        help="Output directory (default: results).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-trial progress.",
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="List available scenarios and exit.",
    )
    parser.add_argument(
        "--use-estimator", action="store_true",
        help="Run controller off EKF-estimated state.",
    )

    args = parser.parse_args()

    if args.list_scenarios:
        print("Available scenarios:")
        for name in list_scenarios():
            s = get_scenario(name)
            print(f"  {name:<12s}  t_final={s.t_final:.1f}s  dt={s.dt}s")
        sys.exit(0)

    if args.scenario is None:
        parser.error("--scenario is required (or use --list-scenarios)")

    run_evaluation(
        scenario_name=args.scenario,
        n_trials=args.trials,
        seed=args.seed,
        output_dir=args.out,
        verbose=args.verbose,
        use_estimator=args.use_estimator,
    )


if __name__ == "__main__":
    main()
