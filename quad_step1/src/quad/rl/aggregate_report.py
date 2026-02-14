"""Aggregate RL evaluation results into a human-readable console report.

Reads the JSON outputs produced by ``quad.rl.baselines`` or
``quad.rl.run_suite`` and prints summary statistics grouped by
(preset, policy).

Usage
-----
::

    # basic usage
    uv run python -m quad.rl.aggregate_report --input results_rl/20260214_122014

    # recursive search, sorted by return, showing per-file stats
    uv run python -m quad.rl.aggregate_report \\
        --input results_rl/20260214_122014 \\
        --recursive --sort-by return --show-files

    # skip files with fewer than 10 episodes
    uv run python -m quad.rl.aggregate_report \\
        --input results_rl/20260214_122014 --min-episodes 10
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


# ── helpers ────────────────────────────────────────────────────────────────

def _safe_get(d: dict, *keys: str, default: Any = None) -> Any:
    """Traverse nested dicts safely, returning *default* on any miss."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


def _fmt(val: Any, decimals: int = 3, width: int = 10) -> str:
    """Format a numeric value for table display."""
    if val is None:
        return "N/A".rjust(width)
    if isinstance(val, float):
        return f"{val:.{decimals}f}".rjust(width)
    return str(val).rjust(width)


def _header_line(cols: list[tuple[str, int]]) -> str:
    return "  ".join(name.rjust(w) for name, w in cols)


def _sep_line(cols: list[tuple[str, int]]) -> str:
    return "  ".join("-" * w for _, w in cols)


# ── file parsing ──────────────────────────────────────────────────────────

def _extract_preset(data: dict) -> str:
    """Resolve preset with fallback chain."""
    preset = data.get("preset")
    if preset is not None:
        return str(preset)
    per_ep = data.get("per_episode")
    if isinstance(per_ep, list) and len(per_ep) > 0:
        ep0 = per_ep[0]
        if isinstance(ep0, dict):
            cfg = ep0.get("episode_config")
            if isinstance(cfg, dict):
                p = cfg.get("preset")
                if p is not None:
                    return str(p)
    return "unknown"


def _extract_git_commits(data: dict) -> set[str]:
    """Collect distinct git_commit values from episode_config blocks."""
    commits: set[str] = set()
    per_ep = data.get("per_episode")
    if not isinstance(per_ep, list):
        return commits
    for ep in per_ep:
        if not isinstance(ep, dict):
            continue
        cfg = ep.get("episode_config")
        if isinstance(cfg, dict):
            gc = cfg.get("git_commit")
            if gc is not None:
                commits.add(str(gc))
    return commits


def _count_term_codes(data: dict) -> Counter:
    """Tally term_code values across per_episode entries."""
    counts: Counter = Counter()
    per_ep = data.get("per_episode")
    if not isinstance(per_ep, list):
        return counts
    for ep in per_ep:
        if not isinstance(ep, dict):
            continue
        tc = ep.get("term_code")
        if tc is not None:
            counts[str(tc)] += 1
    return counts


def parse_result_file(path: Path) -> dict[str, Any] | None:
    """Parse a single JSON result file and return a normalised record.

    Returns ``None`` if the file cannot be parsed or is missing critical
    keys.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  [WARN] skipping {path}: {exc}", file=sys.stderr)
        return None

    if not isinstance(data, dict):
        print(f"  [WARN] skipping {path}: top-level is not an object", file=sys.stderr)
        return None

    record: dict[str, Any] = {
        "file": str(path),
        "policy": data.get("policy"),
        "seed": data.get("seed"),
        "preset": _extract_preset(data),
        "git_commits": _extract_git_commits(data),
        "episodes": data.get("episodes"),
        "return_mean": data.get("return_mean"),
        "return_std": data.get("return_std"),
        "success_rate": data.get("success_rate"),
        "crash_rate": data.get("crash_rate"),
        "mean_episode_length": data.get("mean_episode_length"),
        "wall_time_s": data.get("wall_time_s"),
        "term_codes": _count_term_codes(data),
    }
    return record


# ── aggregation ───────────────────────────────────────────────────────────

def _mean(vals: list[float | None]) -> float | None:
    cleaned = [v for v in vals if v is not None]
    return statistics.mean(cleaned) if cleaned else None


def _stdev(vals: list[float | None]) -> float | None:
    cleaned = [v for v in vals if v is not None]
    if len(cleaned) < 2:
        return 0.0 if len(cleaned) == 1 else None
    return statistics.stdev(cleaned)


GroupKey = tuple[str, str]  # (preset, policy)


def aggregate(records: list[dict[str, Any]]) -> dict[GroupKey, dict[str, Any]]:
    """Group records by (preset, policy) and compute aggregate stats."""
    groups: dict[GroupKey, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        key: GroupKey = (r["preset"], r.get("policy") or "unknown")
        groups[key].append(r)

    agg: dict[GroupKey, dict[str, Any]] = {}
    for key, recs in groups.items():
        sr = [r["success_rate"] for r in recs]
        cr = [r["crash_rate"] for r in recs]
        rm = [r["return_mean"] for r in recs]
        mel = [r["mean_episode_length"] for r in recs]

        combined_tc: Counter = Counter()
        total_ep = 0
        git_commits: set[str] = set()
        for r in recs:
            combined_tc.update(r["term_codes"])
            ep = r.get("episodes")
            if isinstance(ep, (int, float)):
                total_ep += int(ep)
            git_commits |= r["git_commits"]

        agg[key] = {
            "num_runs": len(recs),
            "num_episodes_total": total_ep,
            "success_rate_mean": _mean(sr),
            "success_rate_std": _stdev(sr),
            "crash_rate_mean": _mean(cr),
            "crash_rate_std": _stdev(cr),
            "return_mean_mean": _mean(rm),
            "return_mean_std": _stdev(rm),
            "mean_episode_length_mean": _mean(mel),
            "term_codes": combined_tc,
            "git_commits": git_commits,
        }
    return agg


# ── report printing ───────────────────────────────────────────────────────

_SORT_KEYS = {
    "success": "success_rate_mean",
    "return": "return_mean_mean",
    "crash": "crash_rate_mean",
}


def _print_file_table(records: list[dict[str, Any]]) -> None:
    """Print a per-file headline table."""
    cols: list[tuple[str, int]] = [
        ("file", 48),
        ("policy", 10),
        ("seed", 6),
        ("episodes", 9),
        ("success", 9),
        ("crash", 9),
        ("ret_mean", 10),
        ("wall_s", 8),
    ]
    print()
    print("Per-file results")
    print("=" * 120)
    print(_header_line(cols))
    print(_sep_line(cols))
    for r in records:
        fname = Path(r["file"]).name
        if len(fname) > 48:
            fname = "…" + fname[-47:]
        row = "  ".join([
            fname.rjust(48),
            _fmt(r.get("policy"), width=10),
            _fmt(r.get("seed"), width=6),
            _fmt(r.get("episodes"), width=9),
            _fmt(r.get("success_rate"), decimals=3, width=9),
            _fmt(r.get("crash_rate"), decimals=3, width=9),
            _fmt(r.get("return_mean"), decimals=2, width=10),
            _fmt(r.get("wall_time_s"), decimals=1, width=8),
        ])
        print(row)
    print()


def _print_summary_table(
    agg: dict[GroupKey, dict[str, Any]],
    sort_by: str,
) -> None:
    """Print grouped summary table."""
    sort_key = _SORT_KEYS.get(sort_by, "success_rate_mean")
    # Sort descending for success/return, ascending for crash
    reverse = sort_by != "crash"
    items = sorted(
        agg.items(),
        key=lambda kv: kv[1].get(sort_key) or 0.0,
        reverse=reverse,
    )

    cols: list[tuple[str, int]] = [
        ("preset", 14),
        ("policy", 12),
        ("runs", 6),
        ("episodes", 9),
        ("success", 9),
        ("±σ", 8),
        ("crash", 9),
        ("±σ", 8),
        ("return", 10),
        ("±σ", 10),
        ("ep_len", 8),
    ]

    print()
    print("Summary  (grouped by preset × policy, sorted by {})".format(sort_by))
    print("=" * 120)
    print(_header_line(cols))
    print(_sep_line(cols))

    for (preset, policy), stats in items:
        row = "  ".join([
            preset.rjust(14),
            policy.rjust(12),
            _fmt(stats["num_runs"], width=6),
            _fmt(stats["num_episodes_total"], width=9),
            _fmt(stats["success_rate_mean"], decimals=3, width=9),
            _fmt(stats["success_rate_std"], decimals=3, width=8),
            _fmt(stats["crash_rate_mean"], decimals=3, width=9),
            _fmt(stats["crash_rate_std"], decimals=3, width=8),
            _fmt(stats["return_mean_mean"], decimals=2, width=10),
            _fmt(stats["return_mean_std"], decimals=2, width=10),
            _fmt(stats["mean_episode_length_mean"], decimals=1, width=8),
        ])
        print(row)
    print()


def _print_term_breakdown(agg: dict[GroupKey, dict[str, Any]]) -> None:
    """Print top-5 termination codes per group."""
    print("Termination breakdown  (top 5 per group)")
    print("-" * 60)
    for (preset, policy), stats in sorted(agg.items()):
        tc: Counter = stats["term_codes"]
        if not tc:
            continue
        top5 = tc.most_common(5)
        total = sum(tc.values())
        parts = [f"{code}: {cnt} ({100*cnt/total:.1f}%)" for code, cnt in top5]
        print(f"  {preset:>14s} / {policy:<12s}  {', '.join(parts)}")
    print()


def _print_footer(
    input_dir: Path,
    num_files: int,
    all_commits: set[str],
) -> None:
    """Print a short footer with metadata."""
    print("-" * 60)
    print(f"  input directory : {input_dir}")
    print(f"  files parsed    : {num_files}")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"  report time     : {ts}")
    if all_commits:
        commit_str = ", ".join(sorted(all_commits))
        label = "  git commit(s)   : " + commit_str
        print(label)
        if len(all_commits) > 1:
            print("  ⚠  WARNING: multiple distinct git commits detected!")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m quad.rl.aggregate_report",
        description="Aggregate RL evaluation JSON results into a summary report.",
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to directory containing result JSON files.",
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        default=False,
        help="Search for *.json recursively under --input.",
    )
    p.add_argument(
        "--sort-by",
        choices=["success", "return", "crash"],
        default="success",
        help="Column to sort the summary table by (default: success).",
    )
    p.add_argument(
        "--show-files",
        action="store_true",
        default=False,
        help="Print per-file headline statistics before the summary.",
    )
    p.add_argument(
        "--min-episodes",
        type=int,
        default=0,
        metavar="N",
        help="Skip result files with fewer than N episodes.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    input_dir: Path = args.input

    if not input_dir.is_dir():
        print(f"ERROR: {input_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    # discover JSON files
    pattern = "**/*.json" if args.recursive else "*.json"
    json_files = sorted(input_dir.glob(pattern))

    if not json_files:
        print(f"No *.json files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    # parse
    records: list[dict[str, Any]] = []
    for jf in json_files:
        rec = parse_result_file(jf)
        if rec is None:
            continue
        ep = rec.get("episodes")
        if isinstance(ep, (int, float)) and ep < args.min_episodes:
            continue
        records.append(rec)

    if not records:
        print("No valid result files after filtering.", file=sys.stderr)
        sys.exit(1)

    # optional per-file table
    if args.show_files:
        _print_file_table(records)

    # aggregate + print
    agg = aggregate(records)
    _print_summary_table(agg, sort_by=args.sort_by)
    _print_term_breakdown(agg)

    # footer
    all_commits: set[str] = set()
    for r in records:
        all_commits |= r["git_commits"]
    _print_footer(input_dir, len(records), all_commits)


if __name__ == "__main__":
    main()
