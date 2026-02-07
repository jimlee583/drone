"""
Smoke test for the QuadRacingEnv.

Run with::

    python -m quad.envs.smoke_test
    uv run python -m quad.envs.smoke_test

Runs two short episodes:
  1. Zero-action policy (pure SE(3) baseline follows waypoints)
  2. Random-action policy
and prints a summary for each.  No plots, no GUI.
"""

from __future__ import annotations

import sys
import time

import numpy as np


def _run_episode(
    env,
    policy: str = "zero",
    max_steps: int = 400,
) -> dict:
    """Run one episode and return a summary dict."""
    obs, info = env.reset(seed=0)

    total_reward = 0.0
    step = 0
    terminated = False
    truncated = False
    term_reason = ""

    for step in range(1, max_steps + 1):
        if policy == "zero":
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        elif policy == "random":
            action = env.action_space.sample()
        else:
            raise ValueError(f"Unknown policy: {policy}")

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            term_reason = info.get("term_reason", "unknown")
            break

    return {
        "policy": policy,
        "steps": step,
        "total_reward": total_reward,
        "wps_reached": info.get("wps_reached", 0),
        "laps_done": info.get("laps_done", 0),
        "terminated": terminated,
        "truncated": truncated,
        "term_reason": term_reason,
        "final_pos": info.get("position", np.zeros(3)),
        "sim_time": info.get("sim_time", 0.0),
    }


def _print_summary(result: dict) -> None:
    print(f"\n{'=' * 55}")
    print(f"  Policy: {result['policy']}")
    print(f"{'=' * 55}")
    print(f"  Env steps       : {result['steps']}")
    print(f"  Sim time        : {result['sim_time']:.2f} s")
    print(f"  Total reward    : {result['total_reward']:+.2f}")
    print(f"  Waypoints hit   : {result['wps_reached']}")
    print(f"  Laps completed  : {result['laps_done']}")
    print(f"  Terminated      : {result['terminated']}")
    print(f"  Truncated       : {result['truncated']}")
    print(f"  Reason          : {result['term_reason'] or 'n/a'}")
    pos = result["final_pos"]
    print(f"  Final position  : ({pos[0]:+.2f}, {pos[1]:+.2f}, {pos[2]:+.2f})")


def main() -> None:
    # Import here so errors give a clear message if gymnasium is missing.
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        print(
            "ERROR: gymnasium is not installed.\n"
            "Install it with:  pip install 'gymnasium>=0.29'\n"
            "  or:  uv pip install 'gymnasium>=0.29'",
            file=sys.stderr,
        )
        sys.exit(1)

    from quad.envs.quad_racing_env import QuadRacingEnv, EnvConfig
    from quad.envs.track import circle_track

    track = circle_track(radius=3.0, z=1.5, n_pts=12, wp_radius=1.0, n_laps=1)
    cfg = EnvConfig(
        dt_sim=0.005,
        control_decimation=10,
        max_steps=500,
    )
    env = QuadRacingEnv(track=track, config=cfg, render_mode="human")

    print("=" * 55)
    print("  QuadRacingEnv Smoke Test")
    print("=" * 55)

    # --- Zero policy ---
    t0 = time.monotonic()
    result_zero = _run_episode(env, policy="zero", max_steps=400)
    wall_zero = time.monotonic() - t0
    _print_summary(result_zero)
    print(f"  Wall-clock time : {wall_zero:.2f} s")

    # --- Random policy ---
    t0 = time.monotonic()
    result_rand = _run_episode(env, policy="random", max_steps=200)
    wall_rand = time.monotonic() - t0
    _print_summary(result_rand)
    print(f"  Wall-clock time : {wall_rand:.2f} s")

    env.close()

    # --- Final verdict ---
    print("\n" + "=" * 55)
    ok = True
    if result_zero["total_reward"] < -500:
        print("  [WARN] Zero-policy reward very low — baseline may not track.")
        ok = False
    if result_zero["term_reason"] == "crash":
        print("  [WARN] Zero-policy crashed.")
        ok = False
    if result_zero["wps_reached"] == 0:
        print("  [WARN] Zero-policy reached 0 waypoints.")
        # Not necessarily a failure if the track is large

    if ok:
        print("  [OK] Smoke test completed successfully.")
    else:
        print("  [WARN] Smoke test finished with warnings (see above).")
    print("=" * 55)


if __name__ == "__main__":
    main()
