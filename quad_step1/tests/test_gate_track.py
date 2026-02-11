"""Sanity tests for gate-plane crossing detection state machine."""

import numpy as np
import pytest

from quad.envs.gates import Gate, waypoints_to_gates
from quad.envs.gate_track import GateTrack


# ---- Test 1: Single gate straight-line pass --------------------------------

def test_single_gate_forward_pass():
    gate = Gate(center_w=[0, 0, 0], normal_w=[1, 0, 0],
                radius_m=1.0, half_thickness_m=0.2)
    gt = GateTrack([gate], n_laps=1)
    gt.reset(np.array([-2.0, 0, 0]))

    positions = [np.array([x, 0, 0]) for x in np.linspace(-2, 2, 20)]
    crossed_count = 0
    for prev, new in zip(positions[:-1], positions[1:]):
        result = gt.advance_if_crossed(prev, new)
        if result.crossed:
            crossed_count += 1
        assert not result.wrong_dir, "Should not trigger wrong_dir on forward pass"
        assert not result.lateral_miss, "On-axis pass should not miss laterally"

    assert crossed_count == 1, f"Expected exactly 1 crossing, got {crossed_count}"
    assert gt.done, "Single-lap track should be done"


# ---- Test 2: Reverse pass triggers wrong_dir --------------------------------

def test_reverse_pass_triggers_wrong_dir():
    gate = Gate(center_w=[0, 0, 0], normal_w=[1, 0, 0],
                radius_m=1.0, half_thickness_m=0.2)
    gt = GateTrack([gate], n_laps=1)
    gt.reset(np.array([2.0, 0, 0]))  # ahead of gate (wrong side)

    positions = [np.array([x, 0, 0]) for x in np.linspace(2, -2, 20)]
    wrong_dir_count = 0
    for prev, new in zip(positions[:-1], positions[1:]):
        result = gt.advance_if_crossed(prev, new)
        if result.wrong_dir:
            wrong_dir_count += 1
        assert not result.crossed, "Reverse pass should not count as crossing"

    assert wrong_dir_count == 1, f"Expected 1 wrong_dir, got {wrong_dir_count}"
    assert not gt.done, "Track should not be done after wrong-dir"


# ---- Test 3: Cross plane outside radius triggers lateral_miss ---------------

def test_lateral_miss():
    gate = Gate(center_w=[0, 0, 0], normal_w=[1, 0, 0],
                radius_m=0.5, half_thickness_m=0.2)
    gt = GateTrack([gate], n_laps=1)
    gt.reset(np.array([-2.0, 3.0, 0]))  # behind gate, 3m lateral offset

    positions = [np.array([x, 3.0, 0]) for x in np.linspace(-2, 2, 20)]
    miss_count = 0
    for prev, new in zip(positions[:-1], positions[1:]):
        result = gt.advance_if_crossed(prev, new)
        if result.lateral_miss:
            miss_count += 1
        assert not result.crossed, "Off-axis pass should not count as crossing"

    assert miss_count >= 1, f"Expected at least 1 lateral_miss, got {miss_count}"
    assert gt.current_gate_idx == 0, "Gate should not advance on lateral miss"


# ---- Test 4: Hover near plane — no spurious crossings ----------------------

def test_hover_no_spurious_crossings():
    gate = Gate(center_w=[0, 0, 0], normal_w=[1, 0, 0],
                radius_m=1.0, half_thickness_m=0.2)
    gt = GateTrack([gate], n_laps=1)
    gt.reset(np.array([-0.1, 0, 0]))  # in the slab

    rng = np.random.default_rng(42)
    for _ in range(50):
        prev = np.array([rng.uniform(-0.15, 0.15), 0, 0])
        new = np.array([rng.uniform(-0.15, 0.15), 0, 0])
        result = gt.advance_if_crossed(prev, new)
        assert not result.crossed, "No crossing while hovering in slab"
        assert not result.wrong_dir, "No wrong_dir while hovering in slab"


# ---- Test 5: Lateral miss then re-approach succeeds -------------------------

def test_lateral_miss_then_reapproach():
    gate = Gate(center_w=[0, 0, 0], normal_w=[1, 0, 0],
                radius_m=0.5, half_thickness_m=0.2)
    gt = GateTrack([gate], n_laps=1)
    gt.reset(np.array([-2.0, 3.0, 0]))  # behind gate, far lateral offset

    # Pass through off-axis (lateral miss)
    result = gt.advance_if_crossed(np.array([-1.0, 3.0, 0]),
                                   np.array([1.0, 3.0, 0]))
    assert result.lateral_miss

    # Return behind the gate, on-axis this time
    result = gt.advance_if_crossed(np.array([1.0, 3.0, 0]),
                                   np.array([-1.0, 0.0, 0]))
    assert not result.wrong_dir, "Return from lateral miss should NOT be wrong_dir"

    # Approach on-axis and cross
    result = gt.advance_if_crossed(np.array([-1.0, 0.0, 0]),
                                   np.array([1.0, 0.0, 0]))
    assert result.crossed, "On-axis re-approach should succeed"
    assert gt.done


# ---- Test 6: Multi-gate lap completion --------------------------------------

def test_multi_gate_lap():
    wps = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0]], dtype=np.float64)
    gates = waypoints_to_gates(wps, radius_m=1.0, half_thickness_m=0.2,
                               closed=False)
    gt = GateTrack(gates, n_laps=1)
    gt.reset(np.array([-1.0, 0, 0]))

    for i, cx in enumerate([0, 3, 6]):
        prev = np.array([cx - 1.0, 0, 0])
        new = np.array([cx + 1.0, 0, 0])
        result = gt.advance_if_crossed(prev, new)
        assert result.crossed, f"Gate {i} should be crossed"

    assert gt.done, "Track should be done after all gates"
    assert gt.laps_done == 1


# ---- Test 7: Large dt skip (fast crossing in single step) -------------------

def test_large_dt_skip():
    gate = Gate(center_w=[0, 0, 0], normal_w=[1, 0, 0],
                radius_m=1.0, half_thickness_m=0.2)
    gt = GateTrack([gate], n_laps=1)
    gt.reset(np.array([-10.0, 0, 0]))

    result = gt.advance_if_crossed(np.array([-10.0, 0, 0]),
                                   np.array([10.0, 0, 0]))
    assert result.crossed, "Large jump should still detect crossing"
    assert gt.done


# ---- Test 8: Progress metric sign ------------------------------------------

def test_progress_metric_sign():
    gate = Gate(center_w=[0, 0, 0], normal_w=[1, 0, 0],
                radius_m=1.0, half_thickness_m=0.2)
    gt = GateTrack([gate], n_laps=1)
    gt.reset(np.array([-5.0, 0, 0]))

    d1 = gt.progress_metric(np.array([-3.0, 0, 0]))
    d2 = gt.progress_metric(np.array([-1.0, 0, 0]))
    assert d2 > d1, "Forward motion should increase signed distance"
    assert (d2 - d1) > 0, "Progress (d_new - d_prev) should be positive for forward motion"

    d3 = gt.progress_metric(np.array([-3.0, 0, 0]))
    assert (d3 - d2) < 0, "Backward motion should yield negative progress"
