"""
Gate-plane track with crossing detection and progress tracking.

:class:`GateTrack` wraps a list of :class:`~quad.envs.gates.Gate` objects and
provides a state machine that:

* tracks which gate the drone should cross next,
* detects correct and wrong-direction crossings using a **hysteresis** scheme,
* computes a signed-distance progress metric for smooth reward shaping.

Sign convention  (see ``gates.py`` for full discussion)
-------------------------------------------------------
``d(p) = dot(normal_w, p - center_w)``

* d < 0 → behind the gate (approaching).
* d > 0 → past the gate.

Hysteresis crossing detection
-----------------------------
A naive ``sign(d)`` check fails when the drone crosses slowly (a few cm
per env step) or hovers near the plane.  Instead we use a **slab**
defined by ``half_thickness_m``:

* The drone is marked *approaching* once ``d < -half_thickness`` at any
  point since the last gate advance (or reset).
* A correct crossing is registered when *approaching* **and**
  ``d > +half_thickness``, confirming the drone is solidly past the plane.
* At that instant the lateral offset (in-plane distance from gate centre)
  is checked against ``radius_m``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from quad.envs.gates import Gate


# ---------------------------------------------------------------------------
# Crossing result tuple
# ---------------------------------------------------------------------------

class CrossingResult(NamedTuple):
    """Outcome of a crossing check for one env step."""
    crossed: bool         # gate was correctly crossed
    wrong_dir: bool       # gate was crossed in the wrong direction
    lateral_miss: bool    # plane was crossed but outside the gate radius


# ---------------------------------------------------------------------------
# GateTrack
# ---------------------------------------------------------------------------

class GateTrack:
    """Ordered sequence of gates with lap counting and crossing detection.

    Parameters
    ----------
    gates : list[Gate]
        Ordered gate list.  Normals must point in the direction of travel.
    n_laps : int
        Number of complete passes through all gates required for success.
    """

    def __init__(
        self,
        gates: Sequence[Gate],
        n_laps: int = 1,
    ) -> None:
        if len(gates) == 0:
            raise ValueError("GateTrack requires at least one gate")
        self.gates: List[Gate] = list(gates)
        self.n_laps: int = n_laps

        # Mutable state (reset in `reset()`)
        self.current_gate_idx: int = 0
        self.laps_done: int = 0
        # Hysteresis flags
        self._was_behind: bool = False   # d was < -ht at some point
        self._was_ahead: bool = False    # d was > +ht at some point (wrong-dir)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_gates(self) -> int:
        return len(self.gates)

    @property
    def done(self) -> bool:
        """True when the required number of laps has been completed."""
        return self.laps_done >= self.n_laps

    # ------------------------------------------------------------------
    # Gate accessors
    # ------------------------------------------------------------------

    def current_gate(self) -> Gate:
        """The gate the drone must cross next."""
        return self.gates[self.current_gate_idx]

    def next_gate(self) -> Gate:
        """Look-ahead: the gate *after* the current one (wraps)."""
        idx = (self.current_gate_idx + 1) % self.n_gates
        return self.gates[idx]

    # ------------------------------------------------------------------
    # Signed-distance helpers
    # ------------------------------------------------------------------

    def signed_distance(self, pos: NDArray[np.float64]) -> float:
        """Signed distance from *pos* to the **current** gate's plane.

        See module docstring for sign convention.
        """
        return self.current_gate().signed_distance(pos)

    def progress_metric(self, pos: NDArray[np.float64]) -> float:
        """Convenience alias for :meth:`signed_distance`."""
        return self.signed_distance(pos)

    # ------------------------------------------------------------------
    # Crossing detection (hysteresis scheme)
    # ------------------------------------------------------------------

    def advance_if_crossed(
        self,
        prev_pos: NDArray[np.float64],
        new_pos: NDArray[np.float64],
    ) -> CrossingResult:
        """Check whether the drone has crossed the current gate.

        Uses hysteresis: the drone must have been solidly *behind* the
        gate (``d < -half_thickness``) at some earlier point, and now be
        solidly *past* the gate (``d > +half_thickness``), to count as a
        correct crossing.

        If a correct crossing is detected the gate index is advanced (and
        ``laps_done`` incremented when wrapping).

        Parameters
        ----------
        prev_pos, new_pos : ndarray, shape (3,)
            Drone position at the start and end of the env step.

        Returns
        -------
        CrossingResult
            ``(crossed, wrong_dir, lateral_miss)`` flags.
        """
        gate = self.current_gate()
        d_prev = gate.signed_distance(prev_pos)
        d_new = gate.signed_distance(new_pos)
        ht = gate.half_thickness_m

        # Snapshot before update (needed for wrong-direction guard)
        was_behind_pre = self._was_behind

        # --- Update hysteresis flags ---
        if d_prev < -ht or d_new < -ht:
            self._was_behind = True
        if d_prev > ht or d_new > ht:
            self._was_ahead = True

        crossed = False
        wrong_dir = False
        lateral_miss = False

        # ------------------------------------------------------------------
        # Correct crossing: was behind at some point → now solidly past
        # ------------------------------------------------------------------
        if self._was_behind and d_new > ht:
            # Lateral check: project current position onto the gate plane
            # and measure distance from gate centre.
            p_proj = np.asarray(new_pos, dtype=np.float64) - d_new * gate.normal_w
            lateral = float(np.linalg.norm(p_proj - gate.center_w))

            if lateral <= gate.radius_m:
                crossed = True
                # Full reset — successful crossing completes the approach
                self._was_behind = False
                self._was_ahead = False
            else:
                lateral_miss = True
                # Keep _was_behind so re-approach works; clear _was_ahead
                self._was_ahead = False

        # ------------------------------------------------------------------
        # Wrong-direction crossing: was solidly past → now behind
        # ------------------------------------------------------------------
        elif self._was_ahead and d_new < -ht and not was_behind_pre:
            wrong_dir = True
            # Reset
            self._was_ahead = False
            self._was_behind = True  # now behind again

        # --- Advance on successful crossing ---
        if crossed:
            self.current_gate_idx += 1
            if self.current_gate_idx >= self.n_gates:
                self.current_gate_idx = 0
                self.laps_done += 1
            # Initialise hysteresis for the new gate
            d_new_gate = self.signed_distance(new_pos)
            self._was_behind = d_new_gate < -self.current_gate().half_thickness_m
            self._was_ahead = d_new_gate > self.current_gate().half_thickness_m

        return CrossingResult(crossed=crossed, wrong_dir=wrong_dir, lateral_miss=lateral_miss)

    # ------------------------------------------------------------------
    # Reset / copy
    # ------------------------------------------------------------------

    def reset(self, position: Optional[NDArray[np.float64]] = None) -> None:
        """Reset progress for a new episode.

        Parameters
        ----------
        position : ndarray or None
            Starting position.  Used to initialise hysteresis flags.
        """
        self.current_gate_idx = 0
        self.laps_done = 0
        if position is not None:
            d = self.signed_distance(position)
            ht = self.current_gate().half_thickness_m
            self._was_behind = d < -ht
            self._was_ahead = d > ht
        else:
            self._was_behind = False
            self._was_ahead = False

    def copy(self) -> GateTrack:
        """Return an independent copy (gates are deep-copied)."""
        gt = GateTrack(
            gates=[g.copy() for g in self.gates],
            n_laps=self.n_laps,
        )
        gt.current_gate_idx = self.current_gate_idx
        gt.laps_done = self.laps_done
        gt._was_behind = self._was_behind
        gt._was_ahead = self._was_ahead
        return gt
