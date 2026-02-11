"""
Gate definitions for gate-plane racing.

A **gate** is a circular aperture in 3-D space defined by:

* ``center_w`` – centre position in world frame.
* ``normal_w`` – unit-length normal vector **pointing in the direction of
  travel**.  The drone approaches from the *negative* side
  (``d = dot(normal, p - center) < 0``) and crosses to the *positive* side
  (``d > 0``).
* ``radius_m``  – radius of the circular opening.
* ``half_thickness_m`` – half-thickness of a slab region around the plane.
  Crossings are only registered when the signed distance transitions from
  below ``-half_thickness`` to above ``+half_thickness`` (or vice-versa),
  which avoids spurious triggers from numerical noise.

Sign convention (used everywhere)
---------------------------------
``d(p) = dot(normal_w, p - center_w)``

* d < 0 → drone is *behind* the gate (approaching).
* d > 0 → drone has *passed through* the gate.
* Correct crossing: ``d_prev < -half_thickness`` **and** ``d_new > +half_thickness``.
* Wrong-direction crossing: ``d_prev > +half_thickness`` **and**
  ``d_new < -half_thickness``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _safe_normalize(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return unit vector; fall back to +X if *v* is (near-)zero."""
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return (v / n).astype(np.float64)


# ---------------------------------------------------------------------------
# Gate dataclass
# ---------------------------------------------------------------------------

@dataclass
class Gate:
    """A single gate plane in 3-D space.

    Attributes
    ----------
    center_w : ndarray, shape (3,)
        Gate centre in world frame [m].
    normal_w : ndarray, shape (3,)
        Unit normal pointing in the **direction of travel**.
    radius_m : float
        Radius of the circular opening [m].
    half_thickness_m : float
        Half-width of the slab used for crossing detection [m].
    name : str or None
        Optional human-readable label.
    """

    center_w: NDArray[np.float64]
    normal_w: NDArray[np.float64]
    radius_m: float = 0.5
    half_thickness_m: float = 0.2
    name: Optional[str] = None

    def __post_init__(self) -> None:
        self.center_w = np.asarray(self.center_w, dtype=np.float64)
        self.normal_w = np.asarray(self.normal_w, dtype=np.float64)
        # Ensure unit length
        self.normal_w = _safe_normalize(self.normal_w)

    def signed_distance(self, p: NDArray[np.float64]) -> float:
        """Signed distance from *p* to this gate's plane.

        Positive → past the gate (in the direction of normal).
        Negative → approaching the gate.
        """
        return float(np.dot(self.normal_w, np.asarray(p) - self.center_w))

    def copy(self) -> Gate:
        return Gate(
            center_w=self.center_w.copy(),
            normal_w=self.normal_w.copy(),
            radius_m=self.radius_m,
            half_thickness_m=self.half_thickness_m,
            name=self.name,
        )


# ---------------------------------------------------------------------------
# Helper constructors
# ---------------------------------------------------------------------------

def gate_from_waypoint(
    prev: Optional[NDArray[np.float64]],
    curr: NDArray[np.float64],
    nxt: Optional[NDArray[np.float64]],
    radius_m: float = 0.5,
    half_thickness_m: float = 0.2,
    name: Optional[str] = None,
) -> Gate:
    """Create a gate from a sequence of waypoints.

    The gate plane is placed at *curr* with its normal along the path
    tangent estimated from the neighbouring waypoints:

    * Both *prev* and *nxt* given → ``tangent = normalize(nxt - prev)``
    * Only *prev* → ``tangent = normalize(curr - prev)``
    * Only *nxt*  → ``tangent = normalize(nxt - curr)``
    * Neither     → ``tangent = +X``

    Parameters
    ----------
    prev, curr, nxt : ndarray or None
        Previous, current, and next waypoint positions.
    radius_m : float
        Gate opening radius [m].
    half_thickness_m : float
        Slab half-thickness for crossing detection [m].
    name : str or None
        Optional label.
    """
    curr = np.asarray(curr, dtype=np.float64)

    if prev is not None and nxt is not None:
        tangent = np.asarray(nxt, dtype=np.float64) - np.asarray(prev, dtype=np.float64)
    elif prev is not None:
        tangent = curr - np.asarray(prev, dtype=np.float64)
    elif nxt is not None:
        tangent = np.asarray(nxt, dtype=np.float64) - curr
    else:
        tangent = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    normal = _safe_normalize(tangent)

    return Gate(
        center_w=curr.copy(),
        normal_w=normal,
        radius_m=radius_m,
        half_thickness_m=half_thickness_m,
        name=name,
    )


def waypoints_to_gates(
    waypoints: NDArray[np.float64],
    radius_m: float = 0.5,
    half_thickness_m: float = 0.2,
    closed: bool = True,
) -> List[Gate]:
    """Convert an (N, 3) waypoint array into a list of :class:`Gate` objects.

    Parameters
    ----------
    waypoints : ndarray, shape (N, 3)
        Ordered waypoint positions.
    radius_m : float
        Gate opening radius [m].
    half_thickness_m : float
        Slab half-thickness [m].
    closed : bool
        If *True* the track wraps (last waypoint connects back to first).
        If *False* the first/last gates use one-sided tangent estimates.

    Returns
    -------
    list[Gate]
        One gate per waypoint, normals aligned with the local path tangent.
    """
    wps = np.asarray(waypoints, dtype=np.float64)
    n = len(wps)
    gates: List[Gate] = []

    for i in range(n):
        if closed:
            prev = wps[(i - 1) % n]
            nxt = wps[(i + 1) % n]
        else:
            prev = wps[i - 1] if i > 0 else None
            nxt = wps[i + 1] if i < n - 1 else None
        gate = gate_from_waypoint(
            prev=prev,
            curr=wps[i],
            nxt=nxt,
            radius_m=radius_m,
            half_thickness_m=half_thickness_m,
            name=f"gate_{i}",
        )
        gates.append(gate)

    return gates
