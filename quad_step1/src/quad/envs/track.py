"""
Waypoint track definitions and progress logic for the QuadRacingEnv.

A track is an ordered list of 3-D waypoints.  The environment advances
to the next waypoint when the quadrotor comes within ``wp_radius`` of the
current one.  One "lap" means visiting every waypoint in order once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Track data class
# ---------------------------------------------------------------------------

@dataclass
class WaypointTrack:
    """Ordered waypoint list with progress tracking.

    Attributes
    ----------
    waypoints : ndarray, shape (N, 3)
        World-frame positions of the waypoints.
    wp_radius : float
        Acceptance radius [m] for considering a waypoint reached.
    n_laps : int
        Number of laps required for success (0 = single pass).
    current_wp : int
        Index of the *next* waypoint to visit.
    laps_done : int
        Number of completed laps so far.
    prev_dist : float
        Distance to the current waypoint at the *previous* env step,
        used for computing the progress reward.
    """

    waypoints: NDArray[np.float64]          # (N, 3)
    wp_radius: float = 0.5
    n_laps: int = 1
    current_wp: int = 0
    laps_done: int = 0
    prev_dist: float = np.inf

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_waypoints(self) -> int:
        return self.waypoints.shape[0]

    @property
    def done(self) -> bool:
        """True when all required laps have been completed."""
        return self.laps_done >= self.n_laps

    @property
    def current_waypoint(self) -> NDArray[np.float64]:
        return self.waypoints[self.current_wp]

    @property
    def next_waypoint(self) -> NDArray[np.float64]:
        """Lookahead: the waypoint *after* the current one (wraps)."""
        idx = (self.current_wp + 1) % self.n_waypoints
        return self.waypoints[idx]

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def reset(self, position: NDArray[np.float64]) -> None:
        """Reset track progress for a new episode."""
        self.current_wp = 0
        self.laps_done = 0
        self.prev_dist = float(np.linalg.norm(self.current_waypoint - position))

    def update(self, position: NDArray[np.float64]) -> tuple[float, bool]:
        """Advance waypoint logic and return progress info.

        Parameters
        ----------
        position : ndarray, shape (3,)
            Current quadrotor position in world frame.

        Returns
        -------
        progress : float
            Signed distance reduction toward the current waypoint since the
            last call (positive = getting closer).
        wp_reached : bool
            Whether a waypoint was reached during this call.
        """
        wp = self.current_waypoint
        dist = float(np.linalg.norm(wp - position))
        progress = self.prev_dist - dist  # positive when closing in

        wp_reached = False
        if dist < self.wp_radius:
            wp_reached = True
            self.current_wp += 1
            if self.current_wp >= self.n_waypoints:
                self.current_wp = 0
                self.laps_done += 1
            # Reset prev_dist to new waypoint
            dist = float(np.linalg.norm(self.current_waypoint - position))

        self.prev_dist = dist
        return progress, wp_reached

    # ------------------------------------------------------------------
    # Copy helper (for env.reset)
    # ------------------------------------------------------------------

    def copy(self) -> WaypointTrack:
        return WaypointTrack(
            waypoints=self.waypoints.copy(),
            wp_radius=self.wp_radius,
            n_laps=self.n_laps,
            current_wp=self.current_wp,
            laps_done=self.laps_done,
            prev_dist=self.prev_dist,
        )


# ---------------------------------------------------------------------------
# Track constructors
# ---------------------------------------------------------------------------

def circle_track(
    radius: float = 3.0,
    z: float = 1.5,
    n_pts: int = 12,
    wp_radius: float = 0.5,
    n_laps: int = 1,
) -> WaypointTrack:
    """Create a circular waypoint track in the XY plane.

    Parameters
    ----------
    radius : float
        Circle radius [m].
    z : float
        Altitude [m].
    n_pts : int
        Number of waypoints around the circle.
    wp_radius : float
        Acceptance radius [m].
    n_laps : int
        Laps required for success.
    """
    angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    waypoints = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        np.full(n_pts, z),
    ])
    return WaypointTrack(
        waypoints=waypoints,
        wp_radius=wp_radius,
        n_laps=n_laps,
    )


def figure8_track(
    a: float = 3.0,
    b: float = 1.5,
    z: float = 1.5,
    n_pts: int = 20,
    wp_radius: float = 0.5,
    n_laps: int = 1,
) -> WaypointTrack:
    """Create a figure-8 (lemniscate of Gerono) waypoint track.

    Parameters
    ----------
    a : float
        X semi-axis [m].
    b : float
        Y semi-axis [m].
    z : float
        Altitude [m].
    n_pts : int
        Number of waypoints.
    wp_radius : float
        Acceptance radius [m].
    n_laps : int
        Laps required for success.
    """
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    waypoints = np.column_stack([
        a * np.sin(t),
        b * np.sin(t) * np.cos(t),
        np.full(n_pts, z),
    ])
    return WaypointTrack(
        waypoints=waypoints,
        wp_radius=wp_radius,
        n_laps=n_laps,
    )


def line_track(
    length: float = 10.0,
    z: float = 1.5,
    n_pts: int = 10,
    wp_radius: float = 0.5,
    n_laps: int = 1,
) -> WaypointTrack:
    """Create a straight-line waypoint track along the +X axis.

    Parameters
    ----------
    length : float
        Total track length [m].
    z : float
        Altitude [m].
    n_pts : int
        Number of waypoints (evenly spaced).
    wp_radius : float
        Acceptance radius [m].
    n_laps : int
        Laps required for success.
    """
    xs = np.linspace(0, length, n_pts)
    waypoints = np.column_stack([
        xs,
        np.zeros(n_pts),
        np.full(n_pts, z),
    ])
    return WaypointTrack(
        waypoints=waypoints,
        wp_radius=wp_radius,
        n_laps=n_laps,
    )
