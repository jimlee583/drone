"""
Lightweight Gymnasium wrappers for QuadRacingEnv.

These are optional and do **not** modify the base environment.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Clip action (safety net – the env already clips internally)
# ---------------------------------------------------------------------------

class ClipActionWrapper(gym.ActionWrapper):
    """Clip actions to the environment's action space bounds.

    ``QuadRacingEnv.step`` already clips internally, so this wrapper is
    purely a safety net for policies that might occasionally exceed [-1, 1].
    """

    def action(self, action: NDArray) -> NDArray:
        return np.clip(action, self.action_space.low, self.action_space.high)


# ---------------------------------------------------------------------------
# Running-mean observation normalisation (OFF by default)
# ---------------------------------------------------------------------------

class NormalizeObservation(gym.ObservationWrapper):
    """On-line running-mean / running-std observation normalisation.

    Disabled by default.  Enable via ``NormalizeObservation(env, enabled=True)``.

    The wrapper keeps an exponential moving average of the observation mean
    and variance and returns ``(obs - mean) / (std + eps)``.
    """

    def __init__(
        self,
        env: gym.Env,
        enabled: bool = False,
        clip: float = 10.0,
        epsilon: float = 1e-8,
        alpha: float = 0.01,
    ):
        super().__init__(env)
        self.enabled = enabled
        self.clip = clip
        self.epsilon = epsilon
        self.alpha = alpha

        obs_shape = self.observation_space.shape
        self._mean = np.zeros(obs_shape, dtype=np.float64)
        self._var = np.ones(obs_shape, dtype=np.float64)
        self._count = 0

    def observation(self, obs: NDArray) -> NDArray:
        if not self.enabled:
            return obs

        obs64 = obs.astype(np.float64)
        self._count += 1

        # Exponential moving average
        self._mean = (1.0 - self.alpha) * self._mean + self.alpha * obs64
        self._var = (1.0 - self.alpha) * self._var + self.alpha * (obs64 - self._mean) ** 2

        std = np.sqrt(self._var + self.epsilon)
        normed = (obs64 - self._mean) / std
        normed = np.clip(normed, -self.clip, self.clip)
        return normed.astype(np.float32)


# ---------------------------------------------------------------------------
# Helper: wrap env with optional layers
# ---------------------------------------------------------------------------

def wrap_env(
    env: gym.Env,
    clip_actions: bool = True,
    normalize_obs: bool = False,
) -> gym.Env:
    """Apply optional wrappers to a QuadRacingEnv instance.

    Parameters
    ----------
    env : gym.Env
        Base environment (should be ``QuadRacingEnv``).
    clip_actions : bool
        Wrap with :class:`ClipActionWrapper`.
    normalize_obs : bool
        Wrap with :class:`NormalizeObservation` (running mean/std).

    Returns
    -------
    gym.Env
        Wrapped environment.
    """
    if clip_actions:
        env = ClipActionWrapper(env)
    if normalize_obs:
        env = NormalizeObservation(env, enabled=True)
    return env
