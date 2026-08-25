"""The smallest space/environment protocol that the whole repo runs on.

Deliberately not ``gymnasium``. RTRRL is a *single stream of experience*
algorithm: one environment, batch size one, one update per timestep, no
replay buffer, no vectorised rollout worker. The parts of a modern RL API
that exist to serve batched off-policy training would all be dead weight
here, and a reader trying to follow the algorithm should not have to first
decide which of ``env.step``'s five return values the ``truncated`` flag is.

So: two space classes, one ``Env`` base class, ~80 lines, and
:func:`~rtrrl_playground.envs.gym_adapter.from_gymnasium` for when you do
want to point the agent at a Gymnasium env.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Discrete:
    """``n`` mutually exclusive actions, encoded to the network as a one-hot."""

    n: int

    @property
    def flat_dim(self) -> int:
        return self.n

    def encode(self, a) -> np.ndarray:
        v = np.zeros(self.n, dtype=np.float64)
        if a is not None:
            v[int(a)] = 1.0
        return v

    def sample(self, rng: np.random.Generator):
        return int(rng.integers(self.n))


@dataclass(frozen=True)
class Box:
    """A ``dim``-dimensional continuous action, clipped to ``[low, high]``."""

    dim: int
    low: float = -1.0
    high: float = 1.0

    @property
    def flat_dim(self) -> int:
        return self.dim

    def encode(self, a) -> np.ndarray:
        if a is None:
            return np.zeros(self.dim, dtype=np.float64)
        return np.clip(np.asarray(a, dtype=np.float64), self.low, self.high)

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=self.dim)


class Env:
    """Base class for every environment in this repo.

    Contract::

        obs                                  = env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(action)

    ``terminated`` means the MDP itself ended (the pole fell, the car left
    the track) and bootstrapping must stop; ``truncated`` means only that we
    hit ``max_steps``, and the value of the next state is still a legitimate
    estimate. Getting that distinction wrong silently teaches the agent that
    the world ends after ``max_steps``, which on a *looping* race track is
    exactly the wrong lesson.
    """

    obs_dim: int
    action_space: Discrete | Box
    max_steps: int = 1000
    id: str = "env"

    def reset(self, seed: int | None = None) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def step(self, action):  # pragma: no cover - interface
        raise NotImplementedError

    def render_rollout(self, history, path):  # pragma: no cover - optional
        """Optional: save a picture of one episode. Envs that can, override this."""
        return None
