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


#: Default steering and throttle resolution. Three of each is what every
#: result in this repo before the action-resolution sweep was measured at, so
#: it stays the default -- but it was never a *choice*. The only rationale ever
#: written down (``envs/lanekeep.py``) argues for a flat categorical over two
#: heads, which is an argument about the gradient and holds equally at any
#: resolution; nothing anywhere justified three levels.
#:
#: Three is not free. With steering restricted to {-1, 0, +1} x 22.9 deg there
#: is no intermediate angle in the action set at all, so holding a gentle curve
#: requires dithering between the lock stops: measured on a trained policy, the
#: steering sign reverses every 3 steps (0.15 s) and the car oscillates +/-0.5 m
#: about the centreline. That weave is 67 % of ``curvy``'s half-width and 93 %
#: of master_cup's -- which is why the same policy laps one and cannot stay on
#: the other.
N_STEER, N_THROTTLE = 3, 3


def action_levels(n: int) -> np.ndarray:
    """``n`` evenly spaced commands spanning ``[-1, 1]`` (just ``0`` if n == 1)."""
    if n < 1:
        raise ValueError("need at least one level")
    if n == 1:
        return np.zeros(1)
    return np.linspace(-1.0, 1.0, n)


def action_grid(n_steer: int = N_STEER, n_throttle: int = N_THROTTLE) -> np.ndarray:
    """``(n_steer * n_throttle, 2)`` of ``(steer, throttle)`` in ``[-1, 1]``.

    Row ``a`` is the action the flat categorical head calls ``a``, so this is
    the single definition of what a discrete action *means*. It used to be
    written out four times -- twice in environments and twice in filters, as
    ``(a // 3 - 1, a % 3 - 1)`` -- and the two that matter most had to agree or
    the safety filter would certify one action and the environment execute
    another, silently. At the default resolution this reproduces that formula
    exactly, so no measured result moves.
    """
    st = action_levels(n_steer)
    th = action_levels(n_throttle)
    return np.array([[st[a // n_throttle], th[a % n_throttle]]
                     for a in range(n_steer * n_throttle)], dtype=np.float64)
