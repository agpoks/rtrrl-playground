"""The environment and the filters must agree on what a discrete action means.

This is a correctness test, not a style one. The grid used to be written out
four times -- twice in environments, twice in filters -- as ``(a // 3 - 1,
a % 3 - 1)``. At the default 3x3 the four copies agreed by coincidence of
arithmetic; at any other resolution they do not, and the failure is silent:
the filter certifies that action ``k`` is safe, the environment executes a
different steering angle for the same ``k``, and the certificate is void with
nothing raised anywhere.
"""

import numpy as np
import pytest

from rtrrl_playground.envs.lanekeep import LaneKeep
from rtrrl_playground.safety import PredictiveSafetyFilter, make_safe
from rtrrl_playground.cbf import DiscreteCBFFilter
from rtrrl_playground.spaces import action_grid


def test_default_grid_is_the_historical_formula():
    """Every published number was measured on ``(a // 3 - 1, a % 3 - 1)``."""
    assert np.array_equal(
        action_grid(),
        np.array([[a // 3 - 1, a % 3 - 1] for a in range(9)], dtype=float))


@pytest.mark.parametrize("n_steer,n_throttle", [(3, 3), (5, 3), (7, 3), (3, 5)])
def test_env_decode_matches_its_own_grid(n_steer, n_throttle):
    env = LaneKeep(n_steer=n_steer, n_throttle=n_throttle)
    assert env.action_space.n == n_steer * n_throttle
    for a in range(env.action_space.n):
        assert env._decode(a) == tuple(env.action_grid[a])


@pytest.mark.parametrize("n_steer", [3, 5, 7])
def test_filters_certify_the_action_the_env_executes(n_steer):
    env = LaneKeep(n_steer=n_steer)
    for filt in (PredictiveSafetyFilter(env.track, dt=env.dt,
                                        action_grid=env.action_grid),
                 DiscreteCBFFilter(env.track, dt=env.dt,
                                   action_grid=env.action_grid)):
        assert filt.n_actions == env.action_space.n
        assert np.array_equal(filt._grid, env.action_grid)


def test_make_safe_plumbs_the_grid_without_being_asked():
    """A caller must not have to remember; forgetting is the dangerous case."""
    class _Agent:
        def start(self, obs): return 0
        def step(self, *a): return 0
    env = LaneKeep(n_steer=7)
    safe = make_safe(_Agent(), env)
    assert np.array_equal(safe.filter._grid, env.action_grid)
    assert safe.filter.n_actions == 21
