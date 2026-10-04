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


# --- lookahead heading error ------------------------------------------------
# Added with the feature: the agent can be *told* where the line turns next,
# which is the one thing nine beams in a narrow corridor structurally cannot
# say. Off by default, because every result predating it was measured without.

from rtrrl_playground.envs.track import (HEADING_SCALE, lookahead_features,
                                         lookahead_heading_error)


def test_lookahead_is_off_by_default():
    env = LaneKeep()
    assert env.lookahead is None
    assert env.obs_dim == env.n_beams + int(env.observe_speed)
    assert len(env.reset(seed=0)) == env.obs_dim


@pytest.mark.parametrize("la", [1.0, [0.5, 1.5], [0.5, 1.0, 2.0]])
def test_lookahead_widens_the_observation_by_exactly_its_length(la):
    base, env = LaneKeep(), LaneKeep(lookahead=la)
    n = 1 if np.isscalar(la) else len(la)
    assert env.obs_dim == base.obs_dim + n
    assert len(env.reset(seed=0)) == env.obs_dim


def test_heading_error_is_zero_when_pointed_along_a_straight_line():
    centre = np.stack([np.linspace(0, 50, 400), np.zeros(400)], axis=1)
    assert abs(lookahead_heading_error(centre, 0.0, 0.0, 0.0, 2.0)) < 1e-9
    # turned 30 degrees off the line, the error is that 30 degrees back
    a = lookahead_heading_error(centre, 0.0, 0.0, np.radians(30), 2.0)
    assert np.isclose(np.degrees(a), -30.0, atol=1e-6)


def test_features_are_normalised_and_clipped():
    centre = np.stack([np.linspace(0, 50, 400), np.zeros(400)], axis=1)
    f = lookahead_features(centre, 0.0, 0.0, np.radians(30), [2.0])
    assert np.isclose(f[0], np.radians(-30) / HEADING_SCALE)
    # facing backwards saturates rather than wrapping to a small number
    assert abs(lookahead_features(centre, 0.0, 0.0, np.pi, [2.0])[0]) == 1.0


def test_further_lookaheads_see_more_of_a_curve():
    """The point of more than one distance: the spread is a curvature cue."""
    t = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    circle = np.stack([10 * np.cos(t), 10 * np.sin(t)], axis=1)
    f = lookahead_features(circle, 10.0, 0.0, np.pi / 2, [0.5, 1.0, 2.0])
    assert f[0] < f[1] < f[2]          # the bend only shows up with distance
    assert np.all(f > 0)               # and it is consistently one way


def test_aim_feature_is_pose_free_and_off_by_default():
    """The whole point: it reads the scan, never the pose.

    If this ever started consulting ``env.x``/``y``/``psi`` it would quietly
    acquire a localisation dependency that the deployment story is built on
    not having, and nothing else would notice.
    """
    base, env = LaneKeep(), LaneKeep(aim_feature=True)
    assert base.aim_feature is False
    assert env.obs_dim == base.obs_dim + 1

    env.reset(seed=0)
    a = env._obs()[-1]
    # teleport the car in the map while leaving the scan it sees untouched by
    # restoring it: a map-based feature would move, a scan-based one cannot
    x, y, psi = env.x, env.y, env.psi
    env.x, env.y = x + 500.0, y - 500.0
    moved_scan = env._obs()[-1]           # scan changes -> feature may change
    env.x, env.y, env.psi = x, y, psi
    assert env._obs()[-1] == a            # and returns exactly on restore
    assert np.isfinite(moved_scan)


def test_aim_feature_matches_the_gap_follower_it_is_extracted_from():
    from rtrrl_playground.envs.scripted import free_space_heading
    from rtrrl_playground.envs.track import HEADING_SCALE
    env = LaneKeep(aim_feature=True)
    env.reset(seed=3)
    ranges, _flags = env._last_beams
    expected = np.clip(free_space_heading(ranges, env.beam_angles) / HEADING_SCALE,
                       -1.0, 1.0)
    assert np.isclose(env._obs()[-1], expected)


# --- delta (rate) steering --------------------------------------------------
# Three absolute steering levels contain no angle between 0 and 22.9 deg, so a
# gentle curve can only be held by dithering between the stops. Three *rates*
# can reach and hold any angle. Off by default: every published number is on
# absolute steering.

def test_steer_mode_defaults_to_absolute_and_is_unchanged():
    env = LaneKeep()
    assert env.steer_mode == "absolute"
    env.reset(seed=0)
    for _ in range(20):
        env.step(6)                      # steer +1, throttle 0
    # absolute: full lock is reached and held by the servo lag
    assert env.delta > 0.30


def test_delta_mode_walks_the_servo_and_can_hold_a_middle_angle():
    env = LaneKeep(steer_mode="delta", steer_rate=2.0)
    env.reset(seed=0)
    for _ in range(3):
        env.step(6)                      # three ticks of +rate
    mid = env.delta
    assert 0.0 < mid < 0.40, "should be partway, not at a stop"
    for _ in range(10):
        env.step(3)                      # steer 0 -> hold
    assert np.isclose(env.delta, mid, atol=1e-9), "zero rate must hold the angle"


def test_delta_mode_cannot_exceed_the_lock():
    env = LaneKeep(steer_mode="delta", steer_rate=50.0)
    env.reset(seed=0)
    for _ in range(10):
        env.step(6)
    assert abs(env.delta) <= env.vehicle.steer_max + 1e-12


def test_filter_models_the_same_steering_the_env_does():
    """An absolute-steering filter in front of a rate-steering car certifies a
    plan the car cannot execute, and nothing would raise."""
    env = LaneKeep(steer_mode="delta", steer_rate=2.0)
    safe = make_safe(type("A", (), {"start": lambda s, o: 0,
                                    "step": lambda s, *a: 0})(), env)
    assert safe.filter.model.steer_mode == "delta"
    assert safe.filter._first.steer_mode == "delta"
    assert safe.filter.model.steer_rate == 2.0
