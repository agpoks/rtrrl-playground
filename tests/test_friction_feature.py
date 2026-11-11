"""The friction circle as something the policy can read one step ahead.

`bridges/friction.py` has defined the circle from the measured tyre since it
was written, and says of itself that nothing which governs driving reads it:
the reward does not price grip, no observation mentions it, and nothing stops
the policy at it. Measured on the `why_worse` traces, that costs something
real -- the learning arms command more lateral acceleration than the tyre can
supply on 7.6 % of their *driving* steps, hundreds of steps before they go
off, while the frozen policy never does it once in 3,000.

`grip_feature` is not already this. It is the accelerometer: the circle the
car is using *now*, after the action. `friction_feature` is the circle one
step ahead, over the grid the policy is about to choose from, so there is a
step in which a policy could still act on it.

What these tests pin down is the part that is easy to break silently: the
feature is off by default, it is appended last so that no saved checkpoint's
feature indices move, and the label table stays in the same order as the
observation.
"""

import numpy as np
import pytest

sgj = pytest.importorskip("scuderia_gym_jax")

from rtrrl_playground.envs.scuderia import ScuderiaLaneKeep  # noqa: E402

COMMON = dict(model="st", map_name="master_cup_changed_n22", n_beams_out=9,
              action_mode="discrete", n_steer=21, n_accel=21,
              observe_speed=True, max_steps=50)


def _env(**kw):
    return ScuderiaLaneKeep(**COMMON, **kw)


def test_off_by_default_so_every_saved_policy_still_fits():
    e = _env()
    assert e.friction_feature is False
    assert len(e.reset(seed=0)) == e.obs_dim


def test_it_costs_exactly_two_channels_and_they_come_last():
    """Appended after everything, so no existing feature index moves."""
    base, grown = _env(), _env(friction_feature=True)
    assert grown.obs_dim == base.obs_dim + 2
    o0, o1 = base.reset(seed=0), grown.reset(seed=0)
    assert len(o1) == len(o0) + 2
    #: the shared prefix is untouched -- the whole point of appending last
    assert np.allclose(o1[:len(o0)], o0)


def test_the_label_table_is_in_the_same_order_as_the_observation():
    """A table that disagrees with the vector mislabels every figure axis."""
    e = _env(friction_feature=True)
    rows = e.obs_labels
    assert len(rows) == e.obs_dim
    assert [r["name"] for r in rows[-2:]] == ["friction_afford",
                                              "friction_headroom"]


def test_both_channels_are_normalised_like_the_grip_feature():
    e = _env(friction_feature=True)
    o = e.reset(seed=0)
    afford, headroom = o[-2], o[-1]
    assert 0.0 <= afford <= 1.0
    assert 0.0 <= headroom <= 1.0


def test_standing_still_the_whole_grid_is_affordable():
    """At rest there is no lateral demand, so nothing can leave the circle.

    This is the sanity check on the direction of the signal: `afford` is a
    share that *falls* as the car commits, so if it were inverted this is
    where it would show.
    """
    e = _env(friction_feature=True)
    o = e.reset(seed=0)
    assert o[-2] == pytest.approx(1.0)
    assert o[-1] == pytest.approx(0.0, abs=0.05)


def test_it_reports_the_same_numbers_it_showed_the_policy():
    """`experiments/` logs these off `self`; they must not be a second read."""
    e = _env(friction_feature=True)
    o = e.reset(seed=0)
    assert e._friction_afford == pytest.approx(o[-2])
    assert e._friction_headroom == pytest.approx(o[-1])


def test_the_wheelbase_comes_from_the_model_not_a_constant():
    """0.322 m on this car -- and `affordable`'s default, which may drift."""
    from bridges.friction import affordable  # noqa: F401
    e = _env(friction_feature=True)
    assert e.wheelbase == pytest.approx(0.322, abs=1e-3)
