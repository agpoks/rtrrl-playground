"""The steering axis in delta mode: seeing the wheel, and paying to move it.

`steer_mode="delta"` makes the action a steering *rate*: `_decode` keeps a
setpoint and adds `action * steer_rate * dt` to it. The setpoint is a state
of the control loop, and until `steer_feature` existed it was a *hidden*
one -- nothing in the observation said where the wheel was, so "hold the
angle I am at" was not something the policy could aim for, because it did not
know what that angle was. Measured on the straights, where the error signal
is smallest, the wheel crossed centre 2.3 times a second.

The feature is the *commanded* angle, not `x[0, 2]`: the command is what the
action adds to, and the only one of the two the car knows without a sensor.

`_obs_from` is exercised on a bare instance -- the real constructor starts a
simulator and loads a map, which none of this needs.
"""

import numpy as np
import pytest

from rtrrl_playground.envs.scuderia import ScuderiaLaneKeep, STEER_MAX


def _bare(steer_feature, steer_mode="delta", steer_rate=6.0, dt=0.05):
    env = object.__new__(ScuderiaLaneKeep)
    env._idx = np.array([0, 1, 2])
    env.observe_speed = False
    env.aim_feature = False
    env.drive_features = False
    env.proprio = False
    env.grip_feature = False
    env.lookahead = None
    env.steer_feature = steer_feature
    env.steer_mode = steer_mode
    env.steer_rate = steer_rate
    env.steer_filter = 0.0
    env.dt = dt
    env._steer_cmd = 0.0
    env._steer_out = 0.0
    env._steer_flip = False
    env.action_mode = "continuous"
    env.accel_mode = "accl"
    env.accel_cmd = 10.0
    env.accel_rate = None
    env._accel_cmd = 0.0
    env._accel_jerk = 0.0
    env._v_cmd = 0.0
    env._x = None
    return env


_X = np.zeros((1, 7))
_SCANS = np.full((1, 8), 10.0)


def _obs(env):
    return env._obs_from(_X, _SCANS)


def test_the_feature_is_one_number_appended_at_the_end():
    off, on = _bare(False), _bare(True)
    a, b = _obs(off), _obs(on)
    assert len(b) == len(a) + 1
    # last, so that turning it on does not move any existing feature
    assert np.allclose(b[:-1], a)


def test_it_reads_the_commanded_angle_as_a_fraction_of_lock():
    env = _bare(True)
    for level in (0.0, 0.5, -1.0, 1.0):
        env._steer_cmd = level * STEER_MAX
        assert _obs(env)[-1] == pytest.approx(level)


def test_it_stays_inside_plus_minus_one_at_the_stops():
    env = _bare(True)
    for _ in range(50):                      # drive it hard into the stop
        env._decode(np.array([1.0, 0.0]))
    assert env._steer_cmd == pytest.approx(STEER_MAX)
    assert _obs(env)[-1] == pytest.approx(1.0)


def test_it_tracks_the_integrator_the_action_drives():
    env = _bare(True)
    seen = []
    for _ in range(4):
        env._decode(np.array([1.0, 0.0]))    # +6 rad/s for 50 ms = +0.3 rad
        seen.append(_obs(env)[-1])
    assert seen == pytest.approx([0.75, 1.0, 1.0, 1.0])


def test_the_previous_action_alone_cannot_say_where_the_wheel_is():
    """Why the cell's meta inputs do not already cover this.

    Two histories ending in the *same* action leave the wheel in opposite
    places, so a policy given only its last action -- which is what the
    recurrent cell receives -- cannot tell them apart without integrating,
    and a leaky spiking membrane is a poor integrator.
    """
    left, right = _bare(True), _bare(True)
    for _ in range(3):
        left._decode(np.array([-1.0, 0.0]))
    for _ in range(3):
        right._decode(np.array([+1.0, 0.0]))
    left._decode(np.array([0.0, 0.0]))       # same last action
    right._decode(np.array([0.0, 0.0]))
    assert _obs(left)[-1] == pytest.approx(-1.0)
    assert _obs(right)[-1] == pytest.approx(+1.0)


def test_in_absolute_mode_it_is_just_the_action():
    """Documented as adding nothing there, and it must stay that way."""
    env = _bare(True, steer_mode="absolute")
    for level in (-1.0, -0.25, 0.6):
        env._decode(np.array([level, 0.0]))
        assert _obs(env)[-1] == pytest.approx(level)


def test_it_is_off_unless_asked_for():
    """Every committed result was measured without it."""
    env = _bare(False)
    assert env.steer_feature is False
    assert len(_obs(env)) == len(env._idx)


# -- the wheel-travel price -------------------------------------------------
#
# `steer_move_penalty` charges `|action|` per tick. In delta mode the action
# is the rate, so the charge is proportional to how far the wheel travels --
# holding an angle is free and reaching one costs in proportion to the
# distance. The two costs that came before it do not have that shape:
# `reverse_penalty` fires only once the wheel has already crossed centre, and
# CAPS charges for the rate *changing*, so a constant full-rate command --
# the one that drives the wheel into the stop and holds it there -- is free
# under both of them.

def _reward_env(**kw):
    env = _bare(False)
    env.grip_penalty = 0.0
    env.grip_target = 0.8
    env.reverse_penalty = 0.0
    env.accel_jerk_penalty = 0.0
    env.steer_move_penalty = 0.0
    for k, v in kw.items():
        setattr(env, k, v)
    return env


def _charge(env, level):
    env._decode(np.array([level, 0.0]))
    return -env.steer_move_penalty * env._steer_move


def test_holding_the_wheel_is_free_and_moving_it_is_not():
    env = _reward_env(steer_move_penalty=0.1)
    assert _charge(env, 0.0) == 0.0
    assert _charge(env, 1.0) == pytest.approx(-0.1)
    assert _charge(env, -1.0) == pytest.approx(-0.1)
    assert _charge(env, 0.3) == pytest.approx(-0.03)


def test_the_price_is_proportional_to_the_distance_travelled():
    """Two ways to reach the same angle cost the same."""
    fast, slow = _reward_env(steer_move_penalty=0.1), _reward_env(steer_move_penalty=0.1)
    quick = -sum(_charge(fast, 1.0) for _ in range(2))      # 2 ticks at full rate
    easy = -sum(_charge(slow, 0.5) for _ in range(4))       # 4 at half
    assert fast._steer_cmd == pytest.approx(slow._steer_cmd)
    assert quick == pytest.approx(easy)


def test_a_constant_full_rate_command_is_charged_every_tick():
    """What CAPS does not charge for: the command is unchanging, the wheel
    is not."""
    env = _reward_env(steer_move_penalty=0.1)
    assert sum(_charge(env, 1.0) for _ in range(10)) == pytest.approx(-1.0)


def test_it_is_off_by_default():
    assert _reward_env().steer_move_penalty == 0.0
    env = _reward_env()
    assert _charge(env, 1.0) == 0.0


# -- the mode is not the centre --------------------------------------------

def _head(levels=21):
    from rtrrl_playground.nets.heads import FactoredCategoricalHead
    return FactoredCategoricalHead(4, (levels, levels), rng=np.random.default_rng(0))


def test_the_expected_level_is_zero_when_the_policy_is_symmetric():
    """The case that matters: indifferent between a little left and a little
    right, so the wheel should be held, and the argmax picks a side."""
    h = _head()
    sl = h.slices[0]
    z = np.zeros(h.n_act)
    # a symmetric pair either side of centre, with the right one a hair
    # higher -- the mode is +1 level, the expectation is still 0
    z[sl.start + 9] = 3.0
    z[sl.start + 11] = 3.0001
    h.theta = np.zeros((h.n_act, h.n))
    h.bias = z
    hid = np.zeros(h.n)
    assert h.unflat(h.greedy(hid))[0] == 11
    assert h.unflat(h.mean(hid))[0] == 10          # the hold level


def test_the_expected_level_follows_a_one_sided_policy():
    """It must not just always say zero."""
    h = _head()
    sl = h.slices[0]
    h.theta = np.zeros((h.n_act, h.n))
    h.bias = np.zeros(h.n_act)
    h.bias[sl.start + 17] = 8.0
    assert h.unflat(h.mean(np.zeros(h.n)))[0] == 17


def test_the_two_axes_are_averaged_independently():
    h = _head()
    h.theta = np.zeros((h.n_act, h.n))
    h.bias = np.zeros(h.n_act)
    h.bias[h.slices[0].start + 4] = 8.0
    h.bias[h.slices[1].start + 16] = 8.0
    assert list(h.unflat(h.mean(np.zeros(h.n)))) == [4, 16]


# -- the low-pass on the angle handed to the servo ---------------------------
#
# Eleven interventions on the policy left the wheel crossing centre 7-9
# times a second on the straights: three prices, a steering-rate ladder, the
# entropy coefficient, sticky actions, three read-outs of the distribution,
# an ordinal head and a filter on the state the actor reads. `steer_filter`
# stops asking the policy and puts a lag in the driver instead. It is not a
# rate limit -- `steer_rate` is one of those, and at 2 and 3 rad/s it
# stopped the car cornering without changing how often the wheel reversed.
# A lag discriminates by frequency, which is the whole point.

def _filt_env(tau, rate=6.0):
    env = _bare(False, steer_rate=rate)
    env.steer_filter = tau
    return env


def _drive(env, levels):
    return np.array([env._decode(np.array([u, 0.0]))[0] for u in levels])


def test_a_slow_demand_passes_and_a_fast_one_does_not():
    """The property that separates a lag from a rate limit."""
    n = 60
    slow = [0.15] * n                       # a steady walk to an angle
    fast = [1.0 if i % 2 else -1.0 for i in range(n)]   # a tick-by-tick dither
    for levels, keep in ((slow, 0.75), (fast, 0.45)):
        raw = _drive(_filt_env(0.0), levels)
        lp = _drive(_filt_env(3.0), levels)
        span_raw = raw.max() - raw.min()
        span_lp = lp.max() - lp.min()
        if keep > 0.5:
            assert span_lp > keep * span_raw        # the corner survives
        else:
            assert span_lp < keep * span_raw        # the dither does not


def test_it_is_off_by_default_and_then_the_servo_gets_the_setpoint():
    env = _bare(False)
    assert env.steer_filter == 0.0
    out = _drive(env, [1.0] * 3)
    assert out[-1] == pytest.approx(env._steer_cmd)


def test_the_wheel_feature_reports_what_the_servo_was_given():
    """Or the policy is told it is holding an angle the wheel never got."""
    env = _bare(True, steer_rate=6.0)
    env.steer_filter = 3.0
    for _ in range(5):
        env._decode(np.array([1.0, 0.0]))
    assert _obs(env)[-1] == pytest.approx(env._steer_out / STEER_MAX)
    assert env._steer_out < env._steer_cmd          # and they differ
