"""A jerk limit on the steering rate, and a steering cost gated by the lidar.

Two knobs, both added because the steering axis had nothing the acceleration
axis has. `accel_rate` is a jerk limit in m/s^3 on the acceleration command
and is the only smoothing in this project that never cost distance; the
steering axis had `steer_rate`, which caps how fast the *setpoint* moves,
and nothing at all capping how fast that rate may *change*. The recorded
traces show the policy asking for +6 rad/s and then -6 rad/s on consecutive
50 ms ticks.

`corridor_steer_penalty` is the conditional version of a price that has now
failed three times unconditionally (reversal, CAPS, wheel travel). It
charges for moving the wheel in proportion to how long the corridor ahead
is, so a hairpin -- where the car genuinely needs the whole lock -- is free
and a straight is not. The gate is the minimum of a +/-6 deg lidar cone,
chosen over the +/-12 deg cone `drive_features` already publishes because
measured over 6,000 steps of master_cup it separates straights from corners
with AUC 0.842 against 0.745.

`_decode` and `_obs_from` are exercised on a bare instance: the real
constructor starts a simulator and loads a map, and none of this needs one.
"""

import numpy as np
import pytest

from rtrrl_playground.envs.scuderia import ScuderiaLaneKeep, STEER_MAX

#: a scan that is clear to 10 m everywhere, in the shape `_decode` reads
_ANG = np.linspace(-np.pi, np.pi, 181)


def _bare(steer_jerk_rate=None, corridor_steer_penalty=0.0,
          corridor_feature=False, corridor_lo=3.0, corridor_hi=6.0,
          steer_mode="delta", steer_rate=6.0, dt=0.05, clear=10.0):
    env = object.__new__(ScuderiaLaneKeep)
    env.sensors = ""          # the car's own sensors, off
    env._idx = np.array([0, 1, 2])
    env.observe_speed = env.aim_feature = env.drive_features = False
    env.proprio = env.grip_feature = env.steer_feature = False
    env.lookahead = None
    env.steer_mode, env.steer_rate = steer_mode, steer_rate
    env.steer_filter = 0.0
    env.dt = dt
    env._steer_cmd = env._steer_out = 0.0
    env._steer_flip = False
    env.action_mode = "continuous"
    env.accel_mode = "accl"
    env.accel_cmd = 10.0
    env.accel_rate = None
    env._accel_cmd = env._accel_jerk = env._v_cmd = 0.0
    env._x = None
    env.steer_jerk_rate = steer_jerk_rate
    env._steer_rate_cmd = env._steer_jerk = 0.0
    env.corridor_steer_penalty = corridor_steer_penalty
    env.corridor_lo, env.corridor_hi = corridor_lo, corridor_hi
    env.corridor_feature = corridor_feature
    env._corridor_m, env._corridor_gate = corridor_hi, 0.0
    env.scan_angles = _ANG
    env._scans = np.full((1, len(_ANG)), float(clear))
    return env


def _rate(env, level):
    """One decode at steering `level` in [-1, 1]; returns the rate delivered."""
    env._decode(np.array([level, 0.0]))
    return env._steer_rate_cmd


# --------------------------------------------------------- the jerk limit
def test_without_a_limit_the_rate_can_reverse_in_one_tick():
    """The thing being fixed: this is what the traces show."""
    env = _bare(steer_jerk_rate=None)
    assert _rate(env, +1.0) == pytest.approx(+6.0)
    assert _rate(env, -1.0) == pytest.approx(-6.0)   # 240 rad/s^2 in one tick


@pytest.mark.parametrize("limit", [10.0, 30.0, 60.0])
def test_the_rate_never_changes_faster_than_the_limit(limit):
    env = _bare(steer_jerk_rate=limit)
    prev, worst = 0.0, 0.0
    rng = np.random.default_rng(0)
    for level in rng.uniform(-1.0, 1.0, 400):
        u = _rate(env, float(level))
        worst = max(worst, abs(u - prev) / env.dt)
        prev = u
    assert worst <= limit + 1e-9


def test_the_limit_does_not_cap_the_rate_itself():
    """Given time, it must still reach full `steer_rate` -- it is a jerk
    limit, not a second rate limit. Rate 3 and rate 2 were tried and both
    stopped the car cornering; that must not come back in disguise."""
    env = _bare(steer_jerk_rate=30.0)
    for _ in range(50):
        u = _rate(env, +1.0)
    assert u == pytest.approx(+6.0)


def test_a_held_command_is_free():
    """Holding a constant rate costs nothing, which is what makes this a
    jerk limit and not a price on steering."""
    env = _bare(steer_jerk_rate=10.0)
    for _ in range(40):
        _rate(env, +1.0)
    before = env._steer_rate_cmd
    assert _rate(env, +1.0) == pytest.approx(before)
    assert env._steer_jerk == pytest.approx(0.0)


def test_the_setpoint_still_integrates_the_delivered_rate():
    env = _bare(steer_jerk_rate=30.0)
    expect = 0.0
    for _ in range(12):
        u = _rate(env, +1.0)
        expect = float(np.clip(expect + u * env.dt, -STEER_MAX, STEER_MAX))
        assert env._steer_cmd == pytest.approx(expect)


def test_it_is_refused_in_absolute_mode_rather_than_ignored():
    """In absolute mode the action is an angle, so a limit on "the change in
    the rate" is not a thing; silently dropping it is how a run becomes a
    different experiment."""
    with pytest.raises(ValueError, match="steer_jerk_rate"):
        ScuderiaLaneKeep._check_steer_jerk(30.0, "absolute")
    ScuderiaLaneKeep._check_steer_jerk(30.0, "delta")        # fine
    ScuderiaLaneKeep._check_steer_jerk(None, "absolute")     # also fine


def test_steer_rate_command_reports_what_was_delivered():
    """Every acceleration number in this project was once measured on the
    request while the env delivered something else. Not twice."""
    env = _bare(steer_jerk_rate=10.0)
    env._decode(np.array([1.0, 0.0]))
    assert env.steer_rate_command == pytest.approx(0.5)      # 10 * 0.05
    assert env.steer_rate_command != 6.0                     # what was asked


# ------------------------------------------------------- the corridor gate
def test_the_gate_is_zero_in_a_hairpin_and_one_down_a_straight():
    tight = _bare(corridor_steer_penalty=1.0, clear=2.0)
    _rate(tight, 1.0)
    assert tight._corridor_gate == pytest.approx(0.0)
    open_ = _bare(corridor_steer_penalty=1.0, clear=8.0)
    _rate(open_, 1.0)
    assert open_._corridor_gate == pytest.approx(1.0)


def test_the_gate_ramps_between_the_two():
    env = _bare(corridor_steer_penalty=1.0, corridor_lo=3.0, corridor_hi=6.0,
                clear=4.5)
    _rate(env, 1.0)
    assert env._corridor_gate == pytest.approx(0.5)


def test_the_cue_is_the_narrow_cone_not_the_wide_one():
    """A wall 20 deg off to the side must not shorten the corridor: on
    master_cup the +/-12 deg cone reads the side wall, which is why it
    separates straights from corners so much worse."""
    env = _bare(corridor_steer_penalty=1.0, clear=8.0)
    side = np.abs(_ANG - np.radians(20.0)) < np.radians(2.0)
    env._scans = env._scans.copy()
    env._scans[0][side] = 0.4
    _rate(env, 1.0)
    assert env._corridor_m == pytest.approx(8.0)
    assert env._corridor_gate == pytest.approx(1.0)


def test_something_dead_ahead_does_shorten_it():
    env = _bare(corridor_steer_penalty=1.0, clear=8.0)
    env._scans = env._scans.copy()
    env._scans[0][np.argmin(np.abs(_ANG))] = 2.0
    _rate(env, 1.0)
    assert env._corridor_m == pytest.approx(2.0)
    assert env._corridor_gate == pytest.approx(0.0)


def test_holding_the_wheel_is_free_even_on_the_longest_straight():
    """The whole point: a policy that stops sawing pays nothing. The three
    costs that failed all charged something a straight-running car still
    owes."""
    env = _bare(corridor_steer_penalty=1.0, clear=10.0)
    _rate(env, 0.0)
    assert env._corridor_gate == pytest.approx(1.0)
    assert env._steer_move == pytest.approx(0.0)


def test_the_gate_reads_the_scan_the_policy_acted_on():
    """It is computed in `_decode`, before the simulator runs, so it is the
    corridor the observation was built from -- not the one the car can see
    after it has moved."""
    env = _bare(corridor_steer_penalty=1.0, clear=8.0)
    _rate(env, 1.0)
    assert env._corridor_m == pytest.approx(8.0)
    env._scans = np.full((1, len(_ANG)), 1.0)     # the world changes
    assert env._corridor_m == pytest.approx(8.0)  # the charge does not, yet
    _rate(env, 1.0)
    assert env._corridor_m == pytest.approx(1.0)


# ---------------------------------------------------------- the feature
def test_the_feature_is_one_number_appended_last():
    off, on = _bare(), _bare(corridor_feature=True)
    a = off._obs_from(np.zeros((1, 7)), off._scans)
    b = on._obs_from(np.zeros((1, 7)), on._scans)
    assert len(b) == len(a) + 1
    assert np.allclose(b[:-1], a)


@pytest.mark.parametrize("clear,want", [(2.0, 0.0), (4.5, 0.5), (9.0, 1.0)])
def test_the_feature_is_the_gate_the_cost_uses(clear, want):
    """Pricing something the policy cannot observe is what `steer_feature`
    was added to stop happening; the feature and the gate must be the same
    number."""
    env = _bare(corridor_steer_penalty=1.0, corridor_feature=True, clear=clear)
    _rate(env, 1.0)
    obs = env._obs_from(np.zeros((1, 7)), env._scans)
    assert obs[-1] == pytest.approx(want)
    assert obs[-1] == pytest.approx(env._corridor_gate)


# -------------------------------------------- the two knobs, composed
# A flat jerk limit fails for the same reason the three prices did: the
# steering jerk the policy asks for is the same on a straight as in a
# corner (median 144 against 132 rad/s^2; above 30 on 57 % of straight
# steps against 55 % of corner steps), so one limit cannot quiet the
# straights without binding on more than half the cornering. The corridor
# does tell the two apart, so it is what should decide how hard the limit
# bites.

def test_the_gated_limit_is_the_full_limit_down_a_straight():
    env = _bare(steer_jerk_rate=30.0, clear=10.0)
    env.steer_jerk_corridor = True
    env._jerk_unlimited = 2.0 * env.steer_rate / env.dt
    prev, worst = 0.0, 0.0
    for level in (1.0, -1.0) * 20:
        u = _rate(env, level)
        worst = max(worst, abs(u - prev) / env.dt)
        prev = u
    assert env._corridor_gate == pytest.approx(1.0)
    assert worst <= 30.0 + 1e-9


def test_the_gated_limit_lets_go_in_a_hairpin():
    """Where the car needs the whole lock, it must still get it in one tick."""
    env = _bare(steer_jerk_rate=30.0, clear=2.0)
    env.steer_jerk_corridor = True
    env._jerk_unlimited = 2.0 * env.steer_rate / env.dt
    assert _rate(env, +1.0) == pytest.approx(+6.0)
    assert _rate(env, -1.0) == pytest.approx(-6.0)   # unbound, as before
    assert env._corridor_gate == pytest.approx(0.0)


def test_half_a_gate_is_half_way_between():
    env = _bare(steer_jerk_rate=30.0, clear=4.5)      # gate 0.5
    env.steer_jerk_corridor = True
    env._jerk_unlimited = 2.0 * env.steer_rate / env.dt
    free = env._jerk_unlimited
    _rate(env, +1.0)
    prev = env._steer_rate_cmd
    u = _rate(env, -1.0)
    assert abs(u - prev) / env.dt == pytest.approx(0.5 * (free + 30.0), rel=1e-6)


def test_gating_is_off_unless_asked_for():
    """The flat limit has to keep behaving exactly as the arm that ran."""
    env = _bare(steer_jerk_rate=30.0, clear=10.0)
    assert env.steer_jerk_corridor is False
    assert _rate(env, +1.0) == pytest.approx(1.5)     # 30 * 0.05
