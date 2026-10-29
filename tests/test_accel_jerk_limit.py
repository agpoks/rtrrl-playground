"""The acceleration jerk limit must bind, and must be readable afterwards.

`accel_rate` clips the longitudinal command to one jerk-limited step per
tick. Nothing tested it, and nothing could see it either: a recorded trace
logged the *action* -- the request -- so a limit that silently stopped
working would have left every figure and every jerk number unchanged. That
is how a steering *rate* came to be plotted as a steering angle in this
project, and the acceleration axis had the same hole. `accel_command` closes
it, and these tests pin both halves: the clip happens, and the property
reports what was given rather than what was asked for.

`_decode` is exercised on a bare instance: the real constructor starts a
simulator and loads a map, which this does not need and should not wait for.
"""

import numpy as np
import pytest

from rtrrl_playground.envs.scuderia import ScuderiaLaneKeep, STEER_MAX


def _bare(accel_rate, dt=0.05, accel_cmd=10.0):
    env = object.__new__(ScuderiaLaneKeep)
    env.sensors = ""          # the car's own sensors, off
    env.action_mode = "continuous"
    env.accel_mode = "accl"
    env.accel_cmd = accel_cmd
    env.accel_rate = accel_rate
    env.accel_jerk_penalty = 0.0
    env._accel_cmd = 0.0
    env._accel_jerk = 0.0
    env._v_cmd = 0.0
    env._x = None                      # standstill: v_now = 0
    env.dt = dt
    env.steer_mode = "absolute"
    env.steer_rate = 6.0
    env.steer_filter = 0.0
    env._steer_cmd = 0.0
    env._steer_out = 0.0
    env._steer_move = 0.0
    env._steer_flip = False
    return env


@pytest.mark.parametrize("rate", [5.0, 20.0, 50.0])
def test_the_command_moves_by_at_most_rate_times_dt(rate):
    env = _bare(rate)
    rng = np.random.default_rng(0)
    seen, prev = [], 0.0
    for _ in range(200):
        env._decode(np.array([0.0, rng.uniform(-1.0, 1.0)]))
        seen.append(env.accel_command - prev)
        prev = env.accel_command
    assert max(abs(s) for s in seen) <= rate * env.dt + 1e-9
    # and the limit is actually reached, or the test would pass on a dead axis
    assert max(abs(s) for s in seen) == pytest.approx(rate * env.dt, rel=1e-6)


def test_accel_command_is_the_command_not_the_request():
    env = _bare(50.0)
    env._decode(np.array([0.0, 1.0]))          # asks for +10 m/s2
    assert env.accel_command == pytest.approx(50.0 * 0.05)   # given 2.5
    assert env.accel_command != 10.0


def test_without_a_limit_the_request_is_the_command():
    env = _bare(None)
    for level in (1.0, -1.0, 0.3):
        env._decode(np.array([0.0, level]))
        assert env.accel_command == pytest.approx(level * 10.0)


def test_speed_setpoint_is_exposed():
    env = _bare(None)
    env._decode(np.array([0.0, 1.0]))
    assert env.speed_setpoint == pytest.approx(env._v_cmd)


def test_the_steering_setpoint_is_still_clipped_to_the_lock_angle():
    """The acceleration clip sits next to the steering one; keep it honest."""
    env = _bare(None)
    env.steer_mode = "delta"
    for _ in range(50):
        env._decode(np.array([1.0, 0.0]))
    assert env._steer_cmd == pytest.approx(STEER_MAX)
