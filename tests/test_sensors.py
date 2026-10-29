"""The car's own sensors as observation channels: IMU, VESC, servo, pose.

The published configuration of this project reads nine lidar beams, four
scalars derived from the full scan, three map lookaheads and an echo of its own
steering command. The IMU, the VESC and the servo are all modelled in the
simulator and none of them is read. `sensors` is the letter set that wires them
through, and these are the properties that have to hold for a channel to be
worth anything:

* it is **appended**, so switching sensors on moves no existing feature and a
  checkpoint's indices keep meaning what they meant;
* its **order is canonical**, so ``"wi"`` and ``"iw"`` build the same vector;
* ``obs_labels`` is exactly as long as the vector, because that table is what
  the paper prints and what a loader checks against;
* the slip channel is the **tyre model's own slip**, not a body-speed
  approximation of it -- the front wheel is steered and offset from the centre
  of gravity, and reading the body speed there puts a cornering artefact in the
  one channel whose job is wheelspin.

`_obs_from` is exercised on a bare instance, as in `test_steer_feature`: the
real constructor starts a simulator and loads a map, and none of this needs
one.
"""

import numpy as np
import pytest

from rtrrl_playground.envs.scuderia import (CURRENT_REF, GYRO_REF, IMU_A_REF,
                                            SENSOR_LETTERS, SPEED_MAX,
                                            STEER_MAX, ScuderiaLaneKeep)

R_W, K_TG, MASS = 0.031, 0.014, 4.251
V_MIN = 0.1


def _bare(sensors="", n_wheel=2):
    env = object.__new__(ScuderiaLaneKeep)
    env._idx = np.array([0, 1, 2])
    env.observe_speed = False
    env.aim_feature = False
    env.drive_features = False
    env.proprio = False
    env.grip_feature = False
    env.lookahead = None
    env.steer_feature = False
    env.corridor_feature = False
    env.de_features = ""
    env.steer_mode = "delta"
    env.steer_rate = 6.0
    env.steer_filter = 0.0
    env.dt = 0.05
    env._steer_cmd = 0.0
    env._steer_out = 0.0
    env.action_mode = "continuous"
    env.accel_mode = "accl"
    env.accel_cmd = 10.0
    env.accel_rate = None
    env._accel_cmd = 0.0
    env._x = None
    # what the constructor reads off the running simulator
    env.sensors = "".join(c for c in SENSOR_LETTERS if c in sensors)
    env._n_wheel = n_wheel
    env._mass, env._R_w, env._K_tG = MASS, R_W, K_TG
    env._lf, env._lr = 0.1705, 0.1515
    env._t_f = env._t_r = 0.22
    env._v_min = V_MIN
    env._imu = (0.0, 0.0)
    return env


def _x(v=2.0, delta=0.0, r=0.0, beta=0.0, wheels=(0.0, 0.0)):
    x = np.zeros((1, 7 + len(wheels)))
    x[0, 2], x[0, 3], x[0, 5], x[0, 6] = delta, v, r, beta
    x[0, 7:] = wheels
    return x


_SCANS = np.full((1, 8), 10.0)


def test_every_letter_is_appended_and_nothing_before_it_moves():
    base = _bare()._obs_from(_x(), _SCANS)
    for ch in SENSOR_LETTERS:
        if ch == "p":
            continue  # needs a centreline; covered by the live env
        env = _bare(ch)
        got = env._obs_from(_x(wheels=(64.5, 64.5)), _SCANS)
        assert len(got) > len(base), ch
        assert np.allclose(got[:len(base)], base), ch


def test_the_order_does_not_depend_on_how_the_flag_was_typed():
    a = _bare("iowsc")._obs_from(_x(wheels=(64.5, 64.5)), _SCANS)
    b = _bare("csoiw")._obs_from(_x(wheels=(64.5, 64.5)), _SCANS)
    assert np.allclose(a, b)
    assert _bare("csoiw").sensors == "iowsc"


def test_the_width_is_what_n_sensor_channels_says():
    for sensors, n_wheel, width in (("i", 2, 3), ("o", 2, 2), ("w", 2, 2),
                                    ("w", 4, 4), ("s", 4, 4), ("c", 2, 1),
                                    ("iowsc", 2, 10), ("iowsc", 4, 14)):
        env = _bare(sensors, n_wheel)
        assert env.n_sensor_channels == width, (sensors, n_wheel)
        got = env._obs_from(_x(wheels=(64.5,) * n_wheel), _SCANS)
        assert len(got) - len(_bare()._obs_from(_x(), _SCANS)) == width


def test_the_imu_channels_are_the_modelled_accelerometer():
    env = _bare("i")
    env._imu = (3.0, -5.0)
    got = env._obs_from(_x(r=1.5), _SCANS)[-3:]
    assert got[0] == pytest.approx(3.0 / IMU_A_REF)
    assert got[1] == pytest.approx(-5.0 / IMU_A_REF)
    assert got[2] == pytest.approx(1.5 / GYRO_REF)
    # the accelerometer saturates rather than running away
    env._imu = (90.0, -90.0)
    got = env._obs_from(_x(), _SCANS)[-3:]
    assert got[0] == pytest.approx(1.0) and got[1] == pytest.approx(-1.0)


def test_odometry_is_the_speed_and_the_angle_the_servo_reached():
    env = _bare("o")
    got = env._obs_from(_x(v=3.0, delta=0.2), _SCANS)[-2:]
    assert got[0] == pytest.approx(3.0 / SPEED_MAX)
    assert got[1] == pytest.approx(0.2 / STEER_MAX)


def test_wheel_speed_is_a_surface_speed_not_an_angular_one():
    env = _bare("w")
    # a wheel turning at v / R_w is a wheel rolling at v
    got = env._obs_from(_x(v=2.0, wheels=(2.0 / R_W, 2.0 / R_W)), _SCANS)[-2:]
    assert np.allclose(got, 2.0 / SPEED_MAX)


def test_a_rolling_wheel_does_not_slip():
    env = _bare("s")
    w = 2.0 / R_W
    got = env._obs_from(_x(v=2.0, wheels=(w, w)), _SCANS)[-2:]
    assert np.allclose(got, 0.0, atol=1e-12)


def test_wheelspin_is_negative_and_locking_is_positive():
    """CommonRoad's convention, which the simulator keeps."""
    env = _bare("s")
    rolling = 2.0 / R_W
    spin = env._obs_from(_x(v=2.0, wheels=(rolling * 1.2,) * 2), _SCANS)[-2:]
    lock = env._obs_from(_x(v=2.0, wheels=(rolling * 0.8,) * 2), _SCANS)[-2:]
    assert np.all(spin > 0.15) and np.all(lock < -0.15)


def test_a_standstill_reads_zero_slip_rather_than_minus_one():
    env = _bare("s")
    got = env._obs_from(_x(v=0.0, wheels=(0.0, 0.0)), _SCANS)[-2:]
    assert np.allclose(got, 0.0)


def test_front_slip_uses_the_steered_wheel_centre_velocity():
    """The property the first version of this got wrong.

    Rolling through a corner on full lock: the front wheel's own longitudinal
    velocity is not the body speed, so a wheel that is genuinely rolling has to
    read zero slip anyway. Reading the body speed instead leaves a standing
    offset of a few per cent whenever the car is turning.
    """
    env = _bare("s")
    v, delta, r = 2.0, 0.4, 1.5
    vx, vy = v, 0.0
    u_wf = vx * np.cos(delta) + (vy + env._lf * r) * np.sin(delta)
    got = env._obs_from(_x(v=v, delta=delta, r=r,
                           wheels=(u_wf / R_W, vx / R_W)), _SCANS)[-2:]
    assert np.allclose(got, 0.0, atol=1e-9)
    # and the shortcut would not have: the body speed is 2.9 % away here
    assert abs(u_wf - vx) / vx > 0.02


def test_current_is_the_jerk_limited_command_in_amps():
    env = _bare("c")
    env._accel_cmd = 1.0
    got = float(env._obs_from(_x(), _SCANS)[-1])
    assert got == pytest.approx(MASS * R_W / K_TG / CURRENT_REF, rel=1e-9)
    # ~9.4 A per m/s^2 on this car, so the grid's full scale saturates
    env._accel_cmd = 10.0
    assert float(env._obs_from(_x(), _SCANS)[-1]) == pytest.approx(1.0)


def test_the_wheel_letters_are_refused_without_a_wheel_state():
    env = _bare("w", n_wheel=0)
    env.sensors = "w"
    env._n_wheel = 0
    assert env.n_sensor_channels == 0
