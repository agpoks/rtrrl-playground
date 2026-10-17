r"""The F1TENTH disparity extender, and the five numbers it used to discard.

The extender itself is not new here. `free_space_heading` has computed the
width-corrected scan since the start -- it is what the scripted `GapFollower`
drives on and what `aim_feature` publishes -- but it returned only
:math:`\theta_{safe}` and threw the extended scan away. So the project has
had Feature A of the proposal for a year and none of B--F, and it built its
corridor gate on ``min`` over a *raw* +/-6 deg cone instead, which reads the
side wall wherever the corridor is narrower than the cone is wide.

Two things these tests are for.

**The refactor must not move `aim_feature`.** Every arm in `benchmarks/`
carries `--aim-feature`, and they only pair with a new arm if the heading is
bit-identical to what they were trained on. `test_the_refactor_is_bit_
identical` keeps a frozen copy of the pre-refactor loop and compares.

**The conservative cases must be stated, not implied.** The old loop read an
`inf` range as the furthest free space and pointed the car at it. That is a
lidar dropout, not a straight.

Nothing starts a simulator: the pure functions take a scan, and the env is
exercised on `lanekeep` (no map, no jax) and on a bare scuderia instance.
"""

import numpy as np
import pytest

from rtrrl_playground.envs.scripted import (disparity_extend,
                                            disparity_target,
                                            free_space_heading)
from rtrrl_playground.envs.track import (DE_LETTERS, DE_RATE_SCALE, DE_SWITCH,
                                         HEADING_SCALE, de_features)

ANG = np.linspace(-np.pi / 2, np.pi / 2, 181)   # 1 deg beams, forward sector


def _slot(width_deg, near=0.8, far=10.0, centre_deg=0.0, ang=ANG):
    """A wall at `near` with an opening `width_deg` wide showing `far`."""
    r = np.full(len(ang), float(near))
    gap = np.abs(np.degrees(ang) - centre_deg) <= width_deg / 2
    r[gap] = far
    return r


# ------------------------------------------------- the extender, unchanged
def _frozen(ranges, angles, car_half_width=0.20):
    """The pre-refactor `free_space_heading`, kept here as the reference."""
    ang = np.asarray(angles, dtype=float)
    r = np.asarray(ranges, dtype=float)
    n = len(ang)
    if n < 3:
        return 0.0
    d_ang = float(ang[-1] - ang[0]) / (n - 1)
    ext = r.copy()
    for i in range(n - 1):
        if abs(r[i + 1] - r[i]) < 2 * car_half_width:
            continue
        near, far = (i, i + 1) if r[i] < r[i + 1] else (i + 1, i)
        k = int(np.ceil(np.arctan2(car_half_width, max(r[near], 1e-3))
                        / max(d_ang, 1e-6)))
        lo, hi = ((far, min(far + k + 1, n)) if far > near
                  else (max(far - k, 0), far + 1))
        ext[lo:hi] = np.minimum(ext[lo:hi], r[near])
    best = ext.max()
    tied = np.flatnonzero(ext >= best - 1e-9)
    return float(ang[tied[int(np.argmin(np.abs(ang[tied])))]])


def test_the_refactor_is_bit_identical():
    """Otherwise no existing `--aim-feature` arm pairs with a new one."""
    rng = np.random.default_rng(0)
    for trial in range(300):
        n = int(rng.integers(3, 200))
        ang = np.linspace(-np.pi / 2, np.pi / 2, n)
        if trial % 3 == 0:
            r = rng.uniform(0.05, 10.0, n)
        elif trial % 3 == 1:
            r = np.full(n, 10.0)
        else:
            r = np.clip(3.0 + 2.0 * np.sin(3 * ang) + rng.normal(0, .02, n),
                        .05, 10.0)
        assert free_space_heading(r, ang) == _frozen(r, ang), f"trial {trial}"


def test_the_extension_only_ever_shortens_a_beam():
    """`min`, never `max`: the result is a lower bound on the clearance."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        r = rng.uniform(0.1, 10.0, len(ANG))
        assert np.all(disparity_extend(r, ANG) <= r + 1e-12)


def test_a_doorway_the_car_does_not_fit_through_is_not_chosen():
    """The whole point of Step 3. A 1 deg slot at 10 m is 17 cm wide."""
    r = _slot(1.0)
    assert disparity_extend(r, ANG).max() == pytest.approx(0.8)
    assert disparity_target(r, ANG).d_safe == pytest.approx(0.8)


def test_an_opening_the_car_does_fit_through_survives():
    r = _slot(40.0)
    t = disparity_target(r, ANG)
    assert t.d_safe == pytest.approx(10.0)
    assert abs(t.theta) < np.radians(1.0)


def test_the_exclusion_grows_as_the_edge_gets_closer():
    """`arctan(r_s / d_near)`: a near edge hides more of the scan."""
    wide = [np.count_nonzero(disparity_extend(_slot(40.0, near=d), ANG) < 9.0)
            for d in (0.5, 1.0, 3.0)]
    assert wide[0] > wide[1] > wide[2]


@pytest.mark.parametrize("bad", [np.inf, np.nan, 0.0, -1.0])
def test_an_invalid_range_blocks_a_bearing_instead_of_winning_it(bad):
    """A dropout is not a straight. The old loop would aim at `inf`."""
    r = np.full(len(ANG), 3.0)
    r[20] = bad
    ext = disparity_extend(r, ANG)
    assert np.isfinite(ext).all()
    assert ext[20] == 0.0
    assert abs(disparity_target(r, ANG).theta) < np.radians(1.0)


@pytest.mark.parametrize("bad", [np.inf, np.nan, 0.0, -1.0])
def test_one_dropped_beam_does_not_blank_half_the_scan(bad):
    """Measured: it did. A dropout read as an obstacle at 0 m is a disparity
    with `d_near = 0`, and `arctan(r_s / 0)` is 90 deg of exclusion on each
    side -- so one bad beam of 181 wiped 112 of them and moved the target
    22 deg off straight. An invalid beam is skipped when looking for edges,
    which blocks exactly its own bearing and nothing else."""
    r = np.full(len(ANG), 3.0)
    r[20] = bad
    assert np.count_nonzero(disparity_extend(r, ANG) < 3.0) == 1


def test_a_margin_widens_the_exclusion():
    r = _slot(40.0, near=2.0)
    n = [np.count_nonzero(disparity_extend(r, ANG, margin=m) < 9.0)
         for m in (0.0, 0.1, 0.3)]
    assert n[0] < n[1] < n[2]


def test_a_bigger_disparity_threshold_finds_fewer_edges():
    r = _slot(40.0, near=6.0, far=10.0)      # a 4 m step
    assert np.any(disparity_extend(r, ANG, disparity=2.0) < 9.9)
    assert np.all(disparity_extend(r, ANG, disparity=8.0) == r)


def test_asin_blocks_rather_than_returning_nan_inside_the_safety_radius():
    """`arcsin(r_s / d)` has no value for `d < r_s`; it must not be a NaN."""
    r = _slot(40.0, near=0.05)               # closer than half the car
    ext = disparity_extend(r, ANG, geometry="asin")
    assert np.isfinite(ext).all()
    assert ext.max() == pytest.approx(0.05)


def test_atan_and_asin_agree_while_the_ratio_is_small():
    r = _slot(40.0, near=2.0)                # r_s / d = 0.1
    assert np.allclose(disparity_extend(r, ANG),
                       disparity_extend(r, ANG, geometry="asin"), atol=1e-9)


def test_the_correction_is_in_radians_and_not_in_beams():
    """The angular result must not move when the scan resolution does.

    `n_ext = ceil(alpha / |dtheta|)` with the spacing taken from the angles:
    double the beam density and twice as many beams are overwritten, over the
    same angle. A hard-coded beam count would halve the exclusion instead.
    """
    fine = np.linspace(-np.pi / 2, np.pi / 2, 361)
    coarse = np.linspace(-np.pi / 2, np.pi / 2, 181)
    wf = disparity_target(_slot(40.0, near=2.0, ang=fine), fine).width
    wc = disparity_target(_slot(40.0, near=2.0, ang=coarse), coarse).width
    assert np.degrees(abs(wf - wc)) < 2.0
    nf = np.count_nonzero(disparity_extend(_slot(40.0, near=2.0, ang=fine),
                                           fine) < 9.0)
    nc = np.count_nonzero(disparity_extend(_slot(40.0, near=2.0, ang=coarse),
                                           coarse) < 9.0)
    assert nf == pytest.approx(2 * nc, rel=0.03)


# ------------------------------------------------------------ the target
def test_d_safe_is_the_extended_range_in_the_chosen_direction():
    rng = np.random.default_rng(2)
    for _ in range(40):
        r = np.clip(rng.uniform(0.3, 10.0, len(ANG)), 0.3, 10.0)
        t = disparity_target(r, ANG)
        j = int(np.argmin(np.abs(ANG - t.theta)))
        assert t.ext[j] == pytest.approx(t.d_safe)


def test_an_open_scan_points_straight_ahead_across_the_whole_sector():
    t = disparity_target(np.full(len(ANG), 10.0), ANG)
    assert t.theta == 0.0
    assert t.d_safe == pytest.approx(10.0)
    assert t.width == pytest.approx(np.pi, abs=1e-6)


def test_the_gap_width_is_the_opening_the_car_can_use_not_the_raw_one():
    """The stated criterion: the contiguous run within `gap_frac` of d_safe,
    measured on the *extended* scan -- so it is narrower than the opening a
    protractor would find, by the width correction at each edge."""
    for w in (30.0, 50.0, 70.0):
        t = disparity_target(_slot(w, near=2.0), ANG)
        edge = np.degrees(np.arctan2(0.20, 2.0))
        assert np.degrees(t.width) < w
        assert np.degrees(t.width) == pytest.approx(w - 2 * edge, abs=5.0)


def test_a_blank_wall_reads_as_the_whole_sector():
    """Not a bug to hide: `width` is the run within 10 % of the best, and on
    a uniform wall every bearing is equally bad, so the answer is 180 deg.
    `f6` alone therefore cannot tell a wide-open straight from a dead end --
    which is what `f2` is for, and why the first ablation is `b` and not `f`."""
    flat = disparity_target(np.full(len(ANG), 0.8), ANG)
    assert flat.width == pytest.approx(np.pi, abs=1e-6)
    assert flat.d_safe == pytest.approx(0.8)


def test_width_is_angular_and_is_not_the_metric_width():
    """Documented distinction, pinned: the same gap at twice the range."""
    near = disparity_target(_slot(40.0, far=3.0), ANG)
    far = disparity_target(_slot(40.0, far=9.0), ANG)
    assert near.width == pytest.approx(far.width, abs=np.radians(3))
    # ... while the opening it describes is three times wider in metres
    assert (2 * far.d_safe * np.sin(far.width / 2)
            > 2.5 * 2 * near.d_safe * np.sin(near.width / 2))


# ----------------------------------------------------------- the features
def test_only_the_letters_asked_for_are_returned_and_in_that_order():
    r = _slot(40.0, centre_deg=-30.0)
    ab, _ = de_features(r, ANG, 10.0, spec="ab")
    ba, _ = de_features(r, ANG, 10.0, spec="ba")
    assert len(ab) == 2 and np.allclose(ab, ba[::-1])
    assert len(de_features(r, ANG, 10.0, spec=DE_LETTERS)[0]) == 6


def test_an_unknown_letter_is_refused():
    with pytest.raises(ValueError, match="z"):
        de_features(np.full(len(ANG), 3.0), ANG, 10.0, spec="bz")


def test_feature_a_is_exactly_the_existing_aim_feature():
    """So a disparity arm and every `--aim-feature` arm share one number."""
    rng = np.random.default_rng(3)
    for _ in range(40):
        r = rng.uniform(0.3, 10.0, len(ANG))
        f, _ = de_features(r, ANG, 10.0, spec="a")
        aim = free_space_heading(r, ANG)
        assert f[0] == pytest.approx(np.clip(aim / HEADING_SCALE, -1, 1))


def test_feature_b_is_the_length_of_the_chosen_straight_over_the_range():
    for far in (2.0, 5.0, 9.0):
        f, _ = de_features(_slot(40.0, far=far), ANG, 10.0, spec="b")
        assert f[0] == pytest.approx(far / 10.0)


def test_feature_b_saturates_at_one_and_never_goes_negative():
    assert de_features(np.full(len(ANG), 99.0), ANG, 10.0, "b")[0][0] == 1.0
    assert de_features(np.zeros(len(ANG)), ANG, 10.0, "b")[0][0] == 0.0


def test_feature_b_sees_a_long_straight_where_a_raw_cone_sees_the_wall():
    """Why `b` and not the corridor gate: a corridor narrower than the cone.

    A 1.5 m wide corridor running 8 m ahead. The +/-6 deg cone the gate uses
    subtends +/-0.84 m at 8 m, so its minimum lands on the side wall and the
    gate reads a corner. The extended scan still has 8 m straight ahead.
    """
    ang = np.linspace(-np.pi / 2, np.pi / 2, 721)
    half = 0.75
    with np.errstate(divide="ignore"):
        wall = np.where(np.abs(ang) < 1e-6, np.inf,
                        half / np.maximum(np.abs(np.sin(ang)), 1e-9))
    r = np.minimum(np.where(np.isfinite(wall), wall, 8.0), 8.0)
    cone = np.abs(ang) <= np.radians(6.0)
    assert float(np.min(r[cone])) < 8.0            # the gate reads the wall
    f, _ = de_features(r, ang, 10.0, spec="b")
    assert f[0] == pytest.approx(0.8, abs=0.02)    # `b` reads the straight


def test_feature_c_is_the_width_corrected_clearance_and_never_exceeds_raw():
    r = _slot(1.0, near=0.8)                 # a slot the car cannot use
    raw = float(np.min(r[np.abs(ANG) <= np.radians(6.0)]))
    f, _ = de_features(r, ANG, 10.0, spec="c")
    assert f[0] * 10.0 <= raw + 1e-9
    assert f[0] * 10.0 == pytest.approx(0.8)


def test_the_cone_for_feature_c_is_configurable():
    r = np.full(len(ANG), 9.0)
    r[np.abs(np.degrees(ANG)) > 30.0] = 1.0
    assert de_features(r, ANG, 10.0, "c", cone_deg=6.0)[0][0] == pytest.approx(0.9)
    assert de_features(r, ANG, 10.0, "c", cone_deg=30.0)[0][0] == pytest.approx(0.1)


def test_the_motion_features_read_zero_on_the_first_step_of_an_episode():
    """`prev_theta=None` is a reset, not a jump from the last episode."""
    f, _ = de_features(_slot(40.0, centre_deg=40.0), ANG, 10.0, spec="de")
    assert list(f) == [0.0, 0.0]


def test_feature_d_is_a_rate_and_scales_with_the_step_it_is_divided_by():
    r = _slot(40.0, centre_deg=20.0)
    th = disparity_target(r, ANG).theta
    slow, _ = de_features(r, ANG, 10.0, "d", prev_theta=th - 0.1, dt=0.1)
    fast, _ = de_features(r, ANG, 10.0, "d", prev_theta=th - 0.1, dt=0.05)
    assert fast[0] == pytest.approx(2 * slow[0])
    assert slow[0] == pytest.approx(0.1 / 0.1 / DE_RATE_SCALE)


def test_feature_e_fires_only_past_the_threshold():
    r = _slot(40.0, centre_deg=20.0)
    th = disparity_target(r, ANG).theta
    small = DE_SWITCH * 0.5
    big = DE_SWITCH * 2.0
    assert de_features(r, ANG, 10.0, "e", prev_theta=th - small)[0][0] == 0.0
    assert de_features(r, ANG, 10.0, "e", prev_theta=th - big)[0][0] == 1.0
    # and it is a direction-free magnitude
    assert de_features(r, ANG, 10.0, "e", prev_theta=th + big)[0][0] == 1.0


def test_feature_f_grows_with_the_opening():
    v = [de_features(_slot(w, near=2.0), ANG, 10.0, "f")[0][0]
         for w in (30, 50, 70)]
    assert v[0] < v[1] < v[2] and v[2] <= 1.0


def test_the_returned_heading_is_what_the_next_step_should_be_given():
    r = _slot(40.0, centre_deg=-25.0)
    f, th = de_features(r, ANG, 10.0, spec="a")
    assert th == pytest.approx(disparity_target(r, ANG).theta)
    assert f[0] == pytest.approx(np.clip(th / HEADING_SCALE, -1, 1))


# ------------------------------------------------------------- the wiring
def _lk(**kw):
    from rtrrl_playground.envs.lanekeep import LaneKeep
    return LaneKeep(track="curvy", seed=0, **kw)


@pytest.mark.parametrize("spec", ["b", "ab", "abcdef"])
def test_the_observation_grows_by_one_number_per_letter(spec):
    plain, wide = _lk(), _lk(de_features=spec)
    assert wide.obs_dim == plain.obs_dim + len(spec)
    assert len(wide.reset(seed=0)) == wide.obs_dim


def test_the_features_are_appended_last_so_nothing_already_there_moves():
    """A saved policy, a figure's axis labels and every written-down index."""
    plain, wide = _lk(steer_feature=True), _lk(steer_feature=True,
                                               de_features="bf")
    a, b = plain.reset(seed=0), wide.reset(seed=0)
    assert np.allclose(b[:len(a)], a)


def test_what_the_env_appends_is_what_the_function_computes():
    env = _lk(de_features="b")
    obs = env.reset(seed=0)
    r, _flags = env._last_beams
    f, _ = de_features(r, env.beam_angles, 5.0, spec="b")
    assert obs[-1] == pytest.approx(f[0])


def test_a_bad_letter_is_refused_at_construction_not_at_the_first_step():
    with pytest.raises(ValueError, match="z"):
        _lk(de_features="z")


def test_the_remembered_heading_is_cleared_between_episodes():
    """Or the first step of episode two reads a switch that never happened."""
    env = _lk(de_features="e")
    env.reset(seed=0)
    for _ in range(20):
        env.step(4)
    env.reset(seed=1)
    assert env._de_prev is None or env.reset(seed=2)[-1] == 0.0


def test_the_scuderia_adapter_appends_it_too():
    """On a bare instance: the real constructor starts a simulator."""
    from rtrrl_playground.envs.scuderia import ScuderiaLaneKeep, BEAM_RANGE
    env = object.__new__(ScuderiaLaneKeep)
    env._idx = np.array([0, 90, 180])
    env.observe_speed = env.aim_feature = env.drive_features = False
    env.proprio = env.grip_feature = env.steer_feature = False
    env.corridor_feature = False
    env.lookahead = None
    env.dt = 0.05
    env.scan_angles = ANG
    env.de_features, env.de_cone_deg, env.de_margin = "bf", 6.0, 0.0
    env._de_prev = None
    scans = _slot(40.0, far=7.0)[None, :]
    obs = env._obs_from(np.zeros((1, 7)), scans)
    assert len(obs) == 5                      # three beams plus b and f
    assert obs[3] == pytest.approx(0.7)       # 7 m of a 10 m scan
    assert env._de_prev == pytest.approx(0.0)
    assert BEAM_RANGE == 10.0
