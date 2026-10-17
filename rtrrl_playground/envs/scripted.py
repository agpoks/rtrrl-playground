"""Hand-written reference policies for the two driving tasks.

Nothing learns here. These exist for three reasons:

* **A sanity floor.** If a scripted wall-follower cannot get round the track,
  the environment is broken and no amount of debugging the learning rule will
  help. Every time the track or the sensor changed while this repo was being
  built, these ran first.
* **A yardstick.** "RTRRL reached 420 return" means nothing on its own. These
  policies say what a competent non-learning controller gets, and they are in
  the benchmark tables next to the learned agents for exactly that reason.
* **An honest look at the task.** The overtaker below is about fifteen lines
  and it still crashes half the time, because it commits to a side from one
  frame and has no idea how fast it is closing. That failure -- needing a
  closing *rate* that no single observation contains -- is the reason the
  learned agent is given a recurrent state.

Both take the raw observation and return a discrete action, so they slot in
anywhere a policy does.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

N_BEAMS = 9

#: The default scan these were written against: nine beams over 120 degrees.
#: ``lanekeep`` now takes ``n_beams``/``fov_deg``, and a controller reading
#: ``obs[:9]`` of a 61-beam scan is looking only at the left shoulder, so both
#: policies below take the table they should read.
BEAM_ANGLES = np.deg2rad(np.linspace(-60.0, 60.0, N_BEAMS))


def _act(steer: int, throttle: int) -> int:
    """(steer, throttle) in {-1,0,1}^2 -> the flat 9-way action index."""
    return 3 * (steer + 1) + (throttle + 1)


class WallFollower:
    """Steer towards the side with more room, lift off when the gap closes.

    Two rules, no state, no tuning beyond the two thresholds. On ``lanekeep``
    it laps the track at close to the grip limit, which is a useful thing to
    know before concluding that an agent scoring 500 has learned something
    deep.
    """

    def __init__(self, slow_below: float = 0.55, hard_below: float = 0.30,
                 beam_angles=None):
        self.slow_below = slow_below
        self.hard_below = hard_below
        self.beam_angles = np.asarray(BEAM_ANGLES if beam_angles is None
                                      else beam_angles, dtype=float)

    def __call__(self, obs: np.ndarray) -> int:
        r = obs[:len(self.beam_angles)]
        # Outer third each side, and the centre beam. Written as fractions of
        # the scan rather than as literal indices: `beam_angles` is a
        # constructor argument, so this is also called with 31 or 61 beams,
        # and the old `r[6:]`/`r[:3]`/`r[4]` silently meant "beams 6..60
        # against beams 0..2, steering on a beam 26 degrees off centre" there.
        # At the default nine beams this reduces to exactly those indices.
        n = len(r)
        k = max(1, n // 3)
        left, right = r[n - k:].sum(), r[:k].sum()
        steer = 1 if left > right + 0.02 else (-1 if right > left + 0.02 else 0)
        ahead = r[n // 2]
        throttle = 1 if ahead > self.slow_below else (-1 if ahead < self.hard_below else 0)
        return _act(steer, throttle)

    def reset(self) -> None:
        pass


def free_space_heading(ranges: np.ndarray, angles: np.ndarray,
                       car_half_width: float = 0.20) -> float:
    """Direction of the furthest free space, in radians, from the scan alone.

    The pose-free half of "where does the track go next". A lookahead heading
    error answers that from a map and a localised pose; this answers it from
    the lidar, which matters because the two have very different deployment
    costs: the map version needs a particle filter running at control rate and
    degrades in a way the policy has never seen when localisation drifts, while
    this needs the scan the policy already receives.

    The disparity extension is the part that earns its place. Where two adjacent
    beams differ by more than the car is wide, the far reading belongs to
    something behind an edge the car cannot fit through, so the near reading is
    widened across the angular half-width of the car. Without it the furthest
    beam regularly points through a doorway the car does not fit in.

    Shared with :class:`GapFollower` rather than copied into it -- the same
    rule the scripted baseline drives on is the rule the feature reports, so a
    policy given this feature and the gap-follower are looking at one quantity.
    """
    return disparity_target(ranges, angles, car_half_width).theta


class DisparityTarget(NamedTuple):
    """What the disparity extender knows after one scan.

    ``free_space_heading`` returns only ``theta`` and throws the rest away.
    Every other quantity below was already computed to get it.
    """

    #: direction of the furthest safely reachable space, rad, 0 = straight on
    theta: float
    #: the extended range in that direction, metres -- how long the straight is
    d_safe: float
    #: angular width of the contiguous region the target sits in, rad
    width: float
    #: the width-corrected scan itself, same shape as ``ranges``
    ext: np.ndarray


def disparity_extend(ranges: np.ndarray, angles: np.ndarray,
                     car_half_width: float = 0.20, margin: float = 0.0,
                     disparity: float | None = None,
                     geometry: str = "atan") -> np.ndarray:
    r"""The width-corrected scan :math:`\tilde d`, from the F1TENTH extender.

    Where two adjacent beams differ by more than ``disparity`` metres there is
    an edge, and the far reading belongs to something the car may not fit
    past. The near reading is therefore widened across the angle a body of
    half-width :math:`r_s = w/2 + m` subtends at that range,

    .. math:: \alpha = \arctan(r_s / d_{near}),\qquad
              n = \lceil \alpha / |\Delta\theta| \rceil

    and the beams on the far side are overwritten with
    :math:`\tilde d_j = \min(\tilde d_j, d_{near})` -- never upwards, so
    several disparities compose and the result is a lower bound.

    ``arctan`` and not ``arcsin``: :math:`r_s` is a *lateral* offset at range
    :math:`d`, and the angle such an offset subtends is
    :math:`\arctan(r_s/d)`. :math:`\arcsin` is the tangent-to-a-circle
    version of the same correction; the two agree to 2 % while
    :math:`r_s/d < 0.35` and ``arctan`` has no domain limit, which is why it
    is the default. ``geometry='asin'`` gives the circle version, and treats
    :math:`d_{near} \le r_s` -- the car cannot pass on that side at all --
    by blocking the whole half-scan rather than producing a NaN.

    Conservative handling, stated rather than implied. A non-finite or
    non-positive range is an invalid measurement, not an obstacle at 0 m: it
    reads as 0 so it can never win the argmax, and it is **skipped** when
    looking for edges. Both halves matter. Treating a dropout as a real
    reading of 0 m made it a disparity with :math:`d_{near}\approx 0`, whose
    inflation is :math:`\arctan(r_s/0) \to 90^\circ` on each side -- so one
    dropped beam blanked half the scan and threw the target 22 deg off, which
    is measurably what the first version of this did. Beam spacing comes from
    the actual angles, so a scan at a different resolution is widened by its
    own spacing rather than an assumed constant.

    Factored out of :func:`free_space_heading`, which now calls it, so the
    features built on the extended scan and the heading the scripted
    gap-follower drives on are one piece of arithmetic.
    """
    ang = np.asarray(angles, dtype=float)
    r = np.asarray(ranges, dtype=float)
    r = np.where(np.isfinite(r) & (r > 0.0), r, 0.0)
    n = len(ang)
    if n < 3:
        return r.copy()
    ok = np.isfinite(np.asarray(ranges, dtype=float)) & (r > 0.0)
    r_s = float(car_half_width) + float(margin)
    tau = 2 * float(car_half_width) if disparity is None else float(disparity)
    d_ang = float(ang[-1] - ang[0]) / (n - 1)
    ext = r.copy()
    for i in range(n - 1):
        if not (ok[i] and ok[i + 1]):
            continue
        if abs(r[i + 1] - r[i]) < tau:
            continue
        near, far = (i, i + 1) if r[i] < r[i + 1] else (i + 1, i)
        d_near = max(r[near], 1e-3)
        if geometry == "asin":
            a = (np.pi / 2) if d_near <= r_s else float(np.arcsin(r_s / d_near))
        else:
            a = float(np.arctan2(r_s, d_near))
        k = int(np.ceil(a / max(d_ang, 1e-6)))
        lo, hi = ((far, min(far + k + 1, n)) if far > near
                  else (max(far - k, 0), far + 1))
        ext[lo:hi] = np.minimum(ext[lo:hi], r[near])
    return ext


def disparity_target(ranges: np.ndarray, angles: np.ndarray,
                     car_half_width: float = 0.20, margin: float = 0.0,
                     disparity: float | None = None,
                     geometry: str = "atan",
                     gap_frac: float = 0.9) -> DisparityTarget:
    """Step 4 of the extender, plus the two numbers it discards.

    ``width`` is the contiguous angular run containing the target over which
    the extended scan stays within ``gap_frac`` of ``d_safe``. That is the
    stated clearance criterion -- a *relative* one, so it means the same on a
    5 m straight and a 2 m corner -- and it is an **angular** gap width. The
    metric width of the opening is about ``2 * d_safe * sin(width / 2)`` and
    is a different quantity; nothing here reports it.
    """
    ang = np.asarray(angles, dtype=float)
    n = len(ang)
    if n < 3:
        return DisparityTarget(0.0, 0.0, 0.0,
                               np.asarray(ranges, dtype=float).copy())
    ext = disparity_extend(ranges, ang, car_half_width, margin,
                           disparity, geometry)
    # Break ties toward straight ahead. On an open stretch every beam reads
    # max range, every beam ties for furthest, and a bare argmax returns index
    # zero -- which is hard left. The feature would then tell a policy to turn
    # hard left on every straight, and the scripted gap-follower would do it.
    # Ties are the common case on a wide track, not an edge case.
    best = float(ext.max())
    tied = np.flatnonzero(ext >= best - 1e-9)
    j = int(tied[int(np.argmin(np.abs(ang[tied])))])
    thr = gap_frac * best
    lo = j
    while lo > 0 and ext[lo - 1] >= thr:
        lo -= 1
    hi = j
    while hi < n - 1 and ext[hi + 1] >= thr:
        hi += 1
    return DisparityTarget(float(ang[j]), best,
                           abs(float(ang[hi] - ang[lo])), ext)


class GapFollower:
    """Follow-the-gap with a disparity extender --- the F1TENTH reactive standard.

    ``WallFollower`` steers toward whichever side has more room. That is a
    stable rule on an open corner and the wrong one on a tight one: it aims at
    the *balance* of free space rather than at a direction the car can take, so
    on a hairpin, where both sides are close, it splits the difference and
    drives at the apex.

    This aims at the furthest point instead, after closing the gaps a finite
    beam count opens up. The disparity step is the part that matters: where two
    adjacent beams differ by more than the car is wide, the far reading belongs
    to something behind an edge the car cannot fit past, so the near reading is
    extended across the angular half-width of the car. Without it a
    gap-follower drives confidently into door frames.

    Two thresholds and no state, like the wall-follower, so a comparison
    between them is about the rule and not about the tuning.
    """

    def __init__(self, car_half_width: float = 0.20, slow_below: float = 1.2,
                 hard_below: float = 0.55, max_range: float = 5.0,
                 beam_angles=None):
        self.car_half_width = float(car_half_width)
        self.slow_below, self.hard_below = float(slow_below), float(hard_below)
        self.max_range = float(max_range)
        self.beam_angles = np.asarray(BEAM_ANGLES if beam_angles is None
                                      else beam_angles, dtype=float)

    def __call__(self, obs: np.ndarray) -> int:
        ang = self.beam_angles
        n = len(ang)
        r = np.asarray(obs[:n], dtype=float) * self.max_range
        if n < 3:
            return _act(0, 1)
        aim = free_space_heading(r, ang, self.car_half_width)
        steer = 1 if aim > 0.08 else (-1 if aim < -0.08 else 0)
        ahead = float(r[np.argmin(np.abs(ang))])
        throttle = 1 if ahead > self.slow_below else (-1 if ahead < self.hard_below else 0)
        return _act(steer, throttle)

    def reset(self) -> None:
        pass


class Overtaker(WallFollower):
    """Wall-follower, plus: when a *car* blocks the middle, pick a side and hold it.

    Holding the side is the part that is easy to get wrong, and it is the same
    trap as in ``scuderia_gym_jax``'s ``examples/overtake.py``: if the side is
    recomputed every frame, then the moment the ego draws level with the car it
    is passing, "go round the free side" flips, and the ego steers back into
    it. So the side is latched when the manoeuvre starts and released only once
    the middle beams are clear again.

    What it cannot do -- and deliberately is not patched to do -- is judge
    whether it is closing at 0.3 m/s or 2 m/s, because one frame does not say.
    That is why it still ends up in the back of somebody about half the time.
    """

    def __init__(self, engage_range: float = 0.8, **kw):
        super().__init__(**kw)
        self.engage_range = engage_range
        self.side = 0

    def reset(self) -> None:
        self.side = 0

    def __call__(self, obs: np.ndarray) -> int:
        r, is_car = obs[:N_BEAMS], obs[N_BEAMS:2 * N_BEAMS] > 0.5
        blocked = (is_car[3:6] & (r[3:6] < self.engage_range)).any()
        if not blocked:
            self.side = 0
            return super().__call__(obs)
        if self.side == 0:  # commit once, to the side that is car-free and open
            free_l = 0.0 if is_car[6:].any() else r[6:].min()
            free_r = 0.0 if is_car[:3].any() else r[:3].min()
            self.side = 1 if free_l >= free_r else -1
        throttle = -1 if r[4] < 0.25 else 0  # tuck in if the gap is collapsing
        return _act(self.side, throttle)


SCRIPTED = {"lanekeep": WallFollower, "overtake": Overtaker}
