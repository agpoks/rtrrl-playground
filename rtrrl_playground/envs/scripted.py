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

import numpy as np

N_BEAMS = 9


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

    def __init__(self, slow_below: float = 0.55, hard_below: float = 0.30):
        self.slow_below = slow_below
        self.hard_below = hard_below

    def __call__(self, obs: np.ndarray) -> int:
        r = obs[:N_BEAMS]
        left, right = r[6:].sum(), r[:3].sum()
        steer = 1 if left > right + 0.02 else (-1 if right > left + 0.02 else 0)
        ahead = r[4]
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
