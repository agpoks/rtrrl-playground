"""Overtake -- "learn to overtake": the same car, plus traffic that never yields.

:class:`~rtrrl_playground.envs.lanekeep.LaneKeep` with ``n_opponents`` slower
cars holding the centreline at a constant speed. They do not brake, do not
steer, and do not care that you are there -- exactly the dumb traffic in
``scuderia_gym_jax``'s ``examples/overtake.py``, for the same reason: if the
traffic reacts, you can no longer tell whose behaviour you are looking at.

The observation grows by one channel per beam: **is this return a wall or a
car?** Without that flag a car two metres ahead and a wall two metres ahead
are the same observation, and no amount of recurrence recovers a distinction
the sensor never made. With it, the agent has a lidar plus the world's
crudest object classifier, which is a fair caricature of what the real car
has after ``obstacle_perception``.

What is still hidden is the thing that matters most: **how fast the car
ahead is going**. Its speed is drawn fresh each episode from
``opp_speed_range`` and never observed, so a closing rate can only be
obtained by integrating the range over time. That is a genuine, non-toy
reason to need memory here -- committing to a pass depends entirely on
whether you are closing at 0.3 m/s or 2 m/s, and one frame cannot tell you.

Reward: the same arc-length progress as LaneKeep, ``+2`` each time the ego's
arc length passes an opponent's, ``-5`` and episode over on contact.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.envs.lanekeep import BEAM_ANGLES, BEAM_RANGE, BEAM_STEP, LaneKeep, SPEED_MAX
from rtrrl_playground.spaces import Box, Discrete

CAR_RADIUS = 0.22  # m, half a 0.44 m car -- contact when the centres are closer than 2x this
SEE_RADIUS = 0.30  # m, the radius a beam has to pass within to register a car
OVERTAKE_BONUS = 2.0
CRASH_PENALTY = -5.0


class Overtake(LaneKeep):
    id = "overtake"

    def __init__(self, n_opponents: int = 2, opp_speed_range=(1.5, 2.3),
                 opp_gap: float = 6.0, opp_lateral: float = 0.0,
                 track: str = "oval", action_mode: str = "discrete",
                 observe_speed: bool = False, half_width: float = 0.95,
                 grip_range=(0.6, 1.4), dt: float = 0.05, max_steps: int = 900,
                 start_jitter: float = 0.2, seed: int | None = None):
        # A wider track than LaneKeep's, and not by accident: a pass needs the
        # two cars to be more than 2*CAR_RADIUS apart laterally while both stay
        # inside the boundary, which 0.75 m of half-width does not comfortably
        # allow. Narrow it back down if you want to force the agent to use the
        # opponent's slipstream and wait for a corner exit instead.
        super().__init__(track=track, action_mode=action_mode,
                         observe_speed=observe_speed, half_width=half_width,
                         grip_range=grip_range, dt=dt, max_steps=max_steps,
                         start_jitter=start_jitter, seed=seed)
        self.n_opponents = int(n_opponents)
        self.opp_speed_range = tuple(opp_speed_range)
        self.opp_gap = float(opp_gap)
        self.opp_lateral = float(opp_lateral)
        # beams + "is it a car" flag per beam (+ speedometer, if enabled)
        self.obs_dim = 2 * self.n_beams + int(self.observe_speed)
        self.action_space = Discrete(9) if action_mode == "discrete" else Box(2)

    # -- opponents --------------------------------------------------------
    def _place_opponents(self) -> None:
        """Opponent centres ``(M, 2)`` from their arc lengths, cached.

        Three different consumers want this each step -- the beams, the contact
        check and the renderer -- and recomputing it for each of them was, at
        one point, the single most expensive thing in the environment.
        """
        k = (self._opp_s / self.track.ds).astype(int) % self.track.K
        self._opp_pos = self.track.center[k] + self._opp_d[:, None] * self.track.normal[k]

    def _opp_xy(self) -> np.ndarray:
        return self._opp_pos

    def _obs(self, extra_obstacles=None) -> np.ndarray:
        xy = self._opp_xy()
        ranges, flags = self.track.beam_ranges(
            self.x, self.y, self.psi, BEAM_ANGLES,
            max_range=BEAM_RANGE, step=BEAM_STEP,
            obstacles=xy, obs_radius=SEE_RADIUS,
        )
        self._last_beams = (ranges, flags)
        parts = [ranges / BEAM_RANGE, flags]
        if self.observe_speed:
            parts.append(np.array([self.v / SPEED_MAX]))
        return np.concatenate(parts)

    # -- Env --------------------------------------------------------------
    def _reset_extras(self) -> None:
        m = self.n_opponents
        # Queued up ahead of the ego along the racing line, in the ego's own
        # frame -- so where the ego happens to start on the lap does not change
        # the task, only the corner it meets the first car in.
        self._opp_s = (self._s + self.opp_gap * np.arange(1, m + 1)) % self.track.length
        self._opp_v = self._rng.uniform(*self.opp_speed_range, size=m)
        self._opp_d = np.full(m, self.opp_lateral)
        self._place_opponents()
        self._gap = self._gaps()
        self.overtakes = 0
        self.crashed = False

    def _gaps(self) -> np.ndarray:
        """Signed arc-length gap to each opponent; positive means ahead of us."""
        delta = (self._opp_s - self._s) % self.track.length
        return np.where(delta > self.track.length / 2, delta - self.track.length, delta)

    def step(self, action):
        self._opp_s = (self._opp_s + self._opp_v * self.dt) % self.track.length
        self._place_opponents()
        obs, reward, terminated, truncated, info = super().step(action)

        gap = self._gaps()
        # A pass is the ego's arc length crossing an opponent's from behind:
        # the gap goes from positive (ahead of me) to negative (behind me).
        # Guard on |gap| so that the wrap-around half a lap away, where the
        # sign also flips, is not paid out as an overtake.
        passed = (self._gap > 0) & (gap < 0) & (np.abs(gap) < self.track.length / 4)
        n_passed = int(passed.sum())
        self.overtakes += n_passed
        reward += OVERTAKE_BONUS * n_passed
        self._gap = gap

        if not terminated:
            dist = np.linalg.norm(self._opp_xy() - np.array([self.x, self.y]), axis=1)
            if float(dist.min()) < 2 * CAR_RADIUS:
                reward += CRASH_PENALTY
                terminated, self.crashed = True, True
                obs = np.zeros(self.obs_dim)
                truncated = False

        if self.history:
            self.history[-1]["opp"] = self._opp_xy().copy()
        info.update(overtakes=self.overtakes, crashed=self.crashed,
                    min_gap=float(np.abs(gap).min()))
        return obs, float(reward), terminated, truncated, info

    def render_rollout(self, history=None, path: str = "rollout.png", title: str = ""):
        return super().render_rollout(
            history, path,
            title or f"overtake / {self.track_name} -- {self.overtakes} passes"
                     f"{', crashed' if self.crashed else ''}")
