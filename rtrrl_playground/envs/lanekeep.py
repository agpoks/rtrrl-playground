"""LaneKeep -- "learn to drive": hold a 1:10 car on a closed track, fast.

A kinematic bicycle on the tracks in :mod:`rtrrl_playground.envs.track`, at
20 Hz, with RC-scale numbers (0.33 m wheelbase, 0.4 rad of lock, 4 m/s top
speed). The agent gets five lidar-style beams and nothing else, and is paid
for arc length covered along the centreline.

What makes it a POMDP -- and this is the entire reason it is in this repo --
is what the beams *cannot* tell you. Two things, and the second was added
after the first turned out not to be enough:

**Speed.** Nine ranges to a wall pin down where the car is and which way it is
pointing, and say nothing about how fast it is arriving. Two cars, one
crawling and one on the limit, produce byte-identical observations.

**Grip.** Redrawn every episode from ``grip_range``, never observed. At grip
0.6 the corner limit is 3.0 m/s; at 1.4 it is 4.6 m/s, and the difference is
worth about a hundred points of return. The only way to find out which one you
are on is to drive: ask for more lateral acceleration than the tyres have and
the car runs wide.

**Be honest about how much that buys.** It is not enough to make lanekeep a
task that *requires* memory, and the measurement is in the benchmark table: a
memoryless policy still does well here, because nine beams are a lot of
information and reacting to the forward one is a decent speed controller all
by itself. Lanekeep is where you check that an agent can drive at all.
``overtake`` is where memory earns its keep -- there the missing quantity is
another car's speed, and there is no beam that implies it.

Set ``observe_speed=True`` to hand it the speedometer and turn the task back
into an MDP. That ablation is worth running: it is the control that shows a
memoryless agent's failure here is about the missing state and not about the
learning rule.

Two more things are hidden and worth knowing about, because both give the
recurrence something real to integrate:

* the **steering servo lags** (first-order, 80 ms), so the commanded steering
  angle and the actual one differ during a transient; and
* the beams are **quantised to the ray-march step**, so a smooth approach to
  a wall arrives as a staircase; and
* the **grip** is redrawn every episode from ``grip_range`` and never
  observed, so the speed a corner can be taken at is a property of *this run*
  that has to be discovered by driving.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.envs.track import Track, TRACKS
from rtrrl_playground.envs.vehicle import VehicleParams
from rtrrl_playground.spaces import Box, Discrete, Env

# --- 1:10 RC scale, roughly a Traxxas Slash on a club track -----------------
# These are the defaults of VehicleParams, re-exported as plain floats because
# several modules want them that way. The *env* takes a VehicleParams object,
# so a second vehicle with different numbers is one argument away -- which is
# what tutorial/11 is built on.
_P = VehicleParams()
WHEELBASE = _P.wheelbase
STEER_MAX = _P.steer_max
STEER_TAU = _P.steer_tau
ACCEL_MAX = _P.accel_max
SPEED_MAX = _P.speed_max
DRAG = _P.drag
A_LAT_MAX = _P.a_lat_max

# Beams point right-to-left: index 0 is 60 degrees to starboard, index 8 is 60
# to port, index 4 is straight ahead. Worth stating once, because every scripted
# policy in this repo and every plot of an observation depends on the order.
#
# Nine beams over 120 degrees, not five over 140: the spacing has to be fine
# enough that a car cannot sit *between* two rays. Adjacent beams are 15 deg
# apart, so at 2.5 m they are 0.65 m apart -- about one car's detection
# diameter. With five beams the gap at that range was 1.5 m and traffic
# genuinely disappeared between rays, which is a sensor bug masquerading as a
# hard exploration problem.
BEAM_ANGLES = np.deg2rad(np.linspace(-60.0, 60.0, 9))
BEAM_RANGE = 5.0
BEAM_STEP = 0.15


class LaneKeep(Env):
    id = "lanekeep"

    def __init__(self, track: str = "oval", action_mode: str = "discrete",
                 observe_speed: bool = False, half_width: float = 0.75,
                 grip_range=(0.6, 1.4), vehicle: VehicleParams | None = None,
                 dt: float = 0.05, max_steps: int = 600,
                 start_jitter: float = 0.3, seed: int | None = None):
        if action_mode not in ("discrete", "continuous"):
            raise ValueError("action_mode must be 'discrete' or 'continuous'")
        self.track: Track = (TRACKS[track](half_width=half_width)
                             if isinstance(track, str) else track)
        self.track_name = track if isinstance(track, str) else "custom"
        self.action_mode = action_mode
        self.observe_speed = bool(observe_speed)
        # Grip is redrawn every episode and never observed. Without this the
        # task is reactive: the forward beam alone tells you the corner radius,
        # the corner radius tells you the safe speed, and a memoryless policy
        # matches a hand-written wall-follower. With it, the safe speed depends
        # on a number the agent can only find out by *driving* -- noticing that
        # the car ran wider than the steering asked for. That is a real thing
        # about RC cars (surface, tyre temperature, battery sag) and it is what
        # turns lanekeep from a demonstration into a task that needs memory.
        # Set grip_range=(1.0, 1.0) to switch it off and get the reactive
        # version back.
        self.grip_range = (float(grip_range[0]), float(grip_range[1]))
        self.vehicle = vehicle or VehicleParams()
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.start_jitter = float(start_jitter)
        self.n_beams = len(BEAM_ANGLES)
        self.obs_dim = self.n_beams + int(self.observe_speed)
        # 3 steering choices x 3 throttle choices. A flat 9-way softmax rather
        # than two heads: one categorical distribution is one gradient to derive.
        self.action_space = Discrete(9) if action_mode == "discrete" else Box(2)
        self._rng = np.random.default_rng(seed)
        self._reset_state()

    # -- state ------------------------------------------------------------
    def _reset_state(self):
        self.grip = 1.0
        self.x = self.y = self.psi = self.v = self.delta = 0.0
        self._k = 0
        self._s = 0.0
        self._t = 0
        self._stalled = 0
        self.history: list[dict] = []

    def _decode(self, action) -> tuple[float, float]:
        """Action -> (steer, throttle), both in [-1, 1]."""
        if self.action_mode == "discrete":
            a = int(action)
            return float(a // 3 - 1), float(a % 3 - 1)
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return float(a[0]), float(a[1])

    def _obs(self, extra_obstacles=None) -> np.ndarray:
        ranges, flags = self.track.beam_ranges(
            self.x, self.y, self.psi, BEAM_ANGLES,
            max_range=BEAM_RANGE, step=BEAM_STEP, obstacles=extra_obstacles,
        )
        ranges = self._corrupt(ranges)
        parts = [ranges / BEAM_RANGE]
        if self.observe_speed:
            parts.append(np.array([self.v / self.vehicle.speed_max]))
        self._last_beams = (ranges, flags)
        return np.concatenate(parts)

    def _corrupt(self, ranges: np.ndarray) -> np.ndarray:
        """Sensor defects, applied to the beams before the agent sees them.

        Both default to off. A dropped beam returns ``BEAM_RANGE`` -- the same
        value it returns when it genuinely sees nothing -- because that is what
        a real lidar does, and it is what makes dropout worth simulating: the
        agent cannot tell a miss from an empty corridor.
        """
        p = self.vehicle
        if p.beam_noise > 0:
            ranges = ranges + self._rng.normal(0.0, p.beam_noise, ranges.shape)
        if p.beam_dropout > 0:
            miss = self._rng.random(ranges.shape) < p.beam_dropout
            ranges = np.where(miss, BEAM_RANGE, ranges)
        return np.clip(ranges, 0.0, BEAM_RANGE)

    # -- Env --------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._reset_state()
        k = int(self._rng.integers(self.track.K))
        d0 = self._rng.uniform(-1, 1) * self.start_jitter * self.track.half_width
        psi0 = self.track.heading[k] + self._rng.uniform(-1, 1) * self.start_jitter
        p = self.track.center[k] + d0 * self.track.normal[k]
        self.x, self.y, self.psi = float(p[0]), float(p[1]), float(psi0)
        self.v = 1.0
        self.delta = 0.0
        self._k = k
        self._s = float(self.track.s[k])
        self.grip = float(self._rng.uniform(*self.grip_range))
        self._reset_extras()
        return self._obs()

    def _reset_extras(self) -> None:
        """Hook for subclasses that add world state (see envs/overtake.py).

        Called after the ego is placed and before the first observation, so a
        subclass can put its traffic on the track in time to be seen by it."""

    def _integrate(self, steer: float, throttle: float):
        """One 20 Hz control tick of the kinematic bicycle, with a grip limit.

        The kinematic bicycle on its own has no notion of a tyre, so it will
        take any corner at any speed and the throttle choice is free -- a
        scripted wall-follower laps this track flat out. That makes for a
        useless driving task, so the yaw rate is capped at the rate that
        ``A_LAT_MAX`` of lateral acceleration can actually produce:

            psi_dot = min(v/L * tan(delta), A_LAT_MAX / v)

        which is understeer in its crudest form -- ask for more lateral
        acceleration than the tyres have and you simply do not get it, and the
        car runs wide. It is not a slip-angle model (that is what
        ``scuderia_gym_jax``'s ST/STD models are for); it is the one line that
        makes "how fast can I take this corner" a real question, and therefore
        makes the throttle half of the action space worth learning.
        """
        p = self.vehicle
        # steer_bias is a servo trim that is not quite centred: the commanded
        # zero is not the car's zero. It is the single most common real defect
        # and the one a policy trained in simulation has never seen.
        self.delta += (steer * p.steer_max + p.steer_bias - self.delta) * self.dt / p.steer_tau
        self.v += (throttle * p.accel_max * p.throttle_scale - p.drag * self.v) * self.dt
        self.v = float(np.clip(self.v, 0.0, p.speed_max))
        psi_dot = self.v / p.wheelbase * np.tan(self.delta)
        if self.v > 1e-3:
            grip_limit = p.a_lat_max * self.grip / self.v
            psi_dot = float(np.clip(psi_dot, -grip_limit, grip_limit))
        self.x += self.v * np.cos(self.psi) * self.dt
        self.y += self.v * np.sin(self.psi) * self.dt
        self.psi += psi_dot * self.dt
        self.yaw_rate = psi_dot

    def step(self, action):
        steer, throttle = self._decode(action)
        self._integrate(steer, throttle)
        s, d, k = self.track.frenet(self.x, self.y, self._k, half=24)
        progress = self.track.ds_forward(s, self._s)
        self._s, self._k = s, k
        self._t += 1

        self.history.append(dict(x=self.x, y=self.y, psi=self.psi, v=self.v, d=d))

        off_track = abs(d) > self.track.half_width
        self._stalled = self._stalled + 1 if self.v < 0.2 else 0
        stalled = self._stalled > int(2.0 / self.dt)  # 2 s of not driving

        # Progress, in units of "a full-speed step". Bounded in [-1, 1], which
        # keeps the TD error on the same scale as the crash penalty below and
        # spares us a reward-normalisation knob nobody would enjoy tuning.
        reward = progress / (SPEED_MAX * self.dt)
        terminated = False
        if off_track:
            reward, terminated = -1.0, True
        elif stalled:
            terminated = True
        truncated = bool(self._t >= self.max_steps and not terminated)
        obs = self._obs() if not terminated else np.zeros(self.obs_dim)
        info = {"s": s, "d": d, "v": self.v, "grip": self.grip,
                "off_track": off_track, "stalled": stalled, "laps": 0.0}
        return obs, float(reward), terminated, truncated, info

    # -- pictures ---------------------------------------------------------
    def render_rollout(self, history=None, path: str = "rollout.png", title: str = ""):
        """Save the track and one episode's line, coloured by speed."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist = history if history is not None else self.history
        c, n, hw = self.track.center, self.track.normal, self.track.half_width
        inner, outer = c - hw * n, c + hw * n
        fig, ax = plt.subplots(figsize=(7, 5))
        for b in (inner, outer):
            ax.plot(*np.vstack([b, b[:1]]).T, color="0.35", lw=1.2)
        ax.plot(*np.vstack([c, c[:1]]).T, color="0.8", lw=0.8, ls="--")
        if hist:
            xs = np.array([h["x"] for h in hist])
            ys = np.array([h["y"] for h in hist])
            vs = np.array([h["v"] for h in hist])
            sc = ax.scatter(xs, ys, c=vs, s=4, cmap="viridis", vmin=0, vmax=SPEED_MAX)
            fig.colorbar(sc, ax=ax, label="speed [m/s]")
            ax.plot(xs[0], ys[0], "o", color="tab:green", ms=7, label="start")
            ax.plot(xs[-1], ys[-1], "x", color="tab:red", ms=8, label="end")
            # Traffic, if this rollout had any (see envs/overtake.py). Drawn as
            # a faint trail plus a marker at the end, so a pass reads as the
            # ego line crossing a trail rather than as two dots in space.
            if "opp" in hist[-1]:
                opp = np.stack([h["opp"] for h in hist if "opp" in h])  # (T, M, 2)
                for j in range(opp.shape[1]):
                    ax.plot(opp[:, j, 0], opp[:, j, 1], color="tab:orange", lw=0.7, alpha=0.5)
                ax.plot(opp[-1, :, 0], opp[-1, :, 1], "s", color="tab:orange",
                        ms=6, label="traffic")
            ax.legend(loc="upper right", fontsize=8)
        ax.set_aspect("equal")
        ax.set_title(title or f"{self.id} / {self.track_name}")
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
