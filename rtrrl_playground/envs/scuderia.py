"""An adapter that puts the agents in this repo onto ``scuderia_gym_jax``.

The two driving environments here are toys, and deliberately so: a kinematic
bicycle with a crude grip limit, a bitmap track, and a nine-beam sensor, all
sized so a lesson finishes on a laptop in a minute. The real vehicle models --
ST, STD, STD4W, with Pacejka or brush tyres fitted to actual RC-car recordings
-- live in `scuderia_gym_jax <https://github.com/agpoks/scuderia_gym_jax>`_,
and that is where a result should eventually be reported.

This is the bridge. It wraps a ``ScuderiaEnvJax`` in the same eight-line
:class:`~rtrrl_playground.spaces.Env` interface everything else here speaks, so
the agent code does not change at all::

    from rtrrl_playground.envs.scuderia import ScuderiaLaneKeep
    env = ScuderiaLaneKeep(model="st", map_name="berlin")
    agent = RTRRL(env.obs_dim, env.action_space, cell="lrcu")

Three things are worth knowing before trusting a number that comes out of it:

**The reward is distance travelled, not progress along a racing line.** The
maps that ship with the simulator are occupancy images with no centreline, so
there is no arc length to differentiate. Distance-without-crashing is the
standard stand-in for f1tenth-style RL baselines and it is a genuinely
different objective -- it will happily reward a fast lap of a small loop inside
a wide corridor. If you want the real thing, bring a centreline and replace
:meth:`ScuderiaLaneKeep._reward`.

**One environment, one step at a time.** ``scuderia_gym_jax`` is built to be
``vmap``ped over thousands of cars and ``scan``ned over whole rollouts without
Python in the loop; driving it one step at a time from a Python agent gives up
almost all of that. RTRRL is a batch-size-one algorithm, so there is nothing to
vmap over -- but it does mean the simulator is being used against its grain,
and a step here costs far more than a step of ``lanekeep``.

**The velocity is still hidden.** ``get_obs`` in the simulator returns the full
state vector, which includes the speed; this adapter throws that away and
returns downsampled lidar only, because a POMDP is the point. Pass
``observe_speed=True`` to keep it and turn the task Markov.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.envs.vehicle import VehicleParams
from rtrrl_playground.spaces import Box, Discrete, Env

STEER_MAX = 0.4  # rad, the steering-angle setpoint the simulator's PID takes
SPEED_MAX = 4.0  # m/s
#: How many of the simulator's beams reach the agent, by default. Nine is what
#: ``lanekeep`` gives and it is what every earlier result here was measured on
#: -- but it is a *small fraction* of the real sensor, and measured on a track
#: whose corners are tighter than the car's turning circle it costs a factor of
#: four in distance covered: 0.16 laps at 9 beams over 120 degrees against 0.61
#: at 61 over 270. Raise ``n_beams_out`` for anything that has to place the car
#: on a line rather than merely avoid a wall.
N_BEAMS_OUT = 9
BEAM_RANGE = 10.0


class ScuderiaLaneKeep(Env):
    """Drive a ``scuderia_gym_jax`` car on one of its shipped maps."""

    id = "scuderia-lanekeep"

    def __init__(self, model: str = "st", map_name: str = "berlin",
                 map_ext: str = ".png", tire_model: int | None = None,
                 action_mode: str = "discrete",
                 observe_speed: bool = False, num_beams: int = 108,
                 n_beams_out: int = N_BEAMS_OUT, centreline=None,
                 half_width=None, width_scale: float = 1.0,
                 max_steps: int = 2000, control_repeat: int = 5,
                 start_pose=(0.0, 0.0, 0.0), grip: float | None = None,
                 track_half_width: float | None = None,
                 seed: int = 0, **make_kwargs):
        try:
            import jax
            import jax.numpy as jnp
            import scuderia_gym_jax as sgj
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "ScuderiaLaneKeep needs scuderia_gym_jax and its dependencies:\n"
                "    pip install jax chex\n"
                "    pip install -e /path/to/scuderia_gym_jax\n"
                f"(the import failed with: {exc})"
            ) from exc
        self._jax, self._jnp = jax, jnp

        # tire_model is left to the car config unless asked for: the packed
        # parameter arrays carry a tyre code and ScuderiaEnvJax refuses a spec
        # that disagrees with them, which is the right call and an easy way to
        # get an unhelpful error out of an adapter that guessed.
        if tire_model is not None:
            make_kwargs["tire_model"] = tire_model
        # map_ext matters: the maps that ship with the simulator are a mix of
        # .png and .pgm, and the loader takes the extension as an argument
        # rather than looking. ``levine`` is a .pgm; ``berlin``, ``skirk``,
        # ``vegas`` and ``stata_basement`` are .png.
        self.env = sgj.make(model=model, num_agents=1, produce_scans=True,
                            num_beams=num_beams, map_name=map_name,
                            map_ext=map_ext, collision_on=True, **make_kwargs)
        self.model, self.map_name = model, map_name
        self.num_beams = num_beams
        self.control_repeat = int(control_repeat)  # simulator ticks per agent action
        self.dt = float(self.env.timestep) * self.control_repeat
        self.observe_speed = bool(observe_speed)
        self.action_mode = action_mode
        self.n_beams_out = int(n_beams_out)
        self.obs_dim = self.n_beams_out + int(observe_speed)
        self.action_space = Discrete(9) if action_mode == "discrete" else Box(2)
        self.max_steps = int(max_steps)
        self.start_pose = np.asarray(start_pose, dtype=float)
        self._key = jax.random.key(seed)
        # Which raw beams to keep. Evenly spaced across the full field of view,
        # so the nine numbers the agent sees mean the same thing they do in
        # envs/lanekeep.py: right to left, straight ahead in the middle.
        self._idx = np.linspace(0, num_beams - 1, self.n_beams_out).astype(int)
        # A centreline turns "distance travelled" into "progress along the
        # track", which is what lanekeep pays and therefore the only reward
        # under which a number here is comparable with one from there. Without
        # it the agent is rewarded for a fast lap of any loop inside a wide
        # corridor. Pass an (N, 2) array of metres.
        self.centreline = None if centreline is None else np.asarray(
            centreline, dtype=float)
        if self.centreline is not None:
            d = np.diff(np.vstack([self.centreline, self.centreline[:1]]), axis=0)
            self._cl_s = np.concatenate([[0.0], np.cumsum(np.hypot(d[:, 0], d[:, 1]))])
            self.lap_length = float(self._cl_s[-1])
        self._prev_s = None
        # ``half_width`` ends an episode on lateral deviation from the
        # centreline rather than on hitting something in the map, the way
        # ``lanekeep`` already does. It defaults to None -- the map-collision
        # behaviour below is unchanged -- because on a *shipped* map collision
        # is a fine terminator.
        #
        # It is not fine on an imported one, which is the whole reason this
        # exists. A SLAM occupancy grid marks only the walls the lidar
        # actually saw: measured, 1.6 % of pixels on master_cup against ~30 %
        # for a shipped map. A car that leaves the circuit then drives into
        # open space, never collides, is never terminated and earns nothing --
        # so the run looks like training and teaches nothing. A logical
        # boundary is immune to holes, unclosed walls and the edge of the
        # image, and it is the same quantity a real car's tracking error is
        # measured against, which is why it also transfers to hardware.
        # ``half_width`` takes a scalar or one value per centreline point. Per
        # point is the honest form: a real corridor is not a constant width,
        # and collapsing it to one number has to pick between two bad options
        # -- the narrowest, which terminates a car that is comfortably on
        # track wherever the track is wide, or the median, which lets it leave
        # the track entirely at the pinch points. Measured on master_cup, the
        # per-point half-width runs 0.37 m to 0.78 m, and the 5th percentile
        # (0.54 m) ended a *perfectly tracking* pure-pursuit run in 86 steps.
        #
        # ``width_scale`` multiplies whatever was passed. It exists for
        # training rather than for geometry: starting at 1.3 and annealing to
        # 1.0 gives a policy room to recover early without ever changing the
        # track the numbers are reported against. It does not move the walls.
        if half_width is not None and self.centreline is None:
            raise ValueError(
                "half_width terminates on distance from the centreline, so it "
                "needs one: pass centreline=(N, 2) metres alongside it.")
        if half_width is None:
            self.half_width = None
        else:
            hw = np.atleast_1d(np.asarray(half_width, dtype=float))
            if hw.size not in (1, len(self.centreline)):
                raise ValueError(
                    f"half_width must be a scalar or one value per centreline "
                    f"point ({len(self.centreline)}), got {hw.size}")
            self.half_width = hw * float(width_scale)
        self.width_scale = float(width_scale)
        # jit the bound method once. Without this every control tick re-enters
        # the tracer and a step costs tens of milliseconds -- the simulator is
        # designed to be jitted around a whole `lax.scan` rollout, and stepping
        # it from Python is exactly the usage that does not get that for free.
        self._step_env = self._jax.jit(self.env.step_env)
        self._state = None
        self._x = None
        self.history: list[dict] = []
        self._track = None
        self.track_half_width = (None if track_half_width is None
                                 else float(track_half_width))
        #: This environment terminates on the *per-point* lateral limit while a
        #: safety filter certifies against ``track``'s single scalar, so the two
        #: boundaries are set independently and the scalar is the tighter one.
        #: That leaves a band -- 0.49 m to a median 0.78 m on master_cup -- in
        #: which the episode is still alive and the filter can certify nothing,
        #: and a stopped car there cannot be recovered by braking it harder.
        #: ``lanekeep`` cannot reach such a state, which is why the filter's
        #: recovery is off by default and on here.
        self.filter_recovers_at_standstill = True
        # What a safety filter is told about the car. Most of ``VehicleParams``
        # already describes this vehicle -- 0.40 rad of lock, 4 m/s -- because
        # both simulators model the same 1:10 car. Two fields do not, and
        # neither is cosmetic:
        #
        # ``accel_max`` is 4.0 m/s^2 there and **1.0 m/s^2 here**, measured:
        # full throttle from rest gives 1.13 m/s^2 and full brake from 4 m/s
        # gives 1.00, because ``_decode`` ramps a speed *setpoint* at
        # ``throttle * 1.0 * dt`` and the simulator's PID tracks it. A filter
        # left at 4.0 believes it can shed speed four times faster than the car
        # can, certifies a backup plan that stops in a quarter of the distance,
        # and produces exactly the "crashes *through* the filter" that
        # ``bridges.safety.make_safe_agent`` warns about for optimistic grip.
        #
        # ``drag`` is 0.0 for the same reason: the measured deceleration was a
        # flat 1.00 m/s^2 from 4 m/s down, with no velocity-proportional term
        # (0.15/s would have added 0.6 m/s^2 at that speed and did not).
        #
        # ``wheelbase`` is the simulator's own 0.1705 + 0.1515.
        self.vehicle = VehicleParams(wheelbase=0.322, accel_max=1.0, drag=0.0)
        self.grip = self._read_grip() if grip is None else float(grip)

    def _read_grip(self) -> float:
        """The simulator's own friction coefficient, not a guess at it.

        ``mu`` is packed into the tyre parameter arrays rather than exposed as
        a field; ``simple[13]`` is the one ``scuderia_gym_jax``'s own
        ``tests/test_tire_parity.py`` reads, and on the shipped RC-10 config it
        agrees with ``st[0]``, ``brush[2]`` and ``dugoff[2]`` at 1.1. Reading it
        matters: a barrier certified at 1.0 against tyres worth 1.1 is
        conservative by 10 % everywhere, and one certified the other way round
        is confidently wrong -- see ``bridges.safety.make_safe_agent``'s note on
        a filter given a grip the road does not have.
        """
        try:
            return float(np.asarray(self.env.params.tire.simple).ravel()[13])
        except Exception:
            return 1.0

    # -- the dynamic state, under the names the rest of this project uses --
    #
    # The single-track state is ``[x, y, delta, v, psi, yaw_rate, beta]``,
    # verified against the running simulator rather than read off a docstring.
    # Sideslip and yaw rate are therefore *already here*; they were simply
    # never given names, and that alone is what kept
    # ``bridges.stability.StabilityTrigger`` and
    # ``bridges.safety.make_safe_agent`` off this environment -- both look for
    # ``env.beta`` / ``env.yaw_rate`` / ``env.delta`` / ``env.vx`` /
    # ``env.grip``, find nothing, and either never fire or refuse to attach.
    #
    # Nothing is cached: ``_x`` is refreshed by ``step``/``reset``, and these
    # are read a handful of times per tick against the ~12 ms the simulator
    # itself costs.
    _ST = dict(x=0, y=1, delta=2, v=3, psi=4, yaw_rate=5, beta=6)

    def _st(self, name: str) -> float:
        if self._x is None:
            raise RuntimeError("reset() this environment before reading its state")
        return float(self._x[0, self._ST[name]])

    @property
    def x(self) -> float:
        return self._st("x")

    @property
    def y(self) -> float:
        return self._st("y")

    @property
    def psi(self) -> float:
        return self._st("psi")

    @property
    def delta(self) -> float:
        """Steering angle, rad -- the servo's position, not the setpoint."""
        return self._st("delta")

    @property
    def v(self) -> float:
        """Velocity *magnitude*, m/s. ``vx`` is its longitudinal component."""
        return self._st("v")

    @property
    def vx(self) -> float:
        # beta is the angle between the velocity vector and the body axis, so
        # the longitudinal component is v cos(beta). At the sideslips this car
        # reaches the difference is under a percent -- but the phase-plane
        # barrier divides by this, and "close enough" is how a barrier ends up
        # certified against a speed the car is not doing.
        return self._st("v") * float(np.cos(self._st("beta")))

    @property
    def yaw_rate(self) -> float:
        return self._st("yaw_rate")

    @property
    def beta(self) -> float:
        """Sideslip angle, rad."""
        return self._st("beta")

    @property
    def track(self):
        """A :class:`~rtrrl_playground.envs.track.Track`, for a safety filter.

        Built once, on demand, from the centreline this adapter was given.
        ``Track`` carries a *scalar* half-width while this environment may hold
        one per point, so one number has to stand for the corridor, and the
        choice is not cosmetic. The filter's feasible set has a hard edge at
        ``half_width - margin``: certifiability does not degrade across it, it
        falls off a cliff. Measured on master_cup at 1.5 m/s, over 30 points
        round the lap, as certifiable actions out of nine:

        ==========  ======  ======  ======  ======
        filter hw   d=0.3   d=0.4   d=0.5   d=0.6
        ==========  ======  ======  ======  ======
        0.369 min      8.9     0.0     0.0     0.0
        0.537 p5       9.0     9.0     0.1     0.0
        0.781 median   9.0     9.0     9.0     9.0
        ==========  ======  ======  ======  ======

        So the narrowest value is the *worst* default, not the safest one. This
        environment terminates on the per-point width -- a median of 0.78 m --
        and a filter pinned at the 0.37 m pinch point declares a car
        unrecoverable while the task is still perfectly happy with it. It then
        brakes, the backup controller drives, and with ``credit="executed"`` the
        learner is trained on the backup rather than on its own proposal, which
        is the bias ``experiments/trigger_arms.py``'s revision (1) exists to
        record. Over-conservatism here does not buy safety; it buys a filter
        that is always on.

        The default is therefore the 5th percentile -- the same statistic
        ``bridges.mapimport.save_centreline`` writes into the file's header as
        *the* half-width, so the filter's boundary is the number the track is
        described by. Pass ``track_half_width`` for anything else.

        ``None`` without a centreline, so ``make_safe_agent`` raises its own
        clear error instead of certifying against a track that is not there.
        """
        if self.centreline is None:
            return None
        if self._track is None:
            from rtrrl_playground.envs.track import Track
            if self.track_half_width is not None:
                hw = float(self.track_half_width)
            elif self.half_width is None:
                hw = 1.1
            else:
                hw = float(np.percentile(self.half_width, 5))
            self._track = Track(self.centreline[:, 0], self.centreline[:, 1],
                                half_width=hw)
        return self._track

    # -- helpers ----------------------------------------------------------
    def _split(self):
        self._key, sub = self._jax.random.split(self._key)
        return sub

    def _decode(self, action) -> np.ndarray:
        """Action -> ``[steering angle, speed setpoint]``, the simulator's input."""
        if self.action_mode == "discrete":
            a = int(action)
            steer, throttle = a // 3 - 1, a % 3 - 1
        else:
            a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
            steer, throttle = float(a[0]), float(a[1])
        self._v_cmd = float(np.clip(self._v_cmd + throttle * 1.0 * self.dt, 0.0, SPEED_MAX))
        return np.array([steer * STEER_MAX, self._v_cmd])

    def _pull(self, state):
        """Bring one step's worth of device arrays across, in two transfers.

        Indexing a JAX array (``state.x[0, 3]``) is a *traced operation* that
        dispatches to the device and syncs back; doing that once per field
        costs milliseconds per step and was, measured, the single most
        expensive thing in this adapter. Pull the whole array once, index the
        NumPy copy afterwards.
        """
        return np.asarray(state.x), np.asarray(state.scans)

    def _obs_from(self, x: np.ndarray, scans: np.ndarray) -> np.ndarray:
        beams = np.clip(scans[0][self._idx] / BEAM_RANGE, 0.0, 1.0)
        if self.observe_speed:
            return np.concatenate([beams, [x[0, 3] / SPEED_MAX]])
        return beams

    def _reward(self, x: np.ndarray, x_prev: np.ndarray) -> float:
        """Progress along the centreline if there is one, else distance moved.

        Both are in units of "a full-speed step", so the scale matches
        ``lanekeep``. The distinction is not cosmetic: distance travelled
        rewards a fast lap of any loop inside a wide corridor, and going
        backwards earns the same as going forwards. Arc length along the
        centreline is signed and is what a lap time is made of.
        """
        if self.centreline is None:
            return float(np.linalg.norm(x[0, :2] - x_prev[0, :2])) / (SPEED_MAX * self.dt)
        s = self._arc(x[0, :2])
        if self._prev_s is None:
            self._prev_s = s
            return 0.0
        # shortest signed step around the loop, so crossing the start line
        # does not read as a lap of negative progress
        d = s - self._prev_s
        half = self.lap_length / 2.0
        d = d - self.lap_length if d > half else (d + self.lap_length if d < -half else d)
        self._prev_s = s
        return float(d) / (SPEED_MAX * self.dt)

    def _arc(self, xy) -> float:
        """Arc length of the nearest centreline point, in metres."""
        return self._nearest(xy)[0]

    def _nearest(self, xy) -> tuple[float, float]:
        """``(arc length, lateral distance)`` of the nearest centreline point.

        Distance to the nearest *sample*, not the true distance to the
        polyline, so it over-reads by up to half the sample spacing on a
        straight. Centrelines here are resampled to a fixed spacing
        (``bridges.mapimport._resample``), which bounds that error and keeps
        this one argmin rather than a projection onto every segment.
        """
        d2 = np.sum((self.centreline - xy) ** 2, axis=1)
        k = int(np.argmin(d2))
        return float(self._cl_s[k]), float(np.sqrt(d2[k])), k

    def _limit_at(self, k: int) -> float:
        """The half-width in force at centreline index ``k``."""
        return float(self.half_width[k % len(self.half_width)]
                     if self.half_width.size > 1 else self.half_width[0])

    # -- Env ---------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        jnp = self._jnp
        if seed is not None:
            self._key = self._jax.random.key(seed)
        poses = jnp.asarray(self.start_pose).reshape(1, 3)
        _obs, self._state = self.env.reset(self._split(), poses)
        self._t = 0
        self._v_cmd = 1.0
        self._prev_s = None
        self.history = []
        self._x, self._scans = self._pull(self._state)
        return self._obs_from(self._x, self._scans)

    def step(self, action):
        u = self._jnp.asarray(self._decode(action)).reshape(1, 2)
        x_prev = self._x
        # One key per agent step, reused across the sub-ticks. The key only
        # seeds the lidar noise, which is read once at the end, so splitting
        # per tick buys nothing and costs a dispatch each time.
        key = self._split()
        for _ in range(self.control_repeat):
            # step_env, not step: the simulator's step() auto-resets when every
            # agent is done, which would silently teleport the car mid-episode
            # and hand the agent a transition that never happened.
            _o, self._state, _r, _d, _i = self._step_env(key, self._state, u)
        self._x, self._scans = self._pull(self._state)
        crashed = bool(np.asarray(self._state.collisions)[0])
        reward = self._reward(self._x, x_prev)
        self._t += 1
        x = self._x[0]
        self.history.append(dict(x=float(x[0]), y=float(x[1]), psi=float(x[4]),
                                 v=float(x[3]), d=0.0))
        # Off the track counts as a crash: same -1.0, same termination. The
        # info flag distinguishes them so a run can report which boundary it
        # actually hit -- on an imported map "left the corridor" is the one
        # that fires, and reading it as a collision would misattribute it.
        if self.half_width is not None:
            _s, lateral, k = self._nearest(x[:2])
            limit = self._limit_at(k)
        else:
            lateral, limit = float("nan"), float("nan")
        off_track = bool(self.half_width is not None and lateral > limit)
        if crashed or off_track:
            return (np.zeros(self.obs_dim), -1.0, True, False,
                    {"crashed": bool(crashed), "off_track": off_track,
                     "lateral": lateral})
        truncated = self._t >= self.max_steps
        return (self._obs_from(self._x, self._scans), float(reward), False, truncated,
                {"crashed": False, "off_track": False, "lateral": lateral,
                 "v": float(x[3])})

    # -- pictures ----------------------------------------------------------
    def render_rollout(self, history=None, path: str = "rollout.png", title: str = ""):
        """Plot the driven line over the map's occupancy image."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hist = history if history is not None else self.history
        fig, ax = plt.subplots(figsize=(7, 6))
        sim = getattr(self.env, "scan_sim", None)
        img = getattr(sim, "map_img", None)
        if img is not None:
            ax.imshow(np.asarray(img), cmap="gray", origin="lower")
            ax.set_title((title or f"{self.id} / {self.map_name}") + "  (map pixels)")
        elif hist:
            ax.set_aspect("equal")
            ax.set_title(title or f"{self.id} / {self.map_name}")
        if hist:
            xs = np.array([h["x"] for h in hist])
            ys = np.array([h["y"] for h in hist])
            vs = np.array([h["v"] for h in hist])
            res = getattr(sim, "map_resolution", None)
            org = getattr(sim, "origin", None)
            if img is not None and res and org is not None:
                xs = (xs - float(org[0])) / float(res)
                ys = (ys - float(org[1])) / float(res)
            sc = ax.scatter(xs, ys, c=vs, s=3, cmap="viridis")
            fig.colorbar(sc, ax=ax, label="speed [m/s]")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path
