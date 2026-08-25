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

from rtrrl_playground.spaces import Box, Discrete, Env

STEER_MAX = 0.4  # rad, the steering-angle setpoint the simulator's PID takes
SPEED_MAX = 4.0  # m/s
N_BEAMS_OUT = 9
BEAM_RANGE = 10.0


class ScuderiaLaneKeep(Env):
    """Drive a ``scuderia_gym_jax`` car on one of its shipped maps."""

    id = "scuderia-lanekeep"

    def __init__(self, model: str = "st", map_name: str = "berlin",
                 map_ext: str = ".png", tire_model: int | None = None,
                 action_mode: str = "discrete",
                 observe_speed: bool = False, num_beams: int = 108,
                 max_steps: int = 2000, control_repeat: int = 5,
                 start_pose=(0.0, 0.0, 0.0), seed: int = 0, **make_kwargs):
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
        self.obs_dim = N_BEAMS_OUT + int(observe_speed)
        self.action_space = Discrete(9) if action_mode == "discrete" else Box(2)
        self.max_steps = int(max_steps)
        self.start_pose = np.asarray(start_pose, dtype=float)
        self._key = jax.random.key(seed)
        # Which raw beams to keep. Evenly spaced across the full field of view,
        # so the nine numbers the agent sees mean the same thing they do in
        # envs/lanekeep.py: right to left, straight ahead in the middle.
        self._idx = np.linspace(0, num_beams - 1, N_BEAMS_OUT).astype(int)
        # jit the bound method once. Without this every control tick re-enters
        # the tracer and a step costs tens of milliseconds -- the simulator is
        # designed to be jitted around a whole `lax.scan` rollout, and stepping
        # it from Python is exactly the usage that does not get that for free.
        self._step_env = self._jax.jit(self.env.step_env)
        self._state = None
        self.history: list[dict] = []

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
        """Distance travelled this step, in units of "a full-speed step"."""
        return float(np.linalg.norm(x[0, :2] - x_prev[0, :2])) / (SPEED_MAX * self.dt)

    # -- Env ---------------------------------------------------------------
    def reset(self, seed: int | None = None) -> np.ndarray:
        jnp = self._jnp
        if seed is not None:
            self._key = self._jax.random.key(seed)
        poses = jnp.asarray(self.start_pose).reshape(1, 3)
        _obs, self._state = self.env.reset(self._split(), poses)
        self._t = 0
        self._v_cmd = 1.0
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
        if crashed:
            return np.zeros(self.obs_dim), -1.0, True, False, {"crashed": True}
        truncated = self._t >= self.max_steps
        return (self._obs_from(self._x, self._scans), float(reward), False, truncated,
                {"crashed": False, "v": float(x[3])})

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
