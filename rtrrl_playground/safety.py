"""A predictive safety filter, from scratch, and an honest account of its limits.

References -- Wabersich & Zeilinger, *"A predictive safety filter for
learning-based control of constrained nonlinear dynamical systems"*,
Automatica 2021 (arXiv:1812.05506); Wabersich & Zeilinger, *"Linear model
predictive safety certification for learning-based control"*, CDC 2018; Hewing,
Wabersich, Menner & Zeilinger, *"Learning-Based Model Predictive Control:
Toward Safe Learning in Control"*, Annual Review of Control 2020. See
``papers/README.md``.

## The idea

A learning agent proposes an action. Before it is applied, ask one question:

    if I apply this action, does a *safe backup plan* still exist afterwards?

If yes, apply it unchanged. If no, apply the nearest action for which one does.
The learner is left alone almost all of the time, and constrained only at the
boundary -- which is what makes this different from shaping the reward, wrapping
the action space, or any other scheme that pays for safety everywhere.

Formally the filter solves, at every step,

    min ||u_0 - u_learner||   s.t.  x_{k+1} = f(x_k, u_k),  x_k in X,
                                    u_k in U,  x_N in X_safe

and applies ``u_0``. Here the terminal set ``X_safe`` is **"stopped, and on the
track"**: a car at rest inside the boundary can stay there forever, so it is a
control-invariant set, and reaching it is a certificate that the episode need
not end badly. The backup that reaches it is full braking with the steering
pointed back at the centreline.

## Why it is written this way

The action space is nine discrete actions, so the minimisation above is not a
solver problem -- it is *enumerate and check*, ordered by distance from what the
learner asked for. All nine candidates are rolled out at once as arrays, so a
filtered step costs one vectorised horizon rather than nine sequential ones.
That is the whole implementation, and it is a real predictive safety filter,
not an approximation of one: with a finite input set the argmin is exact.

## Three things it does not do, stated plainly

**It is privileged.** The filter runs on the vehicle *state*, not on the
agent's nine beams. That is not cheating -- on a real car the filter sits on
the state estimator, which is exactly where it belongs -- but it does mean the
guarantee is only ever as good as that estimate, and a filter tested against
ground truth in simulation has not been tested at all in the part that usually
fails.

**It does not know the grip either.** ``LaneKeep`` redraws the tyre grip every
episode and never observes it, and the filter is in the same position: it
predicts with ``assumed_grip``. Set that above the true value and the filter is
confidently wrong -- it certifies a corner the car cannot take, and you get
crashes *through* the filter. Set it to the worst case and you get safety at
the cost of a slow, timid agent. That trade is the actual subject of
``tutorial/10``, and it is the honest version of what a safety filter buys.

**It changes what the learner is learning about.** The action that reaches the
environment is not always the one the policy chose, which makes the update
off-policy in a way TD(lambda) does not account for. See ``credit`` below.
"""

from __future__ import annotations

import math

import numpy as np

from rtrrl_playground.envs.lanekeep import (
    A_LAT_MAX, ACCEL_MAX, DRAG, SPEED_MAX, STEER_MAX, STEER_TAU, WHEELBASE,
)
from rtrrl_playground.envs.vehicle import VehicleParams


class BicycleModel:
    """The environment's own kinematic bicycle, as a batched pure function.

    Batched over candidate actions: state is ``(m, 5)`` for ``m`` candidates, so
    a horizon of ``N`` costs ``N`` vectorised steps rather than ``m * N`` scalar
    ones. Nothing here touches the environment, and the filter never gets to
    call ``env.step`` -- a "predictive" filter that peeked at the real
    simulator would be predicting nothing.
    """

    def __init__(self, dt: float = 0.05, grip: float = 1.0,
                 params: VehicleParams | None = None):
        self.dt, self.grip = float(dt), float(grip)
        # The filter's *belief* about the car, which need not be the car. This
        # is the same mismatch as `grip`, generalised: hand it the simulator's
        # parameters while the plant is a different vehicle and you get exactly
        # the situation a filter faces on real hardware.
        self.params = params or VehicleParams()

    def step(self, s: np.ndarray, steer, throttle) -> np.ndarray:
        """``s = [x, y, psi, v, delta]`` -> next state. Mirrors ``LaneKeep._integrate``."""
        p = self.params
        x, y, psi, v, delta = s.T
        steer = np.asarray(steer, dtype=np.float64)
        throttle = np.asarray(throttle, dtype=np.float64)
        delta = delta + (steer * p.steer_max - delta) * self.dt / p.steer_tau
        v = np.clip(v + (throttle * p.accel_max - p.drag * v) * self.dt, 0.0, p.speed_max)
        psi_dot = v / p.wheelbase * np.tan(delta)
        limit = np.where(v > 1e-3, p.a_lat_max * self.grip / np.maximum(v, 1e-3), np.inf)
        psi_dot = np.clip(psi_dot, -limit, limit)
        return np.stack([x + v * np.cos(psi) * self.dt,
                         y + v * np.sin(psi) * self.dt,
                         psi + psi_dot * self.dt, v, delta], axis=1)


class PredictiveSafetyFilter:
    """Minimal modification of a proposed action, subject to a backup existing.

    ``credit`` decides what the agent is told afterwards, and it is not a
    detail:

    ``"executed"``
        the agent learns about the action that actually happened. The filter
        becomes part of the environment. Simple, and it means the policy can
        never learn *why* it was overridden -- it only sees that something else
        occurred.
    ``"proposed"``
        the agent learns about the action it chose, as though it had been
        applied. Keeps the policy gradient on-policy with respect to the
        policy, and lies about the dynamics.

    Neither is correct. TD(lambda) has no term for "the action was replaced",
    and the literature on learning through a shield does not agree on this
    either, so both are here and ``tutorial/10`` measures the difference rather
    than picking a side.
    """

    def __init__(self, track, horizon: int = 25, dt: float = 0.05,
                 predict_dt_scale: float = 1.0, assumed_grip: float = 1.0,
                 assumed_vehicle: VehicleParams | None = None,
                 margin: float = 0.05, stop_speed: float = 0.25,
                 obstacle_radius: float = 0.44, credit: str = "executed",
                 n_actions: int = 9):
        if credit not in ("executed", "proposed"):
            raise ValueError("credit must be 'executed' or 'proposed'")
        self.track = track
        self.dt = float(dt)
        self.predict_dt = float(dt) * float(predict_dt_scale)
        self.assumed_vehicle = assumed_vehicle or VehicleParams()
        self.model = BicycleModel(dt=self.predict_dt, grip=assumed_grip,
                                  params=self.assumed_vehicle)
        self._first = BicycleModel(dt=float(dt), grip=assumed_grip,
                                   params=self.assumed_vehicle)
        self._brake = np.full(n_actions, -1.0)
        self.horizon, self.margin = int(horizon), float(margin)
        self.stop_speed, self.obstacle_radius = float(stop_speed), float(obstacle_radius)
        self.credit, self.n_actions = credit, int(n_actions)
        self.assumed_grip = float(assumed_grip)
        # (steer, throttle) for each discrete action, and a distance between
        # them so "nearest safe action" means something.
        self._grid = np.array([[a // 3 - 1, a % 3 - 1] for a in range(self.n_actions)],
                              dtype=np.float64)
        self.reset_stats()

    def reset_stats(self) -> None:
        self.n_steps = 0
        self.n_interventions = 0
        self.n_no_safe_action = 0

    @property
    def intervention_rate(self) -> float:
        return self.n_interventions / max(self.n_steps, 1)

    # -- the backup controller --------------------------------------------
    def _project(self, s: np.ndarray):
        """``(on_track, d, psi_ref)`` for a batch of states, in one pass.

        The occupancy bit and the nearest-centreline index come from the *same*
        pair of grid indices. Both are needed at every horizon step -- the
        first for the constraint, the second for the backup controller -- and
        this filter is dispatch-bound on nine-element arrays, so the number of
        NumPy calls per step is the thing that costs, not their size.
        """
        t = self.track
        if t._nidx is None:
            t._build_nearest_grid()
        i, j = t.grid_index(s[:, 0], s[:, 1])
        free = t.free[j, i]
        k = t._nidx[j, i]
        k = np.where(k < 0, 0, k)
        rx = s[:, 0] - t.cx[k]
        ry = s[:, 1] - t.cy[k]
        tx, ty = t.tx[k], t.ty[k]
        return free, -rx * ty + ry * tx, np.arctan2(ty, tx)

    def _backup_action(self, s: np.ndarray, d: np.ndarray, psi_ref: np.ndarray):
        """Full braking, steering back towards the centreline.

        The backup does not have to be good, only *safe*: it has to drive the
        state into the terminal set (stopped, on track) from anywhere the
        filter is willing to certify. Braking gets it stopped; the steering
        term is what stops it from braking in a straight line out of a corner.
        """
        err = np.arctan2(np.sin(psi_ref - s[:, 2]), np.cos(psi_ref - s[:, 2]))
        return np.clip((err - 1.2 * d) / STEER_MAX, -1.0, 1.0)

    # -- the constraint ----------------------------------------------------
    def _unsafe(self, s: np.ndarray, free: np.ndarray, d: np.ndarray, obstacles) -> np.ndarray:
        """True where a predicted state violates the constraints."""
        bad = ~free
        if self.margin > 0:
            bad |= np.abs(d) > (self.track.half_width - self.margin)
        if obstacles is not None and len(obstacles):
            dx = s[:, None, 0] - obstacles[None, :, 0]
            dy = s[:, None, 1] - obstacles[None, :, 1]
            bad |= ((dx * dx + dy * dy) < self.obstacle_radius ** 2).any(axis=1)
        return bad

    def _certify(self, s0: np.ndarray, obstacles) -> np.ndarray:
        """For each candidate first action, does a safe backup plan exist?

        One step of the candidate, then the backup all the way to a stop. A
        candidate is certified if nothing along that trajectory violates a
        constraint **and** the car has reached the terminal set by the end.
        Requiring the terminal set is what makes this a filter with a
        recursive-feasibility argument rather than an N-step lookahead that
        cheerfully drives at a wall N+1 steps away.

        ``predict_dt_scale`` coarsens the prediction grid relative to the
        control rate, which is ordinary MPC practice -- but it defaults to 1.0
        here because on this problem it does not pay. At 2x the grid the filter
        is about twice as fast and *three times* as interventionist (37% of
        steps against 13%), because a braking trajectory predicted in 0.1 s
        jumps is pessimistic about where the car ends up, and pessimism in the
        certificate is refusals in the policy. Cheap and timid was the worse
        trade; the flag is here so you can see that rather than take it on
        trust.
        """
        m = len(s0)
        alive = np.ones(m, dtype=bool)
        brake = self._brake[:m]
        s = s0
        free, d, psi_ref = self._project(s)
        alive &= ~self._unsafe(s, free, d, obstacles)
        for _ in range(self.horizon - 1):
            steer = self._backup_action(s, d, psi_ref)
            s = self.model.step(s, steer, brake)
            free, d, psi_ref = self._project(s)
            alive &= ~self._unsafe(s, free, d, obstacles)
            if not alive.any():
                break
        return alive & (s[:, 3] <= self.stop_speed)

    def _certify_scalar(self, state, action: int, obstacles) -> bool:
        """The same certificate as :meth:`_certify`, for one action, in plain Python.

        This is not premature optimisation, it is the whole reason the filter is
        affordable. The certificate is a 25-step *sequential* recursion on five
        numbers; through NumPy that is ~750 array calls of one element each, and
        NumPy's per-call overhead -- not its arithmetic -- costs about 4 ms.
        The identical arithmetic in Python floats costs about 0.1 ms.

        The fast path runs whenever the proposed action turns out to be safe,
        which on a competent policy is essentially always, so this is the cost
        of the filter in practice. The vectorised nine-candidate version is
        kept for the fallback, where its width finally earns something, and
        ``tests/test_safety.py`` asserts the two agree.
        """
        t = self.track
        if t._nidx is None:
            t._build_nearest_grid()
        free, nidx, cx, cy, tx, ty = t.free, t._nidx, t.cx, t.cy, t.tx, t.ty
        ox, oy, res, nx, ny = t.origin[0], t.origin[1], t.res, t.nx, t.ny
        hw = t.half_width - self.margin if self.margin > 0 else float("inf")
        r2 = self.obstacle_radius ** 2
        grip, dt, pdt = self.assumed_grip, self.dt, self.predict_dt
        vp = self.assumed_vehicle
        steer_max, steer_tau = vp.steer_max, vp.steer_tau
        accel_max, speed_max, drag = vp.accel_max, vp.speed_max, vp.drag
        wheelbase, a_lat_max = vp.wheelbase, vp.a_lat_max

        x, y, psi, v, delta = (float(q) for q in state)
        steer, thr, h = float(self._grid[action, 0]), float(self._grid[action, 1]), dt

        for step in range(self.horizon):
            delta += (steer * steer_max - delta) * h / steer_tau
            v += (thr * accel_max - drag * v) * h
            v = 0.0 if v < 0.0 else (speed_max if v > speed_max else v)
            psi_dot = v / wheelbase * math.tan(delta)
            if v > 1e-3:
                lim = a_lat_max * grip / v
                psi_dot = -lim if psi_dot < -lim else (lim if psi_dot > lim else psi_dot)
            x += v * math.cos(psi) * h
            y += v * math.sin(psi) * h
            psi += psi_dot * h

            i = int((x - ox) / res)
            j = int((y - oy) / res)
            i = 0 if i < 0 else (nx - 1 if i >= nx else i)
            j = 0 if j < 0 else (ny - 1 if j >= ny else j)
            if not free[j, i]:
                return False
            k = int(nidx[j, i])
            if k < 0:
                k = 0
            rx, ry = x - cx[k], y - cy[k]
            tkx, tky = tx[k], ty[k]
            d = -rx * tky + ry * tkx
            if d > hw or -d > hw:
                return False
            if obstacles is not None and len(obstacles):
                for obx, oby in obstacles:
                    if (x - obx) ** 2 + (y - oby) ** 2 < r2:
                        return False
            if step == self.horizon - 1:
                break
            psi_ref = math.atan2(tky, tkx)
            err = math.atan2(math.sin(psi_ref - psi), math.cos(psi_ref - psi))
            steer = (err - 1.2 * d) / steer_max
            steer = -1.0 if steer < -1.0 else (1.0 if steer > 1.0 else steer)
            thr, h = -1.0, pdt
        return v <= self.stop_speed

    def admissible(self, state, obstacles=None) -> np.ndarray:
        """Which of the nine actions have a safe backup from ``state``, as a mask.

        The filter's verdict without the filter's side effects: no statistics
        are updated and no action is chosen. Note that :meth:`_certify` takes
        states the candidate has *already* been applied to, so the candidate's
        own control-rate step happens here first -- which is the same order
        :meth:`__call__` uses, and getting it wrong silently returns the same
        answer nine times.
        """
        s0 = np.repeat(np.asarray(state, dtype=np.float64)[None, :],
                       self.n_actions, axis=0)
        s1 = self._first.step(s0, self._grid[:, 0], self._grid[:, 1])
        return self._certify(s1, obstacles)

    # -- the filter --------------------------------------------------------
    def __call__(self, state, proposed_action: int, obstacles=None):
        """Return ``(action_to_apply, intervened)``.

        ``state`` is ``[x, y, psi, v, delta]``. Candidates are tried in order of
        distance from the proposed action, so an override is the *smallest*
        change that restores a backup, not an arbitrary safe action.
        """
        self.n_steps += 1
        state = np.asarray(state, dtype=np.float64)
        a = int(proposed_action)

        # Certify the proposed action *alone* first. It is safe the large
        # majority of the time -- that is the entire premise of a filter -- and
        # the scalar path makes that common case ~40x cheaper.
        if self._certify_scalar(state, a, obstacles):
            return a, False

        # The candidate's own step is taken at the *control* rate -- that one is
        # really applied, so predicting it on a coarser grid would certify an
        # action the car does not actually take.
        ok = self.admissible(state, obstacles)
        if not ok.any():
            # Nothing is certifiable. This is not a filter failure so much as a
            # report that the state should never have been reached -- with a
            # correct model it is unreachable, and with an optimistic
            # assumed_grip it happens. Brake, and count it.
            self.n_no_safe_action += 1
            self.n_interventions += 1
            one = state[None, :]
            _free, d, psi_ref = self._project(one)
            steer = self._backup_action(one, d, psi_ref)
            s = int(np.clip(np.round(steer[0]), -1, 1))
            return 3 * (s + 1) + 0, True
        order = np.argsort(np.abs(self._grid - self._grid[a]).sum(axis=1), kind="stable")
        for cand in order:
            if ok[cand]:
                self.n_interventions += 1
                return int(cand), True
        raise AssertionError("unreachable")  # pragma: no cover


class SafeAgent:
    """Wrap an agent so every action it emits passes through a filter.

    The wrapper keeps the agent's interface (``start`` / ``step`` / ``greedy``)
    so it drops into the same training loop, and handles the ``credit``
    question: which action the agent is told about afterwards.
    """

    def __init__(self, agent, filt: PredictiveSafetyFilter, state_fn, obstacle_fn=None):
        self.agent, self.filter = agent, filt
        self.state_fn, self.obstacle_fn = state_fn, obstacle_fn
        self.n_params = getattr(agent, "n_params", 0)
        self.cell = getattr(agent, "cell", None)
        self._executed = None

    def _apply(self, proposed):
        obstacles = self.obstacle_fn() if self.obstacle_fn is not None else None
        action, _intervened = self.filter(self.state_fn(), proposed, obstacles)
        self._executed = action
        return action

    @property
    def stats(self):
        s = dict(getattr(self.agent, "stats", {}))
        s["filtered"] = self.filter.intervention_rate
        return s

    def start(self, obs):
        return self._apply(self.agent.start(obs))

    def step(self, obs, reward, terminated, truncated):
        # The agent is *already* holding the action it proposed -- that is what
        # `agent.a` is -- so `credit="proposed"` needs no intervention at all,
        # and `credit="executed"` is the one that has to overwrite it, before
        # `agent.step` uses `agent.a` for both its policy gradient and its
        # meta-RL "previous action" input.
        #
        # Getting this backwards is silent: both settings then behave as
        # "proposed" and produce byte-identical results, which is how the bug
        # was found.
        if self.filter.credit == "executed":
            self.agent.a = self._executed
        nxt = self.agent.step(obs, reward, terminated, truncated)
        return None if nxt is None else self._apply(nxt)

    def eval_policy(self):
        inner = self.agent.eval_policy()
        outer = self

        class _Filtered:
            def reset(self):
                inner.reset()
                outer.filter.reset_stats()

            def observe(self, reward):
                inner.observe(reward)

            def __call__(self, obs):
                return outer._apply(inner(obs))

        pol = _Filtered()
        pol.reset()
        return pol


def make_safe(agent, env, credit: str = "executed", assume_env_vehicle: bool = True,
              **filter_kwargs):
    """Wrap ``agent`` with a filter that reads ``env``'s state and traffic.

    ``assume_env_vehicle`` gives the filter the environment's own parameters --
    a filter that knows the car exactly. Pass ``False``, or an explicit
    ``assumed_vehicle=``, to give it a *different* belief, which is the
    realistic case and the one ``tutorial/11`` uses.
    """
    if assume_env_vehicle:
        filter_kwargs.setdefault("assumed_vehicle", getattr(env, "vehicle", None))
    filt = PredictiveSafetyFilter(env.track, dt=env.dt, credit=credit, **filter_kwargs)

    def state_fn():
        return np.array([env.x, env.y, env.psi, env.v, env.delta])

    obstacle_fn = (lambda: env._opp_xy()) if hasattr(env, "_opp_xy") else None
    return SafeAgent(agent, filt, state_fn, obstacle_fn)
