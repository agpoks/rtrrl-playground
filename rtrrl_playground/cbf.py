"""A control barrier function filter -- the *pointwise* alternative.

References -- Ames, Xu, Grizzle & Tabuada, *"Control Barrier Function Based
Quadratic Programs for Safety Critical Systems"*, IEEE TAC 2017; Ames, Coogan,
Egerstedt, Notomista, Sreenath & Tabuada, *"Control Barrier Functions: Theory
and Applications"*, ECC 2019 (arXiv:1903.11199); Agrawal & Sreenath,
*"Discrete Control Barrier Functions for Safety-Critical Control of Discrete
Systems"*, RSS 2017 -- the discrete-time form used here. See
``papers/README.md``.

## The idea, and how it differs from the predictive filter

Both filters answer "may I apply this action?" and they answer it in
fundamentally different ways.

:mod:`rtrrl_playground.safety` **exhibits a trajectory**: it rolls the model
forward twenty-five steps under a braking backup and checks that the whole path
stays legal and ends stopped. Safety is certified by *producing the plan that
would save you*.

A CBF **evaluates a function**. Define ``h(x)`` positive on the safe set, and
require the action to satisfy one algebraic inequality -- in discrete time
(Agrawal & Sreenath):

    h(x_{t+1})  >=  (1 - alpha) h(x_t),      0 < alpha <= 1

Satisfy that at every step and ``h`` can never cross zero, so the safe set is
forward invariant. No horizon, no backup policy, no rollout: **one model step
instead of twenty-five**.

With a continuous input the constraint is linear in the control and the filter
is the QP ``min ||u - u_L||^2 s.t. CBF``. This repo's action space is nine
discrete actions, so the QP degenerates into enumerate-and-check, exactly as
the predictive filter does -- the argmin is still exact, and the *only*
difference between the two filters is the criterion. That makes the comparison
in ``docs/source/safety.md`` unusually clean: same wrapper, same action set,
same model, same evaluation. One rolls out, one evaluates an inequality.

## The barrier, and the reason a naive one fails

The obvious choice for staying on a track is

    h = half_width - |d|

with ``d`` the lateral offset. It is also **myopic**, and instructively so: it
permits driving at full speed straight at a wall right up until the step before
contact, because until then ``h`` is still comfortably positive and still
decreasing slowly enough. A one-step condition cannot see a braking distance.

That is not a flaw in CBFs, it is a statement about the barrier. The fix is to
put the vehicle's own dynamics into ``h``, and ``h_kind="braking"`` does:

    h = half_width - |d| - lookahead * |v sin(e_psi)|

subtracting the lateral distance the car will cover in ``lookahead`` seconds at
its current lateral closing rate. The barrier now shrinks when you are moving
*towards* a wall, not merely when you are near one.

Both are implemented, ``h_kind`` selects, and the benchmark reports both --
because "CBFs are unsafe here" and "that barrier was unsafe here" are very
different claims and only the second one is true.

## The same three limitations as the predictive filter

It is privileged (it reads the state, not the beams), it inherits its guarantee
entirely from its model of the vehicle -- including the grip it does not know --
and it makes the learner's update off-policy. See
:mod:`rtrrl_playground.safety`, which discusses all three at length; nothing
about them changes because the criterion did.
"""

from __future__ import annotations

import numpy as np

from rtrrl_playground.envs.vehicle import VehicleParams
from rtrrl_playground.safety import BicycleModel

H_KINDS = ("lateral", "braking")


class DiscreteCBFFilter:
    """Discrete-time CBF filter over a finite action set.

    Interface-compatible with
    :class:`~rtrrl_playground.safety.PredictiveSafetyFilter`, so
    :class:`~rtrrl_playground.safety.SafeAgent` wraps either one and every
    comparison between them is like for like.
    """

    def __init__(self, track, dt: float = 0.05, alpha: float = 0.35,
                 h_kind: str = "braking", lookahead: float = 0.45,
                 assumed_grip: float = 1.0, assumed_vehicle: VehicleParams | None = None,
                 margin: float = 0.05, obstacle_radius: float = 0.44,
                 credit: str = "executed", n_actions: int = 9):
        if h_kind not in H_KINDS:
            raise ValueError(f"h_kind must be one of {H_KINDS}")
        if credit not in ("executed", "proposed"):
            raise ValueError("credit must be 'executed' or 'proposed'")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.track = track
        self.dt = float(dt)
        self.alpha = float(alpha)
        self.h_kind, self.lookahead = h_kind, float(lookahead)
        self.assumed_grip = float(assumed_grip)
        self.assumed_vehicle = assumed_vehicle or VehicleParams()
        self.model = BicycleModel(dt=float(dt), grip=assumed_grip,
                                  params=self.assumed_vehicle)
        self.margin, self.obstacle_radius = float(margin), float(obstacle_radius)
        self.credit, self.n_actions = credit, int(n_actions)
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

    # -- the barrier -------------------------------------------------------
    def h(self, states, obstacles=None):
        """``h(x) > 0`` on the safe set, as the minimum over every constraint.

        Batched: ``states`` is ``(m, 5)`` and the result is ``(m,)``. The whole
        filter is one call of this on the nine one-step-ahead states, which is
        the structural advantage over rolling a horizon out -- so it is written
        as array work rather than as a loop over candidates, or the advantage
        exists only on paper.
        """
        t = self.track
        if t._nidx is None:
            t._build_nearest_grid()
        s = np.atleast_2d(np.asarray(states, dtype=np.float64))
        x, y, psi, v = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
        i, j = t.grid_index(x, y)
        free = t.free[j, i]
        k = t._nidx[j, i]
        bad = (~free) | (k < 0)
        k = np.where(k < 0, 0, k)
        rx, ry = x - t.cx[k], y - t.cy[k]
        tx, ty = t.tx[k], t.ty[k]
        d = -rx * ty + ry * tx
        h_track = (t.half_width - self.margin) - np.abs(d)
        if self.h_kind == "braking":
            # Subtract the lateral ground the car will cover in `lookahead`
            # seconds at its current lateral closing rate. Without this the
            # barrier is a pure position constraint and a one-step condition
            # cannot see a braking distance -- see the module docstring.
            psi_ref = np.arctan2(ty, tx)
            e_psi = np.arctan2(np.sin(psi - psi_ref), np.cos(psi - psi_ref))
            h_track = h_track - self.lookahead * np.abs(v * np.sin(e_psi))
        if obstacles is not None and len(obstacles):
            dx = x[:, None] - obstacles[None, :, 0]
            dy = y[:, None] - obstacles[None, :, 1]
            h_obs = np.sqrt(dx * dx + dy * dy).min(axis=1) - self.obstacle_radius
            h_track = np.minimum(h_track, h_obs)
        return np.where(bad, -1.0, h_track)

    # -- the filter --------------------------------------------------------
    def _h_next(self, state, obstacles=None):
        """``(h(x_{t+1}) for all nine actions, the target it must clear)``."""
        state = np.asarray(state, dtype=np.float64)
        target = (1.0 - self.alpha) * float(self.h(state, obstacles)[0])
        s0 = np.repeat(state[None, :], self.n_actions, axis=0)
        s1 = self.model.step(s0, self._grid[:, 0], self._grid[:, 1])
        return self.h(s1, obstacles), target

    def admissible(self, state, obstacles=None) -> np.ndarray:
        """Which of the nine actions satisfy the barrier condition, as a mask.

        The filter's verdict without the filter's side effects: no statistics
        are updated and no action is chosen. Useful for plotting what the
        criterion allows at a state, which is otherwise only observable one
        action at a time through :meth:`__call__`.
        """
        h_next, target = self._h_next(state, obstacles)
        return h_next >= target

    def __call__(self, state, proposed_action: int, obstacles=None):
        """Return ``(action_to_apply, intervened)``.

        One model step per candidate, one inequality per candidate. That is the
        whole filter.
        """
        self.n_steps += 1
        state = np.asarray(state, dtype=np.float64)
        h_now = float(self.h(state, obstacles)[0])
        target = (1.0 - self.alpha) * h_now

        # Check the proposed action alone first. It is safe the large majority
        # of the time -- that is the premise of a filter -- and the predictive
        # filter has the same shortcut, so without this the comparison between
        # them measures an implementation asymmetry rather than the criteria.
        a = int(proposed_action)
        s_one = self.model.step(state[None, :], self._grid[a, 0:1], self._grid[a, 1:2])
        if float(self.h(s_one, obstacles)[0]) >= target:
            return a, False

        h_next, target = self._h_next(state, obstacles)
        ok = h_next >= target
        self.n_interventions += 1
        if not ok.any():
            # No action satisfies the condition. With a valid barrier and a
            # correct model this is unreachable; it happens when the model is
            # wrong (an optimistic grip) or the state was already unsafe. Take
            # the least-bad action -- the one that decreases h least -- and count it.
            self.n_no_safe_action += 1
            return int(np.argmax(h_next)), True
        order = np.argsort(np.abs(self._grid - self._grid[a]).sum(axis=1), kind="stable")
        for cand in order:
            if ok[cand]:
                return int(cand), True
        raise AssertionError("unreachable")  # pragma: no cover


def make_safe_cbf(agent, env, credit: str = "executed", assume_env_vehicle: bool = True,
                  **filter_kwargs):
    """Wrap ``agent`` with a CBF filter. Mirrors ``safety.make_safe`` exactly."""
    from rtrrl_playground.safety import SafeAgent

    if assume_env_vehicle:
        filter_kwargs.setdefault("assumed_vehicle", getattr(env, "vehicle", None))
    filt = DiscreteCBFFilter(env.track, dt=env.dt, credit=credit, **filter_kwargs)

    def state_fn():
        return np.array([env.x, env.y, env.psi, env.v, env.delta])

    obstacle_fn = (lambda: env._opp_xy()) if hasattr(env, "_opp_xy") else None
    return SafeAgent(agent, filt, state_fn, obstacle_fn)
