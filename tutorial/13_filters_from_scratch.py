"""Lesson 13 -- both safety filters from scratch: the maths, and where each fails.

    python tutorial/13_filters_from_scratch.py

A predictive safety filter and a control barrier function, written out in
thirty lines each, imported from nowhere. Then the two experiments that show
what actually separates them -- which is not that one is safer.
"""

# %% [markdown]
# # Lesson 13 — Both filters from scratch
#
# A safety filter sits between the agent and the actuator and answers one
# question per step: **may I apply this action?** Two families answer it in
# opposite ways.
#
# | | certifies safety by | needs | horizon |
# |---|---|---|---|
# | **Predictive** (Wabersich & Zeilinger) | *exhibiting a plan* that stays legal and ends in a safe terminal set | a backup controller + a rollout | $N$ steps |
# | **CBF** (Ames et al.) | *evaluating a function* $h(x)$ and one inequality | a valid barrier $h$ | 1 step |
#
# Both leave the learner alone in the interior and constrain it only at the
# boundary. Neither shapes the reward, restricts the action space, or
# constrains the policy class.

# %%
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import make_env
from rtrrl_playground.envs.scripted import WallFollower

# --- the vehicle model both filters predict with, written out here ---------
WHEELBASE, STEER_MAX, STEER_TAU = 0.33, 0.40, 0.08
ACCEL_MAX, SPEED_MAX, DRAG, A_LAT_MAX = 4.0, 4.0, 0.15, 6.0


def step_model(s, steer, throttle, dt, grip=1.0):
    """One tick of the kinematic bicycle. ``s = [x, y, psi, v, delta]``."""
    x, y, psi, v, delta = s
    delta = delta + (steer * STEER_MAX - delta) * dt / STEER_TAU
    v = min(max(v + (throttle * ACCEL_MAX - DRAG * v) * dt, 0.0), SPEED_MAX)
    psi_dot = v / WHEELBASE * np.tan(delta)
    if v > 1e-3:
        lim = A_LAT_MAX * grip / v
        psi_dot = min(max(psi_dot, -lim), lim)
    return np.array([x + v * np.cos(psi) * dt, y + v * np.sin(psi) * dt,
                     psi + psi_dot * dt, v, delta])


ACTIONS = np.array([[a // 3 - 1, a % 3 - 1] for a in range(9)], dtype=float)


def lateral(track, s):
    """Signed offset from the centreline, and the path heading there."""
    k = int(track.nearest_index(np.array([s[0]]), np.array([s[1]]))[0])
    k = max(k, 0)
    rx, ry = s[0] - track.cx[k], s[1] - track.cy[k]
    tx, ty = track.tx[k], track.ty[k]
    return -rx * ty + ry * tx, np.arctan2(ty, tx)


# %% [markdown]
# ## 1. The predictive safety filter
#
# $$\min_{u_0}\;\|u_0 - u_\text{learner}\| \quad\text{s.t.}\quad
#   x_{k+1}=f(x_k,u_k),\; x_k \in \mathcal{X},\; x_N \in \mathcal{X}_\text{safe}$$
#
# The terminal set is **"stopped, and on the track"** — a car at rest inside
# the boundary can stay there forever, so it is control-invariant, and reaching
# it certifies the episode need not end badly. The backup that gets there is
# full braking with the steering pointed back at the centreline.
#
# Requiring the terminal set is what makes this a filter with a recursive
# feasibility argument, rather than an $N$-step lookahead that cheerfully
# drives at a wall $N{+}1$ steps away.

# %%
def predictive_ok(track, s, action, dt=0.05, N=25, margin=0.05, stop_speed=0.25):
    """Does a safe backup plan exist after applying ``action``?"""
    s = step_model(s, *ACTIONS[action], dt)
    for step in range(N):
        d, psi_ref = lateral(track, s)
        if not track.on_track(np.array(s[0]), np.array(s[1])):
            return False
        if abs(d) > track.half_width - margin:
            return False
        if step == N - 1:
            break
        # the backup: brake hard, steer back towards the line
        err = np.arctan2(np.sin(psi_ref - s[2]), np.cos(psi_ref - s[2]))
        steer = float(np.clip((err - 1.2 * d) / STEER_MAX, -1, 1))
        s = step_model(s, steer, -1.0, dt)
    return s[3] <= stop_speed


# %% [markdown]
# ## 2. The control barrier function
#
# Define $h(x) > 0$ on the safe set and require, in discrete time
# (Agrawal & Sreenath, RSS 2017):
#
# $$h(x_{t+1}) \;\ge\; (1-\alpha)\, h(x_t), \qquad 0 < \alpha \le 1$$
#
# Satisfy that every step and $h$ can never cross zero, so the safe set is
# forward invariant. **One model step, no backup, no horizon.**
#
# ### The barrier, and why the obvious one is wrong
#
# The obvious choice is $h = w - |d|$. It is *myopic*: it permits driving flat
# out at a wall until the step before contact, because until then $h$ is still
# positive and still falling slowly. A one-step condition cannot see a braking
# distance.
#
# Put the dynamics into the barrier instead:
#
# $$h = w - |d| - T_\text{look}\,\big|v\sin e_\psi\big|$$
#
# — subtract the lateral ground the car covers in $T_\text{look}$ seconds at
# its current lateral closing rate. Now the barrier shrinks when you are moving
# *towards* a wall, not merely when you are near one.

# %%
def barrier(track, s, kind="braking", margin=0.05, lookahead=0.45):
    if not track.on_track(np.array(s[0]), np.array(s[1])):
        return -1.0
    d, psi_ref = lateral(track, s)
    h = (track.half_width - margin) - abs(d)
    if kind == "braking":
        e_psi = np.arctan2(np.sin(s[2] - psi_ref), np.cos(s[2] - psi_ref))
        h -= lookahead * abs(s[3] * np.sin(e_psi))
    return float(h)


def cbf_ok(track, s, action, dt=0.05, alpha=0.35, kind="braking"):
    h_now = barrier(track, s, kind)
    h_next = barrier(track, step_model(s, *ACTIONS[action], dt), kind)
    return h_next >= (1.0 - alpha) * h_now


# %% [markdown]
# ## The filter itself is the same three lines either way
#
# Nine discrete actions means the "QP" is enumerate-and-check, ordered by
# distance from what the learner asked for — so the argmin is exact, and **the
# only thing that differs between the two filters is the predicate**.

# %%
def filt(track, s, proposed, ok_fn):
    if ok_fn(track, s, proposed):
        return proposed, False
    order = np.argsort(np.abs(ACTIONS - ACTIONS[proposed]).sum(axis=1), kind="stable")
    for cand in order:
        if ok_fn(track, s, int(cand)):
            return int(cand), True
    return int(np.argmin(np.abs(ACTIONS[:, 1] + 1))), True  # brake, best effort


# %% [markdown]
# ## Experiment 1 — the barrier carries the safety, not the method

# %%
def run(ok_fn, policy_fn, n_ep=15, seed0=100):
    env = make_env("lanekeep")
    off, iv, steps, rets = 0, 0, 0, []
    for ep in range(n_ep):
        obs = env.reset(seed=seed0 + ep)
        pol, R, info = policy_fn(), 0.0, {}
        for _ in range(env.max_steps):
            a = pol(obs)
            if ok_fn is not None:
                s = np.array([env.x, env.y, env.psi, env.v, env.delta])
                a, did = filt(env.track, s, a, ok_fn)
                iv += int(did)
            obs, r, te, tr, info = env.step(a)
            R += r
            steps += 1
            if te or tr:
                break
        off += bool(info.get("off_track"))
        rets.append(R)
    return off / n_ep, float(np.mean(rets)), iv / max(steps, 1)


rng = np.random.default_rng(0)
random_policy = lambda: (lambda o: int(rng.integers(9)))  # noqa: E731

print("  a uniformly random policy -- the stress test\n")
print(f"  {'filter':<34}{'off-track':>11}{'overridden':>12}")
for label, fn in (
    ("none", None),
    ("CBF, h = w - |d|  (naive)", lambda t, s, a: cbf_ok(t, s, a, kind="lateral")),
    ("CBF, h with closing-rate term", lambda t, s, a: cbf_ok(t, s, a, kind="braking")),
    ("predictive, 25-step rollout", predictive_ok),
):
    off, ret, iv = run(fn, random_policy)
    print(f"  {label:<34}{off:>11.0%}{iv:>12.1%}")

print("\n  The naive barrier fails. The same method with the closing-rate term")
print("  does not. That is a statement about the barrier, not about CBFs.")

# %% [markdown]
# ## Experiment 2 — a competent driver, where the difference shows up
#
# A safety filter doing its job is *invisible* to a driver who was not going to
# crash. Whichever filter overrides such a driver more is the more conservative
# one — and one of these two is structurally more conservative than the other.

# %%
print("\n  a scripted wall-follower -- the deployment case\n")
print(f"  {'filter':<34}{'off-track':>11}{'return':>9}{'overridden':>12}")
for label, fn in (
    ("none", None),
    ("CBF, closing-rate barrier", lambda t, s, a: cbf_ok(t, s, a, kind="braking")),
    ("predictive, 25-step rollout", predictive_ok),
):
    off, ret, iv = run(fn, WallFollower)
    print(f"  {label:<34}{off:>11.0%}{ret:>9.0f}{iv:>12.1%}")

print("\n  The CBF clips actions of a driver that was never going to crash.")
print("  A one-step condition cannot tell that a *plan* exists -- only that the")
print("  next state is acceptable -- so it refuses what a rollout certifies.")
print("  That is the price of having no horizon, and it is structural.")

# %% [markdown]
# ## A picture of the barrier
#
# The clearest way to see the difference between the two barriers: plot $h$
# over the track for a car travelling at speed towards the outside wall.

# %%
def plot_barrier(path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    env = make_env("lanekeep")
    track = env.track
    xs = np.linspace(track.origin[0], track.origin[0] + track.nx * track.res, 220)
    ys = np.linspace(track.origin[1], track.origin[1] + track.ny * track.res, 180)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, kind in zip(axes, ("lateral", "braking")):
        H = np.zeros((len(ys), len(xs)))
        for iy, yy in enumerate(ys):
            for ix, xx in enumerate(xs):
                # a car doing 3 m/s, pointed 25 degrees off the path
                H[iy, ix] = barrier(track, np.array([xx, yy, 0.44, 3.0, 0.0]), kind)
        m = ax.pcolormesh(xs, ys, H, cmap="RdYlGn", vmin=-0.8, vmax=0.8, shading="auto")
        ax.contour(xs, ys, H, levels=[0.0], colors="k", linewidths=1.6)
        fig.colorbar(m, ax=ax, label="h(x)")
        c, n, hw = track.center, track.normal, track.half_width
        for b in (c - hw * n, c + hw * n):
            ax.plot(*np.vstack([b, b[:1]]).T, color="0.2", lw=1)
        ax.set_aspect("equal")
        ax.set_title(f"h_kind = {kind!r}"
                     + ("\n(position only — the safe set ignores motion)" if kind == "lateral"
                        else "\n(the safe set shrinks where you are heading at a wall)"))
    fig.suptitle("The barrier, for a car at 3 m/s pointed 25° off the path. "
                 "Black line is h = 0.", y=1.02, fontsize=10)
    fig.tight_layout()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110, bbox_inches="tight")
        print(f"\n  wrote {path}")
    plt.close(fig)


plot_barrier(ROOT / "docs" / "source" / "_static" / "plots" / "barrier.png")

# %% [markdown]
# ## What to take from this
#
# * **Neither method is safer.** The barrier design carries the safety; a bad
#   barrier fails visibly where the same method with a good one does not fail
#   at all.
# * **The pointwise method is structurally more conservative.** It substitutes
#   a local condition for the existence of a plan, and pays for it in
#   interventions on drivers that never needed them.
# * **Their cost profiles are opposite.** The CBF is cheaper when it must
#   intervene, dearer when it need not — because "nothing needed" is one
#   trajectory for the predictive filter and still nine barrier evaluations for
#   the CBF. On a vehicle running a usually-right policy, that favours the
#   rollout, which is the reverse of the usual summary.
# * **Both inherit the same three limitations**, unchanged, because they follow
#   from being a filter and not from the criterion: privileged state access,
#   a guarantee only as good as the vehicle model, and an off-policy update.
#
# ## References
#
# * Wabersich & Zeilinger, *A predictive safety filter for learning-based
#   control of constrained nonlinear dynamical systems*, Automatica 2021 —
#   [arXiv:1812.05506](https://arxiv.org/abs/1812.05506)
# * Ames, Xu, Grizzle & Tabuada, *Control Barrier Function Based Quadratic
#   Programs for Safety Critical Systems*, IEEE TAC 2017
# * Ames, Coogan, Egerstedt, Notomista, Sreenath & Tabuada, *Control Barrier
#   Functions: Theory and Applications*, ECC 2019 —
#   [arXiv:1903.11199](https://arxiv.org/abs/1903.11199)
# * Agrawal & Sreenath, *Discrete Control Barrier Functions for Safety-Critical
#   Control of Discrete Systems*, RSS 2017 — the discrete-time condition used here
