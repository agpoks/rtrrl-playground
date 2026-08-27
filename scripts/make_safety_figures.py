"""Figures for ``docs/source/safety.md``.

    python scripts/make_safety_figures.py            # all four
    python scripts/make_safety_figures.py --only certificate

A safety filter is a geometric object and the prose in ``safety.md`` was
carrying it alone. These four say what the text cannot:

``safety_certificate``   what the certificate *is* -- nine candidate actions,
                         each followed by its braking backup, accepted or not.
``safety_barrier_field`` why ``h_kind="lateral"`` is myopic, in one contour.
``safety_knobs``         the conservatism dial on each filter, swept.
``safety_grip``          what happens when the filter's model is wrong. This
                         is the one to look at before trusting any of it.

Everything is seeded and reproducible; nothing is picked for looks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import make_env  # noqa: E402
from rtrrl_playground.cbf import DiscreteCBFFilter  # noqa: E402
from rtrrl_playground.envs.scripted import WallFollower  # noqa: E402
from rtrrl_playground.safety import PredictiveSafetyFilter  # noqa: E402

OUT = ROOT / "docs" / "source" / "_static" / "plots"
GREEN, RED, INK = "#2a9d5c", "#c1272d", "#222222"
ACTION_NAMES = ["brake", "coast", "accel"]


def _bg(ax, track, zoom=None):
    ax.imshow(track.free, origin="lower", cmap="Greys_r", alpha=0.35,
              extent=[track.origin[0], track.origin[0] + track.nx * track.res,
                      track.origin[1], track.origin[1] + track.ny * track.res])
    ax.plot(track.cx, track.cy, color=INK, lw=0.7, ls=":", alpha=0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    if zoom:
        ax.set_xlim(zoom[0], zoom[1]); ax.set_ylim(zoom[2], zoom[3])


def _admits(filt, state, obstacles=None):
    """Which of the nine actions the filter would allow from ``state``.

    ``_certify`` takes states that have *already* had the candidate applied --
    it certifies the backup from there -- so the candidate step at the control
    rate has to happen first, exactly as ``PredictiveSafetyFilter.__call__``
    does it.
    """
    s0 = np.repeat(np.asarray(state, float)[None, :], filt.n_actions, axis=0)
    s1 = filt._first.step(s0, filt._grid[:, 0], filt._grid[:, 1])
    return filt._certify(s1, obstacles)


def _marginal_state(filt, env, want=(3, 7), seeds=range(60)):
    """Find a state where the filter genuinely disagrees with some actions.

    A competent policy never goes near the boundary, so a state sampled from
    one admits all nine actions and makes a picture of nothing. An *unfiltered*
    random policy has the opposite problem: it leaves the track within a few
    seconds and never lingers in the interesting region either.

    So the search runs the random policy **behind the filter**. That is the
    regime the filter actually exists for -- a policy pushing at the boundary
    and being held off it -- and a step where the filter overrode the proposal
    is by construction a step where the nine answers were not all the same. No
    state is placed by hand, which matters: a hand-placed car would invite the
    suspicion that the picture was arranged to have refusals in it.
    """
    rng = np.random.default_rng(0)
    for seed in seeds:
        obs = env.reset(seed=int(seed))
        for _ in range(env.max_steps):
            s = np.array([env.x, env.y, env.psi, env.v, env.delta])
            n = int(_admits(filt, s).sum())
            if want[0] <= n <= want[1] and env.v > 2.2:
                return s, seed
            a, _ = filt(s, int(rng.integers(9)))
            obs, r, te, tr, _ = env.step(a)
            if te or tr:
                break
    raise RuntimeError("no marginal state found")


# --------------------------------------------------------------------------
def fig_certificate():
    """The certificate itself: one candidate step, then the backup, nine times.

    This is the whole predictive filter in one picture. Each line is *not* a
    plan the car intends to follow -- it is the emergency stop that would still
    be available if that action were taken. The action is admitted if and only
    if its escape route stays on the track and ends stopped.

    The right panel is the same nine answers as the action grid the agent
    actually chooses from, which is the form the result reaches the learner in.
    """
    env = make_env("lanekeep")
    filt = PredictiveSafetyFilter(env.track, dt=env.dt, horizon=25,
                                  assumed_vehicle=getattr(env, "vehicle", None))
    s0, seed = _marginal_state(filt, env)
    ok = _admits(filt, s0)

    fig = plt.figure(figsize=(11.0, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.18)
    ax, gx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    paths = []
    for a in range(9):
        steer, thr = filt._grid[a]
        s = filt._first.step(s0[None, :], steer, thr)
        xs, ys = [s0[0], s[0, 0]], [s0[1], s[0, 1]]
        free, d, psi_ref = filt._project(s)
        for _ in range(filt.horizon - 1):        # the backup, from there on
            st = filt._backup_action(s, d, psi_ref)
            s = filt.model.step(s, st, np.array([-1.0]))
            free, d, psi_ref = filt._project(s)
            xs.append(s[0, 0]); ys.append(s[0, 1])
        paths.append((np.array(xs), np.array(ys)))

    allx = np.concatenate([p[0] for p in paths] + [[s0[0]]])
    ally = np.concatenate([p[1] for p in paths] + [[s0[1]]])
    pad = 0.45
    cx, cy = allx.mean(), ally.mean()
    half = max(np.ptp(allx), np.ptp(ally)) / 2 + pad
    _bg(ax, env.track, zoom=(cx - half * 1.45, cx + half * 1.45, cy - half, cy + half))

    for a, (xs, ys) in enumerate(paths):
        c = GREEN if ok[a] else RED
        ax.plot(xs, ys, color=c, lw=1.8, alpha=0.9, zorder=3)
        ax.plot(xs[-1], ys[-1], "o", ms=5.0, color=c, zorder=4)
    ax.plot(*s0[:2], "*", ms=16, color="#1f4e9c", zorder=5)
    ax.arrow(s0[0], s0[1], 0.22 * np.cos(s0[2]), 0.22 * np.sin(s0[2]),
             head_width=0.07, color="#1f4e9c", zorder=5, length_includes_head=True)
    n_ok = int(ok.sum())
    ax.set_title("each line is the emergency stop that would remain available",
                 fontsize=9.5)
    ax.plot([], [], color=GREEN, lw=1.8, label="backup exists → allowed")
    ax.plot([], [], color=RED, lw=1.8, label="no backup → refused")
    ax.plot([], [], "o", ms=5, color=INK, label="end of backup: stopped, on track")
    ax.legend(loc="lower left", fontsize=7.5, framealpha=0.92)

    # -- the same answer, as the action grid the agent picks from -----------
    grid = np.array([[1.0 if ok[3 * (st + 1) + (th + 1)] else 0.0
                      for th in (-1, 0, 1)] for st in (1, 0, -1)])
    gx.imshow(grid, cmap="RdYlGn", vmin=-0.35, vmax=1.35)
    gx.set_xticks(range(3), ["brake", "coast", "accel"], fontsize=9)
    gx.set_yticks(range(3), ["left", "straight", "right"], fontsize=9)
    for i in range(3):
        for j in range(3):
            gx.text(j, i, "✓" if grid[i, j] else "✗", ha="center", va="center",
                    fontsize=17, color=INK)
    gx.set_title("the nine actions, filtered", fontsize=9.5)
    # Describe what was actually refused rather than asserting a pattern: the
    # state comes out of a search, so the caption has to follow the data.
    bad = [(int(st), int(th)) for st in (-1, 0, 1) for th in (-1, 0, 1)
           if not ok[3 * (st + 1) + (th + 1)]]
    names = {-1: "right", 0: "straight", 1: "left"}
    if bad:
        steers = sorted({names[st] for st, _ in bad})
        thrs = sorted({ACTION_NAMES[th + 1] for _, th in bad})
        gx.set_xlabel(f"refused: {'/'.join(steers)} with {'/'.join(thrs)}", fontsize=8)

    fig.suptitle(f"v = {s0[3]:.1f} m/s, {n_ok} of 9 actions admitted  —  a state "
                 f"reached under a random policy behind the filter (seed {seed})",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / "safety_certificate.png", dpi=160)
    print(f"  wrote safety_certificate.png  ({n_ok}/9 admitted at "
          f"v={s0[3]:.2f}, seed {seed})")
    for a in range(9):
        st, th = filt._grid[a]
        print(f"    steer {st:+.0f}  {ACTION_NAMES[int(th) + 1]:<6} "
              f"{'admitted' if ok[a] else 'REFUSED'}")


# --------------------------------------------------------------------------
def fig_barrier_field():
    """``h`` over (lateral offset, heading error) for both barriers.

    The point is what each one *depends on*. ``lateral`` is a function of
    position alone, so its contours are flat: the same value whether the car is
    running parallel to the wall or driving straight into it. ``braking``
    tilts, because it subtracts where the car is going.
    """
    env = make_env("lanekeep")
    hw = env.track.half_width
    d = np.linspace(-hw, hw, 220)
    e = np.linspace(-0.9, 0.9, 220)
    D, E = np.meshgrid(d, e)
    margin, look = 0.05, 0.45

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for ax, (kind, v) in zip(axes, [("lateral", 2.5), ("braking", 1.0), ("braking", 2.5)]):
        if kind == "lateral":
            H = (hw - margin) - np.abs(D)
        else:
            H = (hw - margin) - np.abs(D) - look * np.abs(v * np.sin(E))
        m = np.abs(H).max()
        ax.contourf(D, E, H, levels=24, cmap="RdYlGn", vmin=-m, vmax=m)
        ax.contour(D, E, H, levels=[0.0], colors="k", linewidths=2.0)
        ax.set_xlabel("lateral offset $d$ [m]")
        ax.set_title(f"$h_\\mathrm{{{kind}}}$   at v = {v} m/s", fontsize=10)
        ax.axhline(0, color="k", lw=0.4, alpha=0.4)
    axes[0].set_ylabel("heading error $e_\\psi$ [rad]")
    fig.suptitle("the barrier is the design choice, not the method  —  "
                 "black line is $h=0$, the edge of the certified set", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "safety_barrier_field.png", dpi=160)
    print("  wrote safety_barrier_field.png")


# --------------------------------------------------------------------------
def _policy(kind, rng):
    """``random`` is the standard stress test; ``fast`` is the one that matters.

    A uniformly random policy on ``LaneKeep`` averages 0.9 m/s, and at that
    speed the tyre-grip limit binds on 0.1% of steps -- the car is simply never
    going fast enough for grip to be the thing that stops it. Any experiment
    about the filter's *model* of grip run under a random policy therefore
    measures nothing, and will report a reassuring flat line.

    ``fast`` holds full throttle and steers at random. It reaches the speed cap,
    the grip limit binds on ~80% of steps, and the filter's belief about grip
    starts to decide whether the car survives.
    """
    if kind == "wall":
        return WallFollower()
    if kind == "fast":
        return lambda o: int(rng.integers(3)) * 3 + 2   # any steer, full throttle
    return lambda o: int(rng.integers(9))


def _run(filt, n_ep=14, seed0=500, policy="random"):
    env = make_env("lanekeep")
    rng = np.random.default_rng(0)
    off, rets = 0, []
    for ep in range(n_ep):
        obs = env.reset(seed=seed0 + ep)
        pol = _policy(policy, rng)
        R, info = 0.0, {}
        for _ in range(env.max_steps):
            a = pol(obs)
            if filt is not None:
                a, _ = filt(np.array([env.x, env.y, env.psi, env.v, env.delta]), a)
            obs, r, te, tr, info = env.step(a)
            R += r
            if te or tr:
                break
        off += bool(info.get("off_track"))
        rets.append(R)
    iv = filt.intervention_rate if filt is not None else 0.0
    if filt is not None:
        filt.reset_stats()
    return off / n_ep, float(np.mean(rets)), iv


def fig_knobs():
    """Each filter's conservatism dial, swept, under the ``fast`` policy.

    Both dials do the same job from opposite directions: the horizon says how
    far ahead a backup must survive, ``alpha`` says how fast ``h`` may decay.
    Turn either one down far enough and the filter stops being a filter.
    """
    env = make_env("lanekeep")
    veh = getattr(env, "vehicle", None)
    horizons = [3, 5, 8, 12, 18, 25, 35]
    alphas = [1.0, 0.8, 0.6, 0.45, 0.35, 0.2, 0.1]
    psf = [_run(PredictiveSafetyFilter(env.track, dt=env.dt, horizon=h,
                                       assumed_vehicle=veh), policy="fast")
           for h in horizons]
    cbf = [_run(DiscreteCBFFilter(env.track, dt=env.dt, alpha=a,
                                  assumed_vehicle=veh), policy="fast")
           for a in alphas]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    for ax, xs, res, xl, ttl in (
            (axes[0], horizons, psf, "backup horizon $N$ [steps]",
             "predictive filter"),
            (axes[1], alphas, cbf, r"CBF decay rate $\alpha$",
             "discrete CBF (braking barrier)")):
        offs = [r[0] for r in res]; ivs = [r[2] for r in res]
        ax.plot(xs, [o * 100 for o in offs], "o-", color=RED, label="episodes off-track")
        ax.plot(xs, [i * 100 for i in ivs], "s-", color="#1f4e9c", label="steps overridden")
        ax.set_xlabel(xl); ax.set_ylabel("%"); ax.set_title(ttl, fontsize=10)
        ax.set_ylim(-3, 103); ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    if alphas[0] > alphas[-1]:
        axes[1].invert_xaxis()
    fig.suptitle("conservatism, swept — full throttle and random steering, 14 episodes each.\n"
                 "A short horizon looks safest because it refuses almost everything; "
                 "the CBF's override rate barely responds to $\\alpha$ at all.",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "safety_knobs.png", dpi=160)
    print("  wrote safety_knobs.png")
    return dict(horizons=horizons, psf=psf, alphas=alphas, cbf=cbf)


# --------------------------------------------------------------------------
def fig_grip():
    """The filter's model is wrong. How wrong may it be?

    ``LaneKeep`` redraws the tyre grip every episode from U(0.6, 1.4) and never
    observes it. The filter predicts with a fixed ``assumed_grip``. Above the
    true value it certifies corners the car cannot take and the guarantee is
    void -- crashes happen *through* the filter. Below it, the filter is safe
    and increasingly unwilling to let anything happen at all.

    This is run under the ``fast`` policy for the reason given in
    :func:`_policy`: under a random one the car never reaches a speed at which
    grip binds, and the whole sweep comes out flat at zero -- which looks like
    a robust filter and is really a test that did not touch the variable.
    """
    env = make_env("lanekeep")
    veh = getattr(env, "vehicle", None)
    grips = [0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.4]
    res = [_run(PredictiveSafetyFilter(env.track, dt=env.dt, assumed_grip=g,
                                       assumed_vehicle=veh), policy="fast")
           for g in grips]

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(grips, [r[0] * 100 for r in res], "o-", color=RED,
            label="episodes off-track  (guarantee void)")
    ax.plot(grips, [r[2] * 100 for r in res], "s-", color="#1f4e9c",
            label="steps overridden  (cost to the learner)")
    ax.axvspan(min(grips), 0.6, color=GREEN, alpha=0.10)
    ax.axvspan(0.6, 1.4, color="#e8a33d", alpha=0.10)
    ax.axvline(0.6, color=INK, lw=1.0, ls="--")
    ax.axvline(1.4, color=INK, lw=1.0, ls="--")
    ax.text(0.62, 96, "true grip is drawn from here, U(0.6, 1.4)",
            fontsize=7.5, color=INK, va="top")
    ax.text(1.44, 88, "optimistic for\nevery episode",
            fontsize=8, color=RED, va="top")
    ax.set_xlabel("assumed_grip  (the filter's belief; the car's is never observed)")
    ax.set_ylabel("%")
    ax.set_title("a safety filter is exactly as good as its model\n"
                 "(full throttle, random steering — the regime where grip binds)",
                 fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "safety_grip.png", dpi=160)
    print("  wrote safety_grip.png")
    for g, r in zip(grips, res):
        print(f"    grip {g:4.2f}   off-track {r[0]:5.0%}   overridden {r[2]:5.1%}")
    return dict(grips=grips, res=res)


FIGS = {"certificate": fig_certificate, "barrier_field": fig_barrier_field,
        "knobs": fig_knobs, "grip": fig_grip}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(FIGS))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (a.only or sorted(FIGS)):
        FIGS[name]()
