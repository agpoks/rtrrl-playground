"""Generate every figure the docs use, into ``docs/source/_static/plots/``.

    python scripts/make_figures.py              # all of them (~10 min)
    python scripts/make_figures.py --quick      # short training, for a smoke test
    python scripts/make_figures.py --only tracks

The figures are committed (``.gitignore`` excludes ``*.png`` *except* under
``docs/source/_static``), because a docs build on Read the Docs cannot train an
agent -- and because a picture of the track is the fastest way to understand
what these environments are.

Everything here is reproducible from a seed; nothing is hand-tuned for looks.
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
from rtrrl_playground.envs.lanekeep import BEAM_ANGLES, BEAM_RANGE, SPEED_MAX  # noqa: E402
from rtrrl_playground.envs.scripted import Overtaker, WallFollower  # noqa: E402
from rtrrl_playground.envs.track import Track  # noqa: E402
from rtrrl_playground.train import rollout, train  # noqa: E402
from rtrrl_playground.utils.load import load_algo  # noqa: E402

OUT = ROOT / "docs" / "source" / "_static" / "plots"


def _track_bg(ax, track, show_center=True):
    """Draw the drivable area from the *bitmap*, which is what the beams hit."""
    ax.imshow(track.free.astype(float), origin="lower", cmap="Greys_r", alpha=0.25,
              extent=[track.origin[0], track.origin[0] + track.nx * track.res,
                      track.origin[1], track.origin[1] + track.ny * track.res])
    c, n, hw = track.center, track.normal, track.half_width
    for b in (c - hw * n, c + hw * n):
        ax.plot(*np.vstack([b, b[:1]]).T, color="0.25", lw=1.4)
    if show_center:
        ax.plot(*np.vstack([c, c[:1]]).T, color="0.7", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def fig_tracks():
    """The two track shapes, with what the car can and cannot fit through."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (name, tr) in zip(axes, (("oval", Track.oval()), ("curvy", Track.curvy()))):
        _track_bg(ax, tr)
        r = 1.0 / np.maximum(np.abs(tr.curvature), 1e-6)
        sc = ax.scatter(tr.center[:, 0], tr.center[:, 1], c=np.minimum(r, 8.0),
                        s=9, cmap="plasma_r", zorder=3)
        fig.colorbar(sc, ax=ax, label="corner radius [m]")
        v = np.sqrt(6.0 * np.minimum(r, 8.0))
        ax.set_title(f"{name}: {tr.length:.0f} m lap, half-width {tr.half_width:.2f} m\n"
                     f"grip-limited speed {v.min():.1f}-{min(v.max(), SPEED_MAX):.1f} m/s")
    fig.suptitle("The tracks, coloured by corner radius — tighter is slower", y=1.0)
    fig.tight_layout()
    _save(fig, "tracks.png")


def fig_observation():
    """What the agent actually sees: nine beams, and nothing else."""
    # Drive with the scripted wall-follower, not with a fixed action: full
    # throttle in a straight line leaves the track at the first corner, and a
    # terminal observation is all zeros, which makes for a very confusing figure.
    env = make_env("lanekeep")
    obs = env.reset(seed=2)
    pol = WallFollower()
    for _ in range(70):
        nxt, _r, te, tr, _i = env.step(pol(obs))
        if te or tr:
            obs = env.reset(seed=2)
            pol.reset()
            continue
        obs = nxt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    _track_bg(ax1, env.track, show_center=False)
    ranges, _flags = env._last_beams
    for ang, r in zip(BEAM_ANGLES, ranges):
        a = env.psi + ang
        ax1.plot([env.x, env.x + r * np.cos(a)], [env.y, env.y + r * np.sin(a)],
                 color="tab:orange", lw=1.4, alpha=0.9, zorder=4)
        ax1.plot(env.x + r * np.cos(a), env.y + r * np.sin(a), ".", color="tab:red", zorder=5)
    ax1.plot(env.x, env.y, "o", color="tab:blue", ms=9, zorder=6)
    lim = 4.0
    ax1.set_xlim(env.x - lim, env.x + lim)
    ax1.set_ylim(env.y - lim, env.y + lim)
    ax1.set_title("nine beams, ±60°, ray-marched against the bitmap")

    ax2.bar(range(9), obs[:9], color="tab:orange")
    ax2.set_xticks(range(9))
    ax2.set_xticklabels([f"{np.degrees(a):+.0f}°" for a in BEAM_ANGLES], fontsize=8)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("range / 5 m")
    ax2.set_title(f"the whole observation\n"
                  f"(hidden: v = {env.v:.2f} m/s, grip = {env.grip:.2f})")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "observation.png")


def _best_episode(make, policy, seeds, key="overtakes"):
    """Pick an episode that actually shows the thing the caption claims.

    Not cherry-picking a *policy* -- the same trained agent, over a fixed and
    stated set of evaluation seeds, choosing the episode with the most passes.
    A figure captioned "a pass is the blue line crossing an orange trail" and
    showing no pass teaches nothing.
    """
    best = None
    for sd in seeds:
        env = make()
        if hasattr(policy, "reset"):
            policy.reset()
        hist, info = _drive(env, policy, sd)
        score = (info.get(key, 0), not info.get("crashed", False), len(hist))
        if best is None or score > best[0]:
            best = (score, hist, info, sd)
    return best[1], best[2], best[3]


def _drive(env, policy, seed):
    obs = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    info = {}
    for _ in range(env.max_steps):
        obs, r, te, tr, info = env.step(policy(obs))
        if hasattr(policy, "observe"):
            policy.observe(r)
        if te or tr:
            break
    return list(env.history), info


def fig_lanekeep(steps, seed=0):
    """A trained agent driving, next to the scripted reference."""
    env = make_env("lanekeep", seed=seed)
    agent = load_algo("rtrrl")(env.obs_dim, env.action_space, cell="ligru", seed=seed)
    out = train(env, agent, steps, progress=False, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (label, pol) in zip(axes, (("scripted wall-follower", WallFollower()),
                                       ("RTRRL / ligru, learned online", agent.eval_policy()))):
        hist, info = _drive(make_env("lanekeep"), pol, 10_000)
        _track_bg(ax, env.track)
        xs = np.array([h["x"] for h in hist])
        ys = np.array([h["y"] for h in hist])
        vs = np.array([h["v"] for h in hist])
        sc = ax.scatter(xs, ys, c=vs, s=5, cmap="viridis", vmin=0, vmax=SPEED_MAX, zorder=3)
        fig.colorbar(sc, ax=ax, label="speed [m/s]")
        ax.plot(xs[0], ys[0], "o", color="tab:green", ms=8, zorder=4)
        ax.set_title(f"{label}\n{len(hist)} steps, "
                     f"{'left the track' if info.get('off_track') else 'stayed on'}")
    fig.suptitle("lanekeep — colour is speed, and the corners are where it has to lift", y=1.0)
    fig.tight_layout()
    _save(fig, "lanekeep.png")
    return out


def fig_overtake(steps, seed=0):
    """A pass, drawn: the ego line crossing a traffic trail and staying ahead."""
    env = make_env("overtake", seed=seed)
    agent = load_algo("rtrrl")(env.obs_dim, env.action_space, cell="ligru", seed=seed)
    train(env, agent, steps, progress=False, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    seeds = list(range(700, 712))
    for ax, (label, pol) in zip(axes, (("scripted overtaker", Overtaker()),
                                       ("RTRRL / ligru, learned online", agent.eval_policy()))):
        hist, info, sd = _best_episode(lambda: make_env("overtake"), pol, seeds)
        _track_bg(ax, make_env("overtake").track)
        opp = np.stack([h["opp"] for h in hist if "opp" in h])
        for j in range(opp.shape[1]):
            ax.plot(opp[:, j, 0], opp[:, j, 1], color="tab:orange", lw=1.1, alpha=0.55, zorder=2)
        ax.plot(opp[-1, :, 0], opp[-1, :, 1], "s", color="tab:orange", ms=7,
                label="traffic", zorder=5)
        xs = np.array([h["x"] for h in hist])
        ys = np.array([h["y"] for h in hist])
        vs = np.array([h["v"] for h in hist])
        sc = ax.scatter(xs, ys, c=vs, s=5, cmap="viridis", vmin=0, vmax=SPEED_MAX, zorder=3)
        fig.colorbar(sc, ax=ax, label="speed [m/s]")
        if info.get("crashed"):
            ax.plot(xs[-1], ys[-1], "X", color="tab:red", ms=13, label="contact", zorder=6)
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(f"{label}\n{info.get('overtakes', 0)} passes, "
                     f"{'crashed' if info.get('crashed') else 'clean'}, {len(hist)} steps"
                     f"  (best of {len(seeds)} eval seeds)")
    fig.suptitle("overtake — a pass is the blue line crossing an orange trail "
                 "and staying ahead of it", y=1.0)
    fig.tight_layout()
    _save(fig, "overtake.png")


def fig_curves(out_lanekeep):
    """One learning curve, so the docs show what online learning looks like."""
    if out_lanekeep is None:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    curve = out_lanekeep["curve"]
    ax.plot([c[0] for c in curve], [c[1] for c in curve], color="tab:blue")
    ax.axhline(575, color="tab:green", ls="--", lw=1, label="scripted wall-follower")
    ax.axhline(23, color="0.6", ls=":", lw=1, label="random")
    ax.set_xlabel("environment steps (one update each)")
    ax.set_ylabel("return, 20-episode mean")
    ax.set_title("lanekeep: RTRRL learning online, one update per timestep")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "learning_curve.png")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="short training, for a smoke test")
    ap.add_argument("--only", default=None,
                    choices=["tracks", "observation", "lanekeep", "overtake"])
    args = ap.parse_args()
    steps = 8_000 if args.quick else 300_000

    if args.only in (None, "tracks"):
        fig_tracks()
    if args.only in (None, "observation"):
        fig_observation()
    out = None
    if args.only in (None, "lanekeep"):
        out = fig_lanekeep(steps)
        fig_curves(out)
    if args.only in (None, "overtake"):
        fig_overtake(steps if not args.quick else 8_000)


if __name__ == "__main__":
    main()
