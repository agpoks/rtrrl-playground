"""Animate what an agent does, and what a safety filter does to it.

Three functions, all returning the path they wrote:

:func:`animate_episode`
    one episode, with the lidar beams the agent is actually steering on. Works
    for ``lanekeep`` and ``overtake`` -- opponents are drawn if the environment
    has them.
:func:`animate_safety`
    the same, with the filter's verdict on all nine actions shown live. This is
    the one that pays for the module: intervention rate is a number in a table,
    and *watching* a filter take the throttle away in a corner is not.
:func:`animate_learning`
    the same policy at several points during training, so the improvement is
    visible rather than asserted by a learning curve.

All three write GIFs via Pillow -- no ffmpeg, no ImageMagick, and a GIF renders
on Read the Docs without a player. Keep ``fps`` and ``max_frames`` modest: an
uncompressed GIF of a 600-step episode is tens of megabytes, so the defaults
subsample.

    from rtrrl_playground.viz import animate_episode
    animate_episode("overtake", policy, "pass.gif")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from rtrrl_playground import make_env
from rtrrl_playground.envs.lanekeep import BEAM_ANGLES, SPEED_MAX

GREEN, RED, INK, BLUE = "#2a9d5c", "#c1272d", "#222222", "#1f4e9c"
ACTION_NAMES = ("brake", "coast", "accel")
STEER_NAMES = ("left", "straight", "right")


def _plt():
    """Import pyplot with a headless backend, unless one is already chosen.

    Animating is something you do from a script on a machine with no display at
    least as often as from a notebook, and ``Agg`` is the difference between a
    GIF and a crash. If the caller has already picked a backend -- a notebook
    has -- leave it alone.
    """
    if matplotlib.get_backend().lower().startswith("agg") or not matplotlib.is_interactive():
        try:
            matplotlib.use("Agg", force=False)
        except Exception:      # pragma: no cover - backend already locked in
            pass
    import matplotlib.pyplot as plt
    return plt


def save_gif(anim, out, fps: int = 20, colors: int = 128) -> Path:
    """Write a :class:`~matplotlib.animation.FuncAnimation` to ``out`` as a GIF.

    Matplotlib's Pillow writer emits full-colour frames with no optimisation,
    which for a 150-frame clip is megabytes of mostly-identical background. The
    re-save quantises to ``colors`` and turns on Pillow's frame differencing;
    on these plots -- flat fills, few hues -- it is worth roughly 3x and there
    is nothing visible to lose. Pass ``colors=0`` to skip it.
    """
    from matplotlib.animation import PillowWriter
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=fps))
    if colors:
        from PIL import Image
        im = Image.open(out)
        # One palette for the whole clip, derived from a *sample spread across
        # it*. Two traps here, both hit while building this:
        #
        # Quantising each frame on its own gives every frame a different
        # palette, which defeats the frame differencing and makes the file 3x
        # LARGER, not marginally.
        #
        # Building the shared palette from frame 0 alone is worse than a size
        # problem: whatever is not on screen at t=0 gets no entry, so in the
        # safety clip -- where nothing is refused in the first frame -- every
        # red cell was silently remapped to the nearest green. The palette has
        # to see the colours the clip will actually use.
        n = im.n_frames
        picks = sorted({int(round(i)) for i in np.linspace(0, n - 1, min(n, 12))})
        strip = []
        for i in picks:
            im.seek(i)
            strip.append(im.convert("RGB"))
        w, h = strip[0].size
        sheet = Image.new("RGB", (w, h * len(strip)))
        for k, fr in enumerate(strip):
            sheet.paste(fr, (0, k * h))
        base = sheet.quantize(colors=colors, method=Image.MEDIANCUT)
        frames = []
        for i in range(im.n_frames):
            im.seek(i)
            frames.append(im.convert("RGB").quantize(palette=base, dither=Image.NONE))
        before = out.stat().st_size
        frames[0].save(out, save_all=True, append_images=frames[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
        if out.stat().st_size > before:      # never make it worse
            anim.save(out, writer=PillowWriter(fps=fps))
    return out


def _track_bg(ax, track, show_center=True):
    """The drivable area from the *bitmap* -- which is what the beams hit."""
    ax.imshow(track.free.astype(float), origin="lower", cmap="Greys_r", alpha=0.25,
              extent=[track.origin[0], track.origin[0] + track.nx * track.res,
                      track.origin[1], track.origin[1] + track.ny * track.res])
    c, n, hw = track.center, track.normal, track.half_width
    for b in (c - hw * n, c + hw * n):
        ax.plot(*np.vstack([b, b[:1]]).T, color="0.25", lw=1.4)
    if show_center:
        ax.plot(*np.vstack([c, c[:1]]).T, color="0.7", lw=0.8, ls="--")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


def _roll(env, policy, seed, max_steps=None, intercept=None):
    """Drive one episode, keeping per-step frames.

    ``intercept(env, obs, proposed) -> (action_to_apply, extra)`` sits between
    the policy and the environment. ``extra`` is stored alongside the pose,
    which is how the safety animation records the filter's nine verdicts
    without this function knowing what a filter is -- and the returned action
    is what is actually applied, which is the part that must not be skipped:
    animating a filter while stepping the environment with the *unfiltered*
    action produces a clip of a car crashing under a caption saying it cannot.
    """
    obs = env.reset(seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    frames, info = [], {}
    for _ in range(max_steps or env.max_steps):
        a = policy(obs)
        extra = None
        if intercept is not None:
            a, extra = intercept(env, obs, a)
        ranges, flags = env._last_beams
        frames.append(dict(x=env.x, y=env.y, psi=env.psi, v=env.v,
                           ranges=np.asarray(ranges).copy(),
                           flags=np.asarray(flags).copy(),
                           opp=env._opp_xy().copy() if hasattr(env, "_opp_xy") else None,
                           extra=extra))
        obs, r, te, tr, info = env.step(a)
        if hasattr(policy, "observe"):
            policy.observe(r)
        if te or tr:
            break
    return frames, info


def _subsample(frames, max_frames):
    if max_frames and len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        return [frames[i] for i in idx]
    return frames


def _draw_car(ax, color=BLUE, size=90):
    return ax.scatter([], [], marker="o", s=size, color=color, zorder=6,
                      edgecolors="white", linewidths=1.2)


# ---------------------------------------------------------------------------
def animate_episode(env_or_name, policy, out, seed: int = 0, fps: int = 20,
                    max_frames: int = 220, beams: bool = True, follow: bool = False,
                    title: str | None = None, max_steps: int | None = None):
    """One episode as a GIF: the car, its trail, and the beams it steers on.

    ``env_or_name`` is an environment or a name for :func:`make_env`. ``follow``
    keeps the camera on the car, which is worth it on ``overtake`` where the
    interesting thing is a few metres wide, and wasteful on ``lanekeep`` where
    the whole point is the racing line around the lap.
    """
    plt = _plt()
    from matplotlib.animation import FuncAnimation

    env = make_env(env_or_name, seed=seed) if isinstance(env_or_name, str) else env_or_name
    frames, info = _roll(env, policy, seed, max_steps=max_steps)
    frames = _subsample(frames, max_frames)

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    _track_bg(ax, env.track)
    trail = ax.scatter([], [], s=7, c=[], cmap="viridis", vmin=0, vmax=SPEED_MAX, zorder=3)
    beam_lines = [ax.plot([], [], lw=1.3, alpha=0.85, color="tab:orange", zorder=4)[0]
                  for _ in BEAM_ANGLES] if beams else []
    opp_dots, = ax.plot([], [], "s", ms=9, color=RED, zorder=5)
    car = _draw_car(ax)
    txt = ax.text(0.015, 0.97, "", transform=ax.transAxes, va="top", fontsize=9,
                  family="monospace",
                  bbox=dict(fc="white", ec="0.7", alpha=0.85, boxstyle="round,pad=0.3"))
    ax.set_title(title or f"{type(env).__name__}: one episode, "
                          f"{len(frames)} frames", fontsize=10)
    if not follow:
        fig.colorbar(trail, ax=ax, label="speed [m/s]", shrink=0.85)

    xs, ys, vs = [], [], []

    def update(k):
        f = frames[k]
        xs.append(f["x"]); ys.append(f["y"]); vs.append(f["v"])
        trail.set_offsets(np.column_stack([xs, ys]))
        trail.set_array(np.asarray(vs))
        car.set_offsets([[f["x"], f["y"]]])
        for line, ang, r, flag in zip(beam_lines, BEAM_ANGLES, f["ranges"], f["flags"]):
            a = f["psi"] + ang
            line.set_data([f["x"], f["x"] + r * np.cos(a)], [f["y"], f["y"] + r * np.sin(a)])
            # A flagged beam is one that hit a car rather than a wall. The agent
            # gets that bit; drawing it in the same colour would hide the only
            # thing that distinguishes an opponent from scenery.
            line.set_color(RED if flag else "tab:orange")
        if f["opp"] is not None and len(f["opp"]):
            opp_dots.set_data(f["opp"][:, 0], f["opp"][:, 1])
        txt.set_text(f"t = {k:3d}\nv = {f['v']:4.1f} m/s")
        if follow:
            ax.set_xlim(f["x"] - 5.0, f["x"] + 5.0)
            ax.set_ylim(f["y"] - 3.3, f["y"] + 3.3)
        return [trail, car, opp_dots, txt, *beam_lines]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=1000 // fps)
    path = save_gif(anim, out, fps=fps)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def animate_safety(env_or_name, policy, filt, out, seed: int = 0, fps: int = 14,
                   max_frames: int = 170, title: str | None = None,
                   max_steps: int | None = None):
    """The filter deciding, live, next to the car it is deciding for.

    Left: the car. Right: the nine actions, green where a safe backup exists
    and red where none does, with the action the agent proposed outlined and
    the one actually executed filled. When those two differ, the filter
    intervened -- and unlike an intervention *rate*, you can see what it took
    away and why.

    ``filt`` is a
    :class:`~rtrrl_playground.safety.PredictiveSafetyFilter` or a
    :class:`~rtrrl_playground.cbf.DiscreteCBFFilter`; both expose the same
    interface, so both animate here.
    """
    plt = _plt()
    from matplotlib.animation import FuncAnimation
    from rtrrl_playground.safety import PredictiveSafetyFilter

    env = make_env(env_or_name, seed=seed) if isinstance(env_or_name, str) else env_or_name

    def intercept(e, _obs, proposed):
        """Ask the filter, record all nine verdicts, and apply what it returns."""
        state = np.array([e.x, e.y, e.psi, e.v, e.delta])
        obstacles = e._opp_xy() if hasattr(e, "_opp_xy") else None
        ok = filt.admissible(state, obstacles)
        applied, intervened = filt(state, int(proposed), obstacles)
        return int(applied), dict(ok=ok, proposed=int(proposed),
                                  applied=int(applied), intervened=bool(intervened))

    filt.reset_stats()
    frames, info = _roll(env, policy, seed, max_steps=max_steps, intercept=intercept)
    frames = _subsample(frames, max_frames)
    if info.get("off_track"):
        raise RuntimeError(
            "the car left the track behind the filter, which should be "
            "impossible with a correct model -- check assumed_grip and "
            "assumed_vehicle before publishing this clip")

    fig = plt.figure(figsize=(11.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.95, 1.0], wspace=0.16)
    ax, gx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    _track_bg(ax, env.track)
    trail = ax.scatter([], [], s=7, c=[], cmap="viridis", vmin=0, vmax=SPEED_MAX, zorder=3)
    opp_dots, = ax.plot([], [], "s", ms=9, color=RED, zorder=5)
    car = _draw_car(ax)
    ovr = ax.scatter([], [], s=150, marker="o", facecolors="none",
                     edgecolors=RED, linewidths=2.0, zorder=7)
    fig.colorbar(trail, ax=ax, label="speed [m/s]", shrink=0.82, pad=0.02)

    grid_im = gx.imshow(np.zeros((3, 3)), cmap="RdYlGn", vmin=-0.35, vmax=1.35)
    gx.set_xticks(range(3), ACTION_NAMES, fontsize=9)
    gx.set_yticks(range(3), STEER_NAMES, fontsize=9)
    marks = [[gx.text(j, i, "", ha="center", va="center", fontsize=15, color=INK)
              for j in range(3)] for i in range(3)]
    prop_box = gx.add_patch(plt.Rectangle((-.5, -.5), 1, 1, fill=False,
                                          ec=BLUE, lw=2.4, zorder=5))
    appl_box = gx.add_patch(plt.Rectangle((-.5, -.5), 1, 1, fill=False,
                                          ec=INK, lw=2.4, ls=":", zorder=6))
    kind = ("predictive filter" if isinstance(filt, PredictiveSafetyFilter)
            else "discrete CBF")
    gx.set_title(f"{kind}: the nine actions", fontsize=10)
    status = gx.set_xlabel("", fontsize=9)
    # Which box is which is not guessable, and the whole panel is unreadable
    # without it.
    gx.plot([], [], color=BLUE, lw=2.4, label="what the agent asked for")
    gx.plot([], [], color=INK, lw=2.4, ls=":", label="what actually ran")
    gx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              fontsize=8, frameon=False)
    fig.suptitle(title or "what the filter takes away, and when", fontsize=10.5)

    xs, ys, vs = [], [], []
    n_ovr = [0]

    def _cell(a):
        return 2 - (a // 3), a % 3          # row: left/straight/right, col: throttle

    def update(k):
        f = frames[k]
        e = f["extra"]
        xs.append(f["x"]); ys.append(f["y"]); vs.append(f["v"])
        trail.set_offsets(np.column_stack([xs, ys]))
        trail.set_array(np.asarray(vs))
        car.set_offsets([[f["x"], f["y"]]])
        if f["opp"] is not None and len(f["opp"]):
            opp_dots.set_data(f["opp"][:, 0], f["opp"][:, 1])

        g = np.zeros((3, 3))
        for a in range(9):
            i, j = _cell(a)
            g[i, j] = 1.0 if e["ok"][a] else 0.0
            marks[i][j].set_text("✓" if e["ok"][a] else "✗")
        grid_im.set_data(g)
        pi, pj = _cell(e["proposed"]); ai, aj = _cell(e["applied"])
        prop_box.set_xy((pj - .5, pi - .5))
        appl_box.set_xy((aj - .5, ai - .5))
        if e["intervened"]:
            n_ovr[0] += 1
            ovr.set_offsets([[f["x"], f["y"]]])
            status.set_text(f"OVERRIDDEN  ({n_ovr[0]}/{k + 1} steps so far)")
            status.set_color(RED)
        else:
            ovr.set_offsets(np.empty((0, 2)))
            status.set_text(f"agent's action allowed  ({n_ovr[0]}/{k + 1} overridden)")
            status.set_color(INK)
        return [trail, car, opp_dots, grid_im, ovr]

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False, interval=1000 // fps)
    path = save_gif(anim, out, fps=fps)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
def animate_learning(env_name, agent_factory, out, checkpoints=(0, 25_000, 100_000),
                     seed: int = 0, fps: int = 18, max_frames: int = 150,
                     title: str | None = None):
    """The same agent at several points in training, driving side by side.

    A learning curve says the return went up. This says what changed: the
    untrained policy leaves the track in a few seconds, and the trained one
    holds a line. Each panel is an evaluation episode at that many steps of
    online learning -- there is no replay buffer to reload, so training simply
    continues from one checkpoint to the next.
    """
    plt = _plt()
    from matplotlib.animation import FuncAnimation
    from rtrrl_playground.train import train

    env = make_env(env_name, seed=seed)
    agent = agent_factory(env)
    runs, done = [], 0
    for c in checkpoints:
        if c > done:
            train(env, agent, c - done, progress=False, seed=seed)
            done = c
        frames, _ = _roll(make_env(env_name, seed=seed + 1), agent.eval_policy(), seed + 1)
        runs.append((c, _subsample(frames, max_frames)))

    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 3.7))
    axes = np.atleast_1d(axes)
    arts = []
    for ax, (c, fr) in zip(axes, runs):
        _track_bg(ax, env.track, show_center=False)
        tr = ax.scatter([], [], s=6, c=[], cmap="viridis", vmin=0, vmax=SPEED_MAX, zorder=3)
        ca = _draw_car(ax, size=70)
        ax.set_title(f"{c:,} steps of online learning".replace(",", " "), fontsize=10)
        # Two panels that both complete the lap look identical, and the reader
        # has no way to tell 25k from 150k. Metres driven separates them.
        lbl = ax.text(0.5, -0.04, "", transform=ax.transAxes, ha="center",
                      va="top", fontsize=9, family="monospace")
        arts.append((tr, ca, fr, [], [], [], lbl))
    fig.suptitle(title or f"{env_name}: the same agent, three points in training",
                 fontsize=11)

    longest = max(len(fr) for _c, fr in runs)

    def update(k):
        out_arts = []
        for tr, ca, fr, xs, ys, vs, lbl in arts:
            if k < len(fr):
                f = fr[k]
                xs.append(f["x"]); ys.append(f["y"]); vs.append(f["v"])
                tr.set_offsets(np.column_stack([xs, ys]))
                tr.set_array(np.asarray(vs))
                ca.set_offsets([[f["x"], f["y"]]])
                d = float(np.hypot(np.diff(xs), np.diff(ys)).sum()) if len(xs) > 1 else 0.0
                lbl.set_text(f"{d:5.1f} m driven   v {f['v']:4.1f} m/s")
            out_arts += [tr, ca, lbl]
        return out_arts

    anim = FuncAnimation(fig, update, frames=longest, blit=False, interval=1000 // fps)
    path = save_gif(anim, out, fps=fps)
    plt.close(fig)
    return path
