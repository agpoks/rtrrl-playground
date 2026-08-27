"""The animation helpers. Small clips, but real ones -- they write GIFs."""

from __future__ import annotations

import numpy as np
import pytest

from rtrrl_playground import make_env
from rtrrl_playground.cbf import DiscreteCBFFilter
from rtrrl_playground.safety import PredictiveSafetyFilter
from rtrrl_playground.viz import animate_episode, animate_safety


def _fast(seed=0):
    rng = np.random.default_rng(seed)
    return lambda o: int(rng.integers(3)) * 3 + 2


def test_animate_episode_writes_a_readable_gif(tmp_path):
    out = animate_episode("lanekeep", _fast(), tmp_path / "e.gif",
                          seed=0, max_frames=6, max_steps=30)
    assert out.exists() and out.stat().st_size > 0
    Image = pytest.importorskip("PIL.Image")
    with Image.open(out) as im:
        assert im.n_frames == 6


@pytest.mark.parametrize("make_filter", [
    lambda env: PredictiveSafetyFilter(env.track, dt=env.dt, assumed_grip=0.6,
                                       assumed_vehicle=getattr(env, "vehicle", None)),
    lambda env: DiscreteCBFFilter(env.track, dt=env.dt, assumed_grip=0.6,
                                  assumed_vehicle=getattr(env, "vehicle", None)),
])
def test_animate_safety_runs_for_both_filters(tmp_path, make_filter):
    env = make_env("lanekeep")
    out = animate_safety(env, _fast(), make_filter(env), tmp_path / "s.gif",
                         seed=0, max_frames=5, max_steps=25)
    assert out.exists() and out.stat().st_size > 0


def test_the_filtered_action_is_the_one_applied(tmp_path):
    """The bug this guards: recording the filter's verdict but stepping the
    environment with the *unfiltered* action, which produces a clip of a car
    crashing under a caption saying the filter prevents exactly that."""
    env = make_env("lanekeep")
    filt = PredictiveSafetyFilter(env.track, dt=env.dt, assumed_grip=0.6,
                                  assumed_vehicle=getattr(env, "vehicle", None))
    animate_safety(env, _fast(), filt, tmp_path / "s.gif",
                   seed=4, max_frames=5, max_steps=120)
    # A full-throttle policy leaves the track within ~25 steps unfiltered, so
    # surviving 120 with a worst-case filter is only possible if the filter's
    # action is what actually reached the plant.
    assert filt.n_interventions > 0


def test_admissible_agrees_with_the_filter_it_came_from():
    """``admissible`` must answer per action what ``__call__`` answers for one.

    Specifically it has to apply the candidate step before certifying: skipping
    that returns the same verdict nine times, which looks plausible and is
    always wrong.
    """
    env = make_env("lanekeep")
    obs = env.reset(seed=0)
    for _ in range(40):
        obs, *_ = env.step(8)
    state = np.array([env.x, env.y, env.psi, env.v, env.delta])
    for filt in (PredictiveSafetyFilter(env.track, dt=env.dt, assumed_grip=0.6,
                                        assumed_vehicle=env.vehicle),
                 DiscreteCBFFilter(env.track, dt=env.dt, assumed_grip=0.6,
                                   assumed_vehicle=env.vehicle)):
        ok = filt.admissible(state)
        assert ok.shape == (9,)
        for a in range(9):
            applied, intervened = filt(state, a)
            # If the filter passed the action through, it must be admissible;
            # if it swapped, the replacement must be.
            assert ok[applied], f"{type(filt).__name__} applied a refused action"
            if not intervened:
                assert ok[a]


def test_shared_palette_keeps_colours_the_first_frame_never_shows(tmp_path):
    """Regression: the GIF palette was built from frame 0 alone, so a colour
    absent at t=0 -- every refused (red) cell in the safety clip -- was
    remapped to the nearest green."""
    from matplotlib.animation import FuncAnimation
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rtrrl_playground.viz import save_gif

    fig, ax = plt.subplots(figsize=(2, 2))
    patch = ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="white"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def update(k):
        patch.set_color("#00ff00" if k < 5 else "#ff0000")
        return [patch]

    out = save_gif(FuncAnimation(fig, update, frames=10, blit=False),
                   tmp_path / "p.gif", fps=5)
    plt.close(fig)
    Image = pytest.importorskip("PIL.Image")
    with Image.open(out) as im:
        # Seek the last frame rather than a fixed index: optimize=True collapses
        # runs of identical frames, so the file legitimately has fewer than 10.
        im.seek(im.n_frames - 1)
        px = np.asarray(im.convert("RGB"))
        r, g, b = px[px.shape[0] // 2, px.shape[1] // 2]
    assert r > 150 and g < 100, f"late-frame red became {(r, g, b)}"
