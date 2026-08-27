"""Architecture diagrams, in the ICRA template's visual language.

    python scripts/make_diagrams.py

The repo's TikZ sources live in ``docs/tikz/`` and are what should go into a
paper -- they are vector, they inherit the document's fonts, and they are what
``\\input{}`` expects. This script renders the *same* figures as PNGs for the
docs, because Read the Docs cannot run LaTeX and a docs page with no picture is
a docs page nobody reads.

The style is copied deliberately from the ICRA template's ``fig_pipeline``:

* **greyscale only** -- no colour survives a black-and-white print;
* rounded 2 pt corners, thin black outlines;
* solid white boxes for things that *compute*, ``black!8`` for the *learned*
  components, dashed ``black!4`` for things that are *data*, ``black!12`` for
  carried *state*;
* ``Latex``-style arrowheads, and italic marginal notes for the claims the
  boxes cannot make themselves.

Keeping the two in step is manual. If you change a diagram, change both.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "docs" / "source" / "_static" / "diagrams"

# The template's four greys, as fractions of black.
LEARNED, PROC, DATA, STATE = "0.92", "white", "0.96", "0.88"
EDGE = "0.15"


def box(ax, xy, w, h, text, fill=PROC, dashed=False, fontsize=9, sub=None):
    """One node, in the template's box style."""
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.0,rounding_size=0.06",
        linewidth=1.0, edgecolor=EDGE, facecolor=fill,
        linestyle=(0, (3, 2)) if dashed else "solid", zorder=3))
    # The subtitle is drawn as its own italic text rather than wrapped in
    # mathtext: "$\it{9 beams}$" renders as "9beams", because mathtext drops
    # the spaces. Two calls, and the spacing stays what it looks like.
    if sub is None:
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                zorder=4, linespacing=1.45)
    else:
        ax.text(x, y + 0.17, text, ha="center", va="center", fontsize=fontsize,
                zorder=4)
        ax.text(x, y - 0.19, sub, ha="center", va="center",
                fontsize=fontsize - 1.2, style="italic", color="0.25", zorder=4)


def arrow(ax, p0, p1, label=None, dashed=False, lblpos=0.5, dx=0.0, dy=0.10,
          via=None):
    """A ``Latex``-headed arrow, optionally routed through waypoints."""
    pts = [p0] + list(via or []) + [p1]
    for a, b in zip(pts[:-1], pts[1:]):
        ax.add_patch(FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=11, linewidth=1.3,
            color=EDGE, shrinkA=0, shrinkB=0, zorder=2,
            linestyle=(0, (4, 2)) if dashed else "solid"))
    if label:
        a, b = pts[max(0, len(pts) - 2)], pts[-1]
        ax.text(a[0] + (b[0] - a[0]) * lblpos + dx,
                a[1] + (b[1] - a[1]) * lblpos + dy,
                label, ha="center", va="bottom", fontsize=8, zorder=5)


def note(ax, xy, text, fontsize=8):
    ax.text(*xy, text, ha="center", va="center", fontsize=fontsize,
            style="italic", color="0.25", zorder=5)


def group(ax, x0, y0, x1, y1, label=None):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.0,rounding_size=0.08",
        linewidth=0.9, edgecolor="0.45", facecolor="none",
        linestyle=(0, (1, 2)), zorder=1))
    if label:
        note(ax, ((x0 + x1) / 2, y0 - 0.22), label)


def _canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# --------------------------------------------------------------------------
def fig_rtrrl():
    """What is carried forward, and what is never carried back."""
    fig, ax = _canvas(11.5, 5.6, (-1.6, 16.2), (-6.0, 1.4))
    box(ax, (0, 0), 2.6, 1.0, "observation", DATA, dashed=True, sub="9 beams")
    box(ax, (0, -1.5), 2.6, 0.95, "previous", DATA, dashed=True, sub="action, reward")
    box(ax, (3.9, -0.7), 3.4, 1.1, "recurrent cell", LEARNED,
        sub="CT-RNN / LTC / LiGRU")
    box(ax, (7.4, -0.7), 1.5, 0.85, "$h_t$", STATE)
    box(ax, (10.6, 0.25), 2.6, 0.95, "actor", LEARNED, sub="$\\pi_\\theta(a\\,|\\,h)$")
    box(ax, (10.6, -1.7), 2.6, 0.95, "critic", LEARNED, sub="$V_w(h)$")
    box(ax, (14.4, -0.7), 2.2, 0.95, "environment", PROC)

    arrow(ax, (1.3, 0), (2.2, -0.35))
    arrow(ax, (1.3, -1.5), (2.2, -1.05))
    arrow(ax, (5.6, -0.7), (6.65, -0.7))
    arrow(ax, (8.15, -0.55), (9.3, 0.25))
    arrow(ax, (8.15, -0.85), (9.3, -1.7))
    arrow(ax, (11.9, 0.25), (13.3, -0.4), "$a_t$", dy=0.05)
    arrow(ax, (14.4, -1.18), (0, -2.0), via=[(14.4, -3.3), (0, -3.3)])
    note(ax, (8.6, -3.05), "$r_t$, $a_t$ fed back as input — this is the "
                           "\"meta\" in meta-RL")

    box(ax, (10.6, -4.7), 2.6, 0.9, r"TD error $\delta_t$", PROC)
    box(ax, (5.4, -4.7), 4.0, 1.0, "online estimator", PROC,
        sub="RFLO / SnAp-1 / UORO / RTRL")
    box(ax, (1.0, -4.7), 2.4, 1.0, "influence", STATE, sub="$\\partial h_t/\\partial\\theta$")
    arrow(ax, (10.6, -2.18), (10.6, -4.25))
    arrow(ax, (9.3, -4.7), (7.4, -4.7))
    arrow(ax, (3.4, -4.7), (2.2, -4.7))
    arrow(ax, (1.0, -4.2), (2.5, -1.25))
    note(ax, (0.6, -2.6), "carried\nto $t{+}1$")
    arrow(ax, (11.9, -4.7), (11.9, -2.18), dashed=True)

    group(ax, -0.4, -5.35, 7.6, -4.05, "carried forward: no backward pass, "
                                       "no replay, no batch")
    note(ax, (3.9, 0.35), "one update per environment step")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "rtrrl_architecture.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  wrote rtrrl_architecture.png")


def fig_safety():
    """The filter as a gate on the action path, not a block in series."""
    fig, ax = _canvas(10.0, 5.4, (-1.8, 11.0), (-5.9, 1.2))
    box(ax, (0, 0), 2.6, 0.95, "policy", LEARNED, sub="or MPCC")
    ax.add_patch(plt.Polygon([(4.3, 0.62), (6.0, 0), (4.3, -0.62), (2.6, 0)],
                             closed=True, facecolor="white", edgecolor=EDGE,
                             linewidth=1.0, zorder=3))
    ax.text(4.3, 0, "backup\nexists?", ha="center", va="center", fontsize=8.5, zorder=4)
    box(ax, (8.6, 0), 2.2, 0.95, "actuator", PROC)
    box(ax, (4.3, -2.5), 3.8, 1.1, "nearest input for which\none does", PROC,
        sub="$\\min\\|u-u_L\\|$")

    arrow(ax, (1.3, 0), (2.6, 0), "$u_L$")
    arrow(ax, (6.0, 0), (7.5, 0), "yes, unchanged")
    arrow(ax, (4.3, -0.62), (4.3, -1.95), "no", dx=0.28, dy=-0.45)
    arrow(ax, (6.2, -2.5), (8.6, -0.48), via=[(8.6, -2.5)])

    box(ax, (0, -2.5), 2.6, 1.0, "vehicle model", DATA, dashed=True,
        sub="+ assumed grip")
    box(ax, (0, -4.6), 2.9, 1.0, "roll the backup", PROC,
        sub="brake, steer to line")
    box(ax, (4.3, -4.6), 3.0, 1.0, r"$x_N \in \mathcal{X}_{safe}$", STATE,
        sub="stopped, on track")
    arrow(ax, (0, -3.0), (0, -4.1))
    arrow(ax, (1.45, -4.6), (2.8, -4.6))
    arrow(ax, (4.3, -4.1), (4.3, -3.05))

    group(ax, -1.6, -5.25, 6.0, -1.85,
          "the guarantee is a statement about the model, not the car")
    note(ax, (4.3, 0.95), "the learner is untouched in the interior")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "safety_pipeline.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  wrote safety_pipeline.png")


def fig_system():
    """Who believes what. The question "how is the system modelled" has four
    different answers depending on which box you are standing in, and that
    mismatch is where most of this repo's measured results come from."""
    fig, ax = _canvas(11.0, 5.2, (-1.8, 15.5), (-5.6, 1.6))
    box(ax, (6.6, 0.55), 7.4, 1.15, "THE PLANT", PROC,
        sub="kinematic bicycle + yaw-rate cap, grip redrawn every episode")
    note(ax, (6.6, 1.45), "the only thing that is actually true")

    box(ax, (0.4, -1.9), 3.0, 1.15, "the agent", LEARNED,
        sub="no model at all")
    box(ax, (4.6, -1.9), 3.0, 1.15, "the filter", DATA, dashed=True,
        sub="same equations, assumed grip")
    box(ax, (8.8, -1.9), 3.0, 1.15, "the MPCC", DATA, dashed=True,
        sub="no yaw-rate cap")
    box(ax, (13.0, -1.9), 3.0, 1.15, "scuderia", DATA, dashed=True,
        sub="slip angles, Pacejka")

    for x in (0.4, 4.6, 8.8, 13.0):
        arrow(ax, (6.6, -0.03), (x, -1.33))

    box(ax, (0.4, -4.3), 3.0, 1.05, "9 lidar beams", STATE,
        sub="+ previous a, r")
    box(ax, (4.6, -4.3), 3.0, 1.05, "true state", STATE, sub="privileged")
    box(ax, (8.8, -4.3), 3.0, 1.05, "true state", STATE, sub="privileged")
    box(ax, (13.0, -4.3), 3.0, 1.05, "the rung up", STATE, sub="a real plant")
    for x in (0.4, 4.6, 8.8, 13.0):
        arrow(ax, (x, -2.47), (x, -3.78))

    note(ax, (6.6, -5.25), "every guarantee on this page is a statement about "
                           "one of the dashed boxes, not about the solid one")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "system_models.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("  wrote system_models.png")


def fig_divergence():
    """Open-loop divergence, from benchmarks/results/models.json."""
    import json
    src = ROOT / "benchmarks" / "results" / "models.json"
    if not src.exists():
        print("  skipped divergence: run benchmarks/models.py first")
        return
    d = json.loads(src.read_text())
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    styles = {"kinematic, grip 0.6": ("0.15", (0, (5, 2))),
              "kinematic, grip 1.4": ("0.35", (0, (5, 2))),
              "REAL_VEHICLE": ("0.5", (0, (1, 1.5))),
              "scuderia ks": ("0.15", "solid"),
              "scuderia st": ("0.35", "solid"),
              "scuderia std": ("0.55", "solid")}
    t = None
    for name, r in sorted(d.items(), key=lambda kv: -kv[1]["final"]):
        c, ls = styles.get(name, ("0.6", "solid"))
        y = r["curve"]
        t = np.arange(len(y)) * 0.05
        ax.plot(t, y, color=c, linestyle=ls, linewidth=1.6, label=name)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance from the repo's kinematic bicycle [m]")
    ax.set_title("same commands, same start, different model\n"
                 "dashed: the same equations with different parameters",
                 fontsize=10)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "model_divergence.png", dpi=170)
    plt.close(fig)
    print("  wrote model_divergence.png")


FIGS = {"rtrrl": fig_rtrrl, "safety": fig_safety, "system": fig_system,
        "divergence": fig_divergence}

if __name__ == "__main__":
    for f in FIGS.values():
        f()
