"""Turn the measured results in ``benchmarks/results/measured.json`` into figures.

    python scripts/make_result_figures.py

The JSON is the single source: the tables in the docs and these plots read the
same numbers, so they cannot drift apart. Each block in it records how it was
produced and the command that reproduces it.

Separate from ``scripts/make_figures.py``, which draws the *environments* and
has to train an agent to do it. This one only reads numbers, so it runs in a
second and can be regenerated on every docs change.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "source" / "_static" / "plots"
DATA = json.loads((ROOT / "benchmarks" / "results" / "measured.json").read_text())


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {(OUT / name).relative_to(ROOT)}")


def fig_cells():
    """The cell comparison, and the ordering inversion between the two tasks."""
    mc = {k: v for k, v in DATA["cells"]["memory_chain"].items() if not k.startswith("_")}
    lk = {k: v for k, v in DATA["cells"]["lanekeep"].items() if not k.startswith("_")}
    ot = {k: v for k, v in DATA["cells"]["overtake"].items() if not k.startswith("_")}
    cells = list(mc)
    x = np.arange(len(cells))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.6))

    ax1.bar(x, [mc[c][0] for c in cells], yerr=[mc[c][1] for c in cells],
            capsize=4, color=["tab:green" if c != "mlp" else "0.6" for c in cells])
    ax1.axhline(1.0, color="tab:blue", ls="--", lw=1, label="optimum")
    ax1.axhline(0.0, color="0.4", lw=1, label="guessing (= no memory)")
    ax1.set_xticks(x); ax1.set_xticklabels(cells, rotation=20, ha="right")
    ax1.set_ylabel("return")
    ax1.set_title("MemoryChain-8 — memory and nothing else")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")

    order = [c for c in cells]
    ax2.bar(x, [lk[c][0] for c in order], yerr=[lk[c][1] for c in order],
            capsize=4, color=["tab:orange" if c != "mlp" else "0.6" for c in order])
    ax2.axhline(DATA["cells"]["lanekeep"]["_scripted"], color="tab:green", ls="--", lw=1,
                label="scripted wall-follower")
    ax2.axhline(DATA["cells"]["lanekeep"]["_random"], color="0.5", ls=":", lw=1, label="random")
    ax2.set_xticks(x); ax2.set_xticklabels(order, rotation=20, ha="right")
    ax2.set_ylabel("return")
    ax2.set_title("lanekeep — driving, which barely needs memory")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")

    # On overtake, return is the wrong headline -- progress alone pays -- so this
    # panel plots the two numbers that say whether it learned to *overtake*.
    sc = DATA["cells"]["overtake"]["_scripted"]
    for c in ot:
        ax3.scatter(ot[c]["crashes"], ot[c]["passes"], s=110,
                    color="0.6" if c == "mlp" else "tab:purple", zorder=3)
        ax3.annotate(c, (ot[c]["crashes"], ot[c]["passes"]), fontsize=8,
                     textcoords="offset points", xytext=(7, 4))
    ax3.scatter(sc["crashes"], sc["passes"], marker="*", s=320, color="tab:green", zorder=4)
    ax3.annotate("scripted", (sc["crashes"], sc["passes"]), fontsize=8, color="tab:green",
                 textcoords="offset points", xytext=(-20, -14))
    ax3.set_xlabel("crash rate")
    ax3.set_ylabel("passes per episode")
    ax3.set_title("overtake — up and to the left is better\n(return is the wrong headline here)")
    ax3.grid(alpha=0.3)

    fig.suptitle("The ordering inverts across tasks — and on overtake every learned cell "
                 "roughly halves the scripted policy's crash rate", y=1.02, fontsize=10)
    fig.tight_layout()
    _save(fig, "results_cells.png")


def fig_safety():
    """Return against crashes-while-learning: the trade the filter actually makes."""
    d = {k: v for k, v in DATA["safety_filter"].items() if not k.startswith("_")}
    names = list(d)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.arange(len(names))
    colors = ["0.5", "tab:blue", "tab:cyan", "tab:green", "tab:red"]

    ax1.bar(x - 0.2, [d[n]["train_off"] for n in names], 0.4, label="during training", color=colors)
    ax1.bar(x + 0.2, [d[n]["eval_off"] for n in names], 0.4, label="at evaluation",
            color=colors, alpha=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
    ax1.set_ylabel("fraction of episodes ending in a wall")
    ax1.set_title("Crashes — the number the filter exists for")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")

    for n, c in zip(names, colors):
        ax2.errorbar(d[n]["train_off"], d[n]["return"][0], yerr=d[n]["return"][1],
                     fmt="o", ms=10, capsize=4, color=c, label=n)
    ax2.set_xlabel("crashes during training")
    ax2.set_ylabel("evaluation return")
    ax2.set_title("Up and to the left is better\n(the worst-case filter is both)")
    ax2.legend(fontsize=7, loc="lower right"); ax2.grid(alpha=0.3)
    fig.suptitle("Safety was better than free at the worst-case grip — and an "
                 "optimistic filter is worse than none", y=1.02, fontsize=10)
    fig.tight_layout()
    _save(fig, "results_safety.png")


def fig_sim_to_real():
    d = DATA["sim_to_real"]["unfiltered"]
    names = [k for k in d if not k.startswith("crash")]
    vals = [d[k][0] for k in names]
    errs = [d[k][1] for k in names]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    colors = ["tab:blue", "tab:red", "tab:green", "0.5"]
    ax.bar(range(len(names)), vals, yerr=errs, capsize=5, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split(". ", 1)[-1] for n in names], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("evaluation return on the other vehicle")
    ax.annotate("", xy=(1, d[names[1]][0]), xytext=(0, d[names[0]][0]),
                arrowprops=dict(arrowstyle="<->", color="tab:red"))
    ax.text(0.5, (d[names[0]][0] + d[names[1]][0]) / 2 + 8, "the sim-to-real gap",
            ha="center", color="tab:red", fontsize=9)
    ax.set_title("Online adaptation closes 45% of the gap — and learning from\n"
                 "scratch on the vehicle beats it, at the same total budget")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, "results_sim_to_real.png")


def fig_estimators():
    e = DATA["estimators"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))
    styles = {"rflo": "-o", "snap1": "-s", "uoro": "-^", "bptt-4": "--v", "bptt-16": "--d"}
    for k, st in styles.items():
        ax1.plot(e["delays"], e[k], st, label=k)
    ax1.axhline(1.0, color="0.6", ls=":", lw=0.8)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("steps the credit must travel back")
    ax1.set_ylabel("cosine to the exact gradient")
    ax1.set_title("What each estimator keeps")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    cost = {k: v for k, v in e["cost"].items() if not k.startswith("_")}
    ks = list(cost)
    ax2.scatter([cost[k][1] for k in ks], [cost[k][0] for k in ks], s=90, color="tab:purple")
    for k in ks:
        ax2.annotate(k, (cost[k][1], cost[k][0]), textcoords="offset points",
                     xytext=(8, 4), fontsize=9)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("microseconds per step")
    ax2.set_ylabel("influence carried [KiB]")
    ax2.set_title("What each estimator costs (n=32)")
    ax2.grid(alpha=0.3, which="both")
    fig.suptitle("RTRL is exact and 32x the memory; UORO is unbiased and, per sample, "
                 "the worst of the four", y=1.02, fontsize=10)
    fig.tight_layout()
    _save(fig, "results_estimators.png")


def main():
    fig_cells()
    fig_safety()
    fig_sim_to_real()
    fig_estimators()


if __name__ == "__main__":
    main()
