"""Lesson 6 -- learn to overtake: the same car, plus traffic that never yields.

    python tutorial/06_learn_to_overtake.py
    python tutorial/06_learn_to_overtake.py --steps 600000 --cells lrcu ltc

The task where partial observability stops being a teaching device and starts
being the actual difficulty: whether to commit to a pass depends entirely on
how fast you are closing, and no single frame contains a closing rate.
"""

# %% [markdown]
# # Lesson 6 — Learn to overtake
#
# `overtake` is `lanekeep` with two slower cars holding the racing line. They
# do not brake, do not steer, and do not care that you are there — the same
# dumb traffic as `scuderia_gym_jax`'s `examples/overtake.py`, for the same
# reason: if the traffic reacts, you can no longer tell whose behaviour you are
# looking at.
#
# The observation gains a second channel per beam: **is this return a wall or a
# car?** Without it, a car two metres ahead and a wall two metres ahead are
# identical, and no amount of recurrence recovers a distinction the sensor
# never made.
#
# What stays hidden is the thing that decides everything: **the speed of the
# car in front**, drawn fresh each episode from 1.5–2.3 m/s. A closing rate can
# only be obtained by watching a range change over several frames. This is not
# a contrived difficulty — it is exactly why the scripted overtaker below,
# which is otherwise sensible, ends up in the back of somebody about half the
# time.
#
# Reward: arc-length progress as before, `+2` per car passed, `-5` and episode
# over on contact.

# %%
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import make_env
from rtrrl_playground.envs.scripted import Overtaker
from rtrrl_playground.train import rollout, train
from rtrrl_playground.utils.load import load_algo

RTRRL = load_algo("rtrrl")
ACLambda = load_algo("ac_lambda")

# %% [markdown]
# ## The observation, with a car in it

# %%
env = make_env("overtake")
obs = env.reset(seed=2)
for _ in range(40):
    obs, r, term, trunc, info = env.step(5)  # straight, full throttle
    if (obs[9:18] > 0.5).any():
        break
print("  beams (range, right -> left):", np.round(obs[:9], 2))
print("  is-a-car flags:              ", obs[9:18].astype(int))
print(f"  the flagged return is a car at {obs[:9][obs[9:18] > 0.5].min() * 5:.1f} m;")
print(f"  its speed -- {env._opp_v.min():.2f} m/s -- is not in the observation, "
      "and never will be.")

# %% [markdown]
# ## Reference points

# %%
def score(policy, n=20, seed=700):
    e = make_env("overtake")
    out = rollout(e, policy, n_episodes=n, seed=seed)
    passes = np.mean([i.get("overtakes", 0) for i in out["infos"]])
    crashes = np.mean([bool(i.get("crashed")) for i in out["infos"]])
    return float(out["returns"].mean()), passes, crashes


rng = np.random.default_rng(0)
for name, pol in (("random", lambda o: int(rng.integers(9))),
                  ("scripted overtaker", Overtaker())):
    ret, passes, crashes = score(pol)
    print(f"  {name:<20} return {ret:7.1f}   passes {passes:4.1f}   crash rate {crashes:.0%}")

# %% [markdown]
# ## Train

# %%
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--cells", nargs="+", default=["ctrnn", "lrcu"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    outdir = ROOT / "runs"
    outdir.mkdir(exist_ok=True)
    rows = []

    for label, cell, algo in ([("memoryless control", "mlp", ACLambda)]
                              + [(f"RTRRL / {c}", c, RTRRL) for c in args.cells]):
        e = make_env("overtake")
        agent = algo(e.obs_dim, e.action_space, cell=cell, lr_actor=1e-3, lr_critic=0.1,
                     lr_rnn=1e-3, entropy_coef=0.03, seed=args.seed)
        out = train(e, agent, args.steps, progress=False, seed=args.seed)
        ev = rollout(e, agent.greedy, n_episodes=20, seed=700, keep_history=True)
        passes = np.mean([i.get("overtakes", 0) for i in ev["infos"]])
        crashes = np.mean([bool(i.get("crashed")) for i in ev["infos"]])
        rows.append((label, float(ev["returns"].mean()), passes, crashes, out["curve"]))
        print(f"  {label:<20} return {rows[-1][1]:7.1f}   passes {passes:4.1f}   "
              f"crash rate {crashes:.0%}")
        e.render_rollout(ev["history"], str(outdir / f"lesson06_{cell}.png"),
                         title=f"{label} / overtake")

    plot(rows, outdir / "lesson06_curves.png")
    return rows


def plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, _ret, _p, _c, curve in rows:
        ax.plot([c[0] for c in curve], [c[1] for c in curve], label=label)
    ax.axhline(406, color="tab:green", ls="--", lw=1, label="scripted overtaker")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("return (20-episode mean)")
    ax.set_title("Learning to overtake without a closing-rate sensor")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## What to look for
#
# **Return is the wrong headline number here** and it is worth being explicit
# about why: an agent that never passes anybody and never crashes scores
# respectably, because progress alone pays. The numbers that say whether it has
# learned to *overtake* are the pass count and the crash rate, which is why
# both are printed and why the scripted policy — 2.4 passes, 50% crashes — is
# the reference rather than its return.
#
# In the rendered episode, the traffic leaves an orange trail. A pass is the
# ego's line crossing a trail and staying ahead of it. A crash is where the
# line stops on top of one.
