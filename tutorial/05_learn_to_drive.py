"""Lesson 5 -- learn to drive: RTRRL on a 1:10 car that cannot see its own speed.

    python tutorial/05_learn_to_drive.py
    python tutorial/05_learn_to_drive.py --steps 400000 --cells ctrnn ltc lrcu

Nine lidar beams, a 9-way action (three steering choices x three throttle
choices), and a reward that pays for arc length covered. The speed is not in
the observation, so throttle can only be chosen by a policy that has worked
out how fast it is going from how the beams are changing.
"""

# %% [markdown]
# # Lesson 5 — Learn to drive
#
# The environment is `lanekeep`: a kinematic bicycle at RC scale (0.33 m
# wheelbase, 0.4 rad of lock, 4 m/s top speed) on a 27 m oval with 2.5 m
# corners, at 20 Hz. Two details make it a real task rather than a demo:
#
# * **A grip limit.** The yaw rate is capped at what 0.6 g of lateral
#   acceleration can produce, so cornering speed is bounded and the throttle
#   choice matters. Without it a kinematic bicycle takes any corner at any
#   speed and flat out is always right.
# * **No speedometer.** Five ranges to a wall tell you where you are and which
#   way you point, and nothing about how fast you are arriving.
#
# Three reference points to keep the numbers honest: a hand-written
# wall-follower scores about **575**, a random policy about **23**, and the
# ceiling is 1.0 per step, so **600** over a 600-step episode.
#
# And one warning, because the point of this repo is not to flatter the
# algorithm: **this task does not actually need memory.** The memoryless
# control is in the comparison below and does about as well. Nine beams are a
# lot of information, and reacting to the forward one is a decent speed
# controller by itself. Lesson 5 is where you check that an agent can drive at
# all; lesson 6 is where the memory has to do work.

# %%
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import make_env
from rtrrl_playground.envs.scripted import WallFollower
from rtrrl_playground.train import rollout, train
from rtrrl_playground.utils.load import load_algo

RTRRL = load_algo("rtrrl")
ACLambda = load_algo("ac_lambda")

# %% [markdown]
# ## What the agent actually sees
#
# Worth printing once. Nine numbers in [0, 1], right to left, and that is the
# entire input apart from the previous action and the previous reward.

# %%
env = make_env("lanekeep")
obs = env.reset(seed=0)
print("  beams (right -> left, 1.0 = nothing within 5 m):")
print("   ", np.round(obs, 3))
for _ in range(10):
    obs, r, *_ = env.step(4)  # straight ahead, coasting
print("  ten steps later:")
print("   ", np.round(obs, 3), f"   reward this step: {r:+.3f}")
print(f"\n  hidden from the agent right now: v = {env.v:.2f} m/s, "
      f"steering = {env.delta:+.3f} rad, lateral offset = {env.history[-1]['d']:+.2f} m")

# %% [markdown]
# ## Reference points first
#
# Never look at a learning curve before you know what the numbers mean.

# %%
def score(policy, env_id="lanekeep", n=10, seed=500):
    e = make_env(env_id)
    out = rollout(e, policy, n_episodes=n, seed=seed)
    return float(out["returns"].mean()), float(out["returns"].std())


rng = np.random.default_rng(0)
random_policy = lambda o: int(rng.integers(9))
print(f"  random policy      {score(random_policy)[0]:7.1f}")
print(f"  scripted wall-follower {score(WallFollower())[0]:7.1f}")

# %% [markdown]
# ## Train
#
# One agent per cell, so the comparison is between the recurrent units and not
# between hyperparameters. Everything else — learning rates, entropy bonus,
# eligibility decays — is identical, and identical to the memoryless control.

# %%
def run(cell, steps, seed=0, algo=RTRRL, **kw):
    e = make_env("lanekeep")
    agent = algo(e.obs_dim, e.action_space, cell=cell, seed=seed, **kw)
    out = train(e, agent, steps, progress=False, seed=seed)
    return agent, e, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--cells", nargs="+", default=["ctrnn", "lrcu"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    outdir = ROOT / "runs"
    outdir.mkdir(exist_ok=True)
    curves = {}

    agent, e, out = run("mlp", args.steps, args.seed, algo=ACLambda)
    curves["memoryless control"] = out["curve"]
    print(f"  {'memoryless AC(lambda)':<24} final {np.mean(out['returns'][-20:]):7.1f}")

    for cell in args.cells:
        agent, e, out = run(cell, args.steps, args.seed)
        curves[f"RTRRL / {cell}"] = out["curve"]
        final = float(np.mean(out["returns"][-20:]))
        print(f"  {'RTRRL / ' + cell:<24} final {final:7.1f}   "
              f"({out['train_time_s']:.0f} s, influence {agent.cell.influence_bytes() / 1024:.0f} KiB)")
        ev = rollout(e, agent.eval_policy(), n_episodes=1, seed=10_000, keep_history=True)
        e.render_rollout(ev["history"], str(outdir / f"lesson05_{cell}.png"),
                         title=f"RTRRL / {cell} / lanekeep -- return {ev['returns'][0]:.0f}")

    plot(curves, outdir / "lesson05_curves.png")
    return curves


def plot(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, curve in curves.items():
        xs = [c[0] for c in curve]
        ys = [c[1] for c in curve]
        ax.plot(xs, ys, label=label)
    ax.axhline(575, color="tab:green", ls="--", lw=1, label="scripted wall-follower")
    ax.axhline(23, color="0.6", ls=":", lw=1, label="random")
    ax.set_xlabel("environment steps")
    ax.set_ylabel("return (20-episode mean)")
    ax.set_title("Learning to drive online, one update per timestep")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Reading the pictures
#
# Each `runs/lesson05_<cell>.png` is one greedy episode: the track, and the
# driven line coloured by speed. What to look for, in order of how much it
# tells you:
#
# * **Does it get round at all**, or does the line stop at a wall?
# * **Does the colour change in the corners?** A policy that has not worked out
#   its own speed drives one colour everywhere — it has found a throttle that
#   survives the corners and wastes the straights. Learning to lift *is* the
#   evidence that the recurrent state is carrying velocity.
# * **Does it use the width of the track?** Cutting the apex raises the reward
#   for free, because arc length along the centreline advances faster on the
#   inside of a corner. An agent that has found that is optimising the thing we
#   actually asked for.
