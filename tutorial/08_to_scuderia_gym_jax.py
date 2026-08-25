"""Lesson 8 -- take the agent off the toy and onto the real vehicle models.

    PYTHONPATH=/path/to/scuderia_gym_jax python tutorial/08_to_scuderia_gym_jax.py
    ... --steps 20000 --cell lrcu --map berlin

Requires `scuderia_gym_jax <https://github.com/agpoks/scuderia_gym_jax>`_ (and
``jax``, ``chex``). Everything else in this repo runs on NumPy alone.
"""

# %% [markdown]
# # Lesson 8 — Onto `scuderia_gym_jax`
#
# The `lanekeep` and `overtake` environments are throwaway: a kinematic bicycle
# with a one-line grip limit, a bitmap track, a nine-beam sensor. They exist so
# a lesson finishes on a laptop in a minute.
#
# The real thing — ST / STD / STD4W single- and double-track models, Pacejka
# and brush tyres fitted to actual RC-car recordings, load transfer, steering
# delay, a friction map — is in `scuderia_gym_jax`. This lesson swaps one for
# the other and changes nothing else, because the agent never knew what it was
# driving:
#
# ```python
# env = ScuderiaLaneKeep(model="std", map_name="berlin")
# agent = RTRRL(env.obs_dim, env.action_space, cell="lrcu")   # unchanged
# ```
#
# Three things are genuinely different once you cross over, and it is better to
# know them before the first run than to discover them in a plot.

# %%
import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground.train import rollout, train
from rtrrl_playground.utils.load import load_algo

RTRRL = load_algo("rtrrl")

# %% [markdown]
# ## 1. The reward is not the same objective
#
# The maps that ship with the simulator are occupancy images with no
# centreline, so there is no arc length to differentiate and the adapter pays
# for **distance travelled without crashing** instead. That is the usual
# f1tenth-style stand-in and it is a genuinely different objective: it will
# happily reward tight circles inside a wide corridor. If you want progress
# along a racing line, bring a centreline and replace `_reward` in
# `rtrrl_playground/envs/scuderia.py`.
#
# ## 2. A step costs about a hundred times more
#
# `lanekeep` runs at ~150 microseconds per step. The adapter runs at ~10
# milliseconds, and almost none of that is physics: `scuderia_gym_jax` is built
# to be `jit`-compiled around a whole `lax.scan` rollout and `vmap`ped over
# thousands of cars, and stepping it one tick at a time from a Python agent
# throws all of that away. Measured below, so you can see the number rather
# than take it on trust.
#
# The way out is to port the *agent* into JAX and put the whole loop inside the
# scan. RTRRL is unusually well suited to that — every array in
# `nets/cell.py` is fixed-shape, the update is a pure function of
# `(params, traces, observation)`, and there is no replay buffer, no dynamic
# indexing and no Python control flow in the update. It is the one thing this
# repo is deliberately set up for and deliberately has not done: NumPy first,
# so the algorithm is readable.
#
# ## 3. Batch size one is still batch size one
#
# `vmap` over 4096 environments does not help RTRRL. It is a single-stream
# algorithm: one car, one update per timestep. What `vmap` *is* good for here
# is running many independent **seeds** or hyperparameter settings at once,
# which is a different and still useful thing.

# %%
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--cell", default="lrcu")
    ap.add_argument("--model", default="st", help="ks | st | std | std4w")
    ap.add_argument("--map", dest="map_name", default="berlin")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    from rtrrl_playground.envs.scuderia import ScuderiaLaneKeep

    env = ScuderiaLaneKeep(model=args.model, map_name=args.map_name, seed=args.seed)
    print(f"  {args.model.upper()} vehicle model on '{args.map_name}', "
          f"obs_dim={env.obs_dim}, {env.action_space}")

    obs = env.reset(seed=args.seed)
    rng = np.random.default_rng(0)
    for _ in range(20):
        env.step(int(rng.integers(9)))       # warm the jit
    env.reset(seed=args.seed)
    t0 = time.perf_counter()
    n = 0
    for _ in range(100):
        _o, _r, term, trunc, _i = env.step(int(rng.integers(9)))
        n += 1
        if term or trunc:
            env.reset()
    print(f"  {(time.perf_counter() - t0) / n * 1e6:,.0f} us per agent step "
          f"({env.control_repeat} simulator ticks each, lidar on)")

    agent = RTRRL(env.obs_dim, env.action_space, cell=args.cell, seed=args.seed)
    print(f"  training {args.steps:,} steps -- expect roughly "
          f"{args.steps * 0.012 / 60:.0f} minutes")
    out = train(env, agent, args.steps, log_every=max(args.steps // 10, 1), seed=args.seed)
    print(f"  final return (last 10 episodes): "
          f"{np.mean(out['returns'][-10:]) if len(out['returns']) else float('nan'):.1f}")

    outdir = ROOT / "runs"
    outdir.mkdir(exist_ok=True)
    ev = rollout(env, agent.eval_policy(), n_episodes=1, seed=999, keep_history=True)
    env.render_rollout(ev["history"], str(outdir / f"lesson08_{args.model}_{args.map_name}.png"),
                       title=f"RTRRL / {args.cell} on scuderia_gym_jax / {args.model}")
    print(f"  wrote runs/lesson08_{args.model}_{args.map_name}.png")
    return out


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Where to go from here
#
# * **Bring a centreline** and make the reward progress rather than distance.
#   `scuderia_twin` and the MPCC planners already carry one per track.
# * **Turn on the vehicle model you care about.** `--model std` gives the
#   single-track drift model with independent wheel speeds; `--model std4w`
#   gives four corners with load transfer. Both are far harder to drive than
#   the kinematic bicycle in `lanekeep`, and both are where the argument for a
#   *recurrent* policy stops being pedagogical: sideslip is a state no range
#   sensor reports.
# * **Fine-tune rather than learn from scratch** (lesson 7). On a real vehicle
#   that is not a refinement, it is the only responsible option.
# * **Port the agent to JAX.** See the notes above; the update is already a
#   pure function of fixed-shape arrays.
