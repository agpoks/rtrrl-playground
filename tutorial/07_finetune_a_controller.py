"""Lesson 7 -- clone a controller offline, then improve it online while driving.

    python tutorial/07_finetune_a_controller.py
    python tutorial/07_finetune_a_controller.py --bc-steps 100000 --rl-steps 300000

This is the setting Lemmel, Resch, Farsang, Hasani, Rus & Grosu put on a real
1:10 car (arXiv:2602.02236): behavioural cloning from demonstrations offline,
then RTRRL fine-tuning during deployment. Reproduced here in miniature, with
the scripted overtaker standing in for the human driver.
"""

# %% [markdown]
# # Lesson 7 — Fine-tune a controller online
#
# Learning to drive from scratch by trial and error is the wrong thing to do on
# a real car, and RTRRL's authors do not propose it. The deployment story in
# their follow-up paper is:
#
# 1. **Clone** a controller offline from demonstrations — three laps of a human
#   driving, in their case.
# 2. **Fine-tune it online**, with RTRRL, while the car is running. The policy
#   improves within the first lap, with no human intervention.
#
# What makes that possible is exactly what this repo is about: the update is
# available every timestep, it costs a constant amount of memory, and there is
# no episode boundary to wait for. A method that needs a replay buffer and a
# backward pass cannot do this on the vehicle.
#
# Here the demonstrator is the scripted `Overtaker` from
# `rtrrl_playground/envs/scripted.py` — competent, and reliably bad in one
# specific way: it commits to a side from a single frame, has no idea how fast
# it is closing, and crashes about half the time. **That is what fine-tuning has
# to fix, and it is a defect no amount of cloning can remove**, because the
# demonstrations contain the crashes too.

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

# %% [markdown]
# ## Phase 1 — clone
#
# Run the expert, execute *its* actions, and push the network's policy towards
# them. No reward is involved and the critic is untouched; the only learning
# signal is `d log pi(a_expert)`, routed into the recurrent cell through the
# same feedback-aligned path RL uses.

# %%
def clone(agent, env, expert, steps, lr=1e-2, seed=0):
    obs = env.reset(seed=seed)
    expert.reset()
    agent.start(obs)
    matched, n = 0, 0
    for t in range(steps):
        a_expert = expert(obs)
        obs, reward, terminated, truncated, _ = env.step(a_expert)
        matched += int(agent.a == a_expert)
        n += 1
        out = agent.imitate(obs, a_expert, reward, terminated, truncated, lr=lr)
        if out is None:
            obs = env.reset()
            expert.reset()
            agent.start(obs)
    return matched / max(n, 1)


# %% [markdown]
# ## Phase 2 — fine-tune
#
# Same weights, same recurrent state machinery, learning signal swapped for the
# TD error. The critic is zeroed first: it has never seen a reward, and a
# confident wrong baseline in the first few hundred steps is exactly how a good
# cloned policy gets destroyed.

# %%
def evaluate(env, policy, n=20, seed=700):
    out = rollout(env, policy, n_episodes=n, seed=seed)
    return (float(out["returns"].mean()),
            float(np.mean([i.get("overtakes", 0) for i in out["infos"]])),
            float(np.mean([bool(i.get("crashed")) for i in out["infos"]])),
            out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bc-steps", type=int, default=60_000)
    ap.add_argument("--rl-steps", type=int, default=200_000)
    ap.add_argument("--cell", default="lrcu",
                    help="lrcu is the cell the hardware paper found works best with RTRRL")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    env = make_env("overtake")
    outdir = ROOT / "runs"
    outdir.mkdir(exist_ok=True)

    ret, passes, crashes, _ = evaluate(env, Overtaker())
    print(f"  expert (scripted)            return {ret:7.1f}  passes {passes:4.1f}  "
          f"crashes {crashes:.0%}")

    agent = RTRRL(env.obs_dim, env.action_space, cell=args.cell, lr_actor=1e-3,
                  lr_critic=0.1, lr_rnn=1e-3, entropy_coef=0.03, seed=args.seed)
    agreement = clone(agent, env, Overtaker(), args.bc_steps)
    ret_bc, passes_bc, crashes_bc, ev_bc = evaluate(env, agent.greedy)
    print(f"  after cloning ({args.bc_steps:,} steps, agreement {agreement:.0%}) "
          f"return {ret_bc:7.1f}  passes {passes_bc:4.1f}  crashes {crashes_bc:.0%}")
    env.render_rollout(ev_bc["history"], str(outdir / "lesson07_cloned.png"),
                       title=f"cloned from the scripted overtaker ({args.cell})")

    agent.begin_finetune()
    out = train(env, agent, args.rl_steps, progress=False, seed=args.seed)
    ret_ft, passes_ft, crashes_ft, ev_ft = evaluate(env, agent.greedy)
    print(f"  after RTRRL fine-tuning ({args.rl_steps:,} steps)      "
          f"return {ret_ft:7.1f}  passes {passes_ft:4.1f}  crashes {crashes_ft:.0%}")
    env.render_rollout(ev_ft["history"], str(outdir / "lesson07_finetuned.png"),
                       title=f"after RTRRL fine-tuning ({args.cell})")

    # The control: the same total budget, spent entirely on RL from scratch.
    scratch = RTRRL(env.obs_dim, env.action_space, cell=args.cell, lr_actor=1e-3,
                    lr_critic=0.1, lr_rnn=1e-3, entropy_coef=0.03, seed=args.seed)
    train(env, scratch, args.bc_steps + args.rl_steps, progress=False, seed=args.seed)
    ret_s, passes_s, crashes_s, _ = evaluate(env, scratch.greedy)
    print(f"  RTRRL from scratch, same total budget       "
          f"return {ret_s:7.1f}  passes {passes_s:4.1f}  crashes {crashes_s:.0%}")
    return dict(expert=ret, cloned=ret_bc, finetuned=ret_ft, scratch=ret_s)


if __name__ == "__main__":
    main()

# %% [markdown]
# ## What to look for
#
# * **Cloning should land near the expert and not above it.** Behavioural
#   cloning cannot exceed its demonstrations; if it does, the evaluation is
#   measuring luck. Watch the agreement percentage rather than the return.
# * **Fine-tuning should move the crash rate**, which is the expert's actual
#   defect. The return can improve for boring reasons (driving slightly faster
#   on the straights); the pass count and the crash rate are the numbers that
#   say whether the closing-rate problem got solved.
# * **The from-scratch control matters.** If it matches the fine-tuned agent on
#   the same total budget, the cloning bought nothing here — which is a real
#   possible outcome on a task this small, and a much more interesting result
#   than a table that only shows the flattering comparison.
