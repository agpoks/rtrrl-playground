"""Lesson 1 -- why any of this is necessary: partial observability.

    python tutorial/01_the_problem.py

Before building an algorithm it is worth being precise about what is wrong
with the simple thing. This lesson shows a task no memoryless policy can
solve, measures a memoryless agent failing at it, and measures the same agent
succeeding the moment the missing state is handed back.
"""

# %% [markdown]
# # Lesson 1 — The problem: you cannot see the whole state
#
# An RC car's sensors give you range, not velocity. A camera gives you a
# picture, not a rate. Almost anything you can actually mount on a small
# vehicle observes *positions*, and almost every control decision depends on
# *derivatives*. That gap is the reason this repo exists, so it is worth making
# it concrete before writing any algorithm.
#
# Formally: the environment is a POMDP, the observation `o_t` is not the state
# `s_t`, and a policy of the form `pi(a | o_t)` is choosing one action for
# every situation that *looks* alike — however different those situations are.

# %%
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import make_env
from rtrrl_playground.train import train
from rtrrl_playground.utils.load import load_algo

ACLambda = load_algo("ac_lambda")   # the memoryless online actor-critic
RTRRL = load_algo("rtrrl")          # the recurrent one

# %% [markdown]
# ## The cleanest possible case: MemoryChain
#
# Show the agent a bit at `t = 0`. Show it nothing for `n - 1` steps. Ask for
# the bit back at the end: `+1` right, `-1` wrong, `0` in between.
#
# There is no dynamics to get wrong, no exploration problem, and the optimal
# return is exactly `1`. There is also, for a memoryless policy, *no
# information at all* — at the moment it is asked, the observation is
# identical whichever bit it saw. The best such a policy can do is guess, for
# an expected return of `0`. This is not a hard task; it is an impossible one,
# and that is the point.

# %%
env = make_env("memory-chain", length=8)
print("observation at each step of one episode (bit, time, query flag):")
o = env.reset(seed=0)
for t in range(env.length):
    print(f"  t={t}  obs={np.round(o, 3)}")
    o, r, term, trunc, info = env.step(0)
print(f"  the bit to remember was {info['bit']}; reward at the end was {r:+.0f}")

# %% [markdown]
# ## Measured, not asserted
#
# Run both agents. Same TD(λ), same traces, same learning rates — the *only*
# difference is whether the policy is a function of the current observation or
# of a recurrent state.

# %%
def run(agent_cls, env_id, steps=60_000, seed=0, **env_kw):
    env = make_env(env_id, **env_kw)
    agent = agent_cls(env.obs_dim, env.action_space, lr_actor=1e-3, lr_critic=0.1,
                      lr_rnn=1e-3, entropy_coef=0.03, seed=seed)
    out = train(env, agent, steps, progress=False, seed=seed)
    rets = out["returns"]
    return float(rets[-len(rets) // 5:].mean()), len(rets)


for name, cls in (("memoryless AC(lambda)", ACLambda), ("RTRRL (recurrent)", RTRRL)):
    score, n_eps = run(cls, "memory-chain", steps=60_000, length=8)
    print(f"  {name:<24} MemoryChain-8: mean return over the last fifth "
          f"= {score:+.3f}   ({n_eps} episodes)")

# %% [markdown]
# ## The same point on something that drives
#
# CartPole is the standard demonstration. `obs_mode="vel"` hides the two
# velocities, leaving cart position and pole angle — so the agent can see
# *where* everything is and not *where it is going*. `obs_mode="full"` puts
# them back.
#
# The gap between those two rows is the size of the hole. Everything else in
# this repo is about filling it without a replay buffer.

# %%
for mode in ("full", "vel"):
    score, n_eps = run(ACLambda, "cartpole-vel", steps=60_000, obs_mode=mode)
    label = "MDP (velocities visible)" if mode == "full" else "POMDP (velocities hidden)"
    print(f"  memoryless AC(lambda) on CartPole, {label:<26} return = {score:7.1f}")

# %% [markdown]
# ## And on the actual task
#
# `lanekeep` gives nine lidar beams and nothing else. Five ranges pin down
# where the car is and which way it is pointing; none of them says how fast it
# is arriving. Two cars, one crawling and one on the limit, produce identical
# observations, so a memoryless policy has to pick one throttle for both.
#
# Run it and look at the picture it saves: the memoryless car either creeps
# (safe, slow) or commits to a speed that works on the straight and puts it in
# the wall at the first corner.

# %%
score, n_eps = run(ACLambda, "lanekeep", steps=60_000)
print(f"  memoryless AC(lambda) on lanekeep: return = {score:7.1f} over {n_eps} episodes")
print("  (for scale: a scripted wall-follower gets ~573, and crashing early gets ~30)")

# %% [markdown]
# ## Where this goes
#
# The fix is a policy with an internal state — but a recurrent policy has to be
# *trained*, and the standard way to train one, backpropagation through time,
# wants the whole episode in memory and cannot produce an update until the
# episode is over. Neither of those is available to something learning while it
# drives.
#
# Lesson 2 builds the alternative: gradients that travel forwards.
