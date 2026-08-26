"""Lesson 10 -- a predictive safety filter, and what it actually costs.

    python tutorial/10_safety_filter.py
    python tutorial/10_safety_filter.py --steps 200000 --assumed-grip 0.6

A learning agent proposes an action; before it is applied, the filter asks
whether a safe backup plan still exists afterwards. If yes, the action goes
through untouched. If not, the nearest action for which one does goes through
instead.
"""

# %% [markdown]
# # Lesson 10 — A predictive safety filter
#
# Learning by trial and error means, definitionally, making the errors. On a
# simulated car that is free. On a real one it is a broken car, and it is the
# single biggest reason online RL does not get deployed.
#
# A **predictive safety filter** (Wabersich & Zeilinger,
# [Automatica 2021](https://arxiv.org/abs/1812.05506)) is the cleanest answer
# to that, and its virtue is what it does *not* do. It does not shape the
# reward, does not restrict the action space, does not constrain the policy
# class. It sits between the agent and the actuator and answers one question
# per step:
#
# > If I apply this action, does a **safe backup plan** still exist afterwards?
#
# Yes → apply it unchanged. No → apply the nearest action for which one does.
# The learner is left alone in the interior and constrained only at the
# boundary.
#
# Formally, at every step:
#
# $$\min_{u_0}\;\|u_0 - u_\text{learner}\| \quad \text{s.t.}\quad
#   x_{k+1}=f(x_k,u_k),\; x_k \in \mathcal{X},\; u_k \in \mathcal{U},\;
#   x_N \in \mathcal{X}_\text{safe}$$
#
# Here $\mathcal{X}_\text{safe}$ is **"stopped, and on the track"**: a car at
# rest inside the boundary can stay there forever, so it is control-invariant,
# and reaching it certifies that the episode need not end badly. The backup
# that gets there is full braking with the steering pointed back at the
# centreline. With nine discrete actions the minimisation is not a solver
# problem — it is enumerate-and-check, ordered by distance from what the
# learner asked for, so the argmin is exact.

# %%
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import make_env
from rtrrl_playground.envs.scripted import WallFollower
from rtrrl_playground.safety import PredictiveSafetyFilter, make_safe
from rtrrl_playground.train import rollout, train
from rtrrl_playground.utils.load import load_algo

RTRRL = load_algo("rtrrl")

# %% [markdown]
# ## Does it actually work?
#
# The strongest test is the dumbest policy. A uniformly random policy leaves
# the track in every single episode. Put the filter in front of it and it must
# never leave, ever — and if it does, the filter is not a filter.

# %%
def measure(policy_fn, use_filter, n_ep=20, seed0=100, **filter_kw):
    env = make_env("lanekeep")
    filt = PredictiveSafetyFilter(env.track, dt=env.dt, **filter_kw) if use_filter else None
    off, rets, lens = 0, [], []
    for ep in range(n_ep):
        obs = env.reset(seed=seed0 + ep)
        pol = policy_fn()
        R, info = 0.0, {}
        for t in range(env.max_steps):
            a = pol(obs)
            if filt is not None:
                a, _ = filt(np.array([env.x, env.y, env.psi, env.v, env.delta]), a)
            obs, r, te, tr, info = env.step(a)
            R += r
            if te or tr:
                break
        off += bool(info.get("off_track"))
        rets.append(R)
        lens.append(t + 1)
    return dict(off=off / n_ep, ret=float(np.mean(rets)), length=float(np.mean(lens)),
                iv=filt.intervention_rate if filt else 0.0,
                nosafe=filt.n_no_safe_action if filt else 0)


rng = np.random.default_rng(0)
print("  20 episodes each:\n")
print(f"  {'policy':<26}{'off-track':>10}{'return':>9}{'steps':>8}{'filtered':>10}")
for label, mk, filt in (("random", lambda: (lambda o: int(rng.integers(9))), False),
                        ("random + filter", lambda: (lambda o: int(rng.integers(9))), True),
                        ("wall-follower", WallFollower, False),
                        ("wall-follower + filter", WallFollower, True)):
    r = measure(mk, filt)
    print(f"  {label:<26}{r['off']:>10.0%}{r['ret']:>9.0f}{r['length']:>8.0f}{r['iv']:>10.1%}")

# %% [markdown]
# Two numbers matter there and they say opposite-sounding things.
#
# **The random policy goes from leaving the track every episode to never.**
# That is the guarantee doing its job, on the worst policy available.
#
# **The competent policy is filtered ~0% of the time.** A safety filter that is
# doing its job is *invisible* to a driver who was not going to crash anyway.
# If your filter intervenes constantly on a good policy, it is not a safety
# filter, it is a controller — and you are now training against it rather than
# against the task.
#
# ## Three limitations, stated before the results
#
# **It is privileged.** The filter runs on the vehicle *state*, not on the
# agent's nine beams. That is not cheating — on a real car the filter sits on
# the state estimator, which is where it belongs — but the guarantee is only
# ever as good as that estimate, and a filter tested against ground truth in
# simulation has not been tested in the part that usually fails.
#
# **It does not know the grip either.** `lanekeep` redraws the tyre grip every
# episode and never observes it; the filter predicts with `assumed_grip`. Set
# that above the truth and it certifies corners the car cannot take — you get
# crashes *through* the filter. Set it to the worst case and you get safety
# paid for in timidity.
#
# **It changes what the learner is learning about.** The action that reaches
# the environment is not always the one the policy chose, so the update is
# off-policy in a way TD(λ) has no term for. `credit="executed"` tells the
# agent about the action that happened (the filter becomes part of the
# environment); `credit="proposed"` tells it about the one it chose (keeps the
# gradient on-policy and misreports the dynamics). Neither is correct, so both
# are measured below.

# %%
def train_one(steps, seed, filter_kw=None):
    env = make_env("lanekeep", seed=seed)
    agent = RTRRL(env.obs_dim, env.action_space, seed=seed)
    runner = agent if filter_kw is None else make_safe(agent, env, **filter_kw)
    obs = env.reset(seed=seed)
    a = runner.start(obs)
    rets, R, off, eps = [], 0.0, 0, 0
    for t in range(steps):
        obs, r, te, tr, info = env.step(a)
        R += r
        a = runner.step(obs, r, te, tr)
        if a is None:
            rets.append(R)
            R, eps = 0.0, eps + 1
            off += bool(info.get("off_track"))
            a = runner.start(env.reset())
    ev = rollout(env, runner.eval_policy(), n_episodes=20, seed=9000 + seed)
    return dict(train_off=off / max(eps, 1),
                eval_ret=float(ev["returns"].mean()),
                eval_off=float(np.mean([bool(i.get("off_track")) for i in ev["infos"]])),
                iv=runner.filter.intervention_rate if filter_kw else 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--assumed-grip", type=float, default=None,
                    help="override the filter's grip assumption (the env draws 0.6-1.4)")
    args = ap.parse_args(argv)

    settings = [("no filter", None),
                ("filter, credit=executed", dict(credit="executed")),
                ("filter, credit=proposed", dict(credit="proposed"))]
    if args.assumed_grip is not None:
        settings.append((f"filter, assumed_grip={args.assumed_grip}",
                         dict(credit="executed", assumed_grip=args.assumed_grip)))

    print(f"\n  training {args.steps:,} steps x {args.seeds} seeds\n")
    print(f"  {'setting':<28}{'eval return':>13}{'train off-track':>17}"
          f"{'eval off-track':>16}{'filtered':>10}")
    for name, kw in settings:
        rs = [train_one(args.steps, s, kw) for s in range(args.seeds)]
        print(f"  {name:<28}{np.mean([r['eval_ret'] for r in rs]):13.1f}"
              f"{np.mean([r['train_off'] for r in rs]):17.1%}"
              f"{np.mean([r['eval_off'] for r in rs]):16.1%}"
              f"{np.mean([r['iv'] for r in rs]):10.1%}")


if __name__ == "__main__":
    main()

# %% [markdown]
# ## What to look for
#
# * **Train off-track is the number the filter exists for.** It is the count of
#   episodes that ended in a wall *during learning* — on a real vehicle, the
#   count of crashes. A filter that takes it to zero has done the job even if
#   the final return is unchanged.
# * **Eval return says what safety cost.** A filter that brakes too eagerly
#   caps the speed the agent can ever learn to carry, and the agent then learns
#   a policy shaped by the filter rather than by the task.
# * **`assumed_grip` is the honest knob.** At 0.6 the filter is robust; at 1.4
#   it certifies corners the car cannot take and crashes happen *through* it.
#   A safety filter inherits its guarantee from its model and nothing else.
#
# Measured, at 200k steps and 4 seeds: the unfiltered agent crashed in 61% of
# training episodes and scored 428. The worst-case filter (`assumed_grip=0.6`)
# crashed in **0%** and scored **449** — safety was not merely cheap here, it
# *paid*, because an episode that ends in a wall is an episode that stopped
# earning. The optimistic filter (1.4) crashed in 61% and scored 329, which is
# worse than having no filter at all.
#
# ## References
#
# * Wabersich & Zeilinger, *A predictive safety filter for learning-based
#   control of constrained nonlinear dynamical systems*, Automatica 2021 —
#   [arXiv:1812.05506](https://arxiv.org/abs/1812.05506)
# * Wabersich & Zeilinger, *Linear model predictive safety certification for
#   learning-based control*, CDC 2018 — [arXiv:1803.08552](https://arxiv.org/abs/1803.08552)
# * Hewing, Wabersich, Menner & Zeilinger, *Learning-Based Model Predictive
#   Control: Toward Safe Learning in Control*, Annual Review of Control,
#   Robotics, and Autonomous Systems 2020
# * Ames, Coogan, Egerstedt, Notomista, Sreenath & Tabuada, *Control Barrier
#   Functions: Theory and Applications*, ECC 2019 —
#   [arXiv:1903.11199](https://arxiv.org/abs/1903.11199) (the pointwise
#   alternative to a predictive certificate)
# * García & Fernández, *A Comprehensive Survey on Safe Reinforcement
#   Learning*, JMLR 2015
