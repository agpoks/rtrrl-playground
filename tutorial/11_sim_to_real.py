"""Lesson 11 -- train in simulation, deploy on a different car, close the gap online.

    python tutorial/11_sim_to_real.py
    python tutorial/11_sim_to_real.py --sim-steps 300000 --real-steps 100000 --safe

The whole argument for RTRRL in one experiment: a policy trained in a simulator
meets a vehicle the simulator was wrong about, and fixes itself while driving.
"""

# %% [markdown]
# # Lesson 11 — Sim-to-real, without the real
#
# This is the deployment story from Lemmel, Resch, Farsang, Hasani, Rus & Grosu
# ([arXiv:2602.02236](https://arxiv.org/abs/2602.02236)) — pretrain offline,
# fine-tune online on the vehicle — with the vehicle replaced by *a second
# simulator whose parameters are different*. That substitution is what makes it
# runnable on a laptop, and it is honest as long as the differences are the
# ones that actually separate a model from a car.
#
# So they are. `REAL_VEHICLE` differs from the nominal one in nine parameters,
# none of them observable:
#
# | | sim | "real" |
# |---|---|---|
# | wheelbase | 0.33 m | 0.35 m |
# | servo lag | 0.08 s | 0.12 s |
# | max acceleration | 4.0 m/s² | 3.4 m/s² |
# | drag | 0.15 | 0.22 |
# | grip ceiling | 6.0 m/s² | 5.2 m/s² |
# | **steering trim** | centred | **+0.03 rad** |
# | throttle delivered | 100% | 90% |
# | lidar noise | none | 4 cm |
# | beam dropout | none | 2% |
#
# The steering trim is the interesting one. A servo whose zero is not the car's
# zero means *going straight requires a non-zero command* — and a policy that
# learned in simulation has never once needed to do that.
#
# ## Four conditions
#
# 1. **sim → sim** — what the policy scored where it was trained. The ceiling.
# 2. **sim → real, frozen** — zero-shot transfer. This is the gap.
# 3. **sim → real, RTRRL keeps learning** — the claim.
# 4. **real from scratch, same total budget** — the control that says whether
#    pretraining was worth anything.

# %%
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import REAL_VEHICLE, VehicleParams, make_env
from rtrrl_playground.safety import make_safe
from rtrrl_playground.train import rollout, train
from rtrrl_playground.utils.load import load_algo

RTRRL = load_algo("rtrrl")

# %% [markdown]
# ## Does it fit RTRRL? Yes, and this is why
#
# Three properties, all of which the alternatives lack:
#
# * **The update is available every timestep**, so adaptation happens *within*
#   the first lap rather than after a batch has been collected.
# * **Memory is constant** — one influence array, three eligibility traces —
#   so nothing has to be stored on the vehicle.
# * **Nothing about the agent changes between the two phases.** The same
#   weights, the same cell, the same learning rule; only the source of the
#   experience changes. There is no "deployment mode" to get wrong.
#
# And the architectural reason it should *work*: RTRRL's input is
# `[observation, previous action, previous reward]`. That is a meta-RL
# architecture, and a vehicle it was not trained on is exactly the kind of
# latent context such an architecture is built to infer. It does not have to
# be *told* the wheelbase changed; it has to notice that the same command now
# produces a different outcome.

# %%
def evaluate(agent, vehicle, n=25, seed=9000, with_filter=False, assumed=None, **env_kw):
    """Evaluate on ``vehicle``. ``with_filter`` measures it as it would be deployed.

    An agent that learned behind a safety filter has to be measured *with* one:
    evaluating it naked measures a policy nobody would ever run, and the
    difference between the two numbers is exactly how much it learned to lean
    on the filter.
    """
    env = make_env("lanekeep", vehicle=vehicle, **env_kw)
    policy = agent.eval_policy()
    if with_filter:
        policy = make_safe(agent, env, assume_env_vehicle=False,
                           assumed_vehicle=assumed or VehicleParams(),
                           assumed_grip=0.6).eval_policy()
    ev = rollout(env, policy, n_episodes=n, seed=seed)
    return dict(ret=float(ev["returns"].mean()), sd=float(ev["returns"].std()),
                off=float(np.mean([bool(i.get("off_track")) for i in ev["infos"]])))


def train_on(agent, vehicle, steps, seed, safe=False, assumed=None):
    """Keep learning on a given vehicle. Returns the crash rate *during* learning."""
    env = make_env("lanekeep", vehicle=vehicle, seed=seed)
    runner = agent
    if safe:
        # The filter believes the *simulator's* car, because that is all anyone
        # would have. Conservative grip is what has to absorb the rest.
        runner = make_safe(agent, env, assume_env_vehicle=False,
                           assumed_vehicle=assumed or VehicleParams(),
                           assumed_grip=0.6)
    obs = env.reset(seed=seed)
    a = runner.start(obs)
    off, eps = 0, 0
    for _ in range(steps):
        obs, r, te, tr, info = env.step(a)
        a = runner.step(obs, r, te, tr)
        if a is None:
            eps += 1
            off += bool(info.get("off_track"))
            a = runner.start(env.reset())
    return off / max(eps, 1)


# %%
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sim-steps", type=int, default=200_000)
    ap.add_argument("--real-steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--safe", action="store_true",
                    help="put a predictive safety filter around the on-vehicle phase")
    args = ap.parse_args(argv)

    sim, real = VehicleParams(), REAL_VEHICLE
    print("  the vehicle the policy has never seen:")
    for k, (a, b) in sim.diff(real).items():
        print(f"    {k:16s} {a} -> {b}")

    # --- 1. train in simulation ---------------------------------------
    env = make_env("lanekeep", vehicle=sim, seed=args.seed)
    agent = RTRRL(env.obs_dim, env.action_space, seed=args.seed)
    train(env, agent, args.sim_steps, progress=False, seed=args.seed)
    in_sim = evaluate(agent, sim, seed=9000)
    print(f"\n  1. sim -> sim (the ceiling)          "
          f"{in_sim['ret']:7.1f} +/- {in_sim['sd']:5.1f}   off-track {in_sim['off']:5.0%}")

    # --- 2. zero-shot on the other vehicle -----------------------------
    zero_shot = evaluate(agent, real, seed=9000)
    print(f"  2. sim -> real, frozen (the gap)     "
          f"{zero_shot['ret']:7.1f} +/- {zero_shot['sd']:5.1f}   off-track {zero_shot['off']:5.0%}")

    # --- 3. keep learning, on the vehicle ------------------------------
    import copy
    tuned = copy.deepcopy(agent)
    crash_rate = train_on(tuned, real, args.real_steps, args.seed + 1,
                          safe=args.safe, assumed=sim)
    after = evaluate(tuned, real, seed=9000)
    print(f"  3. + RTRRL on the vehicle            "
          f"{after['ret']:7.1f} +/- {after['sd']:5.1f}   off-track {after['off']:5.0%}"
          f"   (crashed in {crash_rate:.0%} of episodes while adapting)")
    if args.safe:
        with_f = evaluate(tuned, real, seed=9000, with_filter=True, assumed=sim)
        print(f"  3b. ...evaluated WITH its filter     "
              f"{with_f['ret']:7.1f} +/- {with_f['sd']:5.1f}   off-track {with_f['off']:5.0%}")
        after = with_f

    # --- 4. the control: no pretraining --------------------------------
    env2 = make_env("lanekeep", vehicle=real, seed=args.seed)
    scratch = RTRRL(env2.obs_dim, env2.action_space, seed=args.seed)
    train(env2, scratch, args.sim_steps + args.real_steps, progress=False, seed=args.seed)
    from_scratch = evaluate(scratch, real, seed=9000)
    print(f"  4. real from scratch, same budget    "
          f"{from_scratch['ret']:7.1f} +/- {from_scratch['sd']:5.1f}"
          f"   off-track {from_scratch['off']:5.0%}")

    gap = in_sim["ret"] - zero_shot["ret"]
    recovered = (after["ret"] - zero_shot["ret"]) / gap if abs(gap) > 1e-6 else float("nan")
    print(f"\n  transfer gap: {gap:.1f}   closed online: {recovered:+.0%}")
    if from_scratch["ret"] > after["ret"]:
        print("  NOTE: learning from scratch on the vehicle beat fine-tuning at the "
              "same total\n        budget -- on this task the pretraining bought nothing.")
    return dict(sim=in_sim, zero_shot=zero_shot, tuned=after, scratch=from_scratch)


if __name__ == "__main__":
    main()

# %% [markdown]
# ## Reading the four rows
#
# **Row 2 minus row 1 is the sim-to-real gap**, and it is the only honest
# measure of how much your simulator was lying. If it is near zero your
# perturbations were too gentle to be interesting.
#
# **Row 3 minus row 2 is what online learning bought**, which is the claim
# this whole repo is arguing. Watch the crash rate next to it: adaptation that
# costs a broken car is not adaptation, which is what `--safe` is for.
#
# **Row 4 is the row that could embarrass rows 1–3.** If learning from scratch
# on the real vehicle matches the fine-tuned agent at the same total budget,
# the pretraining bought nothing and the honest conclusion is to skip it. On a
# real car that comparison is not available — you cannot spend 300k steps
# crashing — which is exactly why it has to be run here, where it is free.
#
# ## Where this goes next
#
# The substitution in this lesson is a second simulator. The two real
# replacements for it, in increasing order of honesty:
#
# * `envs/scuderia.py` — the same agent on `scuderia_gym_jax`'s ST/STD models
#   with tyre parameters fitted to actual recordings ({doc}`../to_scuderia_gym_jax`
#   in the docs, `tutorial/08` here).
# * `data/rosbag.py` — clone from a recording of the actual car, then fine-tune
#   (`tutorial/09`, and lesson 7 for the cloning half).
