"""Generate the GIFs the docs embed, into ``docs/source/_static/anim/``.

    python scripts/make_animations.py               # all of them
    python scripts/make_animations.py --only safety
    python scripts/make_animations.py --quick       # short training for a smoke test

These are committed, like the figures: Read the Docs cannot train an agent.
Keep an eye on the file sizes printed at the end -- a GIF is uncompressed
frames, and it is easy to add ten megabytes to a docs repo without noticing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import make_env  # noqa: E402
from rtrrl_playground.cbf import DiscreteCBFFilter  # noqa: E402
from rtrrl_playground.envs.scripted import Overtaker, WallFollower  # noqa: E402
from rtrrl_playground.safety import PredictiveSafetyFilter  # noqa: E402
from rtrrl_playground.train import train  # noqa: E402
from rtrrl_playground.utils.load import load_algo  # noqa: E402
from rtrrl_playground.viz import (  # noqa: E402
    animate_episode, animate_learning, animate_safety,
)

OUT = ROOT / "docs" / "source" / "_static" / "anim"


def _trained(env_name, steps, seed=0, cell="ligru"):
    env = make_env(env_name, seed=seed)
    agent = load_algo("rtrrl")(env.obs_dim, env.action_space, cell=cell, seed=seed)
    train(env, agent, steps, progress=False, seed=seed)
    return agent


def anim_lanekeep(quick):
    """A trained agent holding a line, with the nine beams it is steering on."""
    agent = _trained("lanekeep", 20_000 if quick else 150_000)
    return animate_episode("lanekeep", agent.eval_policy(), OUT / "lanekeep.gif",
                           seed=7, max_frames=140, fps=20,
                           title="lanekeep — RTRRL / ligru, learned online, "
                                 "nine beams and nothing else")


def anim_overtake(quick):
    """A pass. The camera follows, because the interesting part is 3 m wide.

    Searches seeds for an episode that actually contains an overtake: the
    caption promises one, and an episode where the agent trails the opponent
    for the whole clip would make the caption a lie.
    """
    agent = _trained("overtake", 20_000 if quick else 250_000)
    pol = agent.eval_policy()
    best, best_seed = -1, 0
    for seed in range(14):
        env = make_env("overtake", seed=seed)
        obs = env.reset(seed=seed)
        if hasattr(pol, "reset"):
            pol.reset()
        passes = 0
        for _ in range(env.max_steps):
            obs, r, te, tr, info = env.step(pol(obs))
            if hasattr(pol, "observe"):
                pol.observe(r)
            # The key is "overtakes". Reading a key the env does not emit --
            # "passes", say -- silently returns the default at every step, so
            # the search reports zero everywhere and picks seed 0 by accident.
            passes = info.get("overtakes", passes)
            if te or tr:
                break
        if passes > best:
            best, best_seed = passes, seed
    if best <= 0:
        raise RuntimeError(
            "no seed produced an overtake; refusing to write a clip captioned "
            "as a pass. Train longer, or widen the seed search.")
    print(f"    seed {best_seed} has {best} pass(es); animating that one")
    return animate_episode("overtake", pol, OUT / "overtake.gif", seed=best_seed,
                           max_frames=170, fps=18, follow=True,
                           title=f"overtake — {best} completed pass(es). "
                                 "Red beams are the ones that hit a car, not a wall.")


def _fast_policy(seed=1):
    """Full throttle, random steering -- the policy the filter has to work for.

    A *uniformly* random policy is the wrong demo twice over. It averages
    0.9 m/s, so it never reaches a speed where the tyre limit binds; and behind
    a filter it does not crash, it **stalls** -- 94 steps, mean speed 0.58 m/s,
    ``stalled: True``. That is a real result (it is the "filter becomes the
    controller" failure mode in ``safety.md``) but it animates to a stationary
    car, and a clip captioned "it never leaves the track" over a car that has
    stopped moving is telling half the truth.

    Full throttle with random steering moves, gets overridden constantly, and
    is the regime the sweeps in ``safety.md`` use.
    """
    rng = np.random.default_rng(seed)
    return lambda o: int(rng.integers(3)) * 3 + 2


def anim_safety(quick):
    """The filter deciding, live. The one that justifies the module.

    ``assumed_grip=0.6`` is the worst case of the environment's U(0.6, 1.4),
    and the only setting that never lets a crash through -- at 1.0 this policy
    puts 7% of episodes into a wall, and animating a "safety filter" over a
    crash would be a poor advertisement for reading the grip sweep.
    """
    env = make_env("lanekeep")
    filt = PredictiveSafetyFilter(env.track, dt=env.dt, assumed_grip=0.6,
                                  assumed_vehicle=getattr(env, "vehicle", None))
    return animate_safety(env, _fast_policy(), filt,
                          OUT / "safety_filter.gif", seed=4, max_frames=150, fps=14,
                          title="full throttle, random steering, behind a predictive "
                                "safety filter — it never leaves the track")


def anim_cbf(quick):
    """The same random policy behind the CBF, for comparison with the above."""
    env = make_env("lanekeep")
    filt = DiscreteCBFFilter(env.track, dt=env.dt, assumed_grip=0.6,
                             assumed_vehicle=getattr(env, "vehicle", None))
    return animate_safety(env, _fast_policy(), filt,
                          OUT / "safety_cbf.gif", seed=4, max_frames=150, fps=14,
                          title="the same policy, the same state, one algebraic "
                                "inequality instead of a rollout")


def anim_learning(quick):
    """The same agent at three points in training, driving side by side."""
    cps = (0, 5_000, 20_000) if quick else (0, 25_000, 150_000)

    def factory(env):
        return load_algo("rtrrl")(env.obs_dim, env.action_space, cell="ligru", seed=0)

    return animate_learning("lanekeep", factory, OUT / "learning.gif",
                            checkpoints=cps, max_frames=130, fps=18,
                            title="lanekeep: no replay buffer, no batch — "
                                  "this is the policy improving as it drives")


ANIMS = {"lanekeep": anim_lanekeep, "overtake": anim_overtake,
         "safety": anim_safety, "cbf": anim_cbf, "learning": anim_learning}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=sorted(ANIMS))
    ap.add_argument("--quick", action="store_true", help="short training, for a smoke test")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    total = 0
    for name in (a.only or sorted(ANIMS)):
        print(f"  {name} ...", flush=True)
        p = ANIMS[name](a.quick)
        kb = p.stat().st_size / 1024
        total += kb
        print(f"    wrote {p.name}  {kb:.0f} KB")
    print(f"  total {total / 1024:.1f} MB")
