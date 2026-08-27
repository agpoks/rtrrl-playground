"""Every RTRRL variant this repo can build, on one axis at a time.

    python benchmarks/all_variants.py --steps 150000 --seeds 4
    python benchmarks/all_variants.py --env memorychain --steps 200000

The point is a *complete inventory*, not a leaderboard. RTRRL here is six
independent choices -- cell, gradient estimator, feedback path, critic update,
critic learning-rate mode, and whether the meta-RL inputs are present -- and
the docs only ever tabulated the first two. Everything else was a default that
had never been shown to be the right one.

Each axis is swept with all the others held at their defaults, so a row says
"this choice, against the default, on this task". Results land in
``benchmarks/results/variants_<env>.json`` and are rendered by
``docs/source/variants.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import make_env  # noqa: E402
from rtrrl_playground.train import rollout, train  # noqa: E402
from rtrrl_playground.utils.load import load_algo  # noqa: E402

#: Every axis, and what varies along it. The first entry of each is the default,
#: so a sweep that does not beat its own first row is a sweep that found nothing.
AXES = {
    "cell": ["ligru", "ctrnn", "ltc", "lrcu", "liquid_gru", "physics_ligru", "mlp"],
    "estimator": ["rflo", "snap1", "uoro", "rtrl", "hybrid", "none"],
    "feedback": ["random", "symmetric"],
    "critic_update": ["true-online", "paper", "accumulating"],
    "critic_lr_mode": ["normalized", "fixed"],
    "meta_inputs": [True, False],
}
DEFAULTS = dict(cell="ligru", estimator="rflo", feedback="random",
                critic_update="true-online", critic_lr_mode="normalized",
                meta_inputs=True)
TUNED = dict(lr_actor=1e-3, lr_critic=0.03, lr_rnn=1e-5, entropy_coef=0.03)


def _job(args):
    axis, value, env_id, steps, seed = args
    kw = dict(DEFAULTS)
    kw[axis] = value
    # physics_ligru only engages on a 9-action space; elsewhere it is a plain
    # LiGRU and saying so beats reporting it as if it were the physics cell.
    env = make_env(env_id, seed=seed)
    agent = load_algo("rtrrl")(env.obs_dim, env.action_space, seed=seed,
                               **kw, **TUNED)
    train(env, agent, steps, progress=False, seed=seed)
    ev = rollout(make_env(env_id, seed=seed + 500), agent.eval_policy(),
                 n_episodes=20, seed=9000 + seed)
    # influence_bytes is a method, not a property -- calling getattr and then
    # int() on it silently produced a TypeError one process deep, which
    # surfaced only as the whole pool dying.
    inf = agent.cell.influence_bytes
    inf = inf() if callable(inf) else inf
    return dict(axis=axis, value=str(value), seed=seed,
                ret=float(np.mean(ev["returns"])),
                influence_bytes=int(inf) if inf else None)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="lanekeep")
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    jobs = []
    for axis, values in AXES.items():
        for v in values:
            for s in range(a.seeds):
                jobs.append((axis, v, a.env, a.steps, s))
    print(f"  {len(jobs)} runs "
          f"({sum(len(v) for v in AXES.values())} variants x {a.seeds} seeds)",
          flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(_job, jobs), 1):
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['axis']:<14} {r['value']:<14} "
                  f"seed {r['seed']} -> {r['ret']:7.1f}", flush=True)

    out = {}
    for axis in AXES:
        out[axis] = {}
        for v in AXES[axis]:
            rs = [r["ret"] for r in rows if r["axis"] == axis and r["value"] == str(v)]
            ib = [r["influence_bytes"] for r in rows
                  if r["axis"] == axis and r["value"] == str(v) and r["influence_bytes"]]
            out[axis][str(v)] = dict(mean=float(np.mean(rs)), sd=float(np.std(rs)),
                                     n=len(rs),
                                     influence_bytes=int(ib[0]) if ib else None)
    out["_meta"] = dict(env=a.env, steps=a.steps, seeds=a.seeds, defaults=DEFAULTS)
    path = Path(a.out or ROOT / "benchmarks" / "results" / f"variants_{a.env}.json")
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"  wrote {path}", flush=True)

    for axis, vals in out.items():
        if axis.startswith("_"):
            continue
        print(f"\n  {axis}")
        for v, d in sorted(vals.items(), key=lambda kv: -kv[1]["mean"]):
            print(f"    {v:<16} {d['mean']:7.1f} +/- {d['sd']:6.1f}")


if __name__ == "__main__":
    main()
