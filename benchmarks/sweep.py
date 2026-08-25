"""Run one agent across a grid of settings, in parallel, and print the grid.

    python benchmarks/sweep.py --env lanekeep --grid cells --steps 200000 --seeds 3
    python benchmarks/sweep.py --env cartpole-vel --grid estimators
    python benchmarks/sweep.py --env overtake --grid lr --steps 400000 --workers 8

This is how the defaults in ``algos/rtrrl/algo.py`` were chosen, including the
ones that differ from the paper's table. Online RL at batch size one is high
variance -- a single seed on these tasks can be off by a factor of two -- so
every cell of the grid is the mean over ``--seeds`` runs and the spread of the
final quintile is printed next to it. A grid without a spread column is not a
result.

One process per configuration: nothing in the agent is threaded, so this scales
linearly with cores, and BLAS is pinned to one thread per worker so the
processes do not fight each other for them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import ENV_IDS, make_env  # noqa: E402
from rtrrl_playground.nets import CELLS, ESTIMATORS  # noqa: E402
from rtrrl_playground.train import train  # noqa: E402
from rtrrl_playground.utils.load import load_algo  # noqa: E402

BASE = dict(lr_actor=1e-3, lr_critic=0.03, lr_rnn=1e-3, entropy_coef=0.03)

GRIDS = {
    "cells": [dict(cell=c) for c in ("mlp", "ctrnn", "ltc", "lrcu", "ligru")],
    "estimators": [dict(estimator=e) for e in ESTIMATORS],
    "feedback": [dict(feedback=f) for f in ("random", "symmetric")],
    "critic": [dict(critic_update=u) for u in ("true-online", "paper", "accumulating")],
    "lr": [dict(lr_actor=a, lr_critic=c)
           for a in (3e-4, 1e-3, 3e-3) for c in (0.01, 0.03, 0.1)],
    "entropy": [dict(entropy_coef=e) for e in (0.0, 0.003, 0.01, 0.03, 0.1)],
    "meta": [dict(meta_inputs=m) for m in (True, False)],
}


def run_one(job):
    env_id, env_kwargs, steps, kw, seed = job
    returns = np.array([])
    try:
        env = make_env(env_id, seed=seed, **env_kwargs)
        cls = load_algo("ac_lambda" if kw.get("cell") == "mlp" else "rtrrl")
        agent = cls(env.obs_dim, env.action_space, seed=seed,
                    **{k: v for k, v in kw.items() if not (k == "cell" and v == "mlp")})
        out = train(env, agent, steps, progress=False, seed=seed)
        returns = out["returns"]
        infos = out["infos"]
        tail = max(len(returns) // 5, 1)
        return {
            "kw": kw, "seed": seed, "status": "ok",
            "final": float(returns[-tail:].mean()) if len(returns) else float("nan"),
            "curve": [float(returns[i * tail:(i + 1) * tail].mean()) for i in range(5)]
            if len(returns) >= 10 else [],
            "passes": float(np.mean([i.get("overtakes", 0) for i in infos[-tail:]] or [0])),
            "crashes": float(np.mean([bool(i.get("crashed")) for i in infos[-tail:]] or [0])),
            "time_s": out["train_time_s"],
        }
    except Exception as exc:  # a diverged run is data, not a reason to lose the grid
        return {"kw": kw, "seed": seed, "status": type(exc).__name__,
                "final": float("nan"), "curve": [], "passes": 0.0, "crashes": 0.0,
                "time_s": 0.0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="lanekeep", choices=ENV_IDS)
    ap.add_argument("--grid", default="cells", choices=sorted(GRIDS))
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    ap.add_argument("--env-kwargs", default="{}")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    args = ap.parse_args()

    from multiprocessing import Pool

    env_kwargs = json.loads(args.env_kwargs)
    jobs = [(args.env, env_kwargs, args.steps, {**BASE, **cfg}, seed)
            for cfg in GRIDS[args.grid] for seed in range(args.seeds)]
    print(f"{len(jobs)} runs ({len(GRIDS[args.grid])} settings x {args.seeds} seeds), "
          f"{args.steps:,} steps each, {args.workers} workers")
    with Pool(args.workers) as pool:
        results = pool.map(run_one, jobs)

    by_setting: dict[str, list] = {}
    for r in results:
        label = ", ".join(f"{k}={v}" for k, v in r["kw"].items() if k not in BASE or BASE[k] != v)
        by_setting.setdefault(label or "default", []).append(r)

    show_passes = args.env == "overtake"
    head = f"\n  {'setting':<34} {'quintile means over training':^41}  {'final':>7} {'sd':>6}"
    if show_passes:
        head += f" {'passes':>7} {'crash':>6}"
    print(head)
    rows = []
    for label, rs in by_setting.items():
        curves = np.array([r["curve"] for r in rs if r["curve"]])
        finals = np.array([r["final"] for r in rs if np.isfinite(r["final"])])
        if not len(curves):
            print(f"  {label:<34} all runs failed: {','.join(r['status'] for r in rs)}")
            continue
        rows.append((float(finals.mean()), label, curves, finals, rs))
    for final, label, curves, finals, rs in sorted(rows, reverse=True):
        line = (f"  {label:<34} " + " ".join(f"{v:7.1f}" for v in curves.mean(axis=0))
                + f"  {final:7.1f} {finals.std():6.1f}")
        if show_passes:
            line += (f" {np.mean([r['passes'] for r in rs]):7.1f}"
                     f" {np.mean([r['crashes'] for r in rs]):6.0%}")
        bad = [r["status"] for r in rs if r["status"] != "ok"]
        print(line + (f"   [{','.join(bad)}]" if bad else ""))

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"sweep_{args.env}_{args.grid}.json"
    path.write_text(json.dumps({"args": vars(args), "results": results}, indent=1))
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
