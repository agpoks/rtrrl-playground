"""Run every agent in a YAML suite on one environment and print a table.

    python benchmarks/run_suite.py --config benchmarks/configs/driving.yaml
    python benchmarks/run_suite.py --config benchmarks/configs/pomdp.yaml --seeds 5

A suite is the "which of these is better" comparison: same environment, same
step budget, same seeds, agents that differ only in the thing being compared.
Scripted policies can appear as an agent (``algo: scripted``) and should --
a learned return means very little without knowing what a hand-written
controller gets on the same task.

Runs one process per (agent, seed) and reports the mean over seeds with its
spread. Online RL at batch size one is high variance; a single-seed table is
not a result.
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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground import make_env  # noqa: E402
from rtrrl_playground.envs.scripted import SCRIPTED  # noqa: E402
from rtrrl_playground.train import rollout, train  # noqa: E402
from rtrrl_playground.utils.load import load_algo  # noqa: E402

DEFAULTS = dict(lr_actor=1e-3, lr_critic=0.03, lr_rnn=1e-3, entropy_coef=0.03)


def run_one(job):
    env_id, env_kwargs, steps, spec, seed = job
    name = spec["name"]
    algo = spec.get("algo", "rtrrl")
    kw = {k: v for k, v in spec.items() if k not in ("name", "algo")}
    try:
        env = make_env(env_id, seed=seed, **env_kwargs)
        if algo == "scripted":
            policy = SCRIPTED[env_id]()
            ev = rollout(env, policy, n_episodes=20, seed=10_000 + seed)
            return dict(name=name, seed=seed, status="ok", params=0, time_s=0.0,
                        final=float(ev["returns"].mean()), influence=0,
                        passes=float(np.mean([i.get("overtakes", 0) for i in ev["infos"]])),
                        crashes=float(np.mean([bool(i.get("crashed")) for i in ev["infos"]])))
        cls = load_algo(algo)
        if algo == "a2c_bptt":
            agent = cls(env.obs_dim, env.action_space, seed=seed,
                        entropy_coef=DEFAULTS["entropy_coef"], **kw)
        else:
            agent = cls(env.obs_dim, env.action_space, seed=seed, **{**DEFAULTS, **kw})
        out = train(env, agent, steps, progress=False, seed=seed)
        ev = rollout(env, agent.eval_policy(), n_episodes=20, seed=10_000 + seed)
        return dict(name=name, seed=seed, status="ok",
                    params=int(agent.n_params), time_s=float(out["train_time_s"]),
                    final=float(ev["returns"].mean()),
                    influence=int(getattr(getattr(agent, "cell", None),
                                                  "influence_bytes", lambda: 0)()),
                    passes=float(np.mean([i.get("overtakes", 0) for i in ev["infos"]])),
                    crashes=float(np.mean([bool(i.get("crashed")) for i in ev["infos"]])))
    except Exception as exc:
        return dict(name=name, seed=seed, status=type(exc).__name__, params=0,
                    time_s=0.0, final=float("nan"), influence=0, passes=0.0, crashes=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override the suite's step budget")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    env_id = cfg["env"]
    steps = args.steps or int(cfg.get("steps", 200_000))
    seeds = args.seeds or int(cfg.get("seeds", 3))
    env_kwargs = cfg.get("env_kwargs", {}) or {}

    from multiprocessing import Pool

    jobs = [(env_id, env_kwargs, steps, spec, s)
            for spec in cfg["agents"] for s in range(seeds)]
    print(f"{env_id}: {len(cfg['agents'])} agents x {seeds} seeds x {steps:,} steps, "
          f"{args.workers} workers")
    with Pool(args.workers) as pool:
        results = pool.map(run_one, jobs)

    by_name: dict[str, list] = {}
    for r in results:
        by_name.setdefault(r["name"], []).append(r)

    show_passes = env_id == "overtake"
    header = f"\n  {'agent':<26} {'return':>9} {'sd':>7} {'params':>8} {'influence':>10} {'train s':>9}"
    if show_passes:
        header += f" {'passes':>7} {'crash':>7}"
    print(header)
    print("  " + "-" * (len(header) - 4))
    rows = []
    for name, rs in by_name.items():
        finals = np.array([r["final"] for r in rs if np.isfinite(r["final"])])
        if not len(finals):
            print(f"  {name:<26} all runs failed: {','.join(r['status'] for r in rs)}")
            continue
        rows.append((finals.mean(), name, finals, rs))
    for mean, name, finals, rs in sorted(rows, reverse=True):
        inf_kb = rs[0]["influence"] / 1024
        line = (f"  {name:<26} {mean:9.1f} {finals.std():7.1f} {rs[0]['params']:8d} "
                f"{inf_kb:9.1f}K {np.mean([r['time_s'] for r in rs]):9.1f}")
        if show_passes:
            line += f" {np.mean([r['passes'] for r in rs]):7.1f} {np.mean([r['crashes'] for r in rs]):6.0%}"
        print(line)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{Path(args.config).stem}.json"
    path.write_text(json.dumps({"config": cfg, "steps": steps, "seeds": seeds,
                                "results": results}, indent=1))
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
