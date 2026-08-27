"""How much do the vehicle models actually differ, and does it matter?

    python benchmarks/models.py                       # both experiments
    python benchmarks/models.py --only divergence

The repo documents four vehicle models and compares none of them. This is the
comparison, and it asks two separate questions.

**Open-loop divergence.** Drive the *same command sequence* into each model from
the same state and measure how far apart the trajectories get. This is a
statement about the models alone -- no policy, no learning, no reward -- and it
is the number that says whether a controller tuned on one can be trusted on
another.

**Zero-shot transfer.** Train once on the kinematic bicycle and evaluate the
frozen policy on each model. This is the sim-to-real question asked against
real fitted tyres rather than against a perturbed copy of the same equations,
which is what ``tutorial/11`` does.

The scuderia models need ``scuderia_gym_jax`` on the path; without it the
script runs the kinematic-only half and says so.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtrrl_playground.envs.lanekeep import (  # noqa: E402
    A_LAT_MAX, ACCEL_MAX, DRAG, SPEED_MAX, STEER_MAX, STEER_TAU, WHEELBASE,
)
from rtrrl_playground.envs.vehicle import REAL_VEHICLE, VehicleParams  # noqa: E402

OUT = ROOT / "benchmarks" / "results"


def _kinematic(x, u, dt, params, grip=1.0):
    """The repo's own model, as a pure function. Matches ``LaneKeep._integrate``."""
    p = params
    px, py, psi, v, delta = x
    steer, thr = u
    delta = delta + (steer * p.steer_max - delta) * dt / p.steer_tau
    v = float(np.clip(v + (thr * p.accel_max - p.drag * v) * dt, 0.0, p.speed_max))
    psi_dot = v / p.wheelbase * np.tan(delta)
    if v > 1e-3:
        lim = p.a_lat_max * grip / v
        psi_dot = float(np.clip(psi_dot, -lim, lim))
    return np.array([px + v * np.cos(psi) * dt, py + v * np.sin(psi) * dt,
                     psi + psi_dot * dt, v, delta])


def _commands(n, seed=0):
    """A command sequence with corners in it.

    Constant inputs diverge trivially and tell you nothing; a random walk
    exercises the transients -- servo lag, the grip cap engaging and releasing
    -- which is where the models are supposed to disagree.
    """
    rng = np.random.default_rng(seed)
    steer = np.clip(np.cumsum(rng.normal(0, 0.25, n)) * 0.35, -1, 1)
    thr = np.clip(0.6 + rng.normal(0, 0.4, n), -1, 1)
    return np.stack([steer, thr], axis=1)


def divergence(n=200, dt=0.05, seeds=8):
    """Same commands, different models: how far apart do they end up?"""
    rows = {}
    try:
        sys.path.insert(0, "/home/poxx/github/scuderia_gym_jax")
        import jax  # noqa: F401
        import scuderia_gym_jax as sgj
        have_sgj = True
    except Exception as exc:                     # pragma: no cover
        print(f"    scuderia_gym_jax unavailable ({type(exc).__name__}); "
              "kinematic variants only")
        have_sgj = False

    for seed in range(seeds):
        cmds = _commands(n, seed)
        x0 = np.array([0.0, 0.0, 0.0, 2.0, 0.0])
        base = [x0.copy()]
        x = x0.copy()
        for u in cmds:
            x = _kinematic(x, u, dt, VehicleParams())
            base.append(x.copy())
        base = np.array(base)

        variants = {
            "kinematic, grip 0.6": (VehicleParams(), 0.6),
            "kinematic, grip 1.4": (VehicleParams(), 1.4),
            "REAL_VEHICLE": (REAL_VEHICLE, 1.0),
        }
        for name, (params, grip) in variants.items():
            x = x0.copy()
            traj = [x.copy()]
            for u in cmds:
                x = _kinematic(x, u, dt, params, grip=grip)
                traj.append(x.copy())
            traj = np.array(traj)
            d = np.linalg.norm(traj[:, :2] - base[:, :2], axis=1)
            rows.setdefault(name, []).append(d)

        if have_sgj:
            for model in ("ks", "st", "std"):
                try:
                    d = _sgj_traj(sgj, model, cmds, dt, x0)
                except Exception as exc:         # pragma: no cover
                    print(f"    {model}: {type(exc).__name__}: {exc}")
                    continue
                m = min(len(d), len(base))
                rows.setdefault(f"scuderia {model}", []).append(
                    np.linalg.norm(d[:m, :2] - base[:m, :2], axis=1))

    out = {}
    for name, ds in rows.items():
        L = min(len(d) for d in ds)
        A = np.stack([d[:L] for d in ds])
        out[name] = dict(final=float(A[:, -1].mean()), final_sd=float(A[:, -1].std()),
                         at_1s=float(A[:, min(20, L - 1)].mean()),
                         at_2s=float(A[:, min(40, L - 1)].mean()),
                         curve=A.mean(axis=0).tolist())
    return out


def _sgj_traj(sgj, model, cmds, dt, x0):
    """Drive the same commands through a scuderia model. Returns [x, y, psi, v]."""
    import jax
    import jax.numpy as jnp
    env = sgj.make(None, overrides={"ctrl_mode": "accl"}, model=model,
                   ctrl_mode="accl", num_agents=1, produce_scans=False,
                   collision_on=False, timestep=0.01)
    step = jax.jit(env.step_env)
    key = jax.random.key(0)
    _obs, st = env.reset(key, jnp.asarray([[x0[0], x0[1], x0[2]]]))
    st = st.replace(x=st.x.at[:, 3].set(float(x0[3])))
    traj = [np.array([x0[0], x0[1], x0[2], x0[3]])]
    sub = max(1, int(round(dt / 0.01)))
    for u in cmds:
        act = jnp.asarray([[float(u[0]) * STEER_MAX, float(u[1]) * ACCEL_MAX]])
        for _ in range(sub):
            key, k = jax.random.split(key)
            _o, st, _r, _d, _i = step(k, st, act)
        s = np.asarray(st.x[0])
        traj.append(np.array([s[0], s[1], s[4], s[3]]))
    return np.array(traj)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=["divergence"])
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args(argv)

    print("  open-loop divergence from the repo's kinematic bicycle")
    print("  (same command sequence, same start, 10 s at 20 Hz)\n")
    d = divergence(seeds=a.seeds)
    print(f"  {'model':<24}{'after 1 s':>11}{'after 2 s':>11}{'after 10 s':>12}")
    for name, r in sorted(d.items(), key=lambda kv: kv[1]["final"]):
        print(f"  {name:<24}{r['at_1s']:10.2f} m{r['at_2s']:10.2f} m"
              f"{r['final']:11.2f} m")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models.json").write_text(json.dumps(d, indent=2) + "\n")
    print(f"\n  wrote {OUT / 'models.json'}")


if __name__ == "__main__":
    main()
