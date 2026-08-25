"""Train RTRRL online on any of the four environments.

    python algos/rtrrl/example.py --env cartpole-vel --steps 200000
    python algos/rtrrl/example.py --env lanekeep --cell lrcu --render
    python algos/rtrrl/example.py --env overtake --cell ltc --steps 400000 --render
    python algos/rtrrl/example.py --env memory-chain --estimator rtrl

Every knob that the tutorial argues about is a flag here: ``--cell`` picks the
recurrent unit, ``--estimator`` picks how its gradient is obtained,
``--feedback`` turns feedback alignment on and off, ``--critic-update`` selects
between the paper's printed critic line and full true-online TD(lambda).

The defaults are the ones that work on these environments, which are not
everywhere the paper's Table 5 values -- see ``--help`` and the "Deviations"
section of ``algos/rtrrl/README.md`` for which ones differ and why.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rtrrl_playground import ENV_IDS, make_env, set_seed  # noqa: E402
from rtrrl_playground.nets import CELLS, ESTIMATORS  # noqa: E402
from rtrrl_playground.train import result_line, rollout, train  # noqa: E402
from algo import RTRRL  # noqa: E402


def build_parser(description: str = __doc__) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="cartpole-vel", choices=ENV_IDS)
    p.add_argument("--steps", type=int, default=200_000, help="environment steps of online learning")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--cell", default="ctrnn", choices=sorted(CELLS))
    p.add_argument("--estimator", default="rflo", choices=list(ESTIMATORS),
                   help="how the recurrent gradient is obtained (see nets/cell.py)")
    p.add_argument("--feedback", default="random", choices=["random", "symmetric"],
                   help="'random' is feedback alignment; 'symmetric' is the true gradient")
    p.add_argument("--critic-update", default="true-online",
                   choices=["true-online", "paper", "accumulating"])
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.9, help="all three eligibility decays")
    p.add_argument("--lr-actor", type=float, default=1e-3)
    p.add_argument("--lr-critic", type=float, default=0.03)
    p.add_argument("--lr-rnn", type=float, default=1e-5,
                   help="the paper's 1e-3 is two orders of magnitude too high here; see the README")
    p.add_argument("--entropy", type=float, default=0.03)
    p.add_argument("--no-meta-inputs", action="store_true",
                   help="stop feeding the previous action and reward back in")
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--render", action="store_true", help="save a picture of one evaluation episode")
    p.add_argument("--out", default=str(ROOT / "runs"), help="where plots and json go")
    p.add_argument("--env-kwargs", default="{}", help="JSON passed to the environment constructor")
    return p


def make_agent(args, env, cls=RTRRL, **extra):
    return cls(env.obs_dim, env.action_space, n_hidden=args.hidden,
               gamma=args.gamma, lam_actor=args.lam, lam_critic=args.lam,
               lam_rnn=args.lam, lr_actor=args.lr_actor, lr_critic=args.lr_critic,
               lr_rnn=args.lr_rnn, entropy_coef=args.entropy, cell=args.cell,
               estimator=args.estimator, feedback=args.feedback,
               critic_update=args.critic_update,
               meta_inputs=not args.no_meta_inputs, seed=args.seed, **extra)


def report(args, env, agent, out, tag: str) -> None:
    """Print the shared RESULT line, save the learning curve, optionally render."""
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    returns = out["returns"]
    tail = returns[-args.eval_episodes:] if len(returns) else np.array([np.nan])
    final = float(np.mean(tail))

    def greedy(obs):
        return agent.greedy(obs)

    if args.render and hasattr(env, "render_rollout"):
        ev = rollout(env, greedy, n_episodes=1, seed=10_000, keep_history=True)
        path = outdir / f"{tag}_{args.env}_{args.cell}_{args.estimator}.png"
        env.render_rollout(ev["history"], str(path),
                           title=f"{tag} / {args.env} / {args.cell} / {args.estimator}")
        print(f"  wrote {path}")

    curve = outdir / f"{tag}_{args.env}_{args.cell}_{args.estimator}.json"
    curve.write_text(json.dumps({
        "args": vars(args), "curve": out["curve"],
        "returns": returns.tolist(), "train_time_s": out["train_time_s"],
    }, indent=1))
    print(result_line(f"{tag}:{args.cell}:{args.estimator}", "return", final,
                      agent.n_params, out["train_time_s"],
                      influence_bytes=getattr(agent, "cell", None)
                      and agent.cell.influence_bytes()))


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    env = make_env(args.env, **json.loads(args.env_kwargs))
    agent = make_agent(args, env)
    print(f"env={args.env} obs_dim={env.obs_dim} actions={env.action_space} "
          f"cell={args.cell} estimator={args.estimator} params={agent.n_params} "
          f"influence={agent.cell.influence_bytes() / 1024:.1f} KiB")
    out = train(env, agent, args.steps, log_every=args.log_every, seed=args.seed)
    report(args, env, agent, out, "rtrrl")


if __name__ == "__main__":
    main()
