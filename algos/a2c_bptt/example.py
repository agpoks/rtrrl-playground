"""Train the truncated-BPTT recurrent A2C baseline.

    python algos/a2c_bptt/example.py --env cartpole-vel --steps 200000
    python algos/a2c_bptt/example.py --env lanekeep --truncation 64 --render

Needs PyTorch (``pip install -e ".[torch]"``); nothing else in the repo does.

``--truncation`` is the flag worth playing with. It is the knob that
eligibility traces do not have: how many steps back the gradient is allowed to
reach. Raise it and credit travels further at a proportional cost in stored
activations; the printed ``graph_steps`` in the result line is that cost.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "algos" / "rtrrl"))

import numpy as np  # noqa: E402

from rtrrl_playground import make_env, set_seed  # noqa: E402
from rtrrl_playground.train import result_line, rollout, train  # noqa: E402
from rtrrl_playground.utils.load import load_algo  # noqa: E402
from example import build_parser  # noqa: E402  (algos/rtrrl/example.py)


def main() -> None:
    parser = build_parser(__doc__)
    parser.add_argument("--truncation", type=int, default=32,
                        help="BPTT window: how many steps the gradient may reach back")
    parser.add_argument("--lr", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                        help="CPU is the right answer at batch size 1; the flag is here to check")
    args = parser.parse_args()
    set_seed(args.seed)
    env = make_env(args.env, **json.loads(args.env_kwargs))
    agent = load_algo("a2c_bptt")(
        env.obs_dim, env.action_space, n_hidden=args.hidden, gamma=args.gamma,
        lam=args.lam, lr=args.lr, truncation=args.truncation,
        entropy_coef=args.entropy, meta_inputs=not args.no_meta_inputs,
        seed=args.seed, device=args.device)
    print(f"env={args.env} obs_dim={env.obs_dim} truncation={args.truncation} "
          f"params={agent.n_params} (torch/autograd)")
    out = train(env, agent, args.steps, log_every=args.log_every, seed=args.seed)

    returns = out["returns"]
    final = float(np.mean(returns[-args.eval_episodes:])) if len(returns) else float("nan")
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    if args.render and hasattr(env, "render_rollout"):
        ev = rollout(env, agent.greedy, n_episodes=1, seed=10_000, keep_history=True)
        path = outdir / f"a2c_bptt_{args.env}.png"
        env.render_rollout(ev["history"], str(path), title=f"a2c-bptt / {args.env}")
        print(f"  wrote {path}")
    print(result_line("a2c_bptt", "return", final, agent.n_params, out["train_time_s"],
                      graph_steps=agent.peak_graph_steps))


if __name__ == "__main__":
    main()
