"""Train the memoryless AC(lambda) control on any environment.

    python algos/ac_lambda/example.py --env cartpole-vel --steps 200000
    python algos/ac_lambda/example.py --env cartpole-vel --env-kwargs '{"obs_mode":"full"}'
    python algos/ac_lambda/example.py --env lanekeep --render

The first two commands are the point of this file. The same agent, the same
hyperparameters, the same number of steps -- on the POMDP and then on the MDP
version of the same task. The gap between them is the size of the hole that
memory has to fill, measured rather than asserted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "algos" / "rtrrl"))

from rtrrl_playground import make_env, set_seed  # noqa: E402
from rtrrl_playground.train import train  # noqa: E402
from example import build_parser, make_agent, report  # noqa: E402  (algos/rtrrl/example.py)
from rtrrl_playground.utils.load import load_algo  # noqa: E402


def main() -> None:
    parser = build_parser(__doc__)
    args = parser.parse_args()
    args.cell, args.estimator = "mlp", "rflo"  # what makes this the control
    set_seed(args.seed)
    env = make_env(args.env, **json.loads(args.env_kwargs))
    agent = make_agent(args, env, cls=load_algo("ac_lambda"))
    print(f"env={args.env} obs_dim={env.obs_dim} actions={env.action_space} "
          f"MEMORYLESS control, params={agent.n_params}")
    out = train(env, agent, args.steps, log_every=args.log_every, seed=args.seed)
    report(args, env, agent, out, "ac_lambda")


if __name__ == "__main__":
    main()
