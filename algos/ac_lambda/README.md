# AC(λ) — the memoryless control

**Not a different algorithm.** This is
[`RTRRL`](../rtrrl) with the recurrent cell replaced by a feedforward one
(`cell="mlp"`) and the meta-RL inputs switched off. Same TD(λ), same Dutch
trace, same true-online critic update, same entropy bonus, same learning rates,
same code. What is gone is the state that carries information between
timesteps.

**Papers:** Sutton, *"Learning to predict by the methods of temporal
differences"*, Machine Learning 1988; van Seijen, Mahmood, Pilarski, Machado &
Sutton, *"True Online Temporal-Difference Learning"*, JMLR 2016 —
[arXiv:1512.04087](https://arxiv.org/abs/1512.04087). See
[`papers/README.md`](../../papers/README.md).

## Why it exists

Every environment in this repo is partially observable by construction — no
velocity is ever in an observation — so a policy that is a function of the
current observation alone cannot reach the optimum, whatever its learning rate.
This agent is how that claim is *measured* rather than asserted, and it is in
every benchmark table for that reason.

It also keeps the repo honest in the other direction. On
[`lanekeep`](../../rtrrl_playground/envs/lanekeep.py) this control comes third
of nine, because nine lidar beams are a lot of information and reacting to the
forward one is a decent speed controller by itself. On `memory-chain` it scores
exactly 0.00 — it guesses, which is the best a memoryless policy can do there.
Those two results together are worth more than either alone.

## Files

- `algo.py` — about twenty lines. It is a subclass, and deliberately so: if the
  control shared *most* of the code with the thing it is controlling for, a
  reader would be right to wonder about the rest.
- `example.py` — the same flags as `algos/rtrrl/example.py`.

## Run it

```bash
python algos/ac_lambda/example.py --env cartpole-vel
python algos/ac_lambda/example.py --env cartpole-vel --env-kwargs '{"obs_mode":"full"}'
```

Those two commands are the point of this folder: the same agent, the same
budget, on the POMDP and then on the MDP version of one task. The gap between
them is the size of the hole memory has to fill.
