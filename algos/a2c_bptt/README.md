# A2C with truncated BPTT — the gradient control

**Paper:** Mnih, Badia, Mirza, Graves, Lillicrap, Harley, Silver &
Kavukcuoglu, *"Asynchronous Methods for Deep Reinforcement Learning"*,
ICML 2016 — [arXiv:1602.01783](https://arxiv.org/abs/1602.01783). See
[`papers/README.md`](../../papers/README.md).

Same cell as [`RTRRL`](../rtrrl) (a CT-RNN, same equations as
[`nets/ctrnn.py`](../../rtrrl_playground/nets/ctrnn.py)), same heads, same
environments. The only thing that changes is **when and how the gradient is
obtained**.

| | RTRRL | A2C + truncated BPTT |
|---|---|---|
| gradient | carried forwards, one influence array | replayed backwards through `T` stored steps |
| update timing | every step, during the episode | every `T` steps, after |
| memory | constant in `T` | linear in `T` |
| credit reach | traces decay; nothing is cut | exactly `T` steps, then a cliff |
| framework | hand-derived, NumPy | `torch.autograd` |

## The one `.backward()` in the repo

This file calls it; nothing else does. That single line is what RTRRL is buying
its way out of, and having the alternative sitting here in sixty lines makes
the cost concrete rather than rhetorical.

So does the truncation window, which is the knob traces do not have. Raise
`--truncation` and credit reaches further back at a proportional cost in stored
activations; the `graph_steps` field in the printed result line is that cost.
For scale: an `overtake` episode is 900 steps, and the graph for all of it does
not fit in the budget an RC car has between control ticks.

## Files

- `algo.py` — `CTRNNTorch` plus a GAE(λ) advantage actor-critic with a rolling
  window. Exposes the same `start` / `step` interface as the online agents, so
  it drops into the same training loop — which is itself the asymmetry being
  illustrated: the caller cannot tell that this one is not learning online, but
  the machine can.
- `example.py` — adds `--truncation`, `--lr` and `--device`.

## Run it

```bash
pip install -e ".[torch]"
python algos/a2c_bptt/example.py --env lanekeep --steps 200000
python algos/a2c_bptt/example.py --env lanekeep --truncation 64 --render
```

`--device cuda` exists so you can check, not because it helps: at batch size
one with 32 hidden units, dispatch dominates and the CPU wins.
