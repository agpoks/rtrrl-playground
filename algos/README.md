# Algorithms

Three agents. They share the training loop, the environments, the cells and
the heads, so a difference in a result is a difference in the learning rule and
nothing else.

| | [`rtrrl`](rtrrl) | [`ac_lambda`](ac_lambda) | [`a2c_bptt`](a2c_bptt) |
|---|---|---|---|
| memory in the policy | recurrent state | **none** | recurrent state |
| recurrent gradient | carried forwards (RFLO / RTRL / SnAp / UORO) | not needed | replayed backwards (BPTT) |
| updates | every step | every step | every `T` steps |
| memory cost | constant in episode length | constant | linear in `T` |
| credit horizon | traces decay, nothing is truncated | traces | exactly `T` steps |
| framework | NumPy, gradients by hand | NumPy | PyTorch autograd |

`ac_lambda` is the *memory* control: literally `rtrrl` with the recurrent cell
swapped for a feedforward one. `a2c_bptt` is the *gradient* control: the same
recurrent cell, trained the standard way. Between them they isolate the two
claims RTRRL makes.

## Why NumPy

The online agents are NumPy and their gradients are written out by hand. That
is a deliberate choice, and it is worth defending because it looks like the
wrong one:

* **Autograd is the thing being avoided.** The entire point of RFLO is that no
  backward pass happens. Building it on top of a framework whose main feature
  is the backward pass would obscure exactly what the reader is here to see.
* **There is no batch.** RTRRL is a single-stream algorithm — one environment,
  one update per timestep, 32 hidden units. At that size a GPU is slower than a
  CPU and a tensor framework's per-op dispatch dominates everything. Measured:
  the NumPy agent runs about five times faster than the equivalent in torch.
* **NumPy ports to JAX almost mechanically**, which is where this is going
  (see [`tutorial/08`](../tutorial/08_to_scuderia_gym_jax.py)). Every array is
  fixed-shape and the update is a pure function of `(params, traces, obs)`.

`a2c_bptt` uses PyTorch, because there the backward pass *is* the algorithm.
It is the only file in the repo that needs it: `pip install -e ".[torch]"`.

## The common result line

Every `example.py` prints one line the benchmark runner parses:

```
RESULT: model=<name> metric_name=return metric=<value> params=<n> train_time_s=<t> ...
```

Same convention as the other playgrounds in this family.
