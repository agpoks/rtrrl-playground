# Benchmarks

Two kinds of comparison live here, and they answer different questions.

**Suites** (`configs/*.yaml`, run by `run_suite.py`) put several agents on one
environment with identical settings and print a table. That is the "which of
these is better" question.

**Sweeps** (`sweep.py`) run one agent across a grid of hyperparameters or
ablations, in parallel, and print the grid. That is the "does this knob matter"
question — and it is how the defaults in `algos/rtrrl/algo.py` were chosen,
including the two that differ from the paper's table.

```bash
python benchmarks/run_suite.py --config benchmarks/configs/driving.yaml
python benchmarks/run_suite.py --config benchmarks/configs/pomdp.yaml
python benchmarks/sweep.py --env lanekeep --grid cells --steps 200000 --seeds 3
python benchmarks/sweep.py --env cartpole-vel --grid estimators
```

`sweep.py` uses one process per configuration (`--workers`, default: cores
minus one). Nothing in the agent is threaded, so this scales linearly, and it
is the only reason a grid of thirty 300k-step runs is a coffee break rather
than an afternoon.

## Reading the tables honestly

* **Return is not always the headline.** On `overtake`, an agent that never
  passes anybody and never crashes scores respectably, because progress alone
  pays. The pass count and the crash rate are what say whether it learned to
  overtake, so both are in the table.
* **Report the seed spread.** Online RL at batch size one is high variance;
  a single seed on these tasks can be off by a factor of two, and a table of
  best-of-N is not a result.
* **The scripted policies are in the table on purpose.** "RTRRL reached 420"
  means nothing without knowing that a fifteen-line wall-follower reaches 573.
