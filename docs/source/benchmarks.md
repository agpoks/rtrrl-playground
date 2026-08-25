# Benchmarks

Two kinds of comparison, answering different questions.

**Suites** (`benchmarks/configs/*.yaml`, run by `run_suite.py`) put several
agents on one environment with identical settings. "Which of these is better?"

**Sweeps** (`benchmarks/sweep.py`) run one agent across a grid of settings or
ablations. "Does this knob matter?" -- and it is how the defaults in
`algos/rtrrl/algo.py` were chosen.

```bash
python benchmarks/run_suite.py --config benchmarks/configs/driving.yaml
python benchmarks/sweep.py --env lanekeep --grid cells --seeds 3
python benchmarks/sweep.py --env cartpole-vel --grid estimators
```

Both run one process per configuration. Nothing in the agent is threaded and
BLAS is pinned to one thread per worker, so they scale linearly with cores --
which is the only reason a grid of sixty 300k-step runs is a coffee break.

## How to read these tables

**Report the seed spread, and mean over enough seeds.** Online RL at batch size
one is high variance and these tasks are *bimodal*: a run either finds a
driving policy or collapses onto a bad deterministic one, and the mean of those
two is a number that describes neither. Every table here carries the standard
deviation across seeds next to the mean, and where the spread is comparable to
the mean that is the result, not noise to be averaged away.

**The scripted policies are in the table on purpose.** "RTRRL reached 420"
means nothing without knowing that a fifteen-line wall-follower reaches ~575 on
the same task. Both scripted references live in
`rtrrl_playground/envs/scripted.py` and both appear as agents in the suites.

**On `overtake`, return is the wrong headline.** An agent that never passes
anybody and never crashes scores respectably, because progress alone pays. The
pass count and the crash rate are what say whether it learned to overtake, so
`run_suite.py` prints both for that environment.

## Reference points

Measured on this repo's environments, 20 evaluation episodes:

| policy | `lanekeep` | `overtake` |
|---|---|---|
| random | ~23 | ~22 |
| scripted (`WallFollower` / `Overtaker`) | ~575 | ~313 return, 1.6 passes, 65% crashes |
| theoretical ceiling | 600 (1.0/step x 600 steps) | -- |

The scripted overtaker's crash rate is not a bug in it. It commits to a side
from a single frame and has no idea how fast it is closing -- which is exactly
the defect a recurrent policy is supposed to fix, and the reason `overtake` is
in this repo.

## The honest state of the results

Two findings from the sweeps behind the defaults, both of which a
results-only table would hide:

**Training the recurrent weights with RFLO can be worse than not training
them.** On `lanekeep`, the frozen-reservoir ablation (`--estimator none`, so
only the heads learn) scores above several trained configurations. That is a
real and reproducible measurement on this task at this budget, not a claim
about the algorithm in general -- the plausible reading is that a
δ-modulated biased gradient moves the critic's own features under it, and the
two-timescale separation that fixes the actor/critic interaction has no
equivalent for the cell. The sweep tooling is here so the question can be
chased rather than asserted.

**`lanekeep` does not need memory.** The memoryless control does about as well
as the recurrent agents. Nine beams are a lot of information and reacting to
the forward one is a decent speed controller. `overtake` and `memory-chain`
are where the memory has to do work.
