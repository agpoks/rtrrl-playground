# Measured results

Everything below is from this repo's own environments on one 16-core laptop
CPU, with the sweep and suite runners in `benchmarks/`. They are not the
paper's numbers on the paper's benchmarks; see {doc}`the_algorithm` for the
scope.

Two conventions, for a reason. **Every number is a mean over seeds with its
spread**, because these tasks are bimodal -- a run either finds a driving
policy or collapses onto a bad deterministic one, and the mean of those two
describes neither. And **the scripted policies are in the tables**, because a
learned return means nothing without knowing what fifteen lines of hand-written
control achieve on the same task.

## MemoryChain-8 -- the pure memory test

200k steps, 3 seeds, library defaults except the cell. Optimal is `+1.000`;
guessing is `0.000`, and a memoryless policy provably cannot beat guessing.

| agent | final return | sd |
|---|---|---|
| RTRRL / **lrcu** | **+0.94** | 0.03 |
| RTRRL / ligru | +0.88 | 0.04 |
| RTRRL / ctrnn | +0.26 | 0.40 |
| RTRRL / ltc | +0.01 | 0.02 |
| AC(λ), memoryless | −0.01 | 0.02 |

This is the cleanest result in the repo and the one worth reading first.

* **The memoryless control lands exactly on zero.** Not "worse", not "slower" --
  it guesses, which is the best a policy without memory can do here. That is
  the whole premise of the repo, measured.
* **The liquid cell wins**, and by a lot. The LRCU is the cell that Lemmel,
  Resch, Farsang, Hasani, Rus & Grosu found works best with RTRRL on real
  hardware ([arXiv:2602.02236](https://arxiv.org/abs/2602.02236)); it is
  reassuring to see the same ordering fall out of an independent
  reimplementation on a task neither paper used.
* **The CT-RNN -- RTRRL's own cell -- struggles**, at `+0.26` with a standard
  deviation of `0.40`, meaning some seeds solve it and some never leave zero.

## LaneKeep -- can it drive at all

300k steps, 4 seeds, library defaults. Reference points: a scripted
wall-follower gets ~575, random ~23, and the ceiling is 600.

| agent | final return | sd |
|---|---|---|
| RTRRL / ligru | 460 | 63 |
| RTRRL / ctrnn, `--estimator uoro` | 458 | 80 |
| **AC(λ), memoryless** | **447** | 67 |
| RTRRL / ctrnn, `--feedback symmetric` | 389 | 102 |
| RTRRL / ctrnn | 384 | 108 |
| RTRRL / ltc | 336 | 196 |
| RTRRL / ctrnn, `--estimator none` (reservoir) | 246 | 143 |
| RTRRL / ctrnn, `--estimator snap1` | 236 | 136 |
| RTRRL / lrcu | 193 | 190 |

**The memoryless control is third.** That is the honest headline. Nine beams
are a lot of information and reacting to the forward one is a decent speed
controller, so lanekeep does not need memory -- it is where you check that an
agent can drive, not where memory is demonstrated. Note also that the cell
ordering is *inverted* relative to MemoryChain: the LRCU is best where memory
is everything and worst where it is not needed, which is what you would expect
from a more expressive state that is harder to read reactively.

**Feedback alignment costs almost nothing.** `symmetric` (the true gradient
into the cell) scores 389 against `random`'s 384, well inside the seed spread.
That is the whole surprise of the Lillicrap et al. result, reproduced here for
free.

**The seed spread is the result, not noise around it.** At sd ≈ 100 on a mean
of 400, individual seeds run from ~300 to ~570, and in several rows one seed
collapses to near zero. Any single-seed comparison between these rows would be
meaningless.

## What the recurrent learning rate does

`lanekeep`, 300k steps, 4 seeds, CT-RNN. The paper's `alpha_R` is 1e-3.

| `--lr-rnn` | 0 | 1e-5 | 1e-4 | 1e-3 |
|---|---|---|---|---|
| final return | 239 | 338 | 362 | 20 |

Two orders of magnitude below the paper's value is where this lands, and 1e-3
collapses almost every seed. On MemoryChain the same knob barely matters at
all between 1e-5 and 1e-2 -- so this is not a universal correction, it is a
statement about a task where the recurrence has little to learn and a biased,
δ-modulated gradient mostly moves the critic's features around underneath it.

## Gradient estimators

CT-RNN, n=32. Alignment measured in `tutorial/02_gradients_online.py`,
cost measured on one core.

| estimator | cosine to exact | influence | µs/step |
|---|---|---|---|
| `rtrl` | 1.00 | 432 KiB | ~1000 |
| `uoro` (1 sample) | 0.10 | 13.8 KiB | ~90 |
| `snap1` | 0.72 | 13.5 KiB | ~55 |
| `rflo` | 0.72 | 13.5 KiB | ~45 |

Averaging a few hundred UORO samples of the *same* gradient takes its alignment
towards 1; RFLO's 0.72 does not improve with any number of samples. See
{doc}`estimators`.

## With a safety filter

RTRRL on `lanekeep`, 200k steps, 4 seeds — see {doc}`safety`.

| filter | eval return | crashes **during training** | crashes at eval | steps overridden |
|---|---|---|---|---|
| none | 428 ± 97 | 61.3% | 30.0% | — |
| `assumed_grip=1.0` (default), `credit=executed` | 354 ± 136 | 21.3% | 6.2% | 15.8% |
| `assumed_grip=1.0`, `credit=proposed` | 356 ± 216 | 20.6% | 13.8% | 30.7% |
| **`assumed_grip=0.6` (worst case)** | **449 ± 155** | **0.0%** | **0.0%** | 17.5% |
| `assumed_grip=1.4` (optimistic) | 329 ± 53 | 60.8% | 51.2% | 7.8% |

The worst-case filter crashed zero times in 200k steps of learning and still
finished with the highest return. The optimistic one is worse than no filter.

## Sim-to-real

A policy trained in simulation, deployed on a vehicle that differs in nine
unobservable parameters (see {doc}`physics`), then allowed to keep learning.

| condition | return | sd | off-track |
|---|---|---|---|
| 1. sim → sim (the ceiling) | 439.0 | 83 | 30% |
| 2. sim → real, **frozen** (the gap) | 367.9 | 118 | 38% |
| 3. + RTRRL adapting on the vehicle | **399.9** | 123 | 36% |
| 4. real from scratch, same total budget | **453.9** | 88 | 27% |

*(4 seeds, 300k steps in sim + 150k on the vehicle. Crashed in 48% of episodes
while adapting; **45% of the transfer gap closed online**.)*

With a predictive safety filter around the on-vehicle phase:

| condition | return | sd | off-track |
|---|---|---|---|
| 3. + RTRRL behind a filter, evaluated **naked** | 194.1 | 130 | 73% |
| 3b. ...the same agent evaluated **with its filter** | 343.9 | 203 | **11%** |

*(Crashed in 5% of episodes while adapting, against 48% unfiltered.)*

Four things, and three of them are uncomfortable:

**Online adaptation works, and it is modest.** 45% of the transfer gap closes
in 150k on-vehicle steps. Real, reproducible, and not the tenfold result a
headline would want.

**Learning from scratch on the vehicle beat fine-tuning** at the same total
budget (454 vs 400). On this task the pretraining bought nothing — the
simulator's policy is a *worse* starting point than random, because it encodes
a vehicle that does not exist. That is exactly the row this experiment was
designed to be able to embarrass itself with, and on a real car you could never
run it: you cannot spend 450k steps crashing. It is free here, so it gets run.

**An agent that learns behind a safety filter learns to lean on it.** Trained
filtered and then evaluated naked it scores 194 — far worse than never
adapting at all. Evaluated *with* its filter, as it would actually be deployed,
it scores 344 at an 11% off-track rate. Both numbers are true and reporting
only the second would be dishonest; the gap between them is the size of the
dependency it acquired.

**The safety filter still did its job.** Crashes during adaptation fell from
48% to 5%. If the vehicle is real, that trade — 400 → 344 in return, 48% → 5%
in crashes — is not obviously the wrong one. If it is simulated, it clearly is.

## Reproducing

```bash
python benchmarks/sweep.py --env memory-chain --grid cells --steps 200000 --seeds 3
python benchmarks/sweep.py --env lanekeep --grid cells --steps 300000 --seeds 4
python benchmarks/sweep.py --env lanekeep --grid estimators --steps 300000 --seeds 4
python benchmarks/run_suite.py --config benchmarks/configs/driving.yaml
python tutorial/10_safety_filter.py --steps 200000 --seeds 4
python tutorial/11_sim_to_real.py --sim-steps 300000 --real-steps 150000
python tutorial/11_sim_to_real.py --safe
```

Each of those is a few minutes on 15 cores.
