<p align="center">
  <img src="docs/source/_static/logo-banner.svg" alt="rtrrl-playground" width="560">
</p>

# rtrrl-playground

A tutorial playground for **learning while you drive**: reinforcement learning
that updates *every timestep*, from a single stream of experience, with no
replay buffer, no batch, and no backward pass. The algorithm is **RTRRL**
(Real-Time Recurrent Reinforcement Learning, Lemmel & Grosu, AAAI 2025), built
here from scratch and taken apart piece by piece on two small RC-car tasks:
**learn to drive**, and **learn to overtake**.

Fifth companion project, after
[`liquid-nn-playground`](https://github.com/agpoks/liquid-nn-playground),
[`sciml-playground`](https://github.com/agpoks/sciml-playground),
[`cnn-playground`](https://github.com/agpoks/cnn-playground) and
[`transformer-playground`](https://github.com/agpoks/transformer-playground) --
same layout and the same philosophy: every mechanism is hand-written from
primitives. Here that means literally every derivative. **Nothing in the online
agents calls `.backward()`**, because not calling it is the point.

## Start here

```bash
git clone https://github.com/agpoks/rtrrl-playground.git
cd rtrrl-playground
pip install -e .
python tutorial/04_rtrrl_from_scratch.py      # the whole algorithm, one file
```

[`tutorial/`](tutorial) is eight lessons in order, each a runnable script and a
matching notebook:

| # | lesson | what it establishes |
|---|---|---|
| 1 | [The problem](tutorial/01_the_problem.py) | Partial observability, measured. Your sensors give positions; your decisions need rates. |
| 2 | [Online gradients](tutorial/02_gradients_online.py) | RTRL, UORO, SnAp-1, RFLO and truncated BPTT, graded against the exact gradient in alignment, memory and microseconds. |
| 3 | [Eligibility traces](tutorial/03_traces.py) | How one scalar credits an action ten steps back, at one number per parameter. |
| 4 | [**RTRRL from scratch**](tutorial/04_rtrrl_from_scratch.py) | The whole algorithm, ~100 lines of NumPy, importing nothing from this repo but an environment. |
| 5 | [Learn to drive](tutorial/05_learn_to_drive.py) | A 1:10 car with nine lidar beams, no speedometer, and a grip level that changes every episode. |
| 6 | [Learn to overtake](tutorial/06_learn_to_overtake.py) | Traffic whose closing speed no single frame contains. |
| 7 | [Fine-tune a controller](tutorial/07_finetune_a_controller.py) | Clone offline, improve online while driving -- the deployment story from Lemmel et al. (2026). |
| 8 | [Onto `scuderia_gym_jax`](tutorial/08_to_scuderia_gym_jax.py) | Swap the toy bicycle for real ST/STD vehicle models, and what changes when you do. |
| 9 | [Clone from a real bag](tutorial/09_clone_from_a_real_bag.py) | Clone a real driver from a ROS 2 recording, and rebuild the circuit it was recorded on. |
| 10 | [Safety filter](tutorial/10_safety_filter.py) | A predictive safety filter from scratch: never leave the track *while learning*, and what that costs. |
| 11 | [Sim-to-real](tutorial/11_sim_to_real.py) | Train in simulation, deploy on a vehicle the simulator was wrong about, and close the gap online. |

## What is in the box

**One algorithm, factored into three independent choices.** RTRRL is a
recurrent cell, plus a way of getting that cell's gradient, plus a TD(λ)
actor-critic. Each is a flag, and any combination runs:

```bash
python algos/rtrrl/example.py --env lanekeep --cell lrcu --estimator rflo
python algos/rtrrl/example.py --env overtake --cell ltc --estimator rtrl --feedback symmetric
```

| `--cell` | what it adds | paper |
|---|---|---|
| `ctrnn` | a learned but input-*independent* time constant (RTRRL's own) | Funahashi & Nakamura 1993 |
| `ltc` | the time constant becomes a function of the input -- *liquid* | [Hasani et al. 2021](https://arxiv.org/abs/2006.04439) |
| `lrcu` | the *capacitance* becomes one too -- the cell the RTRRL hardware paper recommends | [Farsang et al. 2024](https://arxiv.org/abs/2403.08791) |
| `ligru` | gating without continuous time, as the control | [Ravanelli et al. 2018](https://arxiv.org/abs/1803.10225) |
| `liquid_gru` | a GRU gate read as a *conductance* with a leak floor, so the influence series converges without a numerical cap ([derived from scratch in lesson 12](tutorial/12_cells_from_scratch.py)) | this repo's own |
| `physics_ligru` | three units dead-reckon the agent's own command through the *known* vehicle response; the rest is LiGRU. Untrained they track the hidden steering angle at r=1.00 and the hidden speed at r=0.85 — and at 8 seeds it changes **nothing** measurable about the return, which is [written up as the null result it is](docs/source/cells.md) | this repo's own |
| `mlp` | no recurrence at all, as the other control | -- |

| `--estimator` | what it carries | memory | bias |
|---|---|---|---|
| `rtrl` | the exact influence `dh_i/dtheta_kj` | `n² p` | none |
| `uoro` | a rank-1 random sketch of it | `n + n p` | none, but very noisy |
| `snap1` | the diagonal blocks, true self-recurrence | `n p` | yes |
| `rflo` | the same, decayed by the leak only -- RTRRL's choice | `n p` | yes |
| `none` | nothing; the recurrence is a frozen reservoir | 0 | n/a |

**Two more algorithms, as controls.** [`ac_lambda`](algos/ac_lambda) is the
same agent with the recurrence deleted -- it says what the memory was worth.
[`a2c_bptt`](algos/a2c_bptt) is the same recurrent cell trained the standard
way, with truncated BPTT and PyTorch autograd -- it says what the online
gradient cost. See [`algos/README.md`](algos/README.md).

**The physics is documented**, not buried: a kinematic bicycle with a yaw-rate
cap standing in for understeer, a first-order steering servo, drag, and a
per-episode grip level that is never observed — with every parameter in one
`VehicleParams` dataclass so a *second* vehicle is one argument away. Full
equations, parameter table, sensor model, and an explicit statement of what is
**not** modelled (no slip angle, no tyre curve, no load transfer) in
[`docs/source/physics.md`](docs/source/physics.md). The real vehicle models are
in `scuderia_gym_jax`, one adapter away.

**Four environments**, all partially observable by construction, none of which
ever puts a velocity in the observation:

| id | what it tests |
|---|---|
| `memory-chain` | memory alone; optimal return is exactly 1, so "did it learn" is not a judgement call |
| `cartpole-vel` | control with the velocities hidden -- the RTRRL paper's POMDP |
| `lanekeep` | drive: hold a 27 m oval at the grip limit, with the grip redrawn each episode |
| `overtake` | drive, plus two slower cars whose speed is never observed |

## Layout

```
rtrrl-playground/
├── tutorial/           eight lessons, .py + .ipynb  <- start here
├── algos/<name>/       algo.py, example.py, README.md  (one per algorithm)
├── rtrrl_playground/   shared: envs, cells, heads, traces, training loop
├── benchmarks/         YAML suites, and the parallel sweep runner
├── datasets/           where the data comes from (nothing ships; see the note)
├── papers/             every reference, and why it is here
└── docs/               Sphinx / Read the Docs source
```

## What it looks like when it works

MemoryChain-8, the pure memory test: optimal is `+1.0`, guessing is `0.0`, and
a memoryless policy provably cannot beat guessing.

| agent | final return (3 seeds) |
|---|---|
| RTRRL / **`lrcu`** (liquid resistance + capacitance) | **+0.94** |
| RTRRL / `ligru` | +0.88 |
| RTRRL / `ctrnn` (RTRRL's own cell) | +0.26 |
| AC(λ), memoryless | −0.01 |

The liquid cell winning is not a thumb on the scale: it is the cell the RTRRL
authors found works best on real hardware
([arXiv:2602.02236](https://arxiv.org/abs/2602.02236)), and the same ordering
falling out of an independent reimplementation on a task neither paper used is
about as good as this kind of agreement gets. Full tables, including the ones
that are less flattering, in [`docs/source/benchmark_results.md`](docs/source/benchmark_results.md).

## Honest notes

Three things this repo says that a results table would not:

* **The paper's hyperparameters do not all work here.** The actor learning
  rate and the entropy bonus in particular. The reasons are measured and
  written down in [`algos/rtrrl/README.md`](algos/rtrrl/README.md#deviations-from-the-paper-and-why),
  along with the sweeps behind them.
* **`lanekeep` does not need memory.** Nine beams are a lot of information,
  and a memoryless policy drives it about as well as a hand-written
  wall-follower. That is in the benchmark table rather than quietly omitted.
  `overtake` and `memory-chain` are where memory earns its keep.
* **The frozen-reservoir ablation is competitive.** On `lanekeep`, not
  training the recurrent weights at all beats training them with RFLO at
  several settings. The scope of that claim -- and the three explanations
  worth separating -- is
  [spelled out](algos/rtrrl/README.md#an-uncomfortable-measurement) rather
  than buried.
* **This is a reimplementation for taking apart, not a reproduction.** The
  numbers here are on this repo's own small environments, not the paper's
  benchmark suite. For the authors' code and results, follow the arXiv links
  in [`papers/README.md`](papers/README.md).

## Sim-to-real, without the real

Train in simulation; deploy on a vehicle the simulator was wrong about (longer
wheelbase, slower servo, **a steering trim that is not centred**, less grip,
noisier lidar — nine parameters, none observable); keep learning while driving.

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

This is the deployment story from
[Lemmel et al. 2026](https://arxiv.org/abs/2602.02236), with the vehicle
replaced by a second simulator so it runs on a laptop. See
[`tutorial/11`](tutorial/11_sim_to_real.py).

## Learning without crashing

`rtrrl_playground/safety.py` is a **predictive safety filter**
([Wabersich & Zeilinger, Automatica 2021](https://arxiv.org/abs/1812.05506)),
written from scratch. Before an action reaches the actuator it asks one
question: *if I apply this, does a safe backup plan still exist afterwards?*
Yes → apply it untouched. No → apply the nearest action for which one does.
The terminal set is "stopped, and on the track", and the backup is full braking
with the steering pointed back at the centreline.

```bash
python tutorial/10_safety_filter.py
```

| policy | off-track | return | filtered |
|---|---|---|---|
| random | **100%** | 14 | — |
| random + filter | **0%** | 15 | 16.4% |
| wall-follower | 0% | 569 | — |
| wall-follower + filter | 0% | **569** | **0.0%** |

Both halves matter. The filter takes the worst possible policy from crashing
every episode to never crashing — and it is *invisible* to a competent one, at
0% interventions and an unchanged return. A filter that intervenes constantly
on a good policy is not a safety filter, it is a controller, and you are
training against it rather than against the task.

And with RTRRL learning behind it on `lanekeep` (200k steps, 4 seeds), the
number that matters is crashes **while learning**:

| filter | eval return | crashes **during training** | crashes at eval | steps overridden |
|---|---|---|---|---|
| none | 428 ± 97 | 61.3% | 30.0% | — |
| `assumed_grip=1.0` (default), `credit=executed` | 354 ± 136 | 21.3% | 6.2% | 15.8% |
| `assumed_grip=1.0`, `credit=proposed` | 356 ± 216 | 20.6% | 13.8% | 30.7% |
| **`assumed_grip=0.6` (worst case)** | **449 ± 155** | **0.0%** | **0.0%** | 17.5% |
| `assumed_grip=1.4` (optimistic) | 329 ± 53 | 60.8% | 51.2% | 7.8% |

The worst-case filter never crashed once in 200k steps of learning **and**
finished with the highest return — an episode that ends in a wall is an episode
that stopped paying. The optimistic one is worse than no filter at all. A
safety filter inherits its guarantee from its model and nothing else.

It is also honest about what it cannot do: it is privileged (it reads the
state, not the agent's beams), it does not know the hidden grip either
(`assumed_grip` too high and crashes happen *through* it — there is a test that
asserts exactly that), and it makes the update off-policy in a way TD(λ) has no
term for (`credit={executed,proposed}`, both measured).

## Real recordings

`rtrrl_playground/data/rosbag.py` reads a ROS 2 bag -- `/scan`, a drive
command, odometry -- straight into the format the cloning stage consumes, and
rebuilds the circuit it was recorded on: the human's driven line becomes the
centreline and the recorded occupancy grid becomes the wall bitmap, so the
simulated beams hit the walls the real lidar saw. Pure Python, **no ROS
installation** (`pip install -e ".[bags]"`).

On a 254 s F1TENTH recording (6375 usable demonstrations after dropping the
stationary third), held-out agreement with the human's next command:

| clone | held-out agreement |
|---|---|
| RTRRL / `ctrnn` | **61.5%** |
| memoryless (`mlp`) | 57.1% |
| RTRRL / `lrcu` | 55.4% |
| majority-action baseline | 31.6% |

The baseline to beat is the majority action, not 1/9: a driver holds one
command for many frames, so "keep doing what you were doing" is already a
strong guess. The recurrent clone beats the memoryless one by 4 points, which
is the honest size of what memory buys at predicting a human here.

## Where it is going

The intended destination is
[`scuderia_gym_jax`](https://github.com/agpoks/scuderia_gym_jax) -- ST/STD/STD4W
vehicle models with Pacejka and brush tyres fitted to real RC-car recordings.
[`rtrrl_playground/envs/scuderia.py`](rtrrl_playground/envs/scuderia.py) is a
working adapter, and [`tutorial/08`](tutorial/08_to_scuderia_gym_jax.py) runs
the agent on it unchanged.

## License

MIT, see [`LICENSE`](LICENSE).
