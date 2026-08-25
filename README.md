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
├── papers/             every reference, and why it is here
└── docs/               Sphinx / Read the Docs source
```

## Honest notes

Three things this repo says that a results table would not:

* **The paper's hyperparameters do not all work here.** The actor learning
  rate and the entropy bonus in particular. The reasons are measured and
  written down in [`algos/rtrrl/README.md`](algos/rtrrl/README.md#deviations-from-the-paper-and-why),
  along with the sweeps behind them.
* **`lanekeep` does not need memory.** Nine beams are a lot of information,
  and a memoryless policy drives it about as well as a hand-written
  wall-follower. That is in the benchmark table rather than quietly omitted.
  `overtake` is where memory earns its keep.
* **This is a reimplementation for taking apart, not a reproduction.** The
  numbers here are on this repo's own small environments, not the paper's
  benchmark suite. For the authors' code and results, follow the arXiv links
  in [`papers/README.md`](papers/README.md).

## Where it is going

The intended destination is
[`scuderia_gym_jax`](https://github.com/agpoks/scuderia_gym_jax) -- ST/STD/STD4W
vehicle models with Pacejka and brush tyres fitted to real RC-car recordings.
[`rtrrl_playground/envs/scuderia.py`](rtrrl_playground/envs/scuderia.py) is a
working adapter, and [`tutorial/08`](tutorial/08_to_scuderia_gym_jax.py) runs
the agent on it unchanged.

## License

MIT, see [`LICENSE`](LICENSE).
