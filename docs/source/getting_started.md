# Getting started

## Install

```bash
git clone https://github.com/agpoks/rtrrl-playground.git
cd rtrrl-playground
pip install -e .
```

The online agents need only NumPy and matplotlib. Two extras are optional and
neither is needed for the first seven lessons:

```bash
pip install -e ".[torch]"      # only for algos/a2c_bptt, the BPTT baseline
pip install -e ".[notebooks]"  # jupyter, if you want the .ipynb versions
```

## The one file to read

```bash
python tutorial/04_rtrrl_from_scratch.py
```

About a hundred lines of NumPy: a CT-RNN, an RFLO influence array, three
eligibility traces, and one TD error. It imports an environment from this repo
and nothing else. Everything in `algos/` and `rtrrl_playground/` is a
generalisation of it -- five gradient estimators instead of one, five cells
instead of one, and flags for the ablations.

## Run an agent

```bash
python algos/rtrrl/example.py --env lanekeep --render
python algos/rtrrl/example.py --env overtake --cell lrcu --steps 400000 --render
python algos/ac_lambda/example.py --env lanekeep      # the memoryless control
python algos/a2c_bptt/example.py --env lanekeep       # the BPTT control
```

`--render` saves a picture of one greedy episode into `runs/`: the track and
the driven line, coloured by speed.

## Run the tutorial

```bash
for f in tutorial/0*.py; do python "$f"; done
python scripts/make_notebooks.py     # regenerate the .ipynb versions
```

Lessons 1-4 take a couple of minutes each; 5-7 are training runs and take
longer. All of them take `--steps`.

## Data

Nothing downloads. Three of the four environments generate their own data by
being simulated, and real recordings are yours to supply -- see
[`datasets/README.md`](https://github.com/agpoks/rtrrl-playground/blob/main/datasets/README.md)
and {doc}`real_data`.

## Compare things

```bash
python benchmarks/run_suite.py --config benchmarks/configs/driving.yaml
python benchmarks/sweep.py --env lanekeep --grid cells --seeds 3
```

Both run one process per configuration. Nothing in the agent is threaded, so
they scale linearly with cores.
