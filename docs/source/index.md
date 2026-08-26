# rtrrl-playground

A tutorial playground for **learning while you drive**: reinforcement learning
that updates every timestep, from a single stream of experience, with no replay
buffer, no batch, and no backward pass. The algorithm is **RTRRL** (Lemmel &
Grosu, AAAI 2025), built from scratch and taken apart on two small RC-car
tasks -- learn to drive, and learn to overtake.

Fifth companion project, after `liquid-nn-playground`, `sciml-playground`,
`cnn-playground` and `transformer-playground`. Same philosophy: every mechanism
hand-written from primitives. Here that means every derivative -- nothing in
the online agents calls `.backward()`, because not calling it is the point.

```{toctree}
:maxdepth: 2
:caption: Contents

getting_started
the_algorithm
cells
estimators
environments
real_data
tutorial
benchmarks
benchmark_results
to_scuderia_gym_jax
papers
```
