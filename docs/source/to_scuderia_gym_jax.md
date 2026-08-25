# Onto `scuderia_gym_jax`

`lanekeep` and `overtake` are throwaway: a kinematic bicycle with a one-line
grip limit, a bitmap track, nine beams. They exist so a lesson finishes on a
laptop in a minute.

The real thing -- ST / STD / STD4W single- and double-track models, Pacejka and
brush tyres fitted to actual RC-car recordings, load transfer, steering delay,
a friction map -- is in
[`scuderia_gym_jax`](https://github.com/agpoks/scuderia_gym_jax).
`rtrrl_playground/envs/scuderia.py` wraps it in the same eight-line `Env`
interface, so the agent does not change:

```python
from rtrrl_playground.envs.scuderia import ScuderiaLaneKeep
env = ScuderiaLaneKeep(model="std", map_name="berlin")
agent = RTRRL(env.obs_dim, env.action_space, cell="lrcu")   # unchanged
```

```bash
pip install jax chex
PYTHONPATH=/path/to/scuderia_gym_jax python tutorial/08_to_scuderia_gym_jax.py
```

## Three things that change when you cross over

**The reward is a different objective.** The maps that ship with the simulator
are occupancy images with no centreline, so there is no arc length to
differentiate and the adapter pays for *distance travelled without crashing*
instead. That is the usual f1tenth-style stand-in and it will happily reward
fast circles inside a wide corridor. Bring a centreline -- `scuderia_twin` and
the MPCC planners already carry one per track -- and replace `_reward`.

**A step costs about a hundred times more.** `lanekeep` runs at ~150 µs per
step; the adapter at ~12 ms, and almost none of that is physics.
`scuderia_gym_jax` is built to be `jit`-compiled around a whole `lax.scan`
rollout and `vmap`ped over thousands of cars; stepping it one tick at a time
from a Python agent throws all of that away. Measured, in the same simulator:
613 µs per step called from Python, 101 µs for the same step inside a
`lax.scan`.

**`vmap` does not help.** RTRRL is a single-stream algorithm: one car, one
update per timestep. What `vmap` is good for here is running many independent
seeds or hyperparameter settings at once, which -- given the seed variance in
{doc}`benchmarks` -- is genuinely worth having, but it is a different thing.

## The obvious next step

Port the *agent* into JAX and put the whole loop inside the scan. RTRRL is
unusually well suited to it, and this repo is deliberately set up for it:

* every array in `nets/cell.py` is **fixed-shape** -- the influence, the
  traces, the parameter block all have static shapes known at construction;
* the update is a **pure function** of `(params, traces, observation)`, with
  no replay buffer, no dynamic indexing and no Python control flow;
* the NumPy is written as array expressions rather than loops, so the
  translation is close to mechanical.

It has not been done here on purpose. NumPy first, so the algorithm is
readable; JAX second, when the algorithm is the thing you trust and the
simulator is the thing you are waiting on.

## And on real hardware

That is what Lemmel, Resch, Farsang, Hasani, Rus & Grosu did
([arXiv:2602.02236](https://arxiv.org/abs/2602.02236)): a 1:10 RoboRacer with
an event camera, behavioural cloning offline, RTRRL fine-tuning while driving.
`tutorial/07_finetune_a_controller.py` is that pipeline in miniature, and
`RTRRL.imitate()` is the cloning half -- online, one update per step, with the
same forward-mode gradient, so both phases keep the constant-memory promise.
