# Papers

Every algorithm, cell and environment in this repo traces to one of these.
BibTeX for all of them is in [`references.bib`](references.bib).

## The algorithm

| # | Paper | Year | Link |
|---|---|---|---|
| 1 | Lemmel, Grosu — **Real-Time Recurrent Reinforcement Learning** | AAAI 2025 (preprint 2023) | [arXiv:2311.04830](https://arxiv.org/abs/2311.04830) |
| 2 | Lemmel, Resch, Farsang, Hasani, Rus, Grosu — **Online Fine-Tuning of Pretrained Controllers for Autonomous Driving via Real-Time Recurrent RL** | 2026 | [arXiv:2602.02236](https://arxiv.org/abs/2602.02236) |
| 3 | Lemmel, Grosu — **On the Benefits of Biophysical Synapses** | AAAI 2023 | [arXiv:2303.04944](https://arxiv.org/abs/2303.04944) |

**(1) is what this repo implements.** Three parts — a meta-RL recurrent
architecture, TD(λ) actor-critic with eligibility traces, and RFLO for the
recurrent gradient — composed into an agent that learns from a single stream
of experience with no replay buffer, no batch and no backward pass.

**(2) is why this repo is aimed at RC cars.** The same authors put RTRRL on a
1:10 RoboRacer with an event camera: behavioural cloning offline, then RTRRL
fine-tuning *while driving*, and the policy improves within the first lap.
Two findings from it shape the design here — that the liquid-resistance
liquid-capacitance cell (5) is the one that gets on best with RTRRL, which is
why [`nets/lrcu.py`](../rtrrl_playground/nets/lrcu.py) exists; and that the
natural deployment is fine-tuning a pretrained controller rather than
learning from scratch, which is what
[`tutorial/07_finetune_a_controller.py`](../tutorial) does in miniature.

**(3) is the synapse model** behind the per-synapse nonlinearities in LTC and
LRC. This repo's cells use ordinary linear pre-activations instead and say so
in [`nets/lrcu.py`](../rtrrl_playground/nets/lrcu.py); the paper is the
reference for what is being left out.

## The recurrent cells

| # | Paper | Year | Link | Here |
|---|---|---|---|---|
| 4 | Hasani, Lechner, Amini, Rus, Grosu — **Liquid Time-constant Networks** | AAAI 2021 | [arXiv:2006.04439](https://arxiv.org/abs/2006.04439) | [`nets/ltc.py`](../rtrrl_playground/nets/ltc.py) |
| 5 | Farsang, Neubauer, Grosu — **Liquid Resistance Liquid Capacitance Networks** | NeuroAI @ NeurIPS 2024 | [arXiv:2403.08791](https://arxiv.org/abs/2403.08791) | [`nets/lrcu.py`](../rtrrl_playground/nets/lrcu.py) |
| 6 | Funahashi, Nakamura — **Approximation of dynamical systems by continuous time recurrent neural networks** | Neural Networks 1993 | — | [`nets/ctrnn.py`](../rtrrl_playground/nets/ctrnn.py) |
| 7 | Ravanelli, Brakel, Omologo, Bengio — **Light Gated Recurrent Units for Speech Recognition** | IEEE TETCI 2018 | [arXiv:1803.10225](https://arxiv.org/abs/1803.10225) | [`nets/ligru.py`](../rtrrl_playground/nets/ligru.py) |

Each cell adds exactly one mechanism to the one before it: a fixed learned
time constant (6), a time constant that depends on the input (4), a
*capacitance* that depends on the input as well (5), and — as the control —
gating without any continuous time at all (7).

There is a deliberate gap. **CfC** (Hasani et al., Nature MI 2022) and
**Liquid-S4** are not here; they already live in
[`liquid-nn-playground`](https://github.com/agpoks/liquid-nn-playground) with
proper supervised benchmarks, and this repo links to them rather than
rebuilding them. What is new here is not the cells, it is training them one
timestep at a time from a reward.

## The online gradient

| # | Paper | Year | Link | Here |
|---|---|---|---|---|
| 8 | Williams, Zipser — **A Learning Algorithm for Continually Running Fully Recurrent Neural Networks** | Neural Computation 1989 | — | `estimator="rtrl"` |
| 9 | Murray — **Local online learning in recurrent networks with random feedback** | eLife 2019 | [eLife 43299](https://doi.org/10.7554/eLife.43299) | `estimator="rflo"` |
| 10 | Tallec, Ollivier — **Unbiased Online Recurrent Optimization** | ICLR 2018 | [arXiv:1702.05043](https://arxiv.org/abs/1702.05043) | `estimator="uoro"` |
| 11 | Menick, Elsen, Evci, Osindero, Simonyan, Graves — **A Practical Sparse Approximation for Real Time Recurrent Learning** | ICLR 2021 | [arXiv:2006.07232](https://arxiv.org/abs/2006.07232) | `estimator="snap1"` |
| 12 | Lillicrap, Cownden, Tweed, Akerman — **Random synaptic feedback weights support error backpropagation for deep learning** | Nature Comms 2016 | [doi](https://doi.org/10.1038/ncomms13276) | `feedback="random"` |
| 13 | Bellec et al. — **A solution to the learning dilemma for recurrent networks of spiking neurons (e-prop)** | Nature Comms 2020 | [doi](https://doi.org/10.1038/s41467-020-17236-x) | see note |
| 14 | Marschall, Cho, Savin — **A Unified Framework of Online Learning Algorithms for Training RNNs** | JMLR 2020 | [arXiv:1907.02649](https://arxiv.org/abs/1907.02649) | the map of (8)–(11) |

(8) is the exact thing; (9)–(11) are three different bargains with it, and
they are the `--estimator` flag. (14) is the paper to read if you want the
whole family laid out in one notation — this repo implements four points on
its map.

**e-prop (13) is not a separate estimator here** on purpose. Its eligibility
trace *is* RFLO's influence for a leaky unit, combined with a learning signal
that arrives from outside; RTRRL is, in that framing, e-prop with a TD(λ)
actor-critic supplying the signal. Implementing it as a fifth option would
have produced a near-duplicate of `rflo` and obscured that.

## The learning rule

| # | Paper | Year | Link | Here |
|---|---|---|---|---|
| 15 | Sutton — **Learning to predict by the methods of temporal differences** | Machine Learning 1988 | — | `--critic-update accumulating` |
| 16 | van Seijen, Mahmood, Pilarski, Machado, Sutton — **True Online Temporal-Difference Learning** | JMLR 2016 | [arXiv:1512.04087](https://arxiv.org/abs/1512.04087) | `--critic-update true-online` |
| 17 | Mnih et al. — **Asynchronous Methods for Deep Reinforcement Learning (A3C/A2C)** | ICML 2016 | [arXiv:1602.01783](https://arxiv.org/abs/1602.01783) | [`algos/a2c_bptt`](../algos/a2c_bptt) |

## Safe learning

| # | Paper | Year | Link | Here |
|---|---|---|---|---|
| 20 | Wabersich, Zeilinger — **A predictive safety filter for learning-based control of constrained nonlinear dynamical systems** | Automatica 2021 | [arXiv:1812.05506](https://arxiv.org/abs/1812.05506) | [`safety.py`](../rtrrl_playground/safety.py) |
| 21 | Wabersich, Zeilinger — **Linear model predictive safety certification for learning-based control** | CDC 2018 | [arXiv:1803.08552](https://arxiv.org/abs/1803.08552) | the same idea, linear |
| 22 | Hewing, Wabersich, Menner, Zeilinger — **Learning-Based Model Predictive Control: Toward Safe Learning in Control** | Annu. Rev. Control 2020 | — | the survey |
| 23 | Ames, Coogan, Egerstedt, Notomista, Sreenath, Tabuada — **Control Barrier Functions: Theory and Applications** | ECC 2019 | [arXiv:1903.11199](https://arxiv.org/abs/1903.11199) | [`cbf.py`](../rtrrl_playground/cbf.py) |
| 23b | Ames, Xu, Grizzle, Tabuada — **Control Barrier Function Based Quadratic Programs for Safety Critical Systems** | IEEE TAC 2017 | — | the QP form |
| 23c | Agrawal, Sreenath — **Discrete Control Barrier Functions for Safety-Critical Control of Discrete Systems** | RSS 2017 | — | the discrete-time condition used here |
| 24 | García, Fernández — **A Comprehensive Survey on Safe Reinforcement Learning** | JMLR 2015 | — | the field |

(20) is what [`rtrrl_playground/safety.py`](../rtrrl_playground/safety.py)
implements: rather than shaping the reward or restricting the policy class, put
a filter between the agent and the actuator that asks one question per step --
*does a safe backup plan still exist if I apply this?* -- and applies the
nearest action for which the answer is yes. The learner is untouched in the
interior and constrained only at the boundary, which is why a competent policy
is filtered on ~0% of steps while a random one never leaves the track.

**(23) is the pointwise alternative**, and both are implemented so they can be
compared on the same task, action set, model and wrapper. A CBF certifies
safety by evaluating a scalar function of the current state; a predictive
filter certifies it by exhibiting a trajectory. Measured, the difference is
*not* that one is safer — the barrier design carries the safety, and a naive
positional barrier fails at 47% where the same method with a closing-rate term
does not fail at all. The differences that are real: the pointwise method is
structurally more conservative (a one-step condition cannot tell that a plan
exists), and their cost profiles are opposite. See
[`docs/source/safety.md`](../docs/source/safety.md).

## The environments

| # | Paper | Year | Here |
|---|---|---|---|
| 18 | Barto, Sutton, Anderson — **Neuronlike adaptive elements that can solve difficult learning control problems** | IEEE SMC 1983 | [`envs/cartpole.py`](../rtrrl_playground/envs/cartpole.py) |
| 19 | Osband et al. — **Behaviour Suite for Reinforcement Learning (bsuite)** | ICLR 2020 | [`envs/memory_chain.py`](../rtrrl_playground/envs/memory_chain.py) |

The two driving environments are this repo's own, sized to a 1:10 RC car and
built to be thrown away: they exist so the tutorial can be run on a laptop in
a minute, and the intended destination is
[`scuderia_gym_jax`](https://github.com/agpoks/scuderia_gym_jax), whose ST/STD
vehicle models and real tyre parameters are what a result should eventually be
reported on. See [`docs/source/to_scuderia_gym_jax.md`](../docs/source/to_scuderia_gym_jax.md).
