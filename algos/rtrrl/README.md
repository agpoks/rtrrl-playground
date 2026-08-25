# RTRRL

**Paper:** Julian Lemmel, Radu Grosu, *"Real-Time Recurrent Reinforcement
Learning"*, AAAI 2025 — [arXiv:2311.04830](https://arxiv.org/abs/2311.04830).
Follow-up on real hardware: Lemmel, Resch, Farsang, Hasani, Rus, Grosu,
*"Online Fine-Tuning of Pretrained Controllers for Autonomous Driving via
Real-Time Recurrent RL"*, 2026 —
[arXiv:2602.02236](https://arxiv.org/abs/2602.02236). See
[`papers/README.md`](../../papers/README.md).

## Idea in one paragraph

Three old pieces, composed. A recurrent network is fed
`[observation, previous action, previous reward]` and reads out a policy and a
value — the *meta-RL* architecture. The outer learning rule is TD(λ)
actor-critic with eligibility traces, so one scalar TD error per step can
credit actions taken many steps ago without anything having been stored. And
the recurrent network's own gradient is carried **forwards**, as an influence
array updated in the same pass as the activations (RFLO), so there is never a
backward pass, never a stored history, and never a truncation window. The
result learns from a single stream of experience at constant memory and
constant cost per step — which is what makes it a candidate for something
running on the car rather than in a training rig.

## Files

- `algo.py` — the agent. One class, one update per timestep, no autograd.
- `example.py` — trains it on any of the four environments, with every knob
  the tutorial argues about exposed as a flag.
- `README.md` — this file.

The pieces it composes live in the shared package:
[`nets/cell.py`](../../rtrrl_playground/nets/cell.py) (the five gradient
estimators), [`nets/ctrnn.py`](../../rtrrl_playground/nets/ctrnn.py) and its
three siblings (the cells), [`nets/heads.py`](../../rtrrl_playground/nets/heads.py)
(hand-derived policy/value gradients and feedback alignment), and
[`traces.py`](../../rtrrl_playground/traces.py) (the eligibility traces).

## Run it

```bash
pip install -e .
python algos/rtrrl/example.py --env cartpole-vel --steps 300000
python algos/rtrrl/example.py --env lanekeep  --cell lrcu --render
python algos/rtrrl/example.py --env overtake  --cell ltc --steps 600000 --render
```

Ablations worth an afternoon:

```bash
--estimator rtrl        # the exact gradient: how much is RFLO throwing away?
--estimator uoro        # unbiased but noisy, instead of biased but quiet
--estimator none        # freeze the recurrence: was it a reservoir all along?
--feedback symmetric    # turn feedback alignment off
--critic-update paper   # Algorithm 1's printed critic line, not full true-online
--no-meta-inputs        # stop feeding the last action and reward back in
```

## Timing, which is the part that is easy to get wrong

At step `t` the agent holds `h_t` and the influence `J_t` of `h_t` on the
weights. It draws `a_t` from `h_t`; the environment answers with `r_t` and
`o_{t+1}`; only then does the cell advance to `h_{t+1}`. The TD error compares
`v(h_{t+1})` with `v(h_t)` — but **every trace is updated with the quantities
from time `t`**: `h_t`, `J_t`, and the policy gradient at the action actually
taken. Pairing `J_{t+1}` with `h_t` does not crash, does not warn, and quietly
costs most of the learning.

## Deviations from the paper, and why

The paper's Table 5 hyperparameters are all implemented and reachable; four of
them are not the defaults here, and the reasons are worth reading before you
trust either set of numbers.

| Setting | Paper | Here | Why |
|---|---|---|---|
| `alpha_C` (critic) | 1.0 | 1.0, but **normalised** by ‖φ‖² | The Dutch trace is stable only while `alpha * ‖φ‖² ≤ 1`. With `n` tanh units, `‖φ‖²` starts near 1 (the state is zero, only the bias feature is on) and drifts towards `n` as the state fills out — so 1.0 is fine for a few hundred steps and then diverges. Dividing by `‖φ‖²` pins that product at `alpha_C` forever. `--critic-lr-mode fixed` restores the literal reading. |
| `alpha_A` (actor) | 1e-2 | 1e-3 | Measured, not guessed: see the sweep in [`benchmarks/`](../../benchmarks). At 1e-2 the actor outruns the critic, the persistent early TD-error bias locks the policy onto whichever action it happened to prefer, and the entropy is gone inside a thousand steps. Two-timescale — critic fast, actor slow — is the difference between learning and not. |
| `eta_H` (entropy) | 1e-5 | 3e-2 | The same failure from the other side. The entropy bonus is what keeps the policy alive long enough for the critic to become worth listening to. |
| critic update | Algorithm 1 as printed | full true-online TD(λ) | Algorithm 1 writes `θ_C ← θ_C + δ e_C + α(v − θ_C·h)h`, which is true-online TD(λ) minus its `(v − v_old)e` term. Both are here; `--critic-update paper` gives the printed one. |

One structural deviation as well: the paper's loop draws its first action from
`h = 0`, before the first observation has been fed in, so `o_0` is never seen.
Here the cell consumes `o_0` first and acts from `h_0`. Same algorithm, one
fewer wasted step.

## What it is not

This is a compact, readable reimplementation built for **side-by-side
comparison against its own ablations** on small tasks that run on a laptop.
It is not a reproduction of the paper's numbers on its benchmark suite, and
the environments here are not the ones the paper reports. For the authors'
own code and results, follow the arXiv links above.
