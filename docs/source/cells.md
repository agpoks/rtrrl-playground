# The recurrent cells

Four cells plus a control, chosen so each adds exactly one mechanism to the one
before. All of them plug into any of the five {doc}`estimators`, because the
cell decides what the state does and the estimator decides what you are allowed
to know about how it got there.

| `--cell` | update | adds | reference |
|---|---|---|---|
| `ctrnn` | $h_{t+1} = h_t + (\tanh(W\xi) - h_t)/\tau$ | a learned but input-*independent* time constant | Funahashi & Nakamura 1993 |
| `ltc` | $h_{t+1} = (h_t + \Delta t\, f A)/(1 + \Delta t(1/\tau + f))$ | $\tau$ becomes a function of the input -- *liquid* | [Hasani et al. 2021](https://arxiv.org/abs/2006.04439) |
| `lrcu` | $h_{t+1} = (1 - \epsilon\sigma(f))h_t + \epsilon\tanh(u)\,e$ | the *capacitance* becomes one too | [Farsang et al. 2024](https://arxiv.org/abs/2403.08791) |
| `ligru` | $h_{t+1} = z h_t + (1-z)\tanh(W_c\xi)$ | gating without continuous time, as the control | [Ravanelli et al. 2018](https://arxiv.org/abs/1803.10225) |
| `mlp` | $h_{t+1} = \tanh(W x_t)$ | nothing; no recurrence at all | -- |

**LRCU is here because of the hardware paper.** Lemmel, Resch, Farsang,
Hasani, Rus & Grosu ([arXiv:2602.02236](https://arxiv.org/abs/2602.02236)) put
RTRRL on a real 1:10 RoboRacer with an event camera and found the LRC cell the
one that gets on best with it -- which makes it the most directly relevant
result in the literature to what this playground is for.

## Two things worth knowing before you write your own cell

**Every parameter must belong to exactly one neuron.** `theta` is an $(n, p)$
array: row $i$ is everything neuron $i$ owns. That is not packing convenience,
it is the structural property that makes local online learning possible --
without it the immediate Jacobian is not block-diagonal, RFLO's "keep only
$\partial h_i/\partial\theta_{ij}$" throws away a first-order effect rather
than a second-order one, and the influence array cannot represent what is
happening.

That is why `ligru` is here and a full **GRU is not**. The GRU's reset gate
computes $c = \tanh(W_c[x; r \odot h])$, so $W_r$ belonging to neuron $k$
reaches $h_i$ for every $i$ through $W_c[i,k]$. The boundary is drawn
explicitly in `nets/ligru.py` rather than papered over, because it is a real
and underdiscussed constraint on which architectures admit this kind of
training.

**Bound whatever drives the state.** An LTC settles towards its reversal
potential $A$; an LRCU towards $e$. Leave those unbounded and the state is
unbounded, and over a long online run with no batch to average over that is not
a remote possibility -- it is what happens, at about step 200,000, as a NaN.
Both are clipped in `post_update`, which is also the biophysically honest
choice: a reversal potential is a voltage, not a free scale.

There is a third, subtler one, and it caught this repo too. A unit that has
learned to hold its state perfectly has leak $= 1$, and an influence trace that
never decays is a sum that never converges. `OnlineCell` caps the decay at
`leak_max = 0.99`, an explicit influence horizon of about 100 steps -- the same
idea as $\lambda$ in an eligibility trace.

## Deliberately not here

**CfC** and **Liquid-S4** already live in
[`liquid-nn-playground`](https://github.com/agpoks/liquid-nn-playground) with
proper supervised benchmarks. What is new in this repo is not the cells, it is
training them one timestep at a time from a reward.
