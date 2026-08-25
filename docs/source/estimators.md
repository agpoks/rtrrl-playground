# The five gradient estimators

A recurrent network's gradient has to cross time. Backpropagation through time
stores every activation and replays them backwards; the alternative is to carry
$\partial h/\partial\theta$ **forwards**, in the same pass as the state. The
exact forward version is expensive, and everything interesting here is an
approximation of it.

All five are selected with `--estimator`, share one code path in
`rtrrl_playground/nets/cell.py`, and work with any cell.

| estimator | carries | memory | bias | reference |
|---|---|---|---|---|
| `rtrl` | the exact $J[i,k,j] = \partial h_i/\partial\theta_{kj}$ | $n^2 p$ | none | Williams & Zipser 1989 |
| `uoro` | a rank-1 random sketch $J \approx s\,\tilde\theta^\top$ | $n + np$ | none, but very noisy | Tallec & Ollivier 2018 |
| `snap1` | $J$ restricted to $i = k$, propagated through $\mathrm{diag}(D)$ | $np$ | yes | Menick et al. 2021 |
| `rflo` | the same sparsity, decayed by the leak only | $np$ | yes | Murray 2019 |
| `none` | nothing; the recurrence is a frozen reservoir | 0 | n/a | -- |

## What the difference actually costs

`tutorial/02_gradients_online.py` measures rather than asserts. Cosine
similarity to the exact gradient, on a CT-RNN with 16 units, as a function of
how far back the credit has to travel:

| delay (steps) | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| `rflo` | 0.72 | 0.71 | 0.71 | 0.70 | 0.67 | 0.67 |
| `snap1` | 0.72 | 0.71 | 0.71 | 0.70 | 0.67 | 0.68 |
| `uoro` (one sample) | 0.10 | 0.11 | 0.08 | 0.10 | 0.10 | 0.12 |
| BPTT-4 | 0.91 | 0.92 | 0.92 | 0.93 | 0.91 | 0.91 |
| BPTT-16 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Three things in that table are worth more than the ranking:

**RFLO's bias is bounded, not catastrophic.** It sits around 0.7 and stays
there. The part it keeps -- a neuron's own parameters' influence on its own
state -- is also the bulk of the gradient.

**SnAp-1 is indistinguishable from RFLO here**, and that is an honest negative
result rather than a bug. The one term SnAp-1 keeps that RFLO drops is each
neuron's *self*-recurrence $W_{ii}$, and at fan-in initialisation those
diagonal weights are $O(1/\sqrt{n})$ -- there is very little there to keep. The
two separate on a cell with strong self-recurrence, and not otherwise.

**UORO's per-sample score is not the point.** It is unbiased, which is a claim
about averages. Average $k$ independent UORO estimates of the *same* gradient
and the alignment climbs towards 1; RFLO averages to its own biased answer
however many samples you take. Whether that trade is worth it depends entirely
on how many updates you get to average over -- and at batch size one on a
vehicle, a quiet wrong direction may well beat a loud right one.

## Cost

CT-RNN, $n = 32$, $n_{in} = 20$, $p = 54$, one core:

| estimator | influence | µs/step |
|---|---|---|
| `rtrl` | 432 KiB | ~1000 |
| `uoro` | 13.8 KiB | ~90 |
| `snap1` | 13.5 KiB | ~55 |
| `rflo` | 13.5 KiB | ~45 |
| `none` | 0 | ~45 |

The memory column is the one that decides whether an algorithm fits on a car.
RTRL's $n^2p$ is 32x RFLO's here and grows with $n^2$; the LRCU with $p = 157$
carries 1.2 MiB of influence under RTRL and 39 KiB under RFLO.
