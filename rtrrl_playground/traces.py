"""Eligibility traces: the memory that lets an online update reach backwards.

A single TD error at time ``t`` should not only correct the value of the state
at time ``t``. It is evidence about every state that led there, discounted by
how long ago and by ``lambda``. Storing that as a running sum over parameters
-- one trace vector per parameter vector -- is what makes an update that only
ever looks at the current step still able to credit an action ten steps back.
That is the piece that makes RTRRL work on a task like MemoryChain, where the
only reward arrives at the very end.

Two flavours here, and the difference matters more than it looks:

**Accumulating** ``e <- gamma*lambda*e + x``: the textbook TD(lambda) trace
(Sutton 1988). Simple, and it overshoots -- revisit a state before its trace
has decayed and the trace keeps piling up, so the effective step size on that
feature is larger than the one you configured.

**Dutch** ``e <- gamma*lambda*e + alpha*x - alpha*gamma*lambda*(e.x)*x``: the
trace of true online TD(lambda) (van Seijen et al., JMLR 2016). The extra term
subtracts exactly the part of the accumulation that the new observation makes
redundant. Paired with the small correction term in
:func:`true_online_value_update`, the online updates then match what an
offline lambda-return calculation would have produced -- which is the
guarantee ordinary accumulating traces do not have, and the reason the RTRRL
paper uses this form for the critic. It also absorbs the learning rate into
the trace, which is why ``alpha`` appears here and not at the update site.
"""

from __future__ import annotations

import numpy as np


def accumulating_trace(e: np.ndarray, x: np.ndarray, gamma: float, lam: float) -> np.ndarray:
    return gamma * lam * e + x


def dutch_trace(e: np.ndarray, x: np.ndarray, gamma: float, lam: float, alpha: float) -> np.ndarray:
    gl = gamma * lam
    return gl * e + alpha * x - alpha * gl * float(e @ x) * x


def true_online_value_update(theta: np.ndarray, e: np.ndarray, x: np.ndarray,
                             delta: float, v_old: float, alpha: float) -> np.ndarray:
    """One true-online TD(lambda) critic step, in place-free form.

    ``v_old`` is the value this same state had under the *previous* weights.
    The correction ``alpha (v_old - theta.x) x`` undoes the part of the last
    update that has already been applied to the state we are still sitting on.
    Drop it and you have ordinary TD(lambda) with a Dutch trace, which is not
    the same algorithm and is measurably worse at large ``alpha``.
    """
    return theta + delta * e + alpha * (v_old - float(theta @ x)) * x
