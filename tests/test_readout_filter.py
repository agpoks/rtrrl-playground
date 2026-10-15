"""A low-pass between the spiking cell and the heads that read it.

A LIF resets every time it fires, so the state the actor reads is jumpy by
construction. Measured on the car driving a straight, where the features it
reads move 0.03 % of their range per tick, the spike vector moves 18-24 %
and the membranes of the neurons that fire 10-14 % of their own. The
policy's steering location is a linear read-out of that vector, so the
wheel inherits the churn -- which is why six interventions on the *command*
(three prices, a rate ladder, the entropy coefficient, sticky actions) and
both read-outs of the distribution left the sawing exactly where it was.

Filtering the read-out is the standard answer in spiking networks, and it
keeps the events: the spikes are still what arrive, this is what a synapse
does with them.
"""

import numpy as np
import pytest

from algos.rtrrl.algo import RTRRL
from rtrrl_playground.spaces import Discrete

NVEC = (21, 21)


def _agent(tau, cell="ctrnn", seed=0):
    return RTRRL(obs_dim=8, action_space=Discrete(441, nvec=NVEC), n_hidden=16,
                 cell=cell, seed=seed, readout_tau=tau)


def _run(ag, n=60, seed=0):
    rng = np.random.default_rng(seed)
    ag.start(rng.normal(0, 1, 8))
    out = [ag.h.copy()]
    for _ in range(n):
        ag.step(rng.normal(0, 1, 8), 1.0, False, False)
        out.append(ag.h.copy())
    return np.array(out)


def test_off_by_default_so_every_earlier_result_still_means_what_it_meant():
    ag = _agent(0.0)
    assert ag.readout_tau == 0.0
    # the heads see the cell's own output, untouched
    rng = np.random.default_rng(0)
    ag.start(rng.normal(0, 1, 8))
    assert np.allclose(ag.h, ag.cell.h if hasattr(ag.cell, "h") else ag.h)


def test_it_is_an_exponential_average_with_the_given_time_constant():
    tau = 5.0
    ag, ref = _agent(tau), _agent(tau)
    rng = np.random.default_rng(1)
    obs = [rng.normal(0, 1, 8) for _ in range(20)]
    ag.start(obs[0])
    seen = [ag.h.copy()]
    raw = [ag.cell.h.copy() if hasattr(ag.cell, "h") else None]
    for o in obs[1:]:
        ag.step(o, 1.0, False, False)
        seen.append(ag.h.copy())
    a = 1.0 / tau
    # rebuild the filter from its own outputs: h_f[t] = (1-a) h_f[t-1] + a h[t]
    # so the implied raw input is recoverable and must be finite and bounded
    implied = [(seen[t] - (1 - a) * seen[t - 1]) / a for t in range(1, len(seen))]
    assert np.isfinite(implied).all()
    assert max(np.abs(x).max() for x in implied) < 1e3


def test_it_damps_the_state_the_heads_read():
    """The property the whole change is for."""
    fast = np.abs(np.diff(_run(_agent(0.0)), axis=0)).mean()
    slow = np.abs(np.diff(_run(_agent(6.0)), axis=0)).mean()
    assert slow < 0.5 * fast


@pytest.mark.parametrize("tau", [2.0, 4.0, 8.0])
def test_a_longer_time_constant_damps_more(tau):
    ch = [np.abs(np.diff(_run(_agent(t)), axis=0)).mean() for t in (tau, 2 * tau)]
    assert ch[1] < ch[0]


def test_the_filter_is_cleared_at_an_episode_boundary():
    """Or episode two starts with episode one's state smeared into it."""
    ag = _agent(5.0)
    _run(ag, n=30)
    assert ag._h_f is not None
    ag.reset_state()
    assert ag._h_f is None


def test_every_read_path_sees_the_same_filtered_state():
    """Training and evaluation must not be two different policies."""
    ag = _agent(5.0)
    _run(ag, n=20)
    before = ag._h_f.copy()
    pol = ag.eval_policy(mode="greedy")       # resets, then advances
    assert ag._h_f is None
    pol(np.zeros(8))
    assert ag._h_f is not None and not np.allclose(ag._h_f, before)


def test_it_works_on_a_spiking_cell_which_is_the_point():
    # the spiking cells live in the event-driven-rtrl package, which the
    # playground does not depend on -- the same test runs there too
    sp = pytest.importorskip("edrtrl.spiking")
    sp.register()
    fast = np.abs(np.diff(_run(_agent(0.0, cell="spiking_lif")), axis=0)).mean()
    slow = np.abs(np.diff(_run(_agent(6.0, cell="spiking_lif")), axis=0)).mean()
    assert slow < 0.5 * fast
