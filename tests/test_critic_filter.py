"""A low-pass on the state the *critic* reads, and only the critic.

Why this exists. On a policy that already drives well, the TD error is
almost entirely the critic's own step-to-step jitter rather than reward or
prediction error. Measured on a flawless frozen rollout of the 541 m
master_cup policy, corr(delta, gamma V(s') - V(s)) = 0.93, rising to 0.99
once the progress staircase is taken out of the reward -- a signal-to-noise
of about 2 %. V is a linear read-out of a *spiking* state, so it moves
discontinuously while the car glides, and `step_a = lr_actor * delta`
multiplies that into the actor on every write. The actor's weight norm then
climbs monotonically in every arm that writes and is flat in the one that
does not, which is why `frozen` wins.

`readout_tau` would also quiet it, but it filters what the policy sees too:
that changes the controller and invalidates any warm start. This filters
the critic's view alone. The contract the tests below pin down is that the
policy cannot tell the difference -- same actions, same state trajectory,
bit for bit -- while delta gets quieter.

Because the read-out is linear (V = theta . h + b), filtering h is exactly a
low-pass on V, so the *level* of V survives and with it the -1 - V(s) that a
crash is worth. The jitter does not.
"""

import numpy as np
import pytest

from algos.rtrrl.algo import RTRRL
from rtrrl_playground.spaces import Discrete

NVEC = (21, 21)


def _spiking():
    """The spiking cells live in event-driven-rtrl, which this package does
    not depend on. The jitter this knob exists for is a spiking artefact, so
    the two tests that measure it ask for the real cell and skip without it;
    the contract tests run on any cell."""
    sp = pytest.importorskip("edrtrl.spiking")
    sp.register()
    return "spiking_lif"


def _agent(tau, seed=0, cell="ctrnn"):
    return RTRRL(obs_dim=8, action_space=Discrete(441, nvec=NVEC), n_hidden=24,
                 cell=cell, seed=seed, critic_tau=tau)


def _drive(ag, n=200, seed=0):
    """One episode of fixed inputs, collecting what each side of the agent saw."""
    rng = np.random.default_rng(seed)
    acts = [ag.start(rng.normal(0, 1, 8))]
    hs, deltas = [ag.h.copy()], []
    for t in range(n):
        a = ag.step(rng.normal(0, 1, 8), 0.3, False, False)
        acts.append(a)
        hs.append(ag.h.copy())
        deltas.append(ag.stats["delta"])
    return np.array(acts), np.array(hs), np.array(deltas)


def test_off_by_default_so_every_earlier_result_still_means_what_it_meant():
    ag = _agent(0.0)
    assert ag.critic_tau == 0.0
    _drive(ag, n=20)
    #: not merely equal -- the same array, so the critic's path is untouched
    assert ag._critic_h() is ag.h


def test_the_policy_cannot_tell_the_difference():
    """The whole safety claim: filtering the critic does not move the car.

    Same seed, same inputs, one with the filter and one without. The actions
    and the recurrent state must agree exactly, because the actor reads
    `self.h` and nothing in the filter's path touches it.
    """
    c = _spiking()
    a0, h0, _ = _drive(_agent(0.0, seed=3, cell=c), seed=3)
    a1, h1, _ = _drive(_agent(10.0, seed=3, cell=c), seed=3)
    assert np.array_equal(a0, a1)
    assert np.array_equal(h0, h1)


def test_it_quiets_the_td_error_it_was_added_for():
    """Same trajectory, quieter delta -- which is the point of the knob."""
    c = _spiking()
    _, _, d0 = _drive(_agent(0.0, seed=1, cell=c), seed=1)
    _, _, d1 = _drive(_agent(10.0, seed=1, cell=c), seed=1)
    assert d1.std() < 0.6 * d0.std()


def test_the_level_survives_so_a_crash_is_still_worth_what_it_was():
    """A filter on V must quiet V's *step* without moving V's *level*.

    The only signal in the trace is the terminal step, worth r - V(s), and
    that is the level. A filter that bought a quiet delta by shrinking V
    would have thrown away the one piece of information the run contains.

    Measured on one trajectory under one set of critic weights, so that what
    is compared is the filter and not two different learned critics.
    """
    ag = _agent(12.0, seed=5, cell=_spiking())
    rng = np.random.default_rng(5)
    ag.start(rng.normal(0, 1, 8))
    raw, filt = [], []
    for _ in range(400):
        ag.step(rng.normal(0, 1, 8), 0.3, False, False)
        raw.append(ag.critic.value(ag.h))
        filt.append(ag.critic.value(ag._h_c))
    raw, filt = np.asarray(raw), np.asarray(filt)
    # the level is kept: well inside one standard deviation of the raw signal
    assert abs(filt.mean() - raw.mean()) < 0.5 * raw.std()
    # and the jitter, which is what delta was made of, is largely gone
    assert np.diff(filt).std() < 0.5 * np.diff(raw).std()


def test_the_filter_clears_at_an_episode_boundary():
    """Carrying it across a reset would bootstrap the new episode from the old."""
    ag = _agent(10.0, seed=2)
    _drive(ag, n=50, seed=2)
    assert ag._h_c is not None
    ag.reset_state()
    assert ag._h_c is None
