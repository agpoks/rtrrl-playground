"""A bound on how far the policy may move in one applied update.

``clip`` bounds the *per-tick* gradient. An event-driven run does not apply
per-tick gradients: it accumulates them while it waits and releases the sum
when the trigger fires, so the step the policy actually takes is as long as
the wait was busy -- unbounded in exactly the branch this project is about.

Measured on the car, the two interventions that moved the driving (the
action grid, and the ceiling on the policy's width) are the two that moved
the actor's own magnitude; six prices on what the car *did* with the policy
moved neither and changed nothing. This bounds the magnitude directly.
"""

import numpy as np
import pytest

from algos.rtrrl.algo import RTRRL
from rtrrl_playground.spaces import Discrete

NVEC = (21, 21)


def _agent(trust=0.0, head="ordinal", seed=0, **kw):
    return RTRRL(obs_dim=8, action_space=Discrete(441, nvec=NVEC), n_hidden=16,
                 cell="ctrnn", seed=seed, head=head,
                 actor_trust=trust, **kw)


def _snap(ag):
    out = [ag.actor.theta.copy(), ag.actor.bias.copy()]
    if hasattr(ag.actor, "log_sigma"):
        out.append(ag.actor.log_sigma.copy())
    return out


def _run(ag, n=400, seed=0):
    """Drive it on noise with rewards big enough to make real updates."""
    rng = np.random.default_rng(seed)
    ag.start(rng.normal(0, 1, 8))
    moves = []
    for i in range(n):
        before = _snap(ag)
        ag.step(rng.normal(0, 1, 8), float(rng.normal(1.0, 3.0)),
                False, (i + 1) % 97 == 0)
        after = _snap(ag)
        d = float(np.sqrt(sum(float(((b - a).ravel() @ (b - a).ravel()))
                              for a, b in zip(before, after))))
        if d > 0:
            moves.append(d)
        if (i + 1) % 97 == 0:
            ag.start(rng.normal(0, 1, 8))
    return np.array(moves)


def test_off_by_default_so_every_earlier_result_still_means_what_it_meant():
    ag = _agent()
    assert ag.actor_trust == 0.0
    assert ag.n_actor_capped == 0
    _run(ag)
    # the counters still run, because the radius for a future arm has to be
    # readable off the runs that did not have one
    assert ag.n_actor_writes > 0 and ag.n_actor_capped == 0


def test_a_radius_of_zero_is_bit_identical_to_no_radius():
    """The paired control has to be the same run, not a similar one."""
    a, b = _run(_agent(0.0)), _run(_agent(0.0))
    assert np.array_equal(a, b)
    ag0, ag1 = _agent(0.0), _agent(0.0)
    _run(ag0), _run(ag1)
    assert np.array_equal(ag0.actor.theta, ag1.actor.theta)


@pytest.mark.parametrize("head", ["ordinal", "softmax"])
def test_no_applied_step_is_longer_than_the_radius(head):
    r = 1e-4
    moves = _run(_agent(r, head=head))
    assert len(moves) > 0
    # the write is theta += d with lr folded in already, so the parameter
    # movement IS the update norm -- within float error of the radius
    assert moves.max() <= r * (1 + 1e-9)


def test_it_counts_how_often_it_actually_bound():
    """The sigma ceiling shipped with a diagnostic that did not measure it.

    A trust radius that never binds is not a null result about trust
    regions, it is a run without the knob in it -- so the arm reports the
    fraction of writes it shortened, and that number has to be real.
    """
    loose, tight = _agent(1e9), _agent(1e-9)
    _run(loose), _run(tight)
    assert loose.n_actor_capped == 0
    assert tight.n_actor_capped == tight.n_actor_writes > 0


def test_the_radius_shortens_without_turning_the_update():
    """One factor for every part, so only the length changes."""
    free, bound = _agent(0.0, seed=3), _agent(1e-5, seed=3)
    rng = np.random.default_rng(7)
    obs = [rng.normal(0, 1, 8) for _ in range(40)]
    rew = list(rng.normal(1.0, 3.0, 40))
    for ag in (free, bound):
        ag.start(obs[0])
    d_free = d_bound = None
    for i, (o, r) in enumerate(zip(obs[1:], rew[1:])):
        b_free, b_bound = _snap(free), _snap(bound)
        free.step(o, r, False, False)
        bound.step(o, r, False, False)
        f = np.concatenate([(x - y).ravel() for y, x in zip(b_free, _snap(free))])
        g = np.concatenate([(x - y).ravel() for y, x in zip(b_bound, _snap(bound))])
        if np.linalg.norm(f) > 1e-5 and np.linalg.norm(g) > 0:
            d_free, d_bound = f, g
            break
    assert d_free is not None, "no update large enough to be shortened"
    cos = float(d_free @ d_bound / (np.linalg.norm(d_free) * np.linalg.norm(d_bound)))
    assert cos > 1 - 1e-6
    assert np.linalg.norm(d_bound) < np.linalg.norm(d_free)


def test_the_deferred_branch_is_bounded_too():
    """The branch the bound exists for: one release, many ticks of gradient."""
    ag = _agent(1e-5, spike_threshold=5.0, trigger="evidence")
    moves = _run(ag, n=600)
    assert ag.n_gated_steps > 0, "this arm is not actually deferring"
    assert len(moves) > 0 and moves.max() <= 1e-5 * (1 + 1e-9)
