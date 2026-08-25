"""Recurrent cells, and the online gradient estimators that drive them.

Four cells, chosen so that each one adds exactly one mechanism to the one
before it, and one gated discrete cell as the control:

=========  =====================================================  ==================
cell       what it adds                                           paper
=========  =====================================================  ==================
``ctrnn``  a learned but input-*independent* time constant        Funahashi 1993
``ltc``    the time constant becomes a function of the input      Hasani 2021
``lrcu``   the *capacitance* becomes one too                      Farsang 2024
``ligru``  gating without continuous time, as the control         Ravanelli 2018
=========  =====================================================  ==================

Any of them composes with any of the five estimators in ``cell.py``
(``rtrl``, ``uoro``, ``snap1``, ``rflo``, ``none``), which is the point of the
shared base class: the cell decides what the state does, the estimator decides
what you are allowed to know about how it got there, and the two are
independent choices.
"""

from __future__ import annotations

from rtrrl_playground.nets.cell import ESTIMATORS, OnlineCell
from rtrrl_playground.nets.ctrnn import CTRNN
from rtrrl_playground.nets.heads import CategoricalHead, GaussianHead, ValueHead
from rtrrl_playground.nets.ligru import LiGRU
from rtrrl_playground.nets.lrcu import LRCU
from rtrrl_playground.nets.ltc import LTC
from rtrrl_playground.nets.mlp import MLPCell

CELLS = {"ctrnn": CTRNN, "ltc": LTC, "lrcu": LRCU, "ligru": LiGRU, "mlp": MLPCell}


def make_cell(name: str, n_in: int, n_hidden: int, estimator: str = "rflo", **kw):
    """Build a cell by name. ``kw`` goes to that cell's own constructor."""
    if name not in CELLS:
        raise KeyError(f"unknown cell '{name}'. Known: {', '.join(CELLS)}")
    return CELLS[name](n_in, n_hidden, estimator=estimator, **kw)


__all__ = ["OnlineCell", "ESTIMATORS", "CELLS", "make_cell", "CTRNN", "LTC",
           "LRCU", "LiGRU", "MLPCell", "CategoricalHead", "GaussianHead", "ValueHead"]
