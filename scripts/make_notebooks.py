"""Turn the ``# %%`` cell markers in the tutorial scripts into ``.ipynb`` files.

Every lesson in ``tutorial/`` is a plain Python file you can run with
``python``, marked up with jupytext's cell syntax::

    # %% [markdown]
    # Prose, one comment line per line of markdown.

    # %%
    code

This script converts those to notebooks so the same lesson can be read either
way, with no second copy to keep in sync. Deliberately not a dependency on
jupytext: it is forty lines, and a tutorial repo asking you to install a tool
before you can read it has already lost.

    python scripts/make_notebooks.py          # convert every tutorial/*.py
    python scripts/make_notebooks.py --check  # fail if any notebook is stale
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def split_cells(text: str) -> list[tuple[str, list[str]]]:
    cells: list[tuple[str, list[str]]] = []
    kind, buf = "code", []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# %%"):
            if buf:
                cells.append((kind, buf))
            kind = "markdown" if "[markdown]" in stripped else "code"
            buf = []
            continue
        buf.append(line)
    if buf:
        cells.append((kind, buf))
    return cells


def to_notebook(text: str) -> dict:
    nb_cells = []
    for kind, lines in split_cells(text):
        if kind == "markdown":
            body = [ln[2:] if ln.startswith("# ") else ln.lstrip("#") for ln in lines]
        else:
            body = lines
        while body and not body[0].strip():
            body.pop(0)
        while body and not body[-1].strip():
            body.pop()
        if not body:
            continue
        src = [ln + "\n" for ln in body[:-1]] + [body[-1]]
        nb_cells.append({"cell_type": kind, "metadata": {}, "source": src}
                        | ({"outputs": [], "execution_count": None} if kind == "code" else {}))
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="only report staleness, write nothing")
    args = ap.parse_args()
    stale = []
    for src in sorted((ROOT / "tutorial").glob("*.py")):
        nb_path = src.with_suffix(".ipynb")
        nb = json.dumps(to_notebook(src.read_text()), indent=1) + "\n"
        if args.check:
            if not nb_path.exists() or nb_path.read_text() != nb:
                stale.append(nb_path.name)
            continue
        nb_path.write_text(nb)
        print(f"wrote {nb_path.relative_to(ROOT)}")
    if args.check:
        print("stale:" + (" " + ", ".join(stale) if stale else " none"))
        raise SystemExit(1 if stale else 0)


if __name__ == "__main__":
    main()
