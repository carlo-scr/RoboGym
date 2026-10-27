#!/usr/bin/env python3
"""Build insertion_state_a100.ipynb: a self-contained notebook for a Colab A100 runtime (VS Code client or
browser). The experiment modules are inlined (they are not on the public RoboGym repo), so the VM only needs
the RoboGym clone that the standard bootstrap cell makes. Rebuild after editing any module:

    python experiments/insertion_state/make_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULES = ["common.py", "e1_offset_sweep.py", "analyze_e1.py", "e2_guided.py", "belief.py", "seq_model.py", "e3_recurring.py"]
SRC = HERE / "notebook_src.py"
OUT = HERE / "insertion_state_a100.ipynb"


def cell(kind, text):
    lines = text.strip("\n").split("\n")
    src = [ln + "\n" for ln in lines]
    src[-1] = src[-1].rstrip("\n")
    c = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


def split_cells(text):
    cells, cur, kind = [], [], "code"
    for line in text.split("\n"):
        if line.startswith("# %%"):
            if any(s.strip() for s in cur):
                cells.append((kind, cur))
            kind = "markdown" if "[markdown]" in line else "code"
            cur = []
        else:
            cur.append(line)
    if any(s.strip() for s in cur):
        cells.append((kind, cur))
    return cells


def main():
    cells = []
    for kind, lines in split_cells(SRC.read_text()):
        if kind == "markdown":
            lines = [ln[2:] if ln.startswith("# ") else ("" if ln.strip() == "#" else ln) for ln in lines]
        text = "\n".join(lines)
        if text.strip() == "# @bootstrap":
            text = (ROOT / "notebooks" / "bootstrap_cell.py").read_text()
        elif text.strip() == "# @modules":
            for name in MODULES:
                body = (HERE / name).read_text()
                assert "'''" not in body and not body.rstrip().endswith("\\"), name
                cells.append(cell("code", f"write_module({name!r}, r'''{body}''')"))
            continue
        cells.append(cell(kind, text))
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:02d}"
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
