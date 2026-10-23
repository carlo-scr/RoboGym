#!/usr/bin/env python3
"""Build .ipynb notebooks from the percent-format sources in notebooks/src/ (diffable, runnable as scripts).

    python notebooks/build.py            # build all
    python notebooks/build.py --execute  # build and execute (embeds outputs)
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
BOOTSTRAP = os.path.join(HERE, "bootstrap_cell.py")


def split_cells(text):
    cells, cur, kind = [], [], "code"
    for line in text.split("\n"):
        if line.startswith("# %%"):
            if cur:
                cells.append((kind, cur))
            kind = "markdown" if "[markdown]" in line else "code"
            cur = []
        else:
            cur.append(line)
    if cur:
        cells.append((kind, cur))
    return cells


def strip_md(lines):
    return [ln[2:] if ln.startswith("# ") else ("" if ln.strip() == "#" else ln) for ln in lines]


def trim(lines):
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def build(src_path, out_path):
    cells = []
    for kind, lines in split_cells(open(src_path).read()):
        if kind == "code" and lines and lines[0].strip() == "# @bootstrap":
            lines = open(BOOTSTRAP).read().split("\n")
        lines = trim(strip_md(lines) if kind == "markdown" else lines)
        if not lines:
            continue
        src = [ln + "\n" for ln in lines]
        src[-1] = src[-1].rstrip("\n")
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell.update(outputs=[], execution_count=None)
        cells.append(cell)
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(out_path, "w"), indent=1)
    return out_path


def _kernel_env():
    share = os.path.join(sys.prefix, "share", "jupyter")
    kdir = os.path.join(share, "kernels", "python3")
    os.makedirs(kdir, exist_ok=True)
    json.dump({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Python 3 (robogym)", "language": "python"},
              open(os.path.join(kdir, "kernel.json"), "w"), indent=1)
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["JUPYTER_PATH"] = share + os.pathsep + env.get("JUPYTER_PATH", "")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith(".py") or (a.only and a.only not in fn):
            continue
        out = build(os.path.join(SRC, fn), os.path.join(HERE, fn[:-3] + ".ipynb"))
        print("built", os.path.relpath(out))
        if a.execute:
            subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace", f"--ExecutePreprocessor.timeout={a.timeout}", out],
                           check=True, cwd=HERE, env=_kernel_env())
            print("executed", os.path.relpath(out))


if __name__ == "__main__":
    main()
