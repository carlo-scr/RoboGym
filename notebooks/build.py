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

# Notebooks whose .ipynb is the executed copy (outputs worth keeping): not rebuilt from src; only the bootstrap
# cell is replaced when it changes. Rebuild from src explicitly with --force.
SPLICE_ONLY = {"03_learn_and_steer"}

# Pipeline notebooks: always built without outputs (a stale error output would look like a current failure).
NO_OUTPUTS = {"04_benchmark", "05_offtheshelf", "06_fast_a100"}

# Notebooks derived from another source: the same cells with a different title cell and a code cell inserted
# right before the bootstrap (environment variables the bootstrap reads, e.g. the branch and preset), so the
# pipeline is kept in one file.
VARIANTS = {
    "06_fast_a100": dict(
        src="05_offtheshelf.py",
        title=[
            "# Fast A100 run: off-the-shelf policies in about an hour",
            "Notebook 05 with the `fast` preset preselected: 300 demonstrations per task, ACT / Diffusion Policy / VQ-BeT for",
            "6k steps at batch 64 with the three trainings sharing the GPU, GPU-side evaluation on 20 paired seeds per horizon,",
            "two GIF seeds per policy. Rough A100 budget: demonstrations ~10 min, export ~5 min, training ~15 min per task,",
            "evaluation ~5 min per task, GIFs ~5 min; all three tasks in about an hour. Resumable: rerun after an interruption",
            "and finished stages are skipped. Numbers are indicative (short trainings); the `a100` preset in notebook 05 is the",
            "full run. Add `smolvla` to `ROBOGYM_LEROBOT_POLICIES` in the cell below to fine-tune SmolVLA too (hub download).",
        ],
        preamble=[
            "import os",
            'os.environ.setdefault("ROBOGYM_PRESET", "fast")',
            'os.environ.setdefault("ROBOGYM_BRANCH", "policies-benchmark")  # the branch with the policies / benchmark work; drop once merged into main',
            '# os.environ["ROBOGYM_TASKS"] = "cube_stack"                     # one task first',
            '# os.environ["ROBOGYM_LEROBOT_POLICIES"] = "act,diffusion,vqbet,smolvla"',
            '# os.environ["ROBOGYM_PARALLEL_TRAIN"] = "9"                     # all tasks x policies at once (needs CPU for the loaders)',
        ],
    ),
}


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


def _cell(kind, lines):
    src = [ln + "\n" for ln in lines]
    src[-1] = src[-1].rstrip("\n")
    cell = {"cell_type": kind, "metadata": {}, "source": src}
    if kind == "code":
        cell.update(outputs=[], execution_count=None)
    return cell


def _old_outputs(out_path):
    """Outputs of the existing notebook keyed by code-cell source, so a rebuild keeps embedded outputs for unchanged cells."""
    if not os.path.exists(out_path):
        return {}
    try:
        old = json.load(open(out_path))
    except (OSError, ValueError):
        return {}
    keep = {}
    for c in old.get("cells", []):
        if c.get("cell_type") == "code" and c.get("outputs"):
            keep.setdefault("".join(c["source"]), []).append((c["outputs"], c.get("execution_count")))
    return keep


def splice_bootstrap(out_path):
    """Replace the bootstrap cell of an existing notebook in place (everything else, outputs included, is kept)."""
    nb = json.load(open(out_path))
    boot = trim(open(BOOTSTRAP).read().split("\n"))
    hits = [c for c in nb["cells"] if c["cell_type"] == "code" and "DEFAULT_REPO" in "".join(c["source"])]
    assert len(hits) == 1, f"{out_path}: expected one bootstrap cell, found {len(hits)}"
    hits[0].update(_cell("code", boot))
    json.dump(nb, open(out_path, "w"), indent=1)
    return out_path


def build(src_path, out_path, title=None, preamble=None):
    """title: markdown lines replacing the first markdown cell; preamble: code lines inserted before the bootstrap cell.
    Embedded outputs of the existing notebook are carried over for code cells whose source did not change."""
    cells = []
    old_outputs = {} if os.path.splitext(os.path.basename(out_path))[0] in NO_OUTPUTS else _old_outputs(out_path)
    for kind, lines in split_cells(open(src_path).read()):
        is_bootstrap = kind == "code" and lines and lines[0].strip() == "# @bootstrap"
        if is_bootstrap:
            lines = open(BOOTSTRAP).read().split("\n")
        lines = trim(strip_md(lines) if kind == "markdown" else lines)
        if not lines:
            continue
        if title and kind == "markdown" and not any(c["cell_type"] == "markdown" for c in cells):
            lines = list(title)
        if is_bootstrap and preamble:
            cells.append(_cell("code", list(preamble)))
        cells.append(_cell(kind, lines))
    for c in cells:
        if c["cell_type"] == "code" and old_outputs.get("".join(c["source"])):
            c["outputs"], c["execution_count"] = old_outputs["".join(c["source"])].pop(0)
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
    ap.add_argument("--force", action="store_true", help="also rebuild the SPLICE_ONLY notebooks from their sources")
    a = ap.parse_args()
    for fn in sorted(os.listdir(SRC)):
        if not fn.endswith(".py") or (a.only and a.only not in fn):
            continue
        out = os.path.join(HERE, fn[:-3] + ".ipynb")
        if fn[:-3] in SPLICE_ONLY and os.path.exists(out) and not a.force:
            print("kept (bootstrap spliced)", os.path.relpath(splice_bootstrap(out)))
            continue
        out = build(os.path.join(SRC, fn), out)
        print("built", os.path.relpath(out))
        if a.execute:
            subprocess.run([sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace", f"--ExecutePreprocessor.timeout={a.timeout}", out],
                           check=True, cwd=HERE, env=_kernel_env())
            print("executed", os.path.relpath(out))
    for name, v in VARIANTS.items():
        if a.only and a.only not in name:
            continue
        out = build(os.path.join(SRC, v["src"]), os.path.join(HERE, name + ".ipynb"), title=v.get("title"), preamble=v.get("preamble"))
        print("built", os.path.relpath(out), "from", v["src"])


if __name__ == "__main__":
    main()
