#!/usr/bin/env python3
"""Long-horizon evaluation of policies on the benchmark's test seeds, paired across horizons and policies.

    python scripts/eval_policies.py --task cube_stack --policies expert "expert:exec_noise=0.004" "hier:oracle+scripted" \
        "hier:corrupt(p_swap=0.5)+scripted" "hier:heuristic+scripted(exec_noise=0.003)" chunk:runs/dp_cubes/policy.pt \
        --horizons 2 3 4 --n 20 --out runs/lh_cubes --workers 8
    python scripts/eval_policies.py --task cube_stack --variant '{"mode": "relocate"}' --horizons 3 4 --policies "hier:oracle+scripted" --n 10
    python scripts/eval_policies.py --task cable_plug --policies "hier:llm+scripted" "hier:oracle+scripted" --horizons 1 2 --n 10 --workers 1

Policy specs: see robogym/policies/__init__.py. Writes episodes.parquet, summary.csv, horizon_curve.csv,
attribution.csv, plan_exec_split.csv and results.json (robogym.bench.Results) to --out; re-running resumes.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogym import lh  # noqa: E402
from robogym.config import make_cfg  # noqa: E402
from robogym.datagen import EpisodeSpec  # noqa: E402
from robogym.parallel import run_jobs  # noqa: E402

_POLICIES = {}


def job(env, j):
    from robogym.policies import make_policy

    key = (j["policy"], j.get("device", "cpu"))
    if key not in _POLICIES:
        try:
            import torch

            torch.set_num_threads(1)
        except ImportError:
            pass
        _POLICIES[key] = make_policy(j["policy"], device=j.get("device", "cpu"))
    row = lh.run_episode(env, _POLICIES[key], EpisodeSpec.from_dict(j["spec"]), max_s_scale=j.get("max_s_scale", 1.0))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cube_stack")
    ap.add_argument("--policies", nargs="+", default=["expert", "hier:oracle+scripted"])
    ap.add_argument("--horizons", nargs="*", type=int, default=None)
    ap.add_argument("--n", type=int, default=20, help="episodes per horizon (test-split seeds)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--variant", default="{}", help='extra task variant JSON, e.g. \'{"mode": "relocate"}\'')
    ap.add_argument("--shifts", nargs="*", default=[])
    ap.add_argument("--severity", type=float, default=0.5)
    ap.add_argument("--regime", default=None, help='online physics regime JSON applied from t=0, e.g. \'{"pad_friction": 0.3}\'')
    ap.add_argument("--max-s-scale", type=float, default=1.0, help="time limit as a multiple of the task's default")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bimanual", action="store_true")
    ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    cfg = make_cfg(a.task, bimanual=a.bimanual, **json.loads(a.cfg))
    specs = lh.make_specs(a.task, a.horizons, n=a.n, split=a.split, variant=json.loads(a.variant), shifts=a.shifts, severity=a.severity,
                          regime=json.loads(a.regime) if a.regime else None)
    jobs = [dict(policy=p, spec=sp.to_dict(), device=a.device, max_s_scale=a.max_s_scale) for p in a.policies for sp in specs]
    out = Path(a.out)
    render = any(("image" in p) or p.startswith("lerobot") for p in a.policies)
    res = run_jobs(job, jobs, cfg, out_dir=out / "_jobs", workers=a.workers, render=render, job_key=lambda j: f"{j['policy']}|{j['spec']['seed']}|{j['spec']['variant']}|{j['spec']['tag']}",
                   start_method="spawn" if a.device.startswith("cuda") else None)
    import pandas as pd

    rows = [r["result"] for r in res if r["ok"]]
    errs = [r for r in res if not r["ok"]]
    if errs:
        print(f"{len(errs)} failed jobs, e.g. {errs[0]['error']}")
    if not rows:
        return
    df = pd.DataFrame(rows)
    tables = lh.save(df, out, name=out.name, task=a.task)
    pd.set_option("display.width", 200)
    print("\n== success per policy / horizon ==")
    print(tables["summary"][["policy", "horizon", "n", "success", "lo", "hi", "progress", "n_done", "duration"]].round(2).to_string(index=False))
    print("\n== horizon curve: observed vs independent-subgoal prediction ==")
    print(tables["curve"][["policy", "horizon", "success", "predicted_independent", "compounding", "done_frac"]].round(2).to_string(index=False))
    print("\n== first-failure attribution (counts) ==")
    print(tables["attribution"].round(2).to_string())
    if len(tables["split"]):
        print("\n== plan / execution split (hierarchical policies) ==")
        print(tables["split"].round(2).to_string(index=False))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
