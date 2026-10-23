#!/usr/bin/env python3
"""Expert success rates per task / horizon / shift family, in parallel. Use it after changing physics or
expert parameters; the nominal rows should stay near 100 %.

    python scripts/benchmark_experts.py --tasks cube_stack lego cable_plug --n 40 --workers 8
    python scripts/benchmark_experts.py --tasks lego --n 20 --shifts slippery heavy_object arm_gain_drift --severity 0.7
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogym.config import make_cfg  # noqa: E402
from robogym.datagen import EpisodeSpec  # noqa: E402
from robogym.parallel import run_jobs  # noqa: E402


def job(env, j):
    from robogym.datagen import run_episode

    ep = run_episode(env, EpisodeSpec.from_dict(j["spec"]), cameras=[], save_states=False)
    a = ep["annotations"]
    return dict(task=a["task"], horizon=a["horizon"], cond=j["cond"], success=a["outcome"] == "success", mode=a["failure_mode"], retries=a["retries"], duration=a["duration_s"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["cube_stack", "lego", "cable_plug"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--shifts", nargs="*", default=[])
    ap.add_argument("--severity", type=float, default=0.7)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    import pandas as pd

    for task in a.tasks:
        cfg = make_cfg(task)
        jobs = [dict(spec=EpisodeSpec(seed=s).to_dict(), cond="nominal") for s in range(a.n)]
        for sh in a.shifts:
            jobs += [dict(spec=EpisodeSpec(seed=1000 + s, category="ood", shifts=[sh], shift_severity=a.severity).to_dict(), cond=sh) for s in range(a.n)]
        res = run_jobs(job, jobs, cfg, workers=a.workers, render=False, log=lambda *x: None)
        df = pd.DataFrame([r["result"] for r in res if r["ok"]])
        print(f"\n== {task} ==")
        print(df.groupby(["cond", "horizon"]).agg(n=("success", "size"), success=("success", "mean"), retries=("retries", "mean"), dur_s=("duration", "mean")).round(2).to_string())
        print("failure modes:", df[~df.success].groupby(["cond", "mode"]).size().to_dict())
        if any(not r["ok"] for r in res):
            print("errors:", [r["error"] for r in res if not r["ok"]][:3])


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
