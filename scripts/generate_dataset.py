#!/usr/bin/env python3
"""Generate a labelled episode dataset in parallel.

    python scripts/generate_dataset.py --task cube_stack --out data/cubes_v0 --n 200 --workers 8
    python scripts/generate_dataset.py --task cable_plug --out data/cables --n 2000 --cameras right_wrist front --dart 0.002
    python scripts/generate_dataset.py --task lego --out data/lego_streams --stream mixed --n 40
    MUJOCO_GL=egl python scripts/generate_dataset.py --task lego --out /scratch/lego --n 20000 --workers 90 --cfg '{"camera.width": 224, "camera.height": 224}'

Writes ep_XXXXXX/ folders (see robogym/datagen.py) plus index_<shard>.parquet summaries; re-running resumes.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogym.config import make_cfg  # noqa: E402
from robogym.datagen import CATEGORIES, make_plan  # noqa: E402
from robogym.parallel import episode_job, run_jobs  # noqa: E402
from robogym.streams import make_stream  # noqa: E402
from robogym.tasks.base import get_task_class  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cube_stack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--cameras", nargs="*", default=None, help="cameras to record (default: cfg.camera.record); pass none for no video")
    ap.add_argument("--mix", default="0.6,0.2,0.15,0.05", help="probabilities for " + ",".join(CATEGORIES))
    ap.add_argument("--variants", default=None, help='JSON list of variant dicts to cycle, e.g. \'[{"n":2},{"n":3},{"n":4}]\'')
    ap.add_argument("--dart", type=float, default=0.0, help="DART noise std (m/step) in free-space phases, label = clean expert action")
    ap.add_argument("--stream", default=None, help="TTA protocol instead of a category mix: sudden|gradual|recurring|mixed|in_episode")
    ap.add_argument("--cfg", default="{}", help="JSON dotted EnvCfg overrides")
    ap.add_argument("--no-states", action="store_true", help="skip sim_states.npz (saves ~60%% disk)")
    ap.add_argument("--budget-min", type=float, default=None)
    a = ap.parse_args()

    cfg = make_cfg(a.task, **json.loads(a.cfg))
    task = get_task_class(a.task)(cfg)
    if a.stream:
        plan = make_stream(a.stream, a.n, seed0=a.seed0, dart=a.dart)
    else:
        mix = tuple(zip(CATEGORIES, [float(x) for x in a.mix.split(",")]))
        variants = json.loads(a.variants) if a.variants else None
        plan = make_plan(task, a.n, seed0=a.seed0, mix=mix, variants=variants, dart=a.dart)
    cams = None if a.cameras is None else ([] if a.cameras == ["none"] else a.cameras)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(dict(task=a.task, cfg=cfg.to_dict(), n=a.n, seed0=a.seed0, stream=a.stream, dart=a.dart), open(out / "dataset.json", "w"), indent=1, default=str)
    jobs = [dict(spec=sp.to_dict(), out=str(out / f"ep_{sp.seed:06d}"), cameras=cams, save_states=not a.no_states) for sp in plan]
    res = run_jobs(episode_job, jobs, cfg, out_dir=out / "_jobs", workers=a.workers, render=cams != [], budget_s=None if a.budget_min is None else a.budget_min * 60,
                   job_key=lambda j: j["out"])
    import pandas as pd

    rows = [r["result"] for r in res if r["ok"]]
    if rows:
        df = pd.DataFrame(rows)
        df.to_parquet(out / f"index_{a.seed0:06d}.parquet")
        print(df.groupby(["category", "outcome"]).size().to_string())
        print(df.groupby("failure_mode").size().to_string())
    print(f"{len(rows)} episodes in {out}")


if __name__ == "__main__":
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
