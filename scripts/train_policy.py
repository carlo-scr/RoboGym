#!/usr/bin/env python3
"""Train a Diffusion Policy on a RoboGym dataset and evaluate it on the benchmark's test split.

    python scripts/train_policy.py --data data/cubes_train --task cube_stack --out runs/dp_cubes_state --steps 20000
    python scripts/train_policy.py --data data/cubes_train --task cube_stack --obs image --cams right_wrist front --out runs/dp_cubes_img
    python scripts/train_policy.py --data data/cubes_train --task cube_stack --arch act --out runs/act_cubes_state   # or --arch mlp

Writes policy.pt, train.json (loss curve) and eval.json (success / progress on `--eval-n` test seeds).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogym import bench  # noqa: E402
from robogym.config import make_cfg  # noqa: E402
from robogym.env import RoboGymEnv  # noqa: E402
from robogym.learn.data import list_episodes, load_many  # noqa: E402
from robogym.learn.policy import PolicyCfg, PolicyRunner, load_policy, rollout, save_policy, train_policy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", default="cube_stack")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="dp", choices=["dp", "act", "mlp"], help="Diffusion Policy | Action Chunking Transformer | chunked MLP")
    ap.add_argument("--obs", default="state", choices=["state", "image"])
    ap.add_argument("--action", default="rel", choices=["rel", "abs", "delta"])
    ap.add_argument("--cams", nargs="*", default=["right_wrist", "front"])
    ap.add_argument("--img-hw", nargs=2, type=int, default=[72, 96])
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dims", nargs="*", type=int, default=[64, 128, 256])
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--only-success", action="store_true", help="train on successful episodes only")
    ap.add_argument("--eval-n", type=int, default=20)
    ap.add_argument("--horizon", type=int, default=None, help="task horizon (variant n) for evaluation")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--bimanual", action="store_true")
    ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx = list_episodes(a.data)
    if a.only_success:
        idx = idx[idx.outcome == "success"]
    paths = idx.path.tolist()[: a.max_episodes]
    eps = load_many(paths, cams=a.cams if a.obs == "image" else (), frames=a.obs == "image")
    n_arms = eps[0]["n_arms"]
    cfg = PolicyCfg(arch=a.arch, obs_mode=a.obs, action_mode=a.action, n_arms=n_arms, task_dim=eps[0]["task"].shape[1], cams=tuple(a.cams), img_hw=tuple(a.img_hw), down_dims=tuple(a.dims))
    print(f"{len(eps)} episodes, {sum(len(e['state']) for e in eps)} steps, n_arms {n_arms}, task_dim {cfg.task_dim}")
    t0 = time.time()
    pol, hist = train_policy(eps, cfg, steps=a.steps, bs=a.bs, lr=a.lr, device=a.device)
    save_policy(pol, out / "policy.pt")
    json.dump(dict(cfg=cfg.__dict__, steps=a.steps, loss=hist[::max(1, len(hist) // 500)], train_s=time.time() - t0, n_episodes=len(eps)), open(out / "train.json", "w"), indent=1, default=str)
    # evaluation on the benchmark test split (same seeds for every method)
    env_cfg = make_cfg(a.task, bimanual=a.bimanual or n_arms == 2, **json.loads(a.cfg))
    env = RoboGymEnv(env_cfg, render=a.obs == "image")
    pol = load_policy(out / "policy.pt", device=a.device)
    runner = PolicyRunner(pol, env, device=a.device)
    rows = []
    for sp in bench.split_specs(a.task, "test", n=a.eval_n, horizon=a.horizon):
        from robogym.datagen import build_scenario

        sc = build_scenario(env, sp)
        r = rollout(env, runner, sc, seed=sp.seed)
        rows.append(dict(seed=sp.seed, success=r["success"], progress=r["progress"], duration=r["duration"]))
        print(f"  test seed {sp.seed}: success={r['success']} progress={r['progress']:.2f} t={r['duration']:.1f}s")
    succ = bench.rate([r["success"] for r in rows])
    res = bench.Results(name=out.name, task=a.task, method=f"{a.arch}_{a.obs}_{a.action}", config=dict(steps=a.steps, n_episodes=len(eps)))
    res.add(split="test", metric="success", **succ, progress=float(np.mean([r["progress"] for r in rows])))
    res.save(out / "eval.json")
    print(f"test success {succ['rate']:.2f} [{succ['lo']:.2f}, {succ['hi']:.2f}] (n={succ['n']}), mean progress {np.mean([r['progress'] for r in rows]):.2f}")


if __name__ == "__main__":
    main()
