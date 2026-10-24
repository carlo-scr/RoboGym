#!/usr/bin/env python3
"""RQ1 oracle ranking: from N base-policy states, sample K candidate chunks, execute every candidate from
the exact same simulator state and let the base policy finish. Reports how often candidates disagree in
outcome (steering headroom) and how well each scorer ranks them (AUROC / steering gain).

    python scripts/rq1_oracle.py --policy runs/dp_cubes_state/policy.pt --task cube_stack --n 20 --K 8
    python scripts/rq1_oracle.py --policy ... --wm runs/wm/wm.pt --ae runs/wm/ae.pt   # adds the WM scorers
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogym import bench  # noqa: E402
from robogym.config import make_cfg  # noqa: E402
from robogym.env import RoboGymEnv  # noqa: E402
from robogym.learn.policy import PolicyRunner, load_policy  # noqa: E402
from robogym.learn.steer import FeatureTracker, ProgressScorer, RandomScorer, WMScorer, capture_branches, rq1_metrics, rq1_ranking  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--task", default="cube_stack")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--replans", nargs="*", type=int, default=[2, 5, 8])
    ap.add_argument("--wm", default=None)
    ap.add_argument("--ae", default=None)
    ap.add_argument("--cams", nargs="*", default=["right_wrist", "front"])
    ap.add_argument("--horizon", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    pol = load_policy(a.policy, device=a.device)
    env = RoboGymEnv(make_cfg(a.task, bimanual=pol.cfg.n_arms == 2), render=a.wm is not None or pol.cfg.obs_mode == "image")
    runner = PolicyRunner(pol, env, device=a.device)
    tracker, scorers = None, [RandomScorer(0), ProgressScorer(env)]
    if a.wm:
        from robogym.learn.encoders import load_ae
        from robogym.learn.wm import load_wm

        wm = load_wm(a.wm, device=a.device)
        ae = load_ae(a.ae, device=a.device)
        tracker = FeatureTracker(ae, a.cams, n_arms=pol.cfg.n_arms, H=wm.H, hw=ae.hw)
        scorers += [WMScorer(wm, "alea", device=a.device), WMScorer(wm, "epi", device=a.device)]
    seeds = bench.seeds("shift", a.n)
    branches = capture_branches(env, runner, tracker, seeds, at_replans=tuple(a.replans))
    print(f"{len(branches)} branch states captured")
    succ, prog, scores = rq1_ranking(env, runner, tracker, scorers, branches, K=a.K)
    res, rows = rq1_metrics(succ, scores, prog)
    print(json.dumps(res, indent=1))
    for r in rows:
        print(r)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(summary=res, scorers=rows, succ=succ.tolist(), progress=prog.tolist(), scores={k: v.tolist() for k, v in scores.items()}), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
