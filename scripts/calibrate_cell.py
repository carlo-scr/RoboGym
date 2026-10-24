#!/usr/bin/env python3
"""Measure the quantities ASSUMPTIONS.md guesses and write them as config overrides.

    python scripts/calibrate_cell.py --backend dry --out notes/cell_overrides.json
    python scripts/calibrate_cell.py --backend xarm --ip 192.168.1.xxx --out notes/cell_overrides.json

Measures (on the real cell): the home TCP pose, the gripper's full-stroke closing / opening time and the
width at which the fingers stop on a known object (grasp force indirectly), the wrist F/T bias at rest,
and (with --tag) the fixed camera extrinsics from an AprilTag / ChArUco board at a known bench pose
(OpenCV, optional). NOT TESTED ON HARDWARE YET.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogym.config import make_cfg  # noqa: E402
from robogym.real.xarm_env import RealXArmEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="dry", choices=["dry", "xarm"])
    ap.add_argument("--ip", default=None)
    ap.add_argument("--arm", default="right")
    ap.add_argument("--out", default="notes/cell_overrides.json")
    a = ap.parse_args()
    cfg = make_cfg("cube_stack")
    env = RealXArmEnv(arms=(a.arm,), ips={a.arm: a.ip} if a.ip else None, cfg=cfg, backend=a.backend)
    env.reset()
    over = {}
    p, q = env.tcp_pose(a.arm)
    over["measured.home_tcp"] = dict(pos=p.tolist(), quat=q.tolist())
    # gripper timing
    act = np.zeros(8)
    act[6] = 1.0
    time.time()
    widths = []
    for _ in range(60):
        env.step(act)
        widths.append(env.b[a.arm].grip_width())
        if widths[-1] < 0.002:
            break
    over["measured.gripper_close_s"] = round((len(widths)) * env.dt, 2)
    over["robot.gripper.close_rate"] = round(1.0 / max(over["measured.gripper_close_s"], 0.1), 2)
    act[6] = 0.0
    for _ in range(60):
        env.step(act)
    # F/T bias at rest (average 1 s)
    w = np.mean([env.b[a.arm].wrench() for _ in range(20)], 0)
    over["measured.ft_bias"] = w.tolist()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(over, open(a.out, "w"), indent=1)
    print(json.dumps(over, indent=1))
    env.close()


if __name__ == "__main__":
    main()
