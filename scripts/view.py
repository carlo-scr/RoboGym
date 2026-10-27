#!/usr/bin/env python3
"""Watch an expert / a policy, or replay a recorded episode from its simulator snapshots, with labels.

    python scripts/view.py --task lego --bimanual                       # interactive MuJoCo viewer, expert
    python scripts/view.py --episode data/cubes/ep_000003                # replay the recording (exact)
    python scripts/view.py --episode data/cubes/ep_000003 --video out.mp4 --cam front   # headless, with overlays
    python scripts/view.py --task cable_plug --policy runs/dp/policy.pt --video out.mp4

The interactive mode needs a display (on macOS run it with `mjpython`); `--video` renders the same
thing headless with a timeline overlay (stage, progress, wrench, regime, events).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogym.config import EnvCfg, make_cfg  # noqa: E402
from robogym.datagen import EpisodeSpec, build_episode, load_episode  # noqa: E402
from robogym.env import RoboGymEnv  # noqa: E402
from robogym.scenario import Scenario  # noqa: E402
from robogym.tools import write_video  # noqa: E402


def overlay(img, lines):
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    for i, ln in enumerate(lines):
        d.rectangle((0, 14 * i, 7 * len(ln) + 6, 14 * i + 14), fill=(0, 0, 0))
        d.text((3, 14 * i + 1), ln, fill=(255, 255, 255))
    return np.asarray(im)


def frame(env, cam, extra=()):
    info = env.info()
    w = env.wrench(env.arms[0])
    lines = [f"t={env.time:5.2f}s  stage={info['stage_name']}  progress={info['progress']:.2f}  success={info['success']}",
             f"wrench F=({w[0]:5.1f},{w[1]:5.1f},{w[2]:5.1f}) N   regime: " + ", ".join(f"{k}={v:.2g}" for k, v in env.regime.items() if v not in (1.0, 0, 0.0))]
    lines += list(extra)
    img = env.render_free(width=800, height=500) if cam == "free" else env.render(cam, width=640, height=480, ghost=True)
    return overlay(img, lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cube_stack")
    ap.add_argument("--bimanual", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode", default=None, help="recorded episode folder to replay from its snapshots")
    ap.add_argument("--policy", default=None, help="policy checkpoint instead of the expert")
    ap.add_argument("--shifts", nargs="*", default=[])
    ap.add_argument("--failure", default=None)
    ap.add_argument("--variant", default="{}", help='task variant JSON, e.g. \'{"n": 3, "mode": "handover"}\'')
    ap.add_argument("--video", default=None, help="write a headless video instead of opening the viewer")
    ap.add_argument("--cam", default="free")
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args()

    if a.episode:
        E = load_episode(a.episode)
        cfg = EnvCfg.from_dict(E["features"]["env_cfg"])
        env = RoboGymEnv(cfg)
        env.reset(Scenario.from_dict(E["annotations"]["scenario"]))
        ann = E["annotations"]
        events = {round(t, 2): n for n, t in ann.get("phase_events", [])}
        title = f"{ann['task']} seed {ann['spec']['seed']}  {ann['category']}  outcome={ann['outcome']} ({ann['failure_mode']})"

        def stepper():
            for k in range(len(E["actions64"])):
                env.set_state(dict(physics=E["physics"][k], aux=E["aux"][k]))
                yield [title, f"event: {events.get(round(env.time, 2), '')}"]
                env.step(E["actions64"][k])
    else:
        env = RoboGymEnv(make_cfg(a.task, bimanual=a.bimanual))
        spec = EpisodeSpec(seed=a.seed, category="ood" if a.shifts else ("known_failure" if a.failure else "nominal"), shifts=a.shifts, failure=a.failure, severity=0.8, variant=json.loads(a.variant))
        sc, failure, ex = build_episode(env, spec)
        actor = ex.act
        if a.policy:
            from robogym.learn.policy import PolicyRunner, load_policy

            runner = PolicyRunner(load_policy(a.policy), env, seed=a.seed)
            actor = runner.act
        title = f"{a.task} seed {a.seed} " + ("policy" if a.policy else "expert") + (f" failure={a.failure}" if a.failure else "") + (f" shifts={a.shifts}" if a.shifts else "")

        def stepper():
            while env.time < env.episode_max_s and not env.unstable and not (a.policy is None and ex.done):
                yield [title, f"phase: {ex.phase}" if a.policy is None else ""]
                env.step(actor())

    if a.video:
        frames = []
        stride = max(1, int(round(env.cfg.sim.control_hz / a.fps)))
        for i, lines in enumerate(stepper()):
            if i % stride == 0:
                frames.append(frame(env, a.cam, lines))
        write_video(frames, a.video, fps=a.fps)
        print(f"wrote {a.video} ({len(frames)} frames)")
        return
    import time

    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as v:
        for _lines in stepper():
            v.sync()
            time.sleep(env.dt)
            if not v.is_running():
                break
        print(json.dumps(env.info(), default=str))


if __name__ == "__main__":
    main()
