#!/usr/bin/env python3
"""Stills and videos for the README: hero shots per task, all cameras, the OOD gallery, a failure
montage and one expert episode per task (2x2 mosaic). Output: media/ (gitignored).

    python scripts/render_gallery.py --out media
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robogym import shifts as S  # noqa: E402
from robogym.datagen import EpisodeSpec, build_episode  # noqa: E402
from robogym.env import RoboGymEnv  # noqa: E402
from robogym.scenario import sample_scenario  # noqa: E402
from robogym.tools import mosaic, run_until, write_video  # noqa: E402

TASKS = ("cube_stack", "lego", "cable_plug")
HERO = {"cube_stack": dict(lookat=(0.45, -0.05, 0.08), distance=1.0, azimuth=150, elevation=-22),
        "lego": dict(lookat=(0.46, 0.0, 0.06), distance=0.9, azimuth=160, elevation=-28),
        "cable_plug": dict(lookat=(0.55, 0.0, 0.08), distance=0.85, azimuth=215, elevation=-16)}


def label(img, text):
    from PIL import Image, ImageDraw

    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 8 + 7 * len(text), 18), fill=(0, 0, 0))
    d.text((4, 3), text, fill=(255, 255, 255))
    return np.asarray(im)


def main():
    import imageio.v2 as imageio

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="media")
    ap.add_argument("--video", action="store_true", help="also render one expert episode per task as mp4")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        env = RoboGymEnv(task)
        sc, _, ex = build_episode(env, EpisodeSpec(seed=3))
        run_until(env, ex, "carry", subgoal=0)
        for _ in range(8):
            env.step(ex.act())
        imageio.imwrite(out / f"hero_{task}.png", env.render_free(width=1280, height=800, **HERO[task]))
        cams = [label(env.render(c, width=480, height=360, ghost=True), c) for c in env.cam_ids]
        imageio.imwrite(out / f"cameras_{task}.png", np.concatenate([np.concatenate(cams[:2], 1), np.concatenate(cams[2:4], 1)], 0))
        # OOD gallery: wrist + front camera under each shift
        tiles = []
        for sh in S.ALL_SHIFTS + tuple(env.task.shift_types):
            env.reset(sample_scenario(env.cfg, 5, shifts=[sh], severity=0.8, task=env.task))
            ex2 = env.task.make_expert(env, seed=5)
            run_until(env, ex2, "carry", subgoal=0)
            tiles.append(label(np.concatenate([env.render(env.cfg.camera.record[0], width=320, height=240), env.render("front", width=320, height=240)], 1), sh))
        cols = 3
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[i : i + cols], 1) for i in range(0, len(tiles), cols)]
        imageio.imwrite(out / f"ood_gallery_{task}.png", np.concatenate(rows, 0))
        if a.video:
            sc, _, ex = build_episode(env, EpisodeSpec(seed=11))
            frames = []
            while not ex.done and env.time < env.episode_max_s and not env.unstable:
                env.step(ex.act())
                if env.step_count % 2 == 0:
                    frames.append(label(mosaic(env, ghost=True), f"{task}  t={env.time:4.1f}s  {ex.phase}  progress={env.info()['progress']:.2f}"))
            write_video(frames, out / f"expert_{task}.mp4", fps=10)
        print("rendered", task)
        env.close()


if __name__ == "__main__":
    main()
