"""Notebook / script helpers: image grids, inline video, expert replay to a step, chunk execution."""
from __future__ import annotations

import base64
import io

import numpy as np

from .env import RoboGymEnv


def image_grid(images, titles=None, cols=4, scale=3.2):
    import matplotlib.pyplot as plt

    n = len(images)
    rows = int(np.ceil(n / cols))
    h, w = images[0].shape[:2]
    fig, axes = plt.subplots(rows, cols, figsize=(cols * scale, rows * scale * h / w + 0.3 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            if titles:
                ax.set_title(titles[i], fontsize=9)
    fig.tight_layout()
    return fig


def video_html(frames, fps=20, width=720):
    """Inline mp4 (base64) for notebooks."""
    import imageio.v2 as imageio
    from IPython.display import HTML

    buf = io.BytesIO()
    w = imageio.get_writer(buf, format="mp4", fps=fps, codec="libx264", quality=7, macro_block_size=8, pixelformat="yuv420p")
    for f in frames:
        w.append_data(f)
    w.close()
    b64 = base64.b64encode(buf.getvalue()).decode()
    return HTML(f'<video width="{width}" controls loop autoplay muted><source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>')


def write_video(frames, path, fps=20):
    import imageio.v2 as imageio

    w = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
    for f in frames:
        w.append_data(f)
    w.close()
    return path


def mosaic(env: RoboGymEnv, cams=None, free=True, ghost=False):
    """2x2 mosaic of camera views (+ a free third-person view) at the current state."""
    cams = list(cams or env.cam_ids)[:3 if free else 4]
    imgs = [env.render(c, ghost=ghost) for c in cams]
    if free:
        imgs.append(env.render_free(width=env.cfg.camera.width, height=env.cfg.camera.height, ghost=ghost))
    while len(imgs) < 4:
        imgs.append(np.zeros_like(imgs[0]))
    return np.concatenate([np.concatenate(imgs[:2], 1), np.concatenate(imgs[2:4], 1)], 0)


def run_until(env: RoboGymEnv, ex, phase: str, subgoal: int | None = None, max_s=None):
    """Step the expert until it *enters* `phase` (of `subgoal`, if given): a snapshot point for branching."""
    max_s = max_s or env.episode_max_s
    while not (ex.phase == phase and (subgoal is None or ex.subgoal == subgoal)) and not ex.done and env.time < max_s:
        env.step(ex.act())
    return ex.phase == phase


def finish(env: RoboGymEnv, ex, max_s=None, callback=None):
    """Run the expert to the end (or time-out / instability); returns success."""
    max_s = max_s or env.episode_max_s
    while not ex.done and env.time < max_s and not env.unstable:
        a = ex.act()
        env.step(a)
        if callback:
            callback(env, ex, a)
    return env.info()["success"]


def expert_at(env: RoboGymEnv, spec, step: int):
    """Rebuild an episode deterministically up to `step` (env + expert state) to branch from a recording."""
    from .datagen import build_episode

    _, _, ex = build_episode(env, spec)
    for _ in range(step):
        env.step(ex.act())
    return ex


def chunk_then_expert(env: RoboGymEnv, ex, chunk, max_s=None):
    """Execute an action chunk open-loop (the expert's schedule keeps running underneath, its actions are
    overridden), then hand control back to the expert. Returns success."""
    for a in chunk:
        ex.act()
        env.step(a)
        if ex.done:
            break
    return finish(env, ex, max_s=max_s)


def run_live(cmd, cwd=None, skip=("WARNING",), sink=None):
    """Run a script and stream its output into the notebook (raise with the tail of the log on failure).
    `sink`: a file path; the output then goes there (line-buffered) instead of the notebook."""
    import subprocess

    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd)
    tail = []
    if sink:
        from pathlib import Path

        Path(sink).parent.mkdir(parents=True, exist_ok=True)
    f = open(sink, "a", buffering=1) if sink else None
    try:
        for line in p.stdout:
            if any(k in line for k in skip):
                continue
            if f:
                f.write(line)
            else:
                print(line, end="", flush=True)
            tail = (tail + [line])[-40:]
    finally:
        if f:
            f.close()
    rc = p.wait()
    if rc:
        raise RuntimeError(f"command failed (exit {rc}); last lines:\n" + "".join(tail))
