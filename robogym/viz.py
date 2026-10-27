"""Figures and GIFs for the benchmark: rollout GIFs with captions (front + wrist camera, policy name,
subgoal, outcome), contact sheets of key frames, and the summary figures (horizon curves, first-failure
attribution bars, per-subgoal reliability). Colours are the validated categorical palette of the lab's
chart guidelines, assigned in fixed order; identity is also carried by marker shape / legend / labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]  # fixed order
MARKERS = "osD^vPX*"
INK, MUTED, GRID, SURFACE = "#1a1a19", "#8a8a85", "#e6e6e3", "#fcfcfb"
ATTR_COLORS = {"none": "#1baf7a", "plan": "#2a78d6", "plan_goal": "#6250d6", "exec_grasp": "#eb6834", "exec_place": "#e34948", "late": "#eda100", "unstable": "#8a8a85"}
ATTR_LABELS = {"none": "success", "plan": "plan: wrong object", "plan_goal": "plan: wrong goal", "exec_grasp": "exec: no grasp", "exec_place": "exec: not placed",
               "late": "late failure", "unstable": "sim unstable"}


def _style():
    import matplotlib

    matplotlib.rcParams.update({"text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": MUTED, "font.size": 10,
                                "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def _clean(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


# ---------------------------------------------------------------------- rollout GIFs
def caption_frame(img, lines, height=None, scale=1.0):
    """Stack a caption bar (one or two short lines of text) under an RGB frame."""
    from PIL import Image, ImageDraw, ImageFont

    im = Image.fromarray(np.ascontiguousarray(img))
    if scale != 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    h = height or (16 * len(lines) + 8)
    out = Image.new("RGB", (im.width, im.height + h), (26, 26, 25))
    out.paste(im, (0, 0))
    d = ImageDraw.Draw(out)
    for i, (text, color) in enumerate(lines):
        d.text((6, im.height + 4 + 16 * i), text, fill=color, font=font)
    return np.asarray(out)


def rollout_frames(env, policy, spec, cams=("front", "right_wrist"), stride=3, max_s=None, label=None):
    """Run one evaluation episode with recording and compose captioned mosaic frames (cams side by side).
    Returns (frames uint8 [T,H,W,3], row). Captions: policy, time, the subgoal in progress, outcome."""
    from . import lh

    row = lh.run_episode(env, policy, spec, cams=cams, max_s=max_s)
    fr = row.pop("frames")
    T = min(len(v) for v in fr.values())
    sub_t = json.loads(row["subgoal_t"])
    n_sub = row["horizon"]
    name = label or row["policy"]
    out = []
    for t in range(0, T, stride):
        tile = np.concatenate([fr[c][t] for c in cams], axis=1)
        k = sum(1 for x in sub_t if x is not None and x <= t * env.dt)
        prog = f"subgoal {min(k + 1, n_sub)}/{n_sub}" if k < n_sub else "all subgoals done"
        status = f"t = {t * env.dt:5.1f} s   {prog}"
        last = t >= T - stride
        res = (f"SUCCESS in {row['t_success']:.1f} s" if row["success"] else f"FAILED: {ATTR_LABELS.get(row['attribution'], row['attribution'])} (subgoal {(row['first_fail_k'] or 0) + 1})")
        lines = [(f"{name}  |  {env.task.name}  n={n_sub}", (240, 239, 236)), (status if not last else res, (240, 239, 236) if not last else ((27, 175, 122) if row["success"] else (227, 73, 72)))]
        out.append(caption_frame(tile, lines))
    if out:  # hold the final frame
        out += [out[-1]] * max(1, int(1.0 / (env.dt * stride)))
    return np.stack(out), row


def write_gif(frames, path, fps=7, scale=0.6, colors=64, max_frames=240, mp4=True):
    """Palette-quantised GIF (PIL), downscaled and capped in length (README-sized, a few MB), plus an mp4
    of the full-resolution frames next to it. Returns the GIF path and its size in MB."""
    from PIL import Image

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if mp4:
        from .tools import write_video

        write_video(frames, path.with_suffix(".mp4"), fps=int(round(fps)))
    step = max(1, int(np.ceil(len(frames) / max_frames)))
    ims = []
    for f in frames[::step]:
        im = Image.fromarray(f)
        if scale != 1.0:
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
        ims.append(im.quantize(colors=colors, method=Image.MEDIANCUT))
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=int(1000 / fps * step), loop=0, optimize=True)
    return path, round(path.stat().st_size / 1e6, 2)


def contact_sheet(frames, n=6, path=None):
    """n evenly spaced frames in one row (a static summary of a GIF)."""
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    sheet = np.concatenate([frames[i] for i in idx], axis=1)
    if path:
        from PIL import Image

        Image.fromarray(sheet).save(path)
    return sheet


# ---------------------------------------------------------------------- figures
def horizon_figure(curves: dict, families, path=None, title=None, n_seeds=None):
    """curves: {task: horizon_curve DataFrame}; families: [(label, matcher(policy name) -> bool)] in fixed
    colour order (<= 8). One panel per task, success vs number of subgoals, legend + marker shapes."""
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, len(curves), figsize=(5.0 * len(curves), 3.9), squeeze=False)
    for ax, (task, c) in zip(axes[0], curves.items()):
        for i, (label, match) in enumerate(families[:8]):
            g = c[c.policy.map(match)].sort_values("horizon")
            if len(g):
                ax.plot(g.horizon, g.success, color=SERIES[i], lw=2, marker=MARKERS[i], ms=6, label=label)
        ax.set(title=task.replace("_", " "), xlabel="subgoals in the episode", ylim=(-0.03, 1.03), xticks=sorted(c.horizon.unique()))
        ax.set_ylabel("success rate" if ax is axes[0][0] else "")
        _clean(ax)
    axes[0][-1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title or ("Long-horizon success per policy" + (f" ({n_seeds} paired test seeds per horizon)" if n_seeds else "")), color=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def attribution_figure(attr: dict, path=None, order=None, title="Where the first failure comes from"):
    """attr: {task: attribution crosstab (policies x attribution counts)}. Horizontal 100 % stacked bars,
    2 px surface gaps between segments, labels on segments >= 15 %."""
    import matplotlib.pyplot as plt

    _style()
    keys = [k for k in ATTR_COLORS if k != "unstable"] + ["unstable"]
    fig, axes = plt.subplots(len(attr), 1, figsize=(9, 0.42 * sum(len(t) for t in attr.values()) + 1.6 * len(attr)), squeeze=False)
    for ax, (task, t) in zip(axes[:, 0], attr.items()):
        t = t[[k for k in keys if k in t]].astype(float)
        if order:
            t = t.reindex([p for p in order if p in t.index] + [p for p in t.index if p not in order])
        frac = t.div(t.sum(1).replace(0, np.nan), axis=0).fillna(0)
        left = np.zeros(len(frac))
        y = np.arange(len(frac))[::-1]
        for k in frac.columns:
            v = frac[k].values
            ax.barh(y, v, left=left, color=ATTR_COLORS[k], edgecolor=SURFACE, linewidth=2, height=0.72, label=ATTR_LABELS[k])
            for yi, li, vi in zip(y, left, v):
                if vi >= 0.15:
                    ax.text(li + vi / 2, yi, f"{vi:.0%}", ha="center", va="center", fontsize=8, color="#fcfcfb" if k != "late" else INK)
            left = left + v
        ax.set(yticks=y, yticklabels=[str(p) for p in frac.index], xlim=(0, 1), title=task.replace("_", " "))
        ax.xaxis.set_major_formatter(lambda v, _p: f"{v:.0%}")
        ax.tick_params(axis="y", labelsize=8)
        _clean(ax)
        ax.grid(False, axis="y")
    axes[0, 0].legend(frameon=False, fontsize=8, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.08))
    fig.suptitle(title, color=INK, y=1.0)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def reliability_figure(rel: dict, families, path=None, title="Per-subgoal reliability  r_k = P(subgoal k done | earlier ones done)"):
    """rel: {task: {policy: {k: (r_k, n)}}} from robogym.lh.reliability."""
    import matplotlib.pyplot as plt

    _style()
    fig, axes = plt.subplots(1, len(rel), figsize=(5.0 * len(rel), 3.6), squeeze=False)
    for ax, (task, r) in zip(axes[0], rel.items()):
        fams = [(label, match) for label, match in families[:8] if any(match(p) for p in r)]
        w = 0.8 / max(1, len(fams))
        for i, (label, match) in enumerate(fams):
            pol = next(p for p in r if match(p))
            ks = sorted(r[pol])
            ax.bar(np.array(ks) + (i - len(fams) / 2 + 0.5) * w, [r[pol][k][0] for k in ks], width=w * 0.92, color=SERIES[families.index((label, match))], label=label)
        ax.set(title=task.replace("_", " "), xlabel="subgoal index k", ylim=(0, 1.03), xticks=sorted({k for p in r for k in r[p]}))
        ax.set_ylabel("r_k" if ax is axes[0][0] else "")
        _clean(ax)
        ax.grid(False, axis="x")
    axes[0][-1].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.suptitle(title, color=INK)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def gallery_html(items, path, title="Rollouts"):
    """A small static HTML page listing GIFs with captions (items: [(gif_path, caption)]), relative links."""
    path = Path(path)
    rows = "\n".join(f'<figure style="display:inline-block;margin:8px"><img src="{Path(p).relative_to(path.parent)}" loading="lazy"><figcaption>{c}</figcaption></figure>' for p, c in items)
    path.write_text(f"<!doctype html><meta charset='utf-8'><title>{title}</title><body style='font-family:sans-serif;background:#fcfcfb;color:#1a1a19'><h2>{title}</h2>{rows}</body>")
    return path
