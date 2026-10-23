"""Procedural textures (PNG), generated deterministically on first use into robogym/assets/_generated/
(gitignored) and referenced by absolute path from the MJCF."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "_generated"
# Textured fronts are thin visual planes. Plane local axes: x -> world -y, y -> world +z, normal -> -x
# (towards a robot that looks along +x), so texture-right = world -y, texture-up = world +z.
FRONT_MAT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)


def _font(size):
    for f in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def _cached(name: str, key: str, make) -> str:
    tag = hashlib.md5(key.encode()).hexdigest()[:8]
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    p = ASSET_DIR / f"{name}_{tag}.png"
    if not p.exists():
        make().save(p)
    return str(p)


def wood(seed=0, base=(168, 128, 86)) -> str:
    """Light wood with grain."""

    def make():
        rng = np.random.default_rng(seed)
        w = 512
        y = np.linspace(0, 1, w)[:, None]
        x = np.linspace(0, 1, w)[None]
        grain = np.sin(2 * np.pi * (14 * x + 1.5 * np.sin(6 * y + rng.uniform(0, 6)) + 0.4 * np.sin(31 * x))) * 0.5 + 0.5
        n = rng.normal(0, 1, (w, w))
        v = 0.78 + 0.16 * grain + 0.04 * n
        img = np.clip(np.asarray(base)[None, None] * v[..., None], 0, 255).astype(np.uint8)
        return Image.fromarray(img).filter(ImageFilter.GaussianBlur(0.5))

    return _cached("wood", f"{seed}-{base}", make)


def rubber_mat(color=(52, 56, 62), seed=1) -> str:
    """Dark anti-slip bench mat with a fine dot pattern."""

    def make():
        rng = np.random.default_rng(seed)
        w = 256
        a = np.ones((w, w, 3), np.float32) * np.asarray(color, np.float32) + rng.normal(0, 4, (w, w, 1))
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        for y in range(4, w, 16):
            for x in range(4 + (y // 16 % 2) * 8, w, 16):
                d.ellipse((x, y, x + 4, y + 4), fill=tuple(int(c * 1.25) for c in color))
        return im

    return _cached("rubber_mat", f"{color}-{seed}", make)


def lab_floor() -> str:
    """Grey epoxy floor with a faint tile grid."""

    def make():
        rng = np.random.default_rng(3)
        w = 512
        a = np.full((w, w, 3), 150, np.float32) + rng.normal(0, 3, (w, w, 1))
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, w - 1, w - 1), outline=(125, 126, 128), width=4)
        return im

    return _cached("lab_floor", "v1", make)


def pegboard() -> str:
    """Wall pegboard behind the bench."""

    def make():
        w, h = 512, 512
        im = Image.new("RGB", (w, h), (214, 210, 200))
        d = ImageDraw.Draw(im)
        for y in range(16, h, 32):
            for x in range(16, w, 32):
                d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(120, 118, 112))
        return im

    return _cached("pegboard", "v1", make)


def port_labels(cols: int, start: int, pitch_px: int = 64) -> str:
    """White label strip with port numbers centred over each column."""

    def make():
        w, h = cols * pitch_px, 40
        im = Image.new("RGB", (w, h), (236, 236, 230))
        d = ImageDraw.Draw(im)
        f = _font(26)
        for c in range(cols):
            s = str(start + c)
            tw = d.textlength(s, font=f)
            d.text((c * pitch_px + pitch_px / 2 - tw / 2, 6), s, fill=(20, 20, 20), font=f)
            if c % 6 == 0 and c:
                d.line((c * pitch_px, 4, c * pitch_px, h - 4), fill=(150, 150, 150), width=2)
        return im

    return _cached(f"labels_{cols}_{start}", f"{cols}-{start}-{pitch_px}", make)


def switch_front(ports: int = 24, seed: int = 0) -> str:
    """Desktop switch front: a row of RJ45 ports with link LEDs."""

    def make():
        rng = np.random.default_rng(seed)
        w, h = 1024, 160
        im = Image.new("RGB", (w, h), (52, 56, 62))
        d = ImageDraw.Draw(im)
        pw = 800 // ports
        for c in range(ports):
            x0, y0 = 40 + c * pw, 50
            d.rectangle((x0, y0, x0 + pw - 6, y0 + 60), fill=(8, 8, 10))
            if rng.random() < 0.7:
                d.rectangle((x0 + 2, y0 - 10, x0 + 8, y0 - 3), fill=(40, 220, 70))
        d.rectangle((880, 40, 990, 120), fill=(20, 20, 24), outline=(120, 120, 120), width=2)
        return im

    return _cached(f"switch_{ports}_{seed}", f"{ports}-{seed}", make)


def logo_plate(text="ROBOGYM", color=(240, 240, 240), bg=(40, 42, 48)) -> str:
    def make():
        im = Image.new("RGB", (512, 128), bg)
        d = ImageDraw.Draw(im)
        f = _font(64)
        tw = d.textlength(text, font=f)
        d.text(((512 - tw) / 2, 28), text, fill=color, font=f)
        return im

    return _cached("logo", f"{text}-{color}-{bg}", make)


def checker(c1=(230, 230, 230), c2=(40, 40, 40), n=8) -> str:
    def make():
        w = 256
        a = np.zeros((w, w, 3), np.uint8)
        s = w // n
        for i in range(n):
            for j in range(n):
                a[i * s : (i + 1) * s, j * s : (j + 1) * s] = c1 if (i + j) % 2 == 0 else c2
        return Image.fromarray(a)

    return _cached("checker", f"{c1}-{c2}-{n}", make)
