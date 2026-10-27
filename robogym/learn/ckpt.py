"""Atomic training checkpoints so an interrupted run (Colab disconnect, time budget) resumes where it
stopped: `save_ckpt` writes to a temporary file and renames, `load_ckpt` returns None when there is none."""
from __future__ import annotations

import os
from pathlib import Path

import torch


def save_ckpt(path, obj: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)
    return path


def load_ckpt(path, device="cpu"):
    path = Path(path)
    if not path.exists():
        return None
    return torch.load(path, map_location=device, weights_only=False)


def rng_state():
    import numpy as np

    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state())


def set_rng_state(st):
    import numpy as np

    if st:
        torch.set_rng_state(st["torch"])
        np.random.set_state(st["numpy"])
