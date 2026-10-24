"""Load RoboGym episodes (see robogym.datagen) into arrays for learning.

Observation pieces: `state` (28 per arm, what the robot logs), `task` (privileged task features from the
parquet column `task.features`, a stand-in for perception in the low-dim policy), `frames` (uint8 camera
images decoded from the mp4s).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..env import STATE_PER_ARM


def proprio_idx(n_arms: int) -> list:
    """Per arm: TCP pos (3), quat (4), grip width (1) = state dims 14..22 of each 28-block."""
    return [STATE_PER_ARM * i + j for i in range(n_arms) for j in range(14, 22)]


def list_episodes(root) -> pd.DataFrame:
    root = Path(root)
    idx = [pd.read_parquet(p) for p in sorted(root.glob("index_*.parquet"))]
    if not idx:
        raise FileNotFoundError(f"no index_*.parquet in {root}")
    return pd.concat(idx, ignore_index=True).sort_values("seed").reset_index(drop=True)


def load_episode_arrays(path, cams=(), frames=True):
    """One episode folder -> dict(state [T,S], action [T,A], action_expert, ee_target [T,8n], task [T,F],
    progress [T], stage [T], frames {cam: [T,H,W,3]}, ann, feature_names)."""
    path = Path(path)
    df = pd.read_parquet(path / "episode.parquet")
    ann = json.load(open(path / "annotations.json"))
    fr = json.load(open(path / "feature_report.json"))
    out = dict(
        state=np.stack(df["observation.state"].values).astype(np.float32),
        action=np.stack(df["action"].values).astype(np.float32),
        action_expert=np.stack(df["action.expert"].values).astype(np.float32) if "action.expert" in df else None,
        ee_target=np.stack(df["action.ee_target"].values).astype(np.float32),
        task=np.stack(df["task.features"].values).astype(np.float32),
        progress=df["priv.progress"].values.astype(np.float32),
        stage=df["priv.stage"].values.astype(np.int64),
        ann=ann,
        feature_names=fr["task_feature_names"],
        n_arms=len(fr["env_cfg"]["robot"]["arms"]) if "arms" in fr["env_cfg"]["robot"] else fr["env_cfg"]["robot"]["n_arms"],
        frames={},
    )
    if frames:
        import imageio.v2 as imageio

        for c in cams:
            out["frames"][c] = np.stack(imageio.mimread(path / f"{c}.mp4", memtest=False))[: len(df)]
    return out


def try_load_episode(path, cams=(), frames=True, retries=2, wait_s=2.0):
    import time as _t

    err = None
    for _ in range(retries + 1):
        try:
            ep = load_episode_arrays(path, cams, frames)
            short = [c for c in cams if frames and len(ep["frames"][c]) < len(ep["state"])]
            if short:
                raise ValueError(f"video shorter than episode for {short}")
            return ep
        except Exception as e:  # noqa: BLE001
            err = e
            _t.sleep(wait_s)
    print(f"skipping unreadable episode {Path(path).name}: {type(err).__name__}: {str(err).splitlines()[0][:120]}", flush=True)
    return None


def load_many(paths, cams=(), frames=True, progress=True, workers=8):
    """Decode episodes with a thread pool (video decoding runs in ffmpeg subprocesses)."""
    from concurrent.futures import ThreadPoolExecutor

    eps = []
    with ThreadPoolExecutor(workers) as ex:
        for i, ep in enumerate(ex.map(lambda p: try_load_episode(p, cams, frames), paths)):
            if ep is not None:
                eps.append(ep)
            if progress and (i + 1) % 50 == 0:
                print(f"  loaded {i + 1}/{len(paths)}", flush=True)
    return eps


class Normalizer:
    """Per-dimension min/max scaling to [-1, 1] (constant dims are left centred). `robust` uses the
    0.5/99.5 percentiles instead, so rare large values don't squash the common range."""

    def __init__(self, lo=None, hi=None):
        self.lo, self.hi = lo, hi

    def fit(self, X, robust=False):
        X = np.asarray(X, np.float32).reshape(-1, X.shape[-1])
        if robust:
            self.lo, self.hi = np.percentile(X, 0.5, 0).astype(np.float32), np.percentile(X, 99.5, 0).astype(np.float32)
        else:
            self.lo, self.hi = X.min(0), X.max(0)
        return self

    def scale(self):
        return np.where(self.hi - self.lo > 1e-6, (self.hi - self.lo) / 2, 1.0).astype(np.float32)

    def mid(self):
        return ((self.hi + self.lo) / 2).astype(np.float32)

    def norm(self, X):
        return (X - self.mid()) / self.scale()

    def denorm(self, Xn):
        return Xn * self.scale() + self.mid()

    def state_dict(self):
        return dict(lo=self.lo.tolist(), hi=self.hi.tolist())

    @staticmethod
    def from_state(d):
        return Normalizer(np.asarray(d["lo"], np.float32), np.asarray(d["hi"], np.float32))


def split_by_category(idx: pd.DataFrame, fracs=(0.7, 0.15, 0.15), seed=0):
    """Nominal episodes -> train / cal / test; failures, OOD and regime shifts -> 50/50 train(failure-WM)/test.
    (The benchmark protocol in robogym.bench assigns splits by seed range instead; use that for papers.)"""
    rng = np.random.default_rng(seed)
    out = {}
    nom = idx[idx.category == "nominal"].sample(frac=1, random_state=int(rng.integers(1 << 30)))
    n = len(nom)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    out["nominal_train"], out["nominal_cal"], out["nominal_test"] = nom.iloc[:a], nom.iloc[a:b], nom.iloc[b:]
    for cat in ("known_failure", "ood", "regime_shift"):
        d = idx[idx.category == cat].sample(frac=1, random_state=int(rng.integers(1 << 30)))
        h = len(d) // 2
        out[f"{cat}_train"], out[f"{cat}_test"] = d.iloc[:h], d.iloc[h:]
    return out
