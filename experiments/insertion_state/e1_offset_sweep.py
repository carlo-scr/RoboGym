"""E1: blind compliant insertions from the pre-insertion snapshot with controlled port-estimate errors.
Lateral error on a polar grid, axial error uniform in +-2 mm. Saves every attempt's time series, one
shard per scene (`<out>/scene_<seed>.pkl`); rerunning skips scenes that already have a shard."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import JACKS, Attempt, Probe, make_env, outcome, prepare  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "runs" / "insertion_state"
RADII_MM = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
N_ANG = 12
KEYS_VEC = ("cmd", "tcp", "F", "M", "tau", "tip_true", "Fh")
KEYS_SCALAR = ("t", "angle", "latched", "on_target", "f_plug_face", "f_plug_wedge", "f_plug_back", "f_plug_occ", "f_grip_occ", "f_grip_panel", "f_cable_panel")
PHASES = ("pre", "push", "hold", "deeper", "move", "retract", "reaim", "slide")  # E1 uses the first three


def offsets(seed, radii=RADII_MM):
    rng = np.random.default_rng([seed, 11])
    out = [(0.0, 0.0)]
    for r in radii:
        a0 = rng.uniform(0, 2 * np.pi / N_ANG)
        out += [(r * np.cos(a0 + 2 * np.pi * i / N_ANG), r * np.sin(a0 + 2 * np.pi * i / N_ANG)) for i in range(N_ANG)]
    ax = rng.uniform(-2.0, 2.0, len(out))
    return [np.array([x, y, z]) * 1e-3 for x, (y, z) in zip(ax, out)]


def pack(rows):
    d = {k: np.array([r[k] for r in rows], np.float32) for k in KEYS_VEC + KEYS_SCALAR}
    d["phase"] = np.array([PHASES.index(r["phase"]) for r in rows], np.int8)
    return d


def run_scene(job):
    seed, jack, out_dir, radii = job
    shard = Path(out_dir) / f"scene_{seed}.pkl"
    if shard.exists():
        return seed
    env = make_env(JACKS[jack])
    got = prepare(env, seed)
    res = []
    if got is not None:
        _ex, snap = got
        port = int(env.scenario.task["cables"][0]["port"])
        pb = Probe(env)
        for err in offsets(seed, radii):
            rows = Attempt(env, pb, snap, port, err=err).run()
            res.append(dict(scene=seed, port=port, jack=jack, err=err, outcome=outcome(rows, env.task.tc), unstable=bool(env.unstable), ts=pack(rows)))
            env.unstable = False
    tmp = shard.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(res, f)
    tmp.rename(shard)
    return seed


def load(path):
    """All attempts from a shard directory (or a single pickle from older runs)."""
    path = Path(path)
    if path.is_dir():
        res = []
        for f in sorted(path.glob("scene_*.pkl")):
            res += pickle.load(open(f, "rb"))
        return res
    return pickle.load(open(path, "rb"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, nargs="+", default=None)
    ap.add_argument("--n", type=int, default=36)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--jack", default="nominal", choices=sorted(JACKS))
    ap.add_argument("--radii", default=None, help="json list of radii in mm (default: the full grid)")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--name", default="e1")
    ap.add_argument("--out", default=None, help="shard directory (default runs/insertion_state/<name>)")
    args = ap.parse_args()
    scenes = args.scenes or list(range(args.seed0, args.seed0 + args.n))
    out_dir = Path(args.out) if args.out else OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    radii = tuple(json.loads(args.radii)) if args.radii else RADII_MM
    jobs = [(s, args.jack, str(out_dir), radii) for s in scenes]
    with Pool(args.workers) as p:
        for i, s in enumerate(p.imap_unordered(run_scene, jobs), 1):
            print(f"  scene {s} done ({i}/{len(jobs)})", flush=True)
    res = [a for a in load(out_dir) if a["scene"] in set(scenes)]
    oc = {}
    for a in res:
        oc[a["outcome"]] = oc.get(a["outcome"], 0) + 1
    print(f"{len(res)} attempts over {len({a['scene'] for a in res})} scenes; unstable {sum(a['unstable'] for a in res)}; outcomes {oc}")
    r = np.array([np.linalg.norm(a["err"][1:]) * 1e3 for a in res])
    seated = np.array([a["outcome"] == "seated" for a in res])
    for rr in (0.0,) + radii:
        sel = np.isclose(r, rr, atol=1e-3)
        print(f"  r={rr:4.1f} mm  seated {seated[sel].mean():.2f}  " + " ".join(f"{k}={np.mean([res[i]['outcome'] == k for i in np.where(sel)[0]]):.2f}" for k in ("in_channel", "chamfer", "face", "blocked")))


if __name__ == "__main__":
    main()
