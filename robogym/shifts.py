"""Out-of-distribution shifts, grouped by family, each scaled by a severity in [0, 1].

Severity 0 is the edge of the nominal range; 1 is the most extreme setting we consider plausible.
Physics shifts write into `Scenario.regime` (so they can also be switched on mid-episode, see streams.py).
Task-specific shifts (e.g. dense_ports for cable plugging) are listed by each task in `Task.shift_types`
and handled in `Task.sample`.
"""
from __future__ import annotations

import numpy as np

from .config import EnvCfg
from .scenario import Scenario

SHIFT_GROUPS = {
    "lighting": ("lighting_dim", "lighting_harsh", "lighting_colored", "lighting_flicker"),
    "appearance": ("bench_color", "clutter", "object_color", "object_texture"),
    "sensor": ("camera_shift", "camera_noise", "motion_blur", "overexposure", "lens_occlusion"),
    "physics": ("slippery", "heavy_object", "arm_gain_drift", "weak_gripper", "action_delay", "sticky_bench", "obs_noise"),
}
ALL_SHIFTS = tuple(s for g in SHIFT_GROUPS.values() for s in g)
ALIASES = {"lighting": "lighting_dim", "distractors": "clutter", "slippery_gripper": "slippery"}
OOD_OBJECT_COLORS = ((0.95, 0.45, 0.65, 1), (0.1, 0.75, 0.85, 1), (0.5, 0.2, 0.7, 1), (1.0, 0.45, 0.05, 1), (0.08, 0.08, 0.09, 1))


def lerp(lo, hi, s):
    return float(lo + (hi - lo) * np.clip(s, 0, 1))


def group_of(shift: str) -> str:
    for g, names in SHIFT_GROUPS.items():
        if shift in names:
            return g
    return "task"


def apply_shifts(cfg: EnvCfg, sc: Scenario, rng: np.random.Generator, shifts, s: float):
    """Generic shifts (lighting / appearance / sensor / physics). Task shifts are handled by the task."""
    cams = cfg.camera.all_cameras(cfg.robot)
    if "lighting_dim" in shifts:
        k = lerp(0.5, 0.15, s)
        sc.lights = {n: (np.asarray(v) * k).tolist() for n, v in sc.lights.items()}
        sc.headlight = [k, 1.0, 1.0, 1.0]
    if "lighting_harsh" in shifts:
        sc.lights.update(key=[0.1] * 3, fill=[0.03] * 3, ceil_a=[0.05] * 3, ceil_b=[0.05] * 3, spot=[lerp(1.0, 2.0, s)] * 3)
        sc.headlight = [lerp(0.4, 0.2, s), 1.0, 1.0, 1.0]
    if "lighting_colored" in shifts:
        hue = np.asarray(rng.choice([[1.0, 0.25, 0.2], [0.3, 0.45, 1.0], [1.0, 0.7, 0.25], [0.4, 1.0, 0.5]]))
        hue = 1 + (hue - 1) * lerp(0.5, 1.0, s)
        sc.lights = {n: (np.asarray(v) * hue * 1.2).tolist() for n, v in sc.lights.items()}
        sc.headlight = [1.0, *[float(h) for h in hue]]
    if "lighting_flicker" in shifts:
        sc.flicker = [lerp(0.3, 0.85, s), float(rng.uniform(2.0, 8.0)), float(rng.uniform(0, 2 * np.pi))]
    if "bench_color" in shifts:
        sc.bench_rgba = [*[float(x) for x in rng.choice([[0.25, 0.3, 0.9], [0.9, 0.2, 0.2], [0.2, 0.2, 0.22], [0.3, 0.8, 0.3]])], 1.0]
        sc.mat_rgba = [*[float(x) for x in rng.choice([[0.9, 0.9, 0.9], [0.8, 0.3, 0.1], [0.3, 0.3, 0.9]])], 1.0]
    if "clutter" in shifts:
        n = int(round(lerp(2, cfg.bench.n_distractors, s)))
        sc.distractors = [distractor_item(cfg, rng, i, [], 0.0) for i in range(n)]
    if "camera_shift" in shifts:
        for c in cams:
            dp, dr = rng.normal(0, 1, 3), rng.normal(0, 1, 3)
            sc.cam_jitter[c] = np.concatenate([dp / np.linalg.norm(dp) * lerp(0.006, 0.02, s), dr / np.linalg.norm(dr) * lerp(0.06, 0.2, s)]).tolist()
    if "camera_noise" in shifts:
        sc.image_fx.update(noise=lerp(6, 24, s), salt=lerp(0.001, 0.012, s))
    if "motion_blur" in shifts:
        sc.image_fx.update(blur=int(round(lerp(4, 13, s))), blur_angle=float(rng.uniform(0, np.pi)))
    if "overexposure" in shifts:
        sc.image_fx.update(gain=lerp(1.5, 2.6, s), gamma=lerp(0.85, 0.55, s))
    if "lens_occlusion" in shifts:
        sc.image_fx.update(occlusion=[float(rng.uniform(0.25, 0.75)), float(rng.uniform(0.25, 0.75)), lerp(0.2, 0.42, s)])
    # physics -> regime
    if "slippery" in shifts:
        sc.regime["pad_friction"] = lerp(0.5, 0.15, s)
    if "heavy_object" in shifts:
        sc.regime["object_mass"] = lerp(2.0, 5.0, s)
    if "arm_gain_drift" in shifts:
        sc.regime["arm_gain"] = lerp(0.6, 0.25, s)
    if "weak_gripper" in shifts:
        sc.regime["grip_force"] = lerp(0.4, 0.12, s)
    if "action_delay" in shifts:
        sc.regime["action_delay"] = int(round(lerp(1, 5, s)))
    if "sticky_bench" in shifts:
        sc.regime["bench_friction"] = lerp(1.8, 4.0, s)
    if "obs_noise" in shifts:
        sc.regime["obs_noise"] = lerp(0.002, 0.02, s)
    return sc


def distractor_item(cfg: EnvCfg, rng, i, avoid_xy, min_d):
    """Distractor placement on the bench, at least `min_d` from the points in avoid_xy (n, 2)."""
    b = cfg.bench
    avoid = np.asarray(avoid_xy, float).reshape(-1, 2)
    xy = None
    for _ in range(300):
        xy = np.array([rng.uniform(0.22, b.x_max - 0.12), rng.uniform(-0.42, 0.42)])
        if len(avoid) == 0 or np.min(np.linalg.norm(avoid - xy, axis=1)) > min_d + 0.06:
            break
    return [int(i), float(xy[0]), float(xy[1]), float(rng.uniform(-np.pi, np.pi))]


def relocate_distractors(cfg: EnvCfg, rng, sc: Scenario, task):
    """Re-sample distractor positions away from the task objects (which are sampled after them)."""
    avoid = task.footprint_xy(sc.task) if hasattr(task, "footprint_xy") else []
    return [distractor_item(cfg, rng, d[0], avoid, 0.06) for d in sc.distractors]


def describe(shift: str) -> str:
    return {
        "lighting_dim": "all lights dimmed", "lighting_harsh": "single harsh spotlight", "lighting_colored": "strongly tinted light",
        "lighting_flicker": "fluorescent flicker", "bench_color": "bench / mat recoloured", "clutter": "extra loose objects on the bench",
        "object_color": "task objects in unseen colours", "object_texture": "task objects with unseen texture / gloss",
        "camera_shift": "all cameras displaced and rotated", "camera_noise": "sensor noise + salt", "motion_blur": "directional blur",
        "overexposure": "gain + gamma", "lens_occlusion": "smudge on the lens", "slippery": "low finger-pad friction",
        "heavy_object": "object mass x2-5", "arm_gain_drift": "arm servo gains reduced", "weak_gripper": "gripper force reduced",
        "action_delay": "1-5 control steps of actuation latency", "sticky_bench": "high bench friction", "obs_noise": "proprioception noise",
    }.get(shift, shift)
