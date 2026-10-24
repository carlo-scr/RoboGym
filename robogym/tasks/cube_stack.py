"""Cube stacking: stack n cubes (2..n_cubes) in a given order on top of the bottom cube where it lies.

Variants: "by_size" (largest at the bottom; the goal is readable from the scene) and "pattern" (random
order, shown as a translucent ghost of the finished stack). Long horizon = n cubes.
Stages per cube k >= 1 in the order: reach -> grasp -> carry -> place; progress counts seated cubes with
partial credit for the one in flight.
"""
from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..scene import PARK
from .base import Task, register

PALETTE = {
    "red": (0.85, 0.15, 0.12, 1), "green": (0.2, 0.65, 0.3, 1), "blue": (0.15, 0.35, 0.85, 1), "yellow": (0.95, 0.8, 0.15, 1),
    "white": (0.92, 0.92, 0.9, 1), "grey": (0.55, 0.57, 0.6, 1),
}
NOMINAL_COLORS = ("red", "green", "blue", "yellow", "white", "grey")
SUB = ("reach", "grasp", "carry", "place")


@register
class CubeStack(Task):
    name = "cube_stack"
    failure_types = ("misplace", "wrong_cube", "low_carry")
    shift_types = ("cube_size", "object_color", "object_texture")
    outcome_modes = ("none", "dropped", "toppled", "misplaced", "grasp_missed", "wrong_order", "incomplete", "sim_unstable", "other")

    # ---- model
    def xml(self):
        tc = self.tc
        out, ghosts = [], []
        for i, s in enumerate(tc.sizes[: tc.n_cubes]):
            h = s / 2
            m = tc.density * s**3
            out.append(f'<body name="cube{i}" pos="{G.fmt(PARK + (0.15 * i, -0.3, h + 0.01))}"><freejoint name="cube{i}"/>'
                       f'<geom name="cube{i}" class="cube" size="{h} {h} {h}" mass="{m:.4f}" rgba="0.8 0.2 0.2 1"/>'
                       f'<geom name="cube{i}_edge" class="visual" type="box" size="{h * 0.995} {h * 0.995} {h * 0.995}" rgba="0 0 0 0"/>'
                       f'<site name="cube{i}_top" pos="0 0 {h}" size="0.002" group="4"/></body>')
            ghosts.append(f'<body name="ghost{i}" pos="{G.fmt(PARK + (0.15 * i, -0.6, h))}">'
                          f'<geom name="ghost{i}" class="ghost" type="box" size="{h} {h} {h}" rgba="0.3 0.9 0.3 0.25"/></body>')
        defaults = f'<default class="cube"><geom type="box" contype="1" conaffinity="7" condim="4" friction="{tc.friction} 0.005 0.0005" solimp="0.95 0.99 0.001" solref="0.004 1"/></default>'
        return "\n".join(out + ghosts), [], "", "", defaults

    def bind(self, env):
        m = env.model
        n = self.tc.n_cubes
        self.bodies = [m.body(f"cube{i}").id for i in range(n)]
        self.geoms = [m.geom(f"cube{i}").id for i in range(n)]
        self.geom_sets = [{g} for g in self.geoms]
        self.qadr = [m.jnt_qposadr[m.joint(f"cube{i}").id] for i in range(n)]
        self.dadr = [m.jnt_dofadr[m.joint(f"cube{i}").id] for i in range(n)]
        self.ghosts = [m.body(f"ghost{i}").id for i in range(n)]
        self.ghost_geoms = [m.geom(f"ghost{i}").id for i in range(n)]

    # ---- scenario
    def sample(self, rng, shifts, variant=None):
        tc = self.tc
        variant = variant or {}
        n = int(variant.get("n", rng.integers(2, tc.n_cubes + 1)))
        n = max(2, min(n, tc.n_cubes))
        sizes = np.array(tc.sizes[:n], float)
        if "cube_size" in shifts:
            sizes = sizes * float(rng.uniform(0.7, 0.85) if rng.random() < 0.5 else rng.uniform(1.15, 1.3))
        order = list(range(n)) if tc.variant == "by_size" else [int(k) for k in rng.permutation(n)]
        if "order" in variant:
            order = [int(k) for k in variant["order"]]
        cols = list(rng.choice(NOMINAL_COLORS, size=n, replace=False))
        rgba = [list(PALETTE[c]) for c in cols]
        if "object_color" in shifts:
            from ..shifts import OOD_OBJECT_COLORS

            rgba = [list(OOD_OBJECT_COLORS[int(k)]) for k in rng.choice(len(OOD_OBJECT_COLORS), size=n, replace=False)]
        gloss = float(rng.uniform(0.6, 1.0)) if "object_texture" in shifts else 0.0
        poses = self._sample_poses(rng, sizes, base=order[0])
        return dict(n=n, order=order, sizes=sizes.tolist(), colors=cols, rgba=rgba, gloss=gloss, poses=poses, variant=tc.variant)

    def _sample_poses(self, rng, sizes, base=0):
        """Cubes spread over the spawn area; the others stay >= 0.16 m from the base cube so the open
        gripper (9 cm) never sweeps the growing stack while reaching for them. With two arms the base cube
        sits near the centre line (both arms reach the stack) and the others spread over both sides."""
        x0, x1, y0, y1 = self.tc.spawn
        two = self.cfg.robot.n_arms == 2
        if two:
            x0, x1, y0, y1 = 0.26, 0.48, -0.34, 0.34
        for _ in range(2000):
            P = np.stack([rng.uniform(x0, x1, len(sizes)), rng.uniform(y0, y1, len(sizes))], 1)
            if two:
                P[base] = (rng.uniform(0.30, 0.44), rng.uniform(-0.06, 0.06))  # the stack: reachable by both arms
                for i in range(len(sizes)):
                    if i != base:
                        P[i, 1] = rng.choice([-1, 1]) * rng.uniform(0.14, 0.34)  # clearly on one arm's side
            ok = True
            for i in range(len(sizes)):
                for j in range(i):
                    if np.linalg.norm(P[i] - P[j]) < (sizes[i] + sizes[j]) / 2 * 1.5 + self.tc.min_gap:
                        ok = False
                if i != base and np.linalg.norm(P[i] - P[base]) < 0.16:
                    ok = False
            if ok:
                return [[float(x), float(y), float(rng.uniform(-np.pi / 4, np.pi / 4))] for x, y in P]
        raise RuntimeError("could not place cubes")

    def footprint_xy(self, sc):
        return [p[:2] for p in sc["poses"]]

    def apply(self, env, sc):
        m, o = env.model, env._orig
        for i in range(self.tc.n_cubes):
            g, b = self.geoms[i], self.bodies[i]
            if i < sc["n"]:
                s = sc["sizes"][i]
                k = s / self.tc.sizes[i]
                m.geom_size[g] = o["geom_size"][g] * k
                m.geom_size[g + 1] = o["geom_size"][g + 1] * k
                m.body_mass[b] = o["body_mass"][b] * k**3
                m.body_inertia[b] = o["body_inertia"][b] * k**5
                m.geom_rgba[g] = sc["rgba"][i]
                m.geom_size[self.ghost_geoms[i]] = o["geom_size"][self.ghost_geoms[i]] * k
            m.geom_rgba[self.ghost_geoms[i], 3] = 0.0
        self._regime(env, env.regime)

    def apply_regime(self, env, regime):
        self._regime(env, regime)

    def _regime(self, env, r):
        m, o = env.model, env._orig
        sc = env.scenario.task if env.scenario else None
        for i in range(self.tc.n_cubes):
            g, b = self.geoms[i], self.bodies[i]
            k = (sc["sizes"][i] / self.tc.sizes[i]) if sc and i < sc["n"] else 1.0
            m.body_mass[b] = o["body_mass"][b] * k**3 * r.get("object_mass", 1.0)
            m.body_inertia[b] = o["body_inertia"][b] * k**5 * r.get("object_mass", 1.0)
            m.geom_friction[g, 0] = o["geom_friction"][g, 0] * r.get("object_friction", 1.0)

    def reset_state(self, env, sc):
        d = env.data
        top = env.cfg.bench.top_z
        for i in range(self.tc.n_cubes):
            qa = self.qadr[i]
            if i < sc["n"]:
                x, y, yaw = sc["poses"][i]
                d.qpos[qa : qa + 3] = (x, y, top + sc["sizes"][i] / 2 + 0.002)
                d.qpos[qa + 3 : qa + 7] = G.yaw_quat(yaw)
            else:
                d.qpos[qa : qa + 3] = PARK + (0.15 * i, -0.3, 0.05)
                d.qpos[qa + 3 : qa + 7] = (1, 0, 0, 0)
            d.qvel[self.dadr[i] : self.dadr[i] + 6] = 0

    def after_settle(self, env):
        """Goal poses: the stack grows on the bottom cube where it came to rest."""
        sc = env.scenario.task
        d, m = env.data, env.model
        b0 = self.bodies[sc["order"][0]]
        base_p, base_q = d.xpos[b0].copy(), d.xquat[b0].copy()
        z = base_p[2] + sc["sizes"][sc["order"][0]] / 2
        goals = []
        for k, ci in enumerate(sc["order"]):
            s = sc["sizes"][ci]
            if k == 0:
                goals.append([*base_p.tolist(), *base_q.tolist()])
            else:
                goals.append([base_p[0], base_p[1], z + s / 2, *base_q.tolist()])
                z += s
        sc["goals"] = goals
        show = self.tc.show_goal_ghost or sc.get("variant") == "pattern"
        for k, ci in enumerate(sc["order"]):
            m.body_pos[self.ghosts[ci]] = goals[k][:3]
            m.body_quat[self.ghosts[ci]] = goals[k][3:]
            m.geom_rgba[self.ghost_geoms[ci]] = (*sc["rgba"][ci][:3], 0.28 if (show and k > 0) else 0.0)
        mujoco_forward(env)

    def default_episode_s(self, sc):
        return self.tc.episode_s_per_cube * (sc["n"] - 1) + 4.0

    def horizon(self, sc):
        return sc["n"] - 1

    def subgoal_done(self, env, k):
        pr = self.privileged(env)
        return bool(pr["seated"][k]) and pr["grasped"][env.scenario.task["order"][k]] is None

    def goal_description(self, sc):
        return "stack cubes " + " -> ".join(sc["colors"][i] for i in sc["order"]) + " (bottom to top)"

    # ---- per-step labels
    def privileged(self, env):
        sc = env.scenario.task
        d = env.data
        n = sc["n"]
        pos = np.stack([d.xpos[self.bodies[i]] for i in range(n)])
        quat = np.stack([d.xquat[self.bodies[i]] for i in range(n)])
        vel = np.stack([d.qvel[self.dadr[i] : self.dadr[i] + 6] for i in range(n)])
        grasped = [env.grasped_by(self.geom_sets[i]) for i in range(n)]
        seated = self._seated(sc, pos, quat, vel)
        k_done = 0
        while k_done < n - 1 and seated[k_done + 1]:
            k_done += 1
        return dict(cube_pos=pos, cube_quat=quat, cube_vel=vel, grasped=grasped, seated=seated, n_seated=k_done,
                    order=sc["order"], sizes=np.asarray(sc["sizes"]), goals=np.asarray(sc.get("goals", [[0] * 7] * n)))

    def _seated(self, sc, pos, quat, vel):
        """seated[k]: cube order[k] rests on cube order[k-1] (xy within tol, correct height, level, still)."""
        n = sc["n"]
        tol = self.tc.place_tol_xy
        out = [True]
        for k in range(1, n):
            ci, cb = sc["order"][k], sc["order"][k - 1]
            dz = pos[ci, 2] - pos[cb, 2] - (sc["sizes"][ci] + sc["sizes"][cb]) / 2
            up = G.quat_rotate(quat[ci], [0, 0, 1])
            ok = (np.linalg.norm(pos[ci, :2] - pos[cb, :2]) < tol and abs(dz) < 0.006 and up[2] > 0.97
                  and np.linalg.norm(vel[ci, :3]) < 0.05 and out[-1])
            out.append(bool(ok))
        return out

    def stages(self):
        names = []
        for k in range(1, self.tc.n_cubes):
            names += [f"{s}_{k}" for s in SUB]
        return tuple(names) + ("done",)

    def _substage(self, env, pr, k):
        """Sub-stage of placing cube order[k]: 0 reach, 1 grasp(ed, not lifted), 2 carry, 3 place (near goal)."""
        sc = env.scenario.task
        ci = sc["order"][k]
        g = pr["grasped"][ci]
        p = pr["cube_pos"][ci]
        goal = pr["goals"][k, :3]
        if g is None:
            return 0
        if np.linalg.norm(p[:2] - goal[:2]) < 0.03 and p[2] < goal[2] + 0.04:
            return 3
        if p[2] > sc["sizes"][ci] / 2 + env.cfg.bench.top_z + 0.015:
            return 2
        return 1

    def stage(self, env, pr):
        n = env.scenario.task["n"]
        k = pr["n_seated"] + 1
        if k >= n:
            return len(self.stages()) - 1
        return 4 * (k - 1) + self._substage(env, pr, k)

    def progress(self, env, pr):
        n = env.scenario.task["n"]
        k = pr["n_seated"] + 1
        if k >= n:
            return 1.0
        sub = self._substage(env, pr, k)
        return (pr["n_seated"] + (0.0, 0.25, 0.5, 0.8)[sub]) / (n - 1)

    def success(self, env, pr):
        n = env.scenario.task["n"]
        return pr["n_seated"] == n - 1 and all(g is None for g in pr["grasped"])

    def task_features(self, env, pr):
        """[n, k, progress, grasped flag, moving cube pos(3) quat(4), goal pos(3), goal - cube(3), cube - tcp per arm (6), all cube pos (3 n_cubes)]."""
        sc = env.scenario.task
        n, k = sc["n"], min(pr["n_seated"] + 1, sc["n"] - 1)
        ci = sc["order"][k]
        p, q = pr["cube_pos"][ci], pr["cube_quat"][ci]
        goal = pr["goals"][k, :3]
        rel = []
        for s in env.arms:
            rel.append(p - pr["tcp"][s][0])
        rel = np.concatenate(rel + [np.zeros(3)] * (2 - len(env.arms)))
        allp = np.zeros(3 * self.tc.n_cubes)
        allp[: 3 * n] = pr["cube_pos"].reshape(-1)
        return np.concatenate([[n, k, self.progress(env, pr), float(pr["grasped"][ci] is not None)], p, q, goal, goal - p, rel, allp]).astype(np.float32)

    def feature_names(self):
        return (["n", "k", "progress", "grasped"] + [f"cube_{a}" for a in "xyz"] + [f"cube_q{a}" for a in "wxyz"] + [f"goal_{a}" for a in "xyz"]
                + [f"goal_rel_{a}" for a in "xyz"] + [f"cube_rel_{s}_{a}" for s in ("right", "left") for a in "xyz"]
                + [f"cube{i}_{a}" for i in range(self.tc.n_cubes) for a in "xyz"])

    def record_columns(self, env, pr):
        sc = env.scenario.task
        k = min(pr["n_seated"] + 1, sc["n"] - 1)
        ci = sc["order"][k]
        return dict(n_seated=float(pr["n_seated"]), cube_goal_dist=float(np.linalg.norm(pr["cube_pos"][ci] - pr["goals"][k, :3])),
                    cube_height=float(pr["cube_pos"][ci][2]))

    # ---- outcome labelling
    def outcome(self, env, record, expert):
        pr = self.privileged(env)
        sc = env.scenario.task
        n = sc["n"]
        if env.unstable:
            return "sim_unstable"
        if self.success(env, pr):
            return "none"
        seated_hist = record.get("n_seated", [])
        if seated_hist and max(seated_hist) > pr["n_seated"]:
            return "toppled"
        if pr["n_seated"] >= n - 1:
            return "incomplete"  # everything seated but the expert still holds / touches a cube
        k = pr["n_seated"] + 1
        ci = sc["order"][k]
        # a cube seated out of order?
        pos = pr["cube_pos"]
        for j in range(n):
            if j != ci and j not in sc["order"][: k] and pos[j, 2] > sc["sizes"][j] / 2 + env.cfg.bench.top_z + 0.015 and pr["grasped"][j] is None:
                return "wrong_order"
        grasp_hist = record.get("grasped_k", [])
        if not any(g == k for g in grasp_hist):
            return "grasp_missed"
        if record.get("drops", 0) > 0 or pos[ci, 2] < sc["sizes"][ci] / 2 + env.cfg.bench.top_z + 0.01:
            return "dropped" if record.get("drops", 0) > 0 else "misplaced"
        if pr["grasped"][ci] is not None:
            return "incomplete"
        return "misplaced"

    # ---- expert / failures
    def make_expert(self, env, failure=None, seed=0, noise=True, bimanual=None, recover=False):
        """Scripted expert. With two arms (and bimanual not False) both arms work: each takes the objects on
        its side of the bench and they take turns at the shared structure."""
        from ..experts.cube_stack import CubeStackExpert

        if bimanual is None:
            bimanual = len(env.arms) == 2
        if bimanual and len(env.arms) == 2:
            from ..experts.dual import DualArmExpert

            sc = env.scenario.task
            return DualArmExpert(env, CubeStackExpert, {k: sc["poses"][sc["order"][k]][:2] for k in range(1, sc["n"])}, failure=failure, seed=seed, noise=noise, recover=recover, ordered=True)
        return CubeStackExpert(env, failure=failure, seed=seed, noise=noise, recover=recover)

    def sample_failure(self, kind, s, rng, sc):
        if kind == "misplace":
            a = rng.uniform(0, 2 * np.pi)
            r = 0.008 + 0.03 * s
            return dict(dxy=[float(r * np.cos(a)), float(r * np.sin(a))])
        if kind == "wrong_cube":
            return dict()
        if kind == "low_carry":
            return dict(clearance=float(-0.01 - 0.02 * s))  # carry height below the stack top
        return {}


def mujoco_forward(env):
    import mujoco

    mujoco.mj_forward(env.model, env.data)
