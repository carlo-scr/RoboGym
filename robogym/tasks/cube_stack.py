"""Cube stacking, un-stacking and relocation: move cubes one at a time through a sequence of *moves*.

Modes (variant "mode", default cfg.task.mode):
  stack     stack n cubes (2..n_cubes) in a given order on top of the bottom cube where it lies. Variants
            "by_size" (largest at the bottom, readable from the scene) and "pattern" (random order, shown as
            a translucent ghost of the finished stack).
  unstack   the cubes start stacked; take them down one by one (top first) onto marked spots (disassembly).
  relocate  unstack, move the base cube to a new spot and rebuild the stack there: 2n-1 moves, the
            compositional long-horizon version (later moves depend on where the earlier ones put things).
  handover  two arms: stack, but every cube lies on the far side of the bench from the stack, out of the
            building arm's reach. One arm picks and presents each cube in mid-air, the other takes it and
            places it (robogym.experts.handover). With one arm it falls back to "stack".
A move is (cube, support cube or -1, spot index or -1); it is achieved when the cube rests at its goal: on the
support cube (xy / height / level / still, relative to the support) or on the table spot. Progress counts
achieved moves with partial credit for the one in flight (reach -> grasp -> carry -> place).
Shift "object_shape" swaps the cubes for cylinders of the same size (both geoms are compiled colliding and one
is switched off per episode: a geom compiled with contype = conaffinity = 0 is left out of MuJoCo's bounding
volume hierarchy and can never be switched on later).

Achievement bookkeeping: a move is *achieved* while its cube rests at its goal; in relocate mode later moves
undo earlier ones, so each move also gets a sticky flag (set in order, kept in the snapshot aux) and the
number of done moves counts leading moves that are flagged (or, for the last move of each cube, achieved
now, so a toppled final stack is seen).
"""
from __future__ import annotations

import mujoco
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
MODES = ("stack", "unstack", "relocate", "handover")
SPOT_TOL_XY = 0.025  # a cube counts as on its spot within this (the marker disc is 1.5 x the cube)
MIN_SWEEP = 0.16  # free radius around a stack: the open gripper (9 cm) reaching for a cube must not sweep it
HANDOVER_Z = 0.17  # height of the mid-air hand-over above the bench


@register
class CubeStack(Task):
    name = "cube_stack"
    failure_types = ("misplace", "wrong_cube", "low_carry")
    shift_types = ("cube_size", "object_color", "object_texture", "object_shape")
    outcome_modes = ("none", "dropped", "toppled", "misplaced", "grasp_missed", "wrong_order", "incomplete", "sim_unstable", "other")

    # ---- model
    def xml(self):
        tc = self.tc
        out, ghosts, spots = [], [], []
        for i, s in enumerate(tc.sizes[: tc.n_cubes]):
            h = s / 2
            m = tc.density * s**3
            out.append(f'<body name="cube{i}" pos="{G.fmt(PARK + (0.15 * i, -0.3, h + 0.01))}"><freejoint name="cube{i}"/>'
                       f'<geom name="cube{i}" class="cube" size="{h} {h} {h}" mass="{m:.4f}" rgba="0.8 0.2 0.2 1"/>'
                       f'<geom name="cube{i}_edge" class="visual" type="box" size="{h * 0.995} {h * 0.995} {h * 0.995}" rgba="0 0 0 0"/>'
                       f'<geom name="cube{i}_cyl" class="cube" type="cylinder" size="{h} {h}" mass="0" rgba="0.8 0.2 0.2 0"/>'
                       f'<site name="cube{i}_top" pos="0 0 {h}" size="0.002" group="4"/></body>')
            ghosts.append(f'<body name="ghost{i}" pos="{G.fmt(PARK + (0.15 * i, -0.6, h))}">'
                          f'<geom name="ghost{i}" class="ghost" type="box" size="{h} {h} {h}" rgba="0.3 0.9 0.3 0.25"/></body>')
            spots.append(f'<body name="spot{i}" pos="{G.fmt(PARK + (0.15 * i, -0.9, 0.0))}">'
                         f'<geom name="spot{i}" class="visual" type="cylinder" size="{0.75 * s} 0.0008" rgba="0.95 0.95 0.95 0.6"/></body>')
        defaults = f'<default class="cube"><geom type="box" contype="1" conaffinity="7" condim="4" friction="{tc.friction} 0.005 0.0005" solimp="0.95 0.99 0.001" solref="0.004 1"/></default>'
        return "\n".join(out + ghosts + spots), [], "", "", defaults

    def bind(self, env):
        m = env.model
        n = self.tc.n_cubes
        self.bodies = [m.body(f"cube{i}").id for i in range(n)]
        self.geoms = [m.geom(f"cube{i}").id for i in range(n)]
        self.cyl_geoms = [m.geom(f"cube{i}_cyl").id for i in range(n)]
        self.geom_sets = [{self.geoms[i], self.cyl_geoms[i]} for i in range(n)]
        self.qadr = [m.jnt_qposadr[m.joint(f"cube{i}").id] for i in range(n)]
        self.dadr = [m.jnt_dofadr[m.joint(f"cube{i}").id] for i in range(n)]
        self.ghosts = [m.body(f"ghost{i}").id for i in range(n)]
        self.ghost_geoms = [m.geom(f"ghost{i}").id for i in range(n)]
        self.spot_bodies = [m.body(f"spot{i}").id for i in range(n)]
        self.spot_geoms = [m.geom(f"spot{i}").id for i in range(n)]

    # ---- scenario
    def sample(self, rng, shifts, variant=None):
        tc = self.tc
        variant = variant or {}
        two = self.cfg.robot.n_arms == 2
        mode = str(variant.get("mode", getattr(tc, "mode", "stack")))
        if mode not in MODES:
            raise ValueError(f"unknown cube_stack mode {mode!r}; choose from {MODES}")
        if mode == "handover" and not two:
            mode = "stack"
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
        shape = str(variant.get("shape", "cylinder" if "object_shape" in shifts else "cube"))
        poses, spots, handover = self._sample_layout(rng, sizes, order, mode, two)
        moves = self._moves(order, mode)
        return dict(n=n, mode=mode, order=order, sizes=sizes.tolist(), colors=cols, rgba=rgba, gloss=gloss, shape=shape, poses=poses,
                    spots=spots, moves=moves, stacked=mode in ("unstack", "relocate"), handover=handover, variant=tc.variant)

    @staticmethod
    def _moves(order, mode):
        """[cube, support cube (-1: table spot), spot index (-1: on the support)] in execution order."""
        n = len(order)
        stack = [[order[k], order[k - 1], -1] for k in range(1, n)]
        if mode in ("stack", "handover"):
            return stack
        unstack = [[order[n - 1 - i], -1, i] for i in range(n - 1)]  # top first
        if mode == "unstack":
            return unstack
        return unstack + [[order[0], -1, n - 1]] + stack  # relocate

    def _sample_layout(self, rng, sizes, order, mode, two):
        """Initial cube poses [x, y, yaw], table spots [x, y, yaw] and the hand-over layout.

        stack: cubes spread over the spawn area, the others >= MIN_SWEEP from the base cube; with two arms the
        base sits near the centre line (both arms reach the stack) and the others spread over both sides.
        handover: the stack on one arm's side, every other cube on the far side, out of the builder's reach.
        unstack / relocate: the cubes start stacked at the base pose; spots (n-1, +1 for relocate) are >= MIN_SWEEP
        from the stack and >= 0.12 apart; with two arms the stack (and the relocation spot) sit on the centre
        line and the spots on both sides."""
        n = len(sizes)
        base = order[0]
        x0, x1, y0, y1 = self.tc.spawn
        if two:
            x0, x1, y0, y1 = 0.26, 0.48, -0.34, 0.34
        handover = None
        for _ in range(4000):
            P = np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)], 1)
            S = []
            if mode == "handover":
                side = float(rng.choice([-1.0, 1.0]))  # side of the stack (the receiver's side)
                P[base] = (rng.uniform(0.30, 0.46), side * rng.uniform(0.18, 0.30))
                for i in range(n):
                    if i != base:
                        P[i] = (rng.uniform(0.28, 0.48), -side * rng.uniform(0.16, 0.32))
                handover = dict(receiver="right" if side < 0 else "left", giver="left" if side < 0 else "right",
                                point=[float(rng.uniform(0.38, 0.44)), 0.0, HANDOVER_Z])
            elif mode == "stack" and two:
                P[base] = (rng.uniform(0.30, 0.44), rng.uniform(-0.06, 0.06))  # the stack: reachable by both arms
                for i in range(n):
                    if i != base:
                        P[i, 1] = rng.choice([-1, 1]) * rng.uniform(0.14, 0.34)  # clearly on one arm's side
            elif mode in ("unstack", "relocate"):
                if two:
                    P[base] = (rng.uniform(0.28, 0.32), rng.uniform(-0.05, 0.05))
                n_spots = n - 1 + (mode == "relocate")
                S = np.stack([rng.uniform(x0, x1, n_spots), rng.uniform(y0, y1, n_spots)], 1)
                if two:
                    S[: n - 1, 1] = rng.choice([-1, 1], n - 1) * rng.uniform(0.14, 0.34, n - 1)
                    if mode == "relocate":
                        S[n - 1] = (rng.uniform(0.48, 0.52), rng.uniform(-0.05, 0.05))
            ok = True
            if mode in ("unstack", "relocate"):
                pts = [P[base]] + list(S)
                for i in range(len(pts)):
                    for j in range(i):
                        if np.linalg.norm(pts[i] - pts[j]) < (MIN_SWEEP if 0 in (i, j) else 0.12):
                            ok = False
                if mode == "relocate" and any(np.linalg.norm(S[n - 1] - S[j]) < MIN_SWEEP for j in range(n - 1)):
                    ok = False
            else:
                for i in range(n):
                    for j in range(i):
                        if np.linalg.norm(P[i] - P[j]) < (sizes[i] + sizes[j]) / 2 * 1.5 + self.tc.min_gap:
                            ok = False
                    if i != base and np.linalg.norm(P[i] - P[base]) < MIN_SWEEP:
                        ok = False
            if ok:
                poses = [[float(x), float(y), float(rng.uniform(-np.pi / 4, np.pi / 4))] for x, y in P]
                spots = [[float(x), float(y), float(rng.uniform(-np.pi / 4, np.pi / 4))] for x, y in S]
                return poses, spots, handover
        raise RuntimeError("could not place cubes")

    def footprint_xy(self, sc):
        return [p[:2] for p in sc["poses"]] + [s[:2] for s in sc["spots"]]

    def apply(self, env, sc):
        m, o = env.model, env._orig
        cyl = sc.get("shape", "cube") == "cylinder"
        for i in range(self.tc.n_cubes):
            g, c, b = self.geoms[i], self.cyl_geoms[i], self.bodies[i]
            if i < sc["n"]:
                s = sc["sizes"][i]
                k = s / self.tc.sizes[i]
                m.geom_size[g] = o["geom_size"][g] * k
                m.geom_size[g + 1] = o["geom_size"][g + 1] * k
                m.geom_size[c] = o["geom_size"][c] * k
                m.body_mass[b] = o["body_mass"][b] * k**3
                m.body_inertia[b] = o["body_inertia"][b] * k**5
                m.geom_rgba[g] = (*sc["rgba"][i][:3], 0.0 if cyl else 1.0)
                m.geom_rgba[c] = (*sc["rgba"][i][:3], 1.0 if cyl else 0.0)
                m.geom_size[self.ghost_geoms[i]] = o["geom_size"][self.ghost_geoms[i]] * k
                m.geom_size[self.spot_geoms[i], 0] = o["geom_size"][self.spot_geoms[i], 0] * k
            else:
                m.geom_rgba[g] = (0.8, 0.2, 0.2, 1.0)
                m.geom_rgba[c, 3] = 0.0
                cyl_i = False
            cyl_i = cyl and i < sc["n"]
            m.geom_contype[g], m.geom_conaffinity[g] = (0, 0) if cyl_i else (1, 7)
            m.geom_contype[c], m.geom_conaffinity[c] = (1, 7) if cyl_i else (0, 0)
            m.geom_rgba[self.ghost_geoms[i], 3] = 0.0
            m.geom_rgba[self.spot_geoms[i], 3] = 0.0
            m.body_pos[self.spot_bodies[i]] = PARK + (0.15 * i, -0.9, 0.0)
        top = env.cfg.bench.top_z
        for j, (x, y, _yaw) in enumerate(sc["spots"]):  # tape marks on the bench
            m.body_pos[self.spot_bodies[j]] = (x, y, top + 0.0009)
            m.geom_rgba[self.spot_geoms[j], 3] = 0.6
        self._regime(env, env.regime)
        self._flags = [False] * len(sc["moves"])
        self._latest = {c: k for k, (c, _s, _spot) in enumerate(sc["moves"])}

    def apply_regime(self, env, regime):
        self._regime(env, regime)

    def _regime(self, env, r):
        m, o = env.model, env._orig
        sc = env.scenario.task if env.scenario else None
        for i in range(self.tc.n_cubes):
            g, c, b = self.geoms[i], self.cyl_geoms[i], self.bodies[i]
            k = (sc["sizes"][i] / self.tc.sizes[i]) if sc and i < sc["n"] else 1.0
            m.body_mass[b] = o["body_mass"][b] * k**3 * r.get("object_mass", 1.0)
            m.body_inertia[b] = o["body_inertia"][b] * k**5 * r.get("object_mass", 1.0)
            m.geom_friction[g, 0] = o["geom_friction"][g, 0] * r.get("object_friction", 1.0)
            m.geom_friction[c, 0] = o["geom_friction"][c, 0] * r.get("object_friction", 1.0)

    def reset_state(self, env, sc):
        d = env.data
        top = env.cfg.bench.top_z
        z_stack = top
        for i in range(self.tc.n_cubes):
            qa = self.qadr[i]
            if i >= sc["n"]:
                d.qpos[qa : qa + 3] = PARK + (0.15 * i, -0.3, 0.05)
                d.qpos[qa + 3 : qa + 7] = (1, 0, 0, 0)
            elif not sc.get("stacked"):
                x, y, yaw = sc["poses"][i]
                d.qpos[qa : qa + 3] = (x, y, top + sc["sizes"][i] / 2 + 0.002)
                d.qpos[qa + 3 : qa + 7] = G.yaw_quat(yaw)
            d.qvel[self.dadr[i] : self.dadr[i] + 6] = 0
        if sc.get("stacked"):  # the initial stack, bottom to top, at the base cube's pose
            x, y, yaw = sc["poses"][sc["order"][0]]
            for k, ci in enumerate(sc["order"]):
                qa = self.qadr[ci]
                d.qpos[qa : qa + 3] = (x, y, z_stack + sc["sizes"][ci] / 2 + 0.0015 * (k + 1))
                d.qpos[qa + 3 : qa + 7] = G.yaw_quat(yaw)
                z_stack += sc["sizes"][ci]

    def after_settle(self, env):
        """Nominal goal of every move from the resting poses (ghosts / markers / features); the live goals
        follow the support cubes' actual poses (move_goal)."""
        sc = env.scenario.task
        d, m = env.data, env.model
        n = sc["n"]
        pose = {i: (d.xpos[self.bodies[i]].copy(), d.xquat[self.bodies[i]].copy()) for i in range(n)}
        goals, final = [], {}
        for k, (c, _s, _spot) in enumerate(sc["moves"]):
            gp, gq = self._goal_from(sc, k, pose)
            pose[c] = (gp, gq)
            goals.append([*gp.tolist(), *gq.tolist()])
            final[c] = goals[-1]
        sc["goals"] = goals
        show = self.tc.show_goal_ghost or sc.get("variant") == "pattern" or sc["mode"] != "stack"
        for ci in range(n):
            if ci in final:
                m.body_pos[self.ghosts[ci]] = final[ci][:3]
                m.body_quat[self.ghosts[ci]] = final[ci][3:]
            m.geom_rgba[self.ghost_geoms[ci]] = (*sc["rgba"][ci][:3], 0.28 if (show and ci in final) else 0.0)
        mujoco.mj_forward(env.model, env.data)

    def _goal_from(self, sc, k, pose):
        """Goal (pos, quat) of move k given cube poses `pose` (index -> (pos, quat))."""
        c, s, spot = sc["moves"][k]
        size_c = sc["sizes"][c]
        if s >= 0:
            p, q = pose[s]
            return np.array([p[0], p[1], p[2] + (sc["sizes"][s] + size_c) / 2]), G.yaw_quat(G.yaw_of_quat(q))
        x, y, yaw = sc["spots"][spot]
        return np.array([x, y, self.cfg.bench.top_z + size_c / 2]), G.yaw_quat(yaw)

    def move_goal(self, env, k):
        """Live goal of move k (on the support cube where it is now, or on the spot)."""
        d = env.data
        sc = env.scenario.task
        return self._goal_from(sc, k, {i: (d.xpos[self.bodies[i]], d.xquat[self.bodies[i]]) for i in range(sc["n"])})

    def default_episode_s(self, sc):
        per = self.tc.episode_s_per_cube + (6.0 if sc["mode"] == "handover" else 0.0)
        return per * len(sc["moves"]) + 4.0

    def horizon(self, sc):
        return len(sc["moves"])

    def subgoal_done(self, env, k):
        pr = self.privileged(env)
        return bool(pr["achieved"][k]) and pr["grasped"][env.scenario.task["moves"][k][0]] is None

    def goal_description(self, sc):
        cols = sc["colors"]
        if sc["mode"] in ("stack", "handover"):
            return ("hand over and " if sc["mode"] == "handover" else "") + "stack cubes " + " -> ".join(cols[i] for i in sc["order"]) + " (bottom to top)"
        if sc["mode"] == "unstack":
            return "unstack " + " -> ".join(cols[i] for i in reversed(sc["order"])) + " onto the marked spots"
        return "unstack onto the spots, then rebuild the stack " + " -> ".join(cols[i] for i in sc["order"]) + " on the far spot"

    # ---- per-step labels
    def privileged(self, env):
        sc = env.scenario.task
        d = env.data
        n = sc["n"]
        pos = np.stack([d.xpos[self.bodies[i]] for i in range(n)])
        quat = np.stack([d.xquat[self.bodies[i]] for i in range(n)])
        vel = np.stack([d.qvel[self.dadr[i] : self.dadr[i] + 6] for i in range(n)])
        holders = [env.holders(self.geom_sets[i]) for i in range(n)]
        grasped = [h[0] if h else None for h in holders]
        pose = {i: (pos[i], quat[i]) for i in range(n)}
        goals = np.stack([np.concatenate(self._goal_from(sc, k, pose)) for k in range(len(sc["moves"]))])
        now = self._achieved(sc, pos, quat, vel, goals)
        for k in range(len(now)):  # sticky flags, set in order
            if self._flags[k]:
                continue
            if now[k]:
                self._flags[k] = True
            break
        achieved = [now[k] if self._latest[c] == k else self._flags[k] for k, (c, _s, _spot) in enumerate(sc["moves"])]
        k_done = 0
        while k_done < len(achieved) and achieved[k_done]:
            k_done += 1
        return dict(cube_pos=pos, cube_quat=quat, cube_vel=vel, grasped=grasped, holders=holders, achieved=achieved, achieved_now=now, n_seated=k_done, n_done=k_done,
                    moves=sc["moves"], order=sc["order"], sizes=np.asarray(sc["sizes"]), goals=goals)

    # ---- snapshots (the sticky move flags live outside MjData)
    def aux_size(self):
        return 2 * self.tc.n_cubes - 1

    def get_aux(self, env):
        out = np.zeros(self.aux_size())
        out[: len(self._flags)] = np.asarray(self._flags, float)
        return out

    def set_aux(self, env, aux):
        self._flags = [bool(x) for x in aux[: len(self._flags)]]

    def _achieved(self, sc, pos, quat, vel, goals):
        """achieved[k]: the cube of move k rests at its goal (xy within tol, correct height, level, still)."""
        out = []
        for k, (c, s, _spot) in enumerate(sc["moves"]):
            g = goals[k]
            level = float(np.max(np.abs(G.quat_to_mat(quat[c])[2])))  # some face up (a handed-over cube lands on another face)
            tol = self.tc.place_tol_xy if s >= 0 else SPOT_TOL_XY
            ok = (np.linalg.norm(pos[c, :2] - g[:2]) < tol and abs(pos[c, 2] - g[2]) < 0.006 and level > 0.97 and np.linalg.norm(vel[c, :3]) < 0.05)
            out.append(bool(ok))
        return out

    def stages(self):
        names = []
        for k in range(2 * self.tc.n_cubes - 1):
            names += [f"{s}_{k}" for s in SUB]
        return tuple(names) + ("done",)

    def _substage(self, env, pr, k):
        """Sub-stage of move k: 0 reach, 1 grasped (still supported), 2 carry (free in the air), 3 place (near the goal)."""
        c = pr["moves"][k][0]
        if pr["grasped"][c] is None:
            return 0
        p, goal = pr["cube_pos"][c], pr["goals"][k, :3]
        if np.linalg.norm(p[:2] - goal[:2]) < 0.03 and p[2] < goal[2] + 0.04:
            return 3
        return 1 if env.supported(self.geom_sets[c]) else 2

    def stage(self, env, pr):
        k = pr["n_done"]
        if k >= len(pr["moves"]):
            return len(self.stages()) - 1
        return 4 * k + self._substage(env, pr, k)

    def progress(self, env, pr):
        k = pr["n_done"]
        if k >= len(pr["moves"]):
            return 1.0
        sub = self._substage(env, pr, k)
        return (k + (0.0, 0.25, 0.5, 0.8)[sub]) / len(pr["moves"])

    def success(self, env, pr):
        return pr["n_done"] == len(pr["moves"]) and all(g is None for g in pr["grasped"])

    def task_features(self, env, pr):
        """[n, k, progress, grasped flag, moving cube pos(3) quat(4), goal pos(3), goal - cube(3), cube - tcp per arm (6), all cube pos (3 n_cubes)]."""
        sc = env.scenario.task
        n, k = sc["n"], min(pr["n_done"], len(pr["moves"]) - 1)
        ci = pr["moves"][k][0]
        p, q = pr["cube_pos"][ci], pr["cube_quat"][ci]
        goal = pr["goals"][k, :3]
        rel = [p - pr["tcp"][s][0] for s in env.arms]
        rel = np.concatenate(rel + [np.zeros(3)] * (2 - len(env.arms)))
        allp = np.zeros(3 * self.tc.n_cubes)
        allp[: 3 * n] = pr["cube_pos"].reshape(-1)
        return np.concatenate([[n, k, self.progress(env, pr), float(pr["grasped"][ci] is not None)], p, q, goal, goal - p, rel, allp]).astype(np.float32)

    def feature_names(self):
        return (["n", "k", "progress", "grasped"] + [f"cube_{a}" for a in "xyz"] + [f"cube_q{a}" for a in "wxyz"] + [f"goal_{a}" for a in "xyz"]
                + [f"goal_rel_{a}" for a in "xyz"] + [f"cube_rel_{s}_{a}" for s in ("right", "left") for a in "xyz"]
                + [f"cube{i}_{a}" for i in range(self.tc.n_cubes) for a in "xyz"])

    def record_columns(self, env, pr):
        k = min(pr["n_done"], len(pr["moves"]) - 1)
        ci = pr["moves"][k][0]
        return dict(n_seated=float(pr["n_done"]), cube_goal_dist=float(np.linalg.norm(pr["cube_pos"][ci] - pr["goals"][k, :3])),
                    cube_height=float(pr["cube_pos"][ci][2]))

    # ---- outcome labelling
    def outcome(self, env, record, expert):
        pr = self.privileged(env)
        sc = env.scenario.task
        moves, n = sc["moves"], sc["n"]
        if env.unstable:
            return "sim_unstable"
        if self.success(env, pr):
            return "none"
        hist = record.get("n_seated", [])
        if hist and max(hist) > pr["n_done"]:
            return "toppled"
        if pr["n_done"] >= len(moves):
            return "incomplete"  # everything in place but the expert still holds / touches a cube
        k = pr["n_done"]
        ci = moves[k][0]
        pos = pr["cube_pos"]
        top = env.cfg.bench.top_z
        # a later cube already lifted off the bench and put down somewhere (only meaningful when the cubes
        # still to come started on the bench: stack modes, or the rebuild phase of relocate)
        if sc["mode"] in ("stack", "handover") or (sc["mode"] == "relocate" and k >= n):
            later = {mv[0] for mv in moves[k + 1 :]} - {ci}
            for j in later:
                if pos[j, 2] > sc["sizes"][j] / 2 + top + 0.015 and pr["grasped"][j] is None:
                    return "wrong_order"
        if ci not in record.get("grasped_k", []):
            return "grasp_missed"
        if pr["grasped"][ci] is not None:
            return "incomplete"
        if record.get("drops", 0) > 0:
            return "dropped"
        return "misplaced"

    # ---- expert / failures
    def make_expert(self, env, failure=None, seed=0, noise=True, bimanual=None, recover=False):
        """Scripted expert. With two arms (and bimanual not False) both arms work: each takes the moves on
        its side of the bench and they take turns at the shared structure; in handover mode one arm picks
        and presents, the other takes and places."""
        from ..experts.cube_stack import CubeStackExpert

        if bimanual is None:
            bimanual = len(env.arms) == 2
        sc = env.scenario.task
        if bimanual and len(env.arms) == 2 and sc["mode"] == "handover":
            from ..experts.handover import HandoverExpert

            return HandoverExpert(env, failure=failure, seed=seed, noise=noise, recover=recover)
        if bimanual and len(env.arms) == 2:
            from ..experts.dual import DualArmExpert

            return DualArmExpert(env, CubeStackExpert, self._move_sides(sc), failure=failure, seed=seed, noise=noise, recover=recover, ordered=True)
        return CubeStackExpert(env, failure=failure, seed=seed, noise=noise, recover=recover)

    def _move_sides(self, sc):
        """move index -> xy that decides which arm does it: where the cube is picked up (its spawn pose, or
        the spot an earlier move put it on) for stacking moves, the spot for put-down moves."""
        where = {i: sc["poses"][i][:2] for i in range(sc["n"])}
        out = {}
        for k, (c, s, spot) in enumerate(sc["moves"]):
            out[k] = list(sc["spots"][spot][:2]) if s < 0 else list(where[c])
            where[c] = sc["spots"][spot][:2] if s < 0 else where[s]
        return out

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
