"""Oversized-LEGO assembly: press n bricks onto a studded baseplate following a target model.

Bricks are hollow boxes (roof + 4 walls) with hemispherical studs; a brick seats when its walls pass
the studs of the support below (baseplate or an already snapped brick) and its bottom reaches the
support's top, xy / yaw aligned to the stud lattice. A detent force resists the last millimetres
(press-fit), then a weld fixes the brick (snapped bricks cannot be pulled off again). The target model
is shown as a translucent ghost. Long horizon = number of bricks; the goal structure is a random
stable stacking, so later bricks depend on earlier ones being seated.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .. import geometry as G
from ..scene import PARK
from .base import Task, register

KINDS = {"1x2": (1, 2), "1x4": (1, 4), "2x2": (2, 2), "2x3": (2, 3), "2x4": (2, 4)}  # (studs along x, along y)
PALETTE = {"red": (0.85, 0.12, 0.12, 1), "yellow": (0.97, 0.8, 0.1, 1), "blue": (0.1, 0.35, 0.85, 1), "green": (0.15, 0.6, 0.25, 1)}
NOMINAL_COLORS = tuple(PALETTE)
SUB = ("reach", "grasp", "carry", "align", "press")


@register
class Lego(Task):
    name = "lego"
    failure_types = ("misalign", "partial_press", "wrong_slot")
    shift_types = ("tight_fit", "stiff_snap", "object_color", "object_texture", "plate_shift")
    outcome_modes = ("none", "dropped", "misaligned", "wrong_slot", "not_snapped", "grasp_missed", "incomplete", "sim_unstable", "other")

    def __init__(self, cfg):
        super().__init__(cfg)
        tc = self.tc
        self.P, self.H = tc.pitch, tc.brick_h
        self.kinds = [KINDS[k] for k in tc.bricks]
        self.n_bricks = len(self.kinds)
        self.Nx, self.Ny = tc.plate_studs
        self.plate_top = cfg.bench.top_z + tc.plate_h
        self.wall_t = (2 * tc.pitch - tc.gap - (tc.pitch + 2 * tc.stud_r) - tc.fit_clearance) / 2
        self.tool_tip = (0.0, 0.0, 0.0)  # set in bind(): the held brick's centre in the TCP frame
        self.compliant_gains = (0.15, 0.15, 1.0, 0.3, 0.3, 0.3)  # soft laterally, stiff along the press (TCP z)

    # ---- geometry helpers
    def dims(self, i):
        nx, ny = self.kinds[i]
        return nx * self.P - self.tc.gap, ny * self.P - self.tc.gap

    def studs(self, nx, ny):
        return [((a - (nx - 1) / 2) * self.P, (b - (ny - 1) / 2) * self.P) for a in range(nx) for b in range(ny)]

    def _brick_geoms(self, i, prefix, cls, rgba, with_studs=True):
        tc = self.tc
        lx, ly = self.dims(i)
        H, cav, t, r = self.H, tc.cavity_depth, self.wall_t, tc.stud_r
        out = [f'<geom name="{prefix}_roof" class="{cls}" type="box" pos="0 0 {cav / 2}" size="{lx / 2} {ly / 2} {(H - cav) / 2}" rgba="{G.fmt(rgba, 3)}"/>']
        zc = -H / 2 + cav / 2
        for sgn in (-1, 1):
            out.append(f'<geom name="{prefix}_wx{"p" if sgn > 0 else "n"}" class="{cls}" type="box" pos="{sgn * (lx / 2 - t / 2)} 0 {zc}" size="{t / 2} {ly / 2} {cav / 2}" rgba="{G.fmt(rgba, 3)}"/>')
            out.append(f'<geom name="{prefix}_wy{"p" if sgn > 0 else "n"}" class="{cls}" type="box" pos="0 {sgn * (ly / 2 - t / 2)} {zc}" size="{lx / 2 - t} {t / 2} {cav / 2}" rgba="{G.fmt(rgba, 3)}"/>')
        if with_studs:
            for k, (sx, sy) in enumerate(self.studs(*self.kinds[i])):
                if tc.stud_cyl_h > 0:
                    out.append(f'<geom name="{prefix}_s{k}" class="{cls}" type="capsule" pos="{sx} {sy} {H / 2 + tc.stud_cyl_h / 2}" size="{r} {tc.stud_cyl_h / 2}" rgba="{G.fmt(rgba, 3)}"/>')
                else:
                    out.append(f'<geom name="{prefix}_s{k}" class="{cls}" type="sphere" pos="{sx} {sy} {H / 2}" size="{r}" rgba="{G.fmt(rgba, 3)}"/>')
        return out

    # ---- model
    def xml(self):
        tc = self.tc
        out = []
        for i in range(self.n_bricks):
            lx, ly = self.dims(i)
            m = tc.density * lx * ly * self.H
            out.append(f'<body name="brick{i}" pos="{G.fmt(PARK + (0.12 * i, -0.3, 0.05))}"><freejoint name="brick{i}"/>'
                       f'<inertial pos="0 0 0" mass="{m:.5f}" diaginertia="{G.fmt(G.box_inertia(m, (lx, ly, self.H)))}"/>')
            out += self._brick_geoms(i, f"brick{i}", "lego", (0.8, 0.2, 0.2, 1))
            out.append(f'<site name="brick{i}_bottom" pos="0 0 {-self.H / 2}" size="0.002" group="4"/></body>')
            out.append(f'<body name="ghost{i}" pos="{G.fmt(PARK + (0.12 * i, -0.6, 0.05))}">')
            out.append(f'<geom name="ghost{i}" class="ghost" type="box" size="{lx / 2} {ly / 2} {self.H / 2}" rgba="0.3 0.9 0.3 0.25"/>')
            for k, (sx, sy) in enumerate(self.studs(*self.kinds[i])):
                out.append(f'<geom name="ghost{i}_s{k}" class="ghost" type="sphere" pos="{sx} {sy} {self.H / 2}" size="{tc.stud_r}" rgba="0.3 0.9 0.3 0.25"/>')
            out.append("</body>")
        # baseplate
        hx, hy = self.Nx * self.P / 2, self.Ny * self.P / 2
        out.append(f'<body name="plate" pos="{tc.plate_xy[0]} {tc.plate_xy[1]} {self.cfg.bench.top_z}">')
        out.append(f'<geom name="plate" class="lego" type="box" pos="0 0 {tc.plate_h / 2}" size="{hx} {hy} {tc.plate_h / 2}" rgba="0.55 0.6 0.55 1"/>')
        for a in range(self.Nx):
            for b in range(self.Ny):
                out.append(f'<geom name="plate_s{a}_{b}" class="lego" type="sphere" pos="{(a - (self.Nx - 1) / 2) * self.P} {(b - (self.Ny - 1) / 2) * self.P} {tc.plate_h}" size="{tc.stud_r}" rgba="0.55 0.6 0.55 1"/>')
        out.append('<site name="plate_origin" pos="0 0 0" size="0.002" group="4"/></body>')
        eq = "".join(f'<weld name="snap{i}" body1="latch_anchor" body2="brick{i}" active="false" solref="0.01 1"/>' for i in range(self.n_bricks))
        defaults = (f'<default class="lego"><geom contype="1" conaffinity="7" condim="4" friction="{tc.friction} 0.005 0.0005" '
                    f'solimp="0.95 0.99 0.001" solref="0.005 1"/></default>')
        return "\n".join(out), [], eq, "", defaults

    def bind(self, env):
        m = env.model
        n = self.n_bricks
        self.bodies = [m.body(f"brick{i}").id for i in range(n)]
        self.qadr = [m.jnt_qposadr[m.joint(f"brick{i}").id] for i in range(n)]
        self.dadr = [m.jnt_dofadr[m.joint(f"brick{i}").id] for i in range(n)]
        self.geom_sets = [{g for g in range(m.ngeom) if m.geom_bodyid[g] == b} for b in self.bodies]
        self.stud_geoms = [[m.geom(f"brick{i}_s{k}").id for k in range(len(self.studs(*self.kinds[i])))] for i in range(n)]
        self.wall_geoms = [[m.geom(f"brick{i}_{w}").id for w in ("roof", "wxp", "wxn", "wyp", "wyn")] for i in range(n)]
        self.ghosts = [m.body(f"ghost{i}").id for i in range(n)]
        self.ghost_geoms = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == self.ghosts[i]] for i in range(n)]
        self.snap_eq = [m.equality(f"snap{i}").id for i in range(n)]
        self.plate = m.body("plate").id
        self.plate_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == self.plate}
        self.plate_studs = [m.geom(f"plate_s{a}_{b}").id for a in range(self.Nx) for b in range(self.Ny)]
        self._occ = {}  # level -> set of occupied cells (snapped bricks), plate frame
        w = min(self.dims(0))
        self.tool_tip = (0.0, 0.0, env.grasp_tcp_height(w) - self.H / 2)  # remote compliance about the brick centre

    # ---- scenario
    def sample(self, rng, shifts, variant=None):
        tc = self.tc
        variant = variant or {}
        n = int(variant.get("n", rng.integers(2, self.n_bricks + 1)))
        n = max(1, min(n, self.n_bricks))
        cols = list(rng.choice(NOMINAL_COLORS, size=n, replace=n > len(NOMINAL_COLORS)))
        rgba = [list(PALETTE[c]) for c in cols]
        if "object_color" in shifts:
            from ..shifts import OOD_OBJECT_COLORS

            rgba = [list(OOD_OBJECT_COLORS[int(k)]) for k in rng.choice(len(OOD_OBJECT_COLORS), size=n, replace=False)]
        px0 = tc.plate_xy[0] - (0.05 if self.cfg.robot.n_arms == 2 else 0.0)  # two arms: within both arms' reach
        plate = [px0 + float(rng.uniform(-0.03, 0.03)), tc.plate_xy[1] + float(rng.uniform(-0.03, 0.03)), float(rng.uniform(-0.26, 0.26))]
        if "plate_shift" in shifts:
            plate = [tc.plate_xy[0] + float(rng.choice([-1, 1]) * rng.uniform(0.07, 0.12)), tc.plate_xy[1] + float(rng.choice([-1, 1]) * rng.uniform(0.10, 0.18)), float(rng.uniform(-0.8, 0.8))]
        goals = []
        for _ in range(30):  # a random build order can paint itself into a corner: retry, then settle for fewer bricks
            g = self._sample_goals(rng, n)
            if len(g) > len(goals):
                goals = g
            if len(goals) == n:
                break
        n = len(goals)
        cols, rgba = cols[:n], rgba[:n]
        sides = None
        if self.cfg.robot.n_arms == 2:  # each brick spawns on the side of its goal (that arm builds it)
            sides = [self._goal_side(plate, k, goals[k]) for k in range(n)]
        poses = self._sample_poses(rng, n, plate, sides=sides)
        return dict(n=n, kinds=list(tc.bricks[:n]), colors=cols, rgba=rgba, plate=plate, goals=goals, poses=poses,
                    gloss=float(rng.uniform(0.5, 1.0)) if "object_texture" in shifts else 0.0,
                    stud_scale=float(1 + 0.08 * rng.uniform(0.6, 1.0)) if "tight_fit" in shifts else 1.0,
                    snap_scale=float(rng.uniform(2.0, 3.5)) if "stiff_snap" in shifts else 1.0)

    def _cells(self, i_cell, j_cell, nx, ny):
        return {(i_cell + a, j_cell + b) for a in range(nx) for b in range(ny)}

    def _sample_goals(self, rng, n):
        """Random stable structure in lattice coordinates: [i, j, level, yaw_q] per brick (build order)."""
        Nx, Ny = self.Nx, self.Ny
        occ = {}
        goals = []
        for k in range(n):
            nx0, ny0 = self.kinds[k]
            cands = []
            max_level = max(occ) if occ else -1
            for level in range(0, max_level + 2):
                for yaw_q in ((0,) if nx0 == ny0 else (0, 1)):
                    nx, ny = (nx0, ny0) if yaw_q == 0 else (ny0, nx0)
                    for i in range(0, Nx - nx + 1):
                        for j in range(0, Ny - ny + 1):
                            cells = self._cells(i, j, nx, ny)
                            # top-down insertable: nothing already placed at this level or above these cells
                            if any(cells & occ.get(lv, set()) for lv in range(level, max_level + 1)):
                                continue
                            if level > 0 and len(cells & occ.get(level - 1, set())) < int(np.ceil(0.75 * len(cells))):
                                continue  # >= 3/4 of the footprint rests on bricks below (a cantilevered brick tilts when pressed)
                            # the gripper pads stick out 4 mm on two opposite faces; neighbours (same level or
                            # higher) along the finger axis block the grasp. Square bricks can be grasped along
                            # either axis, long bricks only across their short side.
                            above = set().union(*[occ.get(lv, set()) for lv in range(level, max_level + 1)]) if occ else set()
                            bx = any((a + 1, b) in above or (a - 1, b) in above for (a, b) in cells)
                            by = any((a, b + 1) in above or (a, b - 1) in above for (a, b) in cells)
                            if nx == ny and bx and by:
                                continue
                            if nx != ny and ((nx < ny and bx) or (ny < nx and by)):
                                continue  # fingers must run along the short axis
                            cands.append((level, i, j, yaw_q))
            if not cands:
                break
            levels = np.array([c[0] for c in cands])
            if k > 0 and rng.random() < 0.65 and (levels > 0).any():
                cands = [c for c in cands if c[0] > 0]
            elif k > 0:
                # on the plate but near the structure: within 2 cells of an occupied cell
                near = []
                for c in cands:
                    if c[0] != 0:
                        continue
                    nx, ny = (nx0, ny0) if c[3] == 0 else (ny0, nx0)
                    cells = self._cells(c[1], c[2], nx, ny)
                    if any(abs(a - a2) <= 2 and abs(b - b2) <= 2 for (a, b) in cells for (a2, b2) in occ.get(0, set())):
                        near.append(c)
                cands = near or [c for c in cands if c[0] == 0] or cands
            else:
                cands = [c for c in cands if 1 <= c[1] <= Nx - 3 and 1 <= c[2] <= Ny - 3] or cands
            level, i, j, yaw_q = cands[int(rng.integers(len(cands)))]
            nx, ny = (nx0, ny0) if yaw_q == 0 else (ny0, nx0)
            occ.setdefault(level, set()).update(self._cells(i, j, nx, ny))
            goals.append([int(i), int(j), int(level), int(yaw_q)])
        return goals

    def goal_pose_plate(self, k, goal):
        """Brick k's goal (centre pos, yaw) in the plate frame from lattice coords."""
        i, j, level, yaw_q = goal
        nx0, ny0 = self.kinds[k]
        nx, ny = (nx0, ny0) if yaw_q == 0 else (ny0, nx0)
        cx = (i + (nx - 1) / 2 - (self.Nx - 1) / 2) * self.P
        cy = (j + (ny - 1) / 2 - (self.Ny - 1) / 2) * self.P
        return np.array([cx, cy, self.tc.plate_h + level * self.H + self.H / 2]), yaw_q * np.pi / 2

    def goal_pose_world(self, sc, k):
        p, yaw = self.goal_pose_plate(k, sc["goals"][k])
        px, py, pyaw = sc["plate"]
        q = G.yaw_quat(pyaw)
        return np.array([px, py, self.cfg.bench.top_z]) + G.quat_rotate(q, p), G.yaw_quat(pyaw + yaw)

    def _goal_side(self, plate, k, goal):
        """+1 (left arm) / -1 (right arm): the world-y side of brick k's goal cell."""
        p, _ = self.goal_pose_plate(k, goal)
        px, py, pyaw = plate
        y = py + np.sin(pyaw) * p[0] + np.cos(pyaw) * p[1]
        return 1.0 if y >= 0 else -1.0

    def _sample_poses(self, rng, n, plate, sides=None):
        x0, x1, y0, y1 = self.tc.spawn
        if sides is not None:
            x0, x1 = 0.24, 0.48
        hx, hy = self.Nx * self.P / 2 + 0.06, self.Ny * self.P / 2 + 0.06
        for _ in range(3000):
            P = np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)], 1)
            if sides is not None:
                P[:, 1] = np.asarray(sides) * rng.uniform(0.16, 0.36, n)
            if np.any((np.abs(P[:, 0] - plate[0]) < hx) & (np.abs(P[:, 1] - plate[1]) < hy)):
                continue
            ok = True
            rad = [np.hypot(*self.dims(i)) / 2 for i in range(n)]
            for i in range(n):
                for j in range(i):
                    if np.linalg.norm(P[i] - P[j]) < rad[i] + rad[j] + self.tc.min_gap:
                        ok = False
            if ok:
                return [[float(x), float(y), float(rng.uniform(-np.pi, np.pi))] for x, y in P]
        raise RuntimeError("could not place bricks")

    def footprint_xy(self, sc):
        return [p[:2] for p in sc["poses"]] + [sc["plate"][:2]]

    def apply(self, env, sc):
        m, o, tc = env.model, env._orig, self.tc
        px, py, pyaw = sc["plate"]
        m.body_pos[self.plate] = (px, py, self.cfg.bench.top_z)
        m.body_quat[self.plate] = G.yaw_quat(pyaw)
        for g in self.plate_studs:
            m.geom_size[g, 0] = o["geom_size"][g, 0] * sc["stud_scale"]
        for i in range(self.n_bricks):
            for g in self.stud_geoms[i]:
                m.geom_size[g, 0] = o["geom_size"][g, 0] * sc["stud_scale"]
            active = i < sc["n"]
            rgba = sc["rgba"][i] if active else (0.5, 0.5, 0.5, 1)
            for g in self.geom_sets[i]:
                m.geom_rgba[g] = rgba
            for g in self.ghost_geoms[i]:
                m.geom_rgba[g] = (*rgba[:3], 0.3 if (active and tc.show_goal_ghost) else 0.0)
            if active:
                gp, gq = self.goal_pose_world(sc, i)
                m.body_pos[self.ghosts[i]] = gp
                m.body_quat[self.ghosts[i]] = gq
            else:
                m.body_pos[self.ghosts[i]] = PARK + (0.12 * i, -0.6, 0.05)
        self.snap_force = tc.snap_force * sc["snap_scale"]
        self._regime(env, env.regime)

    def apply_regime(self, env, regime):
        self._regime(env, regime)

    MAX_MASS_SCALE = 2.5  # heavier bricks swing in the finger linkage and destabilise the grasp

    def _regime(self, env, r):
        m, o = env.model, env._orig
        ms = min(float(r.get("object_mass", 1.0)), self.MAX_MASS_SCALE)
        for i in range(self.n_bricks):
            b = self.bodies[i]
            m.body_mass[b] = o["body_mass"][b] * ms
            m.body_inertia[b] = o["body_inertia"][b] * ms
            for g in self.geom_sets[i]:
                m.geom_friction[g, 0] = o["geom_friction"][g, 0] * r.get("object_friction", 1.0)

    def reset_state(self, env, sc):
        d = env.data
        top = env.cfg.bench.top_z
        for i in range(self.n_bricks):
            qa = self.qadr[i]
            if i < sc["n"]:
                x, y, yaw = sc["poses"][i]
                d.qpos[qa : qa + 3] = (x, y, top + self.H / 2 + 0.001)
                d.qpos[qa + 3 : qa + 7] = G.yaw_quat(yaw)
            else:
                d.qpos[qa : qa + 3] = PARK + (0.12 * i, -0.3, 0.05)
                d.qpos[qa + 3 : qa + 7] = (1, 0, 0, 0)
            d.qvel[self.dadr[i] : self.dadr[i] + 6] = 0
            d.eq_active[self.snap_eq[i]] = 0
        self._occ = {}
        self._snapped = [False] * self.n_bricks
        self._snap_cells = [None] * self.n_bricks

    def default_episode_s(self, sc):
        return self.tc.episode_s_per_brick * sc["n"] + 4.0

    def horizon(self, sc):
        return sc["n"]

    def subgoal_done(self, env, k):
        pr = self.privileged(env)
        return bool(pr["at_goal"][k]) and pr["grasped"][k] is None

    def goal_description(self, sc):
        return "assemble " + ", ".join(f"{c} {k} at level {g[2]}" for c, k, g in zip(sc["colors"], sc["kinds"], sc["goals"]))

    # ---- lattice fit
    def plate_frame(self, env, pos, quat):
        d = env.data
        pp, pq = d.xpos[self.plate], d.xquat[self.plate]
        pi, qi = G.pose_inv(pp, pq)
        return G.pose_mul(pi, qi, np.asarray(pos), np.asarray(quat))

    def lattice_fit(self, i, pos_p, quat_p):
        """Residuals of brick i (pose in the plate frame) w.r.t. the nearest valid lattice seat."""
        nx0, ny0 = self.kinds[i]
        yaw = G.yaw_of_quat(quat_p)
        yaw_q = int(np.round(yaw / (np.pi / 2)))
        r_yaw = float(yaw - yaw_q * np.pi / 2)
        nx, ny = (nx0, ny0) if yaw_q % 2 == 0 else (ny0, nx0)
        ic = int(np.clip(np.round(pos_p[0] / self.P - (nx - 1) / 2 + (self.Nx - 1) / 2), 0, self.Nx - nx))
        jc = int(np.clip(np.round(pos_p[1] / self.P - (ny - 1) / 2 + (self.Ny - 1) / 2), 0, self.Ny - ny))
        cx = (ic + (nx - 1) / 2 - (self.Nx - 1) / 2) * self.P
        cy = (jc + (ny - 1) / 2 - (self.Ny - 1) / 2) * self.P
        r_xy = float(np.hypot(pos_p[0] - cx, pos_p[1] - cy))
        z_bottom = pos_p[2] - self.H / 2
        level = int(max(0, np.round((z_bottom - self.tc.plate_h) / self.H)))
        r_z = float(z_bottom - (self.tc.plate_h + level * self.H))
        up = G.quat_rotate(quat_p, [0, 0, 1])
        return dict(i=ic, j=jc, level=level, yaw_q=yaw_q % 4, r_xy=r_xy, r_z=r_z, r_yaw=r_yaw, tilt=float(np.arccos(np.clip(up[2], -1, 1))),
                    cells=self._cells(ic, jc, nx, ny))

    def supported(self, fit):
        if fit["level"] == 0:
            return True
        return len(fit["cells"] & self._occ.get(fit["level"] - 1, set())) >= 2

    # ---- per-step physics hooks
    def pre_step(self, env):
        """Press-fit detent: resistance ramping to snap_force over the last snap_engage of travel."""
        sc = env.scenario.task if env.scenario else None
        if sc is None:
            return
        d = env.data
        for i in range(sc["n"]):
            if self._snapped[i]:
                continue
            b = self.bodies[i]
            pos_p, quat_p = self.plate_frame(env, d.xpos[b], d.xquat[b])
            fit = self.lattice_fit(i, pos_p, quat_p)
            if fit["r_xy"] < 0.004 and 0.0 < fit["r_z"] < self.tc.snap_engage and self.supported(fit):
                f = self.snap_force * (1 - fit["r_z"] / self.tc.snap_engage)
                d.xfrc_applied[b, 2] += f

    def post_step(self, env):
        sc = env.scenario.task if env.scenario else None
        if sc is None:
            return
        d, m, tc = env.data, env.model, self.tc
        for i in range(sc["n"]):
            if self._snapped[i]:
                continue
            b = self.bodies[i]
            pos_p, quat_p = self.plate_frame(env, d.xpos[b], d.xquat[b])
            fit = self.lattice_fit(i, pos_p, quat_p)
            if (fit["r_xy"] <= tc.snap_tol_xy and abs(fit["r_z"]) <= tc.snap_tol_z and abs(fit["r_yaw"]) <= tc.snap_tol_yaw
                    and fit["tilt"] < 0.05 and self.supported(fit)):
                eq = self.snap_eq[i]
                m.eq_data[eq, 3:6] = d.xpos[b]
                m.eq_data[eq, 6:10] = d.xquat[b]
                d.eq_active[eq] = 1
                self._snapped[i] = True
                self._snap_cells[i] = (fit["level"], fit["i"], fit["j"], fit["yaw_q"])
                self._occ.setdefault(fit["level"], set()).update(fit["cells"])

    # ---- snapshots (snap bookkeeping lives outside MjData)
    def aux_size(self):
        return 5 * self.n_bricks

    def get_aux(self, env):
        out = []
        for i in range(self.n_bricks):
            c = self._snap_cells[i] or (-1, -1, -1, -1)
            out += [float(self._snapped[i]), *c]
        return np.asarray(out, float)

    def set_aux(self, env, aux):
        self._occ = {}
        self._snapped, self._snap_cells = [], []
        for i in range(self.n_bricks):
            s, level, ii, jj, yq = aux[5 * i : 5 * i + 5]
            self._snapped.append(bool(s))
            if s:
                self._snap_cells.append((int(level), int(ii), int(jj), int(yq)))
                nx0, ny0 = self.kinds[i]
                nx, ny = (nx0, ny0) if int(yq) % 2 == 0 else (ny0, nx0)
                self._occ.setdefault(int(level), set()).update(self._cells(int(ii), int(jj), nx, ny))
            else:
                self._snap_cells.append(None)

    # ---- labels
    def privileged(self, env):
        sc = env.scenario.task
        d = env.data
        n = sc["n"]
        pos = np.stack([d.xpos[self.bodies[i]] for i in range(n)])
        quat = np.stack([d.xquat[self.bodies[i]] for i in range(n)])
        grasped = [env.grasped_by(self.geom_sets[i]) for i in range(n)]
        fits = [self.lattice_fit(i, *self.plate_frame(env, pos[i], quat[i])) for i in range(n)]
        at_goal = []
        for i in range(n):
            g = sc["goals"][i]
            c = self._snap_cells[i]
            at_goal.append(bool(self._snapped[i] and c is not None and c[0] == g[2] and c[1] == g[0] and c[2] == g[1] and (self.kinds[i][0] == self.kinds[i][1] or c[3] % 2 == g[3] % 2)))
        k = n
        for i in range(n):
            if not at_goal[i]:
                k = i
                break
        goals = np.stack([np.concatenate(self.goal_pose_world(sc, i)) for i in range(n)])
        return dict(brick_pos=pos, brick_quat=quat, grasped=grasped, snapped=list(self._snapped[:n]), at_goal=at_goal, n_done=int(sum(at_goal)),
                    k=k, fits=fits, goals=goals)

    def stages(self):
        names = []
        for k in range(self.n_bricks):
            names += [f"{s}_{k}" for s in SUB]
        return tuple(names) + ("done",)

    def _substage(self, env, pr, k):
        g = pr["grasped"][k]
        p = pr["brick_pos"][k]
        goal = pr["goals"][k, :3]
        fit = pr["fits"][k]
        if g is None:
            return 0
        if np.linalg.norm(p[:2] - goal[:2]) < 0.012 and fit["r_z"] < 0.012:
            return 4
        if np.linalg.norm(p[:2] - goal[:2]) < 0.03:
            return 3
        if p[2] > env.cfg.bench.top_z + self.H / 2 + 0.02:
            return 2
        return 1

    def stage(self, env, pr):
        k = pr["k"]
        if k >= env.scenario.task["n"]:
            return len(self.stages()) - 1
        return len(SUB) * k + self._substage(env, pr, k)

    def progress(self, env, pr):
        n = env.scenario.task["n"]
        k = pr["k"]
        if k >= n:
            return 1.0
        sub = self._substage(env, pr, k)
        return (pr["n_done"] + (0.0, 0.2, 0.45, 0.7, 0.85)[sub]) / n

    def success(self, env, pr):
        return pr["n_done"] == env.scenario.task["n"] and all(g is None for g in pr["grasped"])

    def task_features(self, env, pr):
        """[n, k, progress, grasped, brick pos(3) quat(4), goal pos(3) yaw(1), goal - pos(3), r_xy r_z r_yaw, brick - tcp per arm(6), all brick pos(3 n_bricks)]."""
        sc = env.scenario.task
        n = sc["n"]
        k = min(pr["k"], n - 1)
        p, q = pr["brick_pos"][k], pr["brick_quat"][k]
        gp = pr["goals"][k, :3]
        gyaw = G.yaw_of_quat(pr["goals"][k, 3:])
        fit = pr["fits"][k]
        rel = [p - pr["tcp"][s][0] for s in env.arms] + [np.zeros(3)] * (2 - len(env.arms))
        allp = np.zeros(3 * self.n_bricks)
        allp[: 3 * n] = pr["brick_pos"].reshape(-1)
        return np.concatenate([[n, k, self.progress(env, pr), float(pr["grasped"][k] is not None)], p, q, gp, [gyaw], gp - p,
                               [fit["r_xy"], fit["r_z"], fit["r_yaw"]], *rel, allp]).astype(np.float32)

    def feature_names(self):
        return (["n", "k", "progress", "grasped"] + [f"brick_{a}" for a in "xyz"] + [f"brick_q{a}" for a in "wxyz"] + [f"goal_{a}" for a in "xyz"] + ["goal_yaw"]
                + [f"goal_rel_{a}" for a in "xyz"] + ["r_xy", "r_z", "r_yaw"] + [f"brick_rel_{s}_{a}" for s in ("right", "left") for a in "xyz"]
                + [f"brick{i}_{a}" for i in range(self.n_bricks) for a in "xyz"])

    def record_columns(self, env, pr):
        n = env.scenario.task["n"]
        k = min(pr["k"], n - 1)
        f = pr["fits"][k]
        return dict(n_done=float(pr["n_done"]), r_xy=f["r_xy"], r_z=f["r_z"], r_yaw=f["r_yaw"], tilt=f["tilt"], snapped=float(pr["snapped"][k]))

    def outcome(self, env, record, expert):
        pr = self.privileged(env)
        sc = env.scenario.task
        if env.unstable:
            return "sim_unstable"
        if self.success(env, pr):
            return "none"
        k = pr["k"]
        if any(self._snapped[i] and not pr["at_goal"][i] for i in range(sc["n"])):
            return "wrong_slot"
        if not any(g == k for g in record.get("grasped_k", [])):
            return "grasp_missed"
        if record.get("drops", 0) > 0:
            return "dropped"
        if pr["grasped"][k] is not None:
            return "incomplete"
        fit = pr["fits"][k]
        goal = pr["goals"][k, :3]
        if np.linalg.norm(pr["brick_pos"][k][:2] - goal[:2]) < 0.02 and fit["r_z"] < 0.012:
            return "not_snapped" if (fit["r_xy"] < 0.004 and fit["tilt"] < 0.08) else "misaligned"
        return "misaligned" if np.linalg.norm(pr["brick_pos"][k][:2] - goal[:2]) < 0.05 else "other"

    def make_expert(self, env, failure=None, seed=0, noise=True, bimanual=None, recover=False):
        """Scripted expert. With two arms (and bimanual not False) both arms work: each takes the objects on
        its side of the bench and they take turns at the shared structure."""
        from ..experts.lego import LegoExpert

        if bimanual is None:
            bimanual = len(env.arms) == 2
        if bimanual and len(env.arms) == 2:
            from ..experts.dual import DualArmExpert

            sc = env.scenario.task
            xy = {k: (0.0, self._goal_side(sc["plate"], k, sc["goals"][k])) for k in range(sc["n"])}  # assign by goal side
            return DualArmExpert(env, LegoExpert, xy, failure=failure, seed=seed, noise=noise, recover=recover, ordered=True)
        return LegoExpert(env, failure=failure, seed=seed, noise=noise, recover=recover)

    def sample_failure(self, kind, s, rng, sc):
        if kind == "misalign":
            a = rng.uniform(0, 2 * np.pi)
            r = 0.003 + 0.005 * s
            return dict(dxy=[float(r * np.cos(a)), float(r * np.sin(a))], dyaw=float(rng.normal(0, 0.05 + 0.15 * s)))
        if kind == "partial_press":
            return dict(stop_above=float(0.0015 + 0.003 * s))
        if kind == "wrong_slot":
            return dict(shift=[int(rng.choice([-1, 1])), 0] if rng.random() < 0.5 else [0, int(rng.choice([-1, 1]))])
        return {}


def _forward(env):
    mujoco.mj_forward(env.model, env.data)
