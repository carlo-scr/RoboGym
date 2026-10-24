"""Scripted LEGO expert: grasp each brick across its short side, carry it above its goal cell, align
the *brick* (closed loop on the measured hand-brick transform), press down compliantly until it snaps
(retrying with a small re-alignment if it jams on the studs), release, retreat."""
from __future__ import annotations

import numpy as np

from .. import geometry as G
from .base import Expert, down_quat


class LegoExpert(Expert):
    name = "lego_expert"
    MAX_RETRIES = 2

    def __init__(self, env, failure=None, seed=0, noise=True, **kw):
        super().__init__(env, failure=failure, seed=seed, noise=noise, **kw)
        self.place_noise = self.place_noise / 3.0  # a careful operator: ~0.5 mm placement noise

    def pick_arm(self):
        env = self.env
        if len(env.arms) == 1:
            return env.arms[0]
        return "right" if env.scenario.task["plate"][1] < 0 else "left"

    def _blocked_axes(self, k):
        """Plate-frame axes (0 = x, 1 = y) along which a neighbouring brick (same level or higher) touches
        brick k's goal cells: the finger pads must not point that way."""
        task, sc = self.env.task, self.env.scenario.task
        gi, gj, lvl, yq = sc["goals"][k]
        nx0, ny0 = task.kinds[k]
        nx, ny = (nx0, ny0) if yq % 2 == 0 else (ny0, nx0)
        cells = task._cells(gi, gj, nx, ny)
        occ = set()
        for j in range(sc["n"]):
            if j == k:
                continue
            i2, j2, l2, y2 = sc["goals"][j]
            if l2 < lvl:
                continue
            a0, b0 = task.kinds[j]
            na, nb = (a0, b0) if y2 % 2 == 0 else (b0, a0)
            occ |= task._cells(i2, j2, na, nb)
        blocked = set()
        for (a, b) in cells:
            if (a + 1, b) in occ or (a - 1, b) in occ:
                blocked.add(0)
            if (a, b + 1) in occ or (a, b - 1) in occ:
                blocked.add(1)
        return blocked

    def _grasp_yaw(self, k, brick_yaw, cur_yaw):
        """Grasp yaw: TCP x (across the fingers) along the brick's long axis for 2x4s; for square bricks any
        quarter turn, preferring a finger axis that will not hit a neighbouring brick at the goal."""
        task, sc = self.env.task, self.env.scenario.task
        nx, ny = task.kinds[k]
        if nx != ny:
            # fingers (TCP y) across the short side. Brick dims: nx along local x, ny along local y.
            # TCP x = Rz(c)(-x_world): c = brick_yaw puts TCP x along the brick's x axis (fingers along y).
            base = brick_yaw + (np.pi / 2 if nx < ny else 0.0)  # long axis along y -> fingers must be along x
            cands = base + np.array([0.0, np.pi])
            return float(cands[np.argmin(np.abs(G.wrap(cands - cur_yaw)))])
        blocked = self._blocked_axes(k)
        plate_yaw = sc["plate"][2]
        # the brick is placed with the lattice yaw closest to its current yaw, so the gripper rotation
        # between grasp and goal is the small residual below
        gyaw = G.yaw_of_quat(task.goal_pose_world(sc, k)[1])
        gyaw = gyaw + (np.pi / 2) * np.round((brick_yaw - gyaw) / (np.pi / 2))
        rot = gyaw - brick_yaw
        cands = brick_yaw + np.arange(-4, 5) * np.pi / 2
        best, best_cost = None, np.inf
        for c in cands:
            fy = c + np.pi / 2 + rot - plate_yaw  # finger axis (TCP y) in the plate frame at the goal
            axis = 0 if abs(np.cos(fy)) > abs(np.sin(fy)) else 1
            cost = (10.0 if axis in blocked else 0.0) + abs(G.wrap(c - cur_yaw))
            if cost < best_cost:
                best, best_cost = float(c), cost
        return best

    def plan(self):
        env, _task = self.env, self.env.task
        sc = env.scenario.task
        n = sc["n"]
        for k in self.my_subgoals(range(n)):
            yield from self.subgoal_with_recovery(k, self._place_brick)
        self.subgoal = n
        yield from self.go_home()

    def _place_brick(self, k):
        env, task = self.env, self.env.task
        sc = env.scenario.task
        top = env.cfg.bench.top_z
        H = task.H
        if True:
            self.subgoal = k
            pr = env.privileged()
            bp, bq = pr["brick_pos"][k], pr["brick_quat"][k]
            yaw = self._grasp_yaw(k, G.yaw_of_quat(bq), G.yaw_of_quat(self.tq) - np.pi)
            q_grasp = down_quat(yaw)
            off_axis = G.quat_rotate(q_grasp, [1, 0, 0])
            grasp_p = np.array([bp[0], bp[1], bp[2] - H / 2 + self.grasp_dz(min(task.dims(k)))]) + np.array([*self.grasp_jitter, 0.0]) + self.misgrasp_offset() * off_axis
            struct_top = top + task.tc.plate_h + (max(task._occ) + 1 if task._occ else 0) * H
            reach_z = max(grasp_p[2] + self.approach_h, struct_top + 0.06)
            yield from self.move([grasp_p[0], grasp_p[1], reach_z], q_grasp, speed=0.3, phase="reach")
            yield from self.move(grasp_p, q_grasp, speed=0.15, min_T=0.5, phase="descend")
            yield from self.settle()
            yield from self.grasp()
            # goal (world), with the failure biases the operator believes in
            gp, gq = task.goal_pose_world(sc, k)
            if self._f_here("wrong_slot"):
                sx, sy = self.failure.params["shift"]
                gp = gp + G.quat_rotate(G.yaw_quat(sc["plate"][2]), [sx * task.P, sy * task.P, 0.0])
                self.event("wrong_slot")
            gp = gp + np.array([*self.place_noise, 0.0])
            if self._f_here("misalign"):
                gp = gp + np.array([*self.failure.params["dxy"], 0.0])
                gq = G.quat_mul(G.yaw_quat(self.failure.params["dyaw"]), gq)
                self.event("misalign")
            # the brick's yaw at the goal: the goal yaw class closest to the current brick yaw (square bricks: any quarter turn)
            pr = env.privileged()
            byaw = G.yaw_of_quat(pr["brick_quat"][k])
            nx, ny = task.kinds[k]
            period = np.pi / 2 if nx == ny else np.pi
            gyaw = G.yaw_of_quat(gq)
            gyaw = gyaw + period * np.round((byaw - gyaw) / period)
            gq = G.yaw_quat(gyaw)
            rel = self.object_in_hand(pr["brick_pos"][k], pr["brick_quat"][k])
            z_carry = max(struct_top, gp[2] + H / 2) + 0.05 + H / 2
            yield from self.move(self.tp + [0, 0, max(0.0, z_carry - pr["brick_pos"][k][2])], speed=0.2, min_T=0.4, phase="lift")
            yield from self.wait_turn(k)  # bimanual: bricks go on in build order, one arm at the plate at a time
            struct_top = top + task.tc.plate_h + (max(task._occ) + 1 if task._occ else 0) * H
            z_carry = max(struct_top, gp[2] + H / 2) + 0.05 + H / 2
            above_p, above_q = self.tcp_for_object_pose(gp + [0, 0, z_carry - gp[2]], gq, rel)
            yield from self.carry(above_p, above_q, speed=0.25, min_T=0.8)
            # align just above the studs, closed loop on the measured brick pose
            snapped = False
            for _attempt in range(self.MAX_RETRIES + 1):
                # align just above the studs: closed loop on the measured brick pose (the brick may sit
                # slightly differently in the hand after every move)
                hover = gp + [0, 0, task.tc.stud_r + task.tc.stud_cyl_h + 0.005]
                for it in range(3):
                    pr = env.privileged()
                    rel = self.object_in_hand(pr["brick_pos"][k], pr["brick_quat"][k])
                    p_al, q_al = self.tcp_for_object_pose(hover, gq, rel)
                    yield from self.move(p_al, q_al, speed=0.08, min_T=0.5 if it == 0 else 0.25, phase="align")
                    yield from self.settle(tol=0.0015, T_max=0.5)
                    err = np.linalg.norm(env.privileged()["brick_pos"][k][:2] - hover[:2])
                    if err < 0.0007:
                        break
                # press: compliant (soft laterally, stiff along the approach), force-limited push below the seat
                pr = env.privileged()
                rel = self.object_in_hand(pr["brick_pos"][k], pr["brick_quat"][k])
                stop_above = self.failure.params["stop_above"] if self._f_here("partial_press") else -0.005
                p_pr, q_pr = self.tcp_for_object_pose(gp + [0, 0, stop_above], gq, rel)
                yield from self.move(p_pr, q_pr, speed=0.03, min_T=0.6, compliant=True, phase="press",
                                     push_axis=np.array([0, 0, -1.0]), max_lead=0.008, stop=lambda k=k: task._snapped[k])
                for _ in range(int(0.8 / env.dt)):
                    if task._snapped[k]:
                        break
                    yield self._action()
                snapped = bool(task._snapped[k])
                if snapped or self._f_here("partial_press") or self._f_here("misalign") or self._f_here("wrong_slot"):
                    break
                self.retries += 1
                self.event("jam")
                yield from self.move(self.tp + [0, 0, task.tc.stud_r + 0.01], speed=0.08, min_T=0.4, phase="backoff")
            if snapped:
                self.event("snap")
            yield from self.release()
            yield from self.move(self.tp + [0, 0, 0.07], speed=0.25, min_T=0.4, phase="retreat")
            yield from self.clear_zone()
            self.turn_done(k)
