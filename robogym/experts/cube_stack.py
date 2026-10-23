"""Scripted cube-stacking expert: for each cube in the order, top-down grasp, lift, carry above the
stack, descend, release, retreat; re-measures the cube-in-hand transform after grasping so the *cube*
lands on the goal."""
from __future__ import annotations

import numpy as np

from .. import geometry as G
from .base import Expert, down_quat, grasp_yaw_for


class CubeStackExpert(Expert):
    name = "cube_stack_expert"

    def pick_arm(self):
        env = self.env
        if len(env.arms) == 1:
            return env.arms[0]
        sc = env.scenario.task
        y = np.mean([p[1] for p in sc["poses"][: sc["n"]]])
        return "right" if y < 0 else "left"

    def plan(self):
        env = self.env
        sc = env.scenario.task
        n = sc["n"]
        order = list(sc["order"])
        if self._f("wrong_cube") and n >= 3:
            k = max(1, self.failure.params.get("subgoal", 0))
            k = min(k, n - 2)
            order[k], order[k + 1] = order[k + 1], order[k]
            self.event("wrong_cube")
        top = env.cfg.bench.top_z
        for k in range(1, n):
            self.subgoal = k - 1
            ci = order[k]
            s = sc["sizes"][ci]
            pr = env.privileged()
            cp, cq = pr["cube_pos"][ci], pr["cube_quat"][ci]
            yaw = grasp_yaw_for(G.yaw_of_quat(cq), G.yaw_of_quat(self.tq) + np.pi)
            q_grasp = down_quat(yaw)
            off_axis = G.quat_rotate(q_grasp, [1, 0, 0])  # TCP x: across the fingers (the object's grasp axis)
            jitter = np.array([self.grasp_jitter[0], self.grasp_jitter[1], 0.0])
            grasp_p = np.array([cp[0], cp[1], cp[2] - s / 2 + self.grasp_dz(s)]) + jitter + self.misgrasp_offset() * off_axis
            # reach above, descend, grasp
            stack_top_now = max(pr["cube_pos"][j][2] + sc["sizes"][j] / 2 for j in range(n))
            reach_z = max(grasp_p[2] + self.approach_h, stack_top_now + 0.05)
            yield from self.move([grasp_p[0], grasp_p[1], reach_z], q_grasp, speed=0.3, phase="reach")
            yield from self.move(grasp_p, q_grasp, speed=0.15, min_T=0.5, phase="descend")
            yield from self.settle()
            yield from self.grasp()
            # lift above everything that stands on the bench
            pr = env.privileged()
            rel = self.object_in_hand(pr["cube_pos"][ci], pr["cube_quat"][ci])
            goal_p = np.asarray(sc["goals"][k][:3]) + np.array([*self.place_noise, 0.0])
            if self._f_here("misplace"):
                goal_p = goal_p + np.array([*self.failure.params["dxy"], 0.0])
                self.event("misplace")
            goal_q = G.quat_mul(G.yaw_quat(G.yaw_of_quat(cq) - G.yaw_of_quat(pr["cube_quat"][ci])), pr["cube_quat"][ci])
            # keep the cube's current yaw, but as a pure yaw (level) orientation
            goal_q = G.yaw_quat(grasp_yaw_for(G.yaw_of_quat(np.asarray(sc["goals"][k][3:])), G.yaw_of_quat(pr["cube_quat"][ci])))
            stack_top = max(top + s, goal_p[2] + s / 2)
            clearance = 0.06
            if self._f_here("low_carry"):
                clearance = self.failure.params["clearance"]
                self.event("low_carry")
            z_carry = max(stack_top + clearance + s / 2, top + s / 2 + 0.03)
            lift_p = self.tp + [0, 0, max(0.0, z_carry - pr["cube_pos"][ci][2])]
            yield from self.move(lift_p, speed=0.2, min_T=0.4, phase="lift")
            # carry: move the *cube* to above the goal
            above_p, above_q = self.tcp_for_object_pose(goal_p + [0, 0, z_carry - goal_p[2]], goal_q, rel)
            yield from self.carry(above_p, above_q, speed=0.25, min_T=0.8)
            # descend onto the stack (re-measure the hand-object transform first: the cube may have shifted)
            pr = env.privileged()
            rel = self.object_in_hand(pr["cube_pos"][ci], pr["cube_quat"][ci])
            place_p, place_q = self.tcp_for_object_pose(goal_p + [0, 0, 0.002 + self.release_height()], goal_q, rel)
            yield from self.move(place_p, place_q, speed=0.12, min_T=0.6, phase="descend")
            yield from self.settle(tol=0.003, T_max=0.5, phase="place")
            yield from self.release()
            yield from self.move(self.tp + [0, 0, 0.08], speed=0.25, min_T=0.4, phase="retreat")
        self.subgoal = n - 1
        yield from self.go_home()
