"""Scripted cube expert: for each move, top-down grasp of the cube where it lies (on the bench or on top
of a stack), lift above everything on the bench, carry above the goal, descend, release, retreat.
Re-measures the cube-in-hand transform after grasping so the *cube* lands on the goal, and places it in
the nearest level orientation of the pose it has in the hand (a cube received sideways in a hand-over
still lands flat). Pick and place are separate generators so the hand-over experts can reuse them."""
from __future__ import annotations

import numpy as np

from .. import geometry as G
from .base import Expert, down_quat, grasp_yaw_for


class CubeStackExpert(Expert):
    name = "cube_stack_expert"
    CARRY_CLEARANCE = 0.06

    def pick_arm(self):
        env = self.env
        if len(env.arms) == 1:
            return env.arms[0]
        sc = env.scenario.task
        y = np.mean([p[1] for p in sc["poses"][: sc["n"]]])
        return "right" if y < 0 else "left"

    def plan_moves(self):
        """The move list this expert believes in (the wrong_cube failure swaps two stacking moves)."""
        sc = self.env.scenario.task
        moves = [list(m) for m in sc["moves"]]
        if self._f("wrong_cube"):
            k = max(0, min(self.failure.params.get("subgoal", 0), len(moves) - 2))
            cands = [j for j in range(k, len(moves) - 1) if moves[j][1] >= 0 and moves[j + 1][1] >= 0]
            if cands:
                j = cands[0]
                (c0, s0, _), (c1, s1, _) = moves[j], moves[j + 1]
                moves[j], moves[j + 1] = [c1, s0, -1], [c0, c1 if s1 == c0 else s1, -1]
                self.event("wrong_cube")
        return moves

    def plan(self):
        moves = self.plan_moves()
        for k in self.my_subgoals(range(len(moves))):
            yield from self.subgoal_with_recovery(k, lambda kk: self._do_move(kk, moves))
        self.subgoal = len(moves)
        yield from self.go_home()

    # ------------------------------------------------------------------ pieces
    def scene_top(self, pr, exclude=()):
        """Highest point of anything standing on the bench (except the cubes in `exclude`)."""
        sc = self.env.scenario.task
        top = self.env.cfg.bench.top_z
        return max([top] + [pr["cube_pos"][j][2] + sc["sizes"][j] / 2 for j in range(sc["n"]) if j not in exclude])

    def _pick(self, c):
        """Reach above cube c, descend, grasp (no lift)."""
        env = self.env
        sc = env.scenario.task
        s = sc["sizes"][c]
        pr = env.privileged()
        cp, cq = pr["cube_pos"][c], pr["cube_quat"][c]
        yaw = grasp_yaw_for(G.yaw_of_quat(cq), G.yaw_of_quat(self.tq) + np.pi)
        q_grasp = down_quat(yaw)
        off_axis = G.quat_rotate(q_grasp, [1, 0, 0])  # TCP x: across the fingers (the object's grasp axis)
        jitter = np.array([self.grasp_jitter[0], self.grasp_jitter[1], 0.0])
        grasp_p = np.array([cp[0], cp[1], cp[2] - s / 2 + self.grasp_dz(s)]) + jitter + self.misgrasp_offset() * off_axis
        reach_z = max(grasp_p[2] + self.approach_h, self.scene_top(pr) + 0.05)
        yield from self.move([grasp_p[0], grasp_p[1], reach_z], q_grasp, speed=0.3, phase="reach")
        yield from self.move(grasp_p, q_grasp, speed=0.15, min_T=0.5, phase="descend")
        yield from self.settle()
        yield from self.grasp()

    def goal_of(self, k, moves):
        """Goal pose of move k as this operator believes it (live support pose + its biases)."""
        env = self.env
        sc = env.scenario.task
        c, s, spot = moves[k]
        d = env.data
        pose = {i: (d.xpos[env.task.bodies[i]], d.xquat[env.task.bodies[i]]) for i in range(sc["n"])}
        goal_p, goal_q = env.task._goal_from(dict(sc, moves=moves), k, pose)
        goal_p = goal_p + np.array([*self.place_noise, 0.0])
        if self._f_here("misplace"):
            goal_p = goal_p + np.array([*self.failure.params["dxy"], 0.0])
            self.event("misplace")
        return goal_p, goal_q

    def _place_held(self, k, moves):
        """Lift the held cube above everything, carry it above the goal, descend, release, retreat."""
        env = self.env
        sc = env.scenario.task
        c = moves[k][0]
        s = sc["sizes"][c]
        top = env.cfg.bench.top_z
        pr = env.privileged()
        rel = self.object_in_hand(pr["cube_pos"][c], pr["cube_quat"][c])
        goal_p, goal_q = self.goal_of(k, moves)
        # the cube lands level, keeping the yaw class it has in the hand, aligned to the goal's yaw
        level = G.level_quat(pr["cube_quat"][c])
        yaw = grasp_yaw_for(G.yaw_of_quat(goal_q), G.yaw_of_quat(level))
        goal_q = G.quat_mul(G.yaw_quat(yaw - G.yaw_of_quat(level)), level)
        clearance = self.CARRY_CLEARANCE
        if self._f_here("low_carry"):
            clearance = self.failure.params["clearance"]
            self.event("low_carry")
        z_carry = max(max(self.scene_top(pr, exclude=(c,)), goal_p[2] + s / 2) + clearance + s / 2, top + s / 2 + 0.03)
        lift_p = self.tp + [0, 0, max(0.0, z_carry - pr["cube_pos"][c][2])]
        yield from self.move(lift_p, speed=0.2, min_T=0.4, phase="lift")
        yield from self.wait_turn(k)  # bimanual: the structure is built in order, one arm at a time
        above_p, above_q = self.tcp_for_object_pose(goal_p + [0, 0, z_carry - goal_p[2]], goal_q, rel)
        yield from self.carry(above_p, above_q, speed=0.25, min_T=0.8)
        # descend (re-measure the hand-object transform first: the cube may have shifted)
        pr = env.privileged()
        rel = self.object_in_hand(pr["cube_pos"][c], pr["cube_quat"][c])
        place_p, place_q = self.tcp_for_object_pose(goal_p + [0, 0, 0.002 + self.release_height()], goal_q, rel)
        yield from self.move(place_p, place_q, speed=0.12, min_T=0.6, phase="descend")
        yield from self.settle(tol=0.003, T_max=0.5, phase="place")
        yield from self.release()
        yield from self.move(self.tp + [0, 0, 0.08], speed=0.25, min_T=0.4, phase="retreat")
        yield from self.clear_zone()
        self.turn_done(k)

    def _do_move(self, k, moves):
        self.subgoal = k
        c = moves[k][0]
        if self.coord is not None and self.stacked(c):
            yield from self.wait_turn(k)  # the cube is in the shared structure: taking it off is this arm's turn only
        yield from self._pick(c)
        yield from self._place_held(k, moves)

    def stacked(self, c):
        """Cube c rests on another cube (not on the bench)."""
        env = self.env
        return env.privileged()["cube_pos"][c][2] > env.cfg.bench.top_z + env.scenario.task["sizes"][c] / 2 + 0.01
