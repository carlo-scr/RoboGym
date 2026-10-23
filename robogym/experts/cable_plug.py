"""Scripted cable-plugging expert (port of RobotPluggingSim's state machine): top-down grasp of the
boot, lift, carry to a pre-insertion pose in front of the lit port (plug re-measured in hand), align,
compliant insertion with closed-loop tip servoing and a bounded push, spiral-search retries on a jam,
release, retreat. Repeats for each cable."""
from __future__ import annotations

import numpy as np

from .. import geometry as G
from .base import Expert, down_quat

SEARCH = np.array([[0, 1], [1, 0], [-1, 0], [0, -1], [1, 1], [-1, 1], [1, -1], [-1, -1]], float)


class CablePlugExpert(Expert):
    name = "cable_plug_expert"
    PRE_DIST = 0.020
    OVERSHOOT = 0.003
    MAX_LEAD = 0.005  # force-limited push: ~K x 5 mm, enough for the 10 N latch detent also under some latency
    MAX_RETRIES = 3
    SEARCH_STEP = 0.0006
    JAM_WINDOW = 20  # 1.0 s without depth progress while pushing

    def __init__(self, env, failure=None, seed=0, noise=True, **kw):
        super().__init__(env, failure=failure, seed=seed, noise=noise, **kw)
        r = self.rng
        self.port_noise_yz = r.normal(0, 0.0003, 2) if noise else np.zeros(2)
        self.port_noise_rot = r.normal(0, 0.005, 3) if noise else np.zeros(3)
        self.search = np.zeros(2)
        self.insert_axis = np.array([1.0, 0, 0])

    def pick_arm(self):
        env = self.env
        if len(env.arms) == 1:
            return env.arms[0]
        return "right" if env.scenario.task["cables"][0]["plug_xy"][1] < 0 else "left"

    def _believed_port(self, c, port):
        task = self.env.task
        p_port, q_port = task.port_pose(self.env, port)
        R = G.quat_to_mat(q_port)
        yz = self.port_noise_yz + self.search
        q_err = G.quat_from_rotvec(self.port_noise_rot)
        if self._f_here("misalign"):
            yz = yz + np.asarray(self.failure.params["dyz"])
            q_err = G.quat_mul(G.quat_from_rotvec(self.failure.params["drot"]), q_err)
        return p_port + R @ np.array([0.0, yz[0], yz[1]]), G.quat_mul(q_port, q_err)

    def _pre_insert_goal(self, c, port, dist):
        """TCP pose that puts the plug tip `dist` in front of the believed port, axis aligned."""
        task = self.env.task
        p_port, q_des = self._believed_port(c, port)
        axis = G.quat_rotate(q_des, [1, 0, 0])
        p_plug = p_port - axis * (dist + task.tip_len)
        pl = self.env.privileged()["plugs"][c]
        rel = self.object_in_hand(pl["pos"], pl["quat"])
        self.insert_axis = axis
        return G.pose_mul(p_plug, q_des, rel[0], rel[1])

    def plan(self):
        env, task = self.env, self.env.task
        sc = env.scenario.task
        n = sc["n"]
        top = env.cfg.bench.top_z
        for c in self.my_subgoals(range(n)):
            self.subgoal = c
            port = sc["cables"][c]["port"]
            if self._f_here("wrong_port"):
                port = self.failure.params["port"]
                self.event("wrong_port")
            self.search[:] = 0
            pr = env.privileged()["plugs"][c]
            yaw = G.yaw_of_quat(pr["quat"])
            # TCP x must point along the plug axis (the remote-compliance tool tip is +x of the TCP):
            # TCP x = Rz(yaw_g) (-x_world) = (cos yaw, sin yaw)  ->  yaw_g = yaw + pi
            q_grasp = down_quat(yaw + np.pi)
            ax = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            gxy = pr["grasp"][:2] + (self.grasp_jitter[0] + self.misgrasp_offset()) * ax[:2]
            grasp_p = np.array([gxy[0], gxy[1], top + self.grasp_dz(task.tc.boot_w, clearance=0.0015)])  # fingertips 1.5 mm above the bench
            yield from self.move(grasp_p + [0, 0, self.approach_h], q_grasp, speed=0.3, phase="reach")
            yield from self.move(grasp_p, q_grasp, speed=0.12, min_T=0.5, phase="descend")
            yield from self.settle()
            yield from self.grasp()
            yield from self.lift(0.09, speed=0.2, min_T=0.5)
            yield from self.wait_turn(c)  # bimanual: one arm at the panel at a time
            # carry to the pre-insertion pose (plug re-measured in hand), then a short align hop
            goal = self._pre_insert_goal(c, port, self.PRE_DIST)
            yield from self.carry(goal[0], goal[1], speed=0.22, min_T=1.2)
            goal = self._pre_insert_goal(c, port, self.PRE_DIST)
            yield from self.move(goal[0], goal[1], T=0.7, phase="align")
            yield from self.settle(tol=0.0015, T_max=0.6)
            # insertion with retries
            max_retries = 0 if (self._f_here("misalign") or self._f_here("partial_insert")) else self.MAX_RETRIES
            inserted = False
            for attempt in range(max_retries + 1):
                res = yield from self._insert(c, port)
                if res == "latched":
                    inserted = True
                    break
                if attempt < max_retries and res == "jam":
                    self.retries += 1
                    self.search = self.SEARCH_STEP * np.ceil(self.retries / 8) * SEARCH[(self.retries - 1) % 8]
                    yield from self._backoff(c)
                else:
                    break
            if inserted:
                self.event("latched")
            yield from self.release()
            yield from self.move(self.tp - self.insert_axis * 0.05 + [0, 0, 0.04], speed=0.2, min_T=0.6, phase="retreat")
            yield from self.clear_zone()
            self.turn_done(c)
        self.subgoal = n
        yield from self.go_home()

    def _insert(self, c, port):
        """Two-stage insertion. (A) Stiff, closed-loop approach: steer the plug tip onto the believed port
        axis while moving to the jack face (the servo keeps the lateral error well inside the chamfer).
        (B) Compliant push along the axis with a bounded lead (force-limited), until the latch clicks.
        Returns 'latched' | 'jam' | 'stopped'."""
        env, task, tc = self.env, self.env.task, self.env.task.tc
        self.start_phase("insert")
        pr = env.privileged()["plugs"][c]
        q0 = self.tq.copy()
        servo_p, servo_r = np.zeros(3), np.zeros(3)
        # ---- (A) approach to the face, stiff
        self.compliant = False
        p_start = self.tp.copy()
        to_face = -pr["depth"] - 0.0008  # tip 0.8 mm in front of the face
        n_a = max(4, int(round(max(0.5, to_face / 0.02) / self.speed / env.dt)))
        for i in range(1, n_a + 1):
            u = G.min_jerk(i / n_a)
            pr = env.privileged()["plugs"][c]
            if task.latched[c] >= 0:
                return "latched"
            p_port, _ = self._believed_port(c, port)
            e = p_port - pr["tip"]
            e -= (e @ self.insert_axis) * self.insert_axis
            servo_p = np.clip(servo_p + 0.6 * e, -0.004, 0.004)
            plug_ax = G.quat_rotate(pr["quat"], [1, 0, 0])
            servo_r = np.clip(servo_r + 0.3 * np.cross(plug_ax, self.insert_axis), -0.06, 0.06)
            self.tp = p_start + u * to_face * self.insert_axis + servo_p
            self.tq = G.quat_mul(G.quat_from_rotvec(servo_r), q0)
            yield from self._maybe_failure_step(0.5 * i / n_a)
            yield self._action()
            if pr["depth"] > -0.0015:
                break
        # ---- (B) compliant push
        self.compliant = True
        pr = env.privileged()["plugs"][c]
        frac = self.failure.params["depth_frac"] if self._f_here("partial_insert") else 1.0
        extra = 0.0 if self._f_here("partial_insert") else self.OVERSHOOT
        to_go = tc.seat_depth * frac + extra - pr["depth"]
        p0 = self.tp.copy()
        p1 = p0 + self.insert_axis * to_go
        T = 1.2 * max(0.5, to_go / 0.02) / self.speed
        n = max(1, int(round(T / env.dt)))
        depth_hist = []
        for i in range(1, n + 1 + int(0.4 / env.dt)):  # segment + hold
            u = G.min_jerk(min(1.0, i / n))
            pr = env.privileged()["plugs"][c]
            if task.latched[c] >= 0:
                return "latched"
            self.tp = p0 + u * (p1 - p0)
            p_tcp, _ = self.tcp()
            lead = (self.tp - p_tcp) @ self.insert_axis
            if lead > self.MAX_LEAD:  # force-limited push
                self.tp = self.tp - (lead - self.MAX_LEAD) * self.insert_axis
            yield from self._maybe_failure_step(0.5 + 0.5 * min(1.0, i / n))
            yield self._action()
            if pr["grasped_by"] != self.arm:
                return "stopped"
            depth_hist.append(pr["depth"])
            h = depth_hist[-self.JAM_WINDOW :]
            if len(h) == self.JAM_WINDOW and max(h) - h[0] < 0.0002 and not self._f_here("partial_insert"):
                self.event("jam")
                return "jam"
        return "latched" if task.latched[c] >= 0 else ("stopped" if self._f_here("partial_insert") else "jam")

    def _backoff(self, c):
        """Pull the tip ~6 mm in front of the face (stiffly, so the next approach starts from a clean pose)."""
        pr = self.env.privileged()["plugs"][c]
        self.start_phase("backoff")
        goal = self.tp - self.insert_axis * (pr["depth"] + 0.006 + self.MAX_LEAD)
        yield from self.move(goal, T=0.4, compliant=False)
