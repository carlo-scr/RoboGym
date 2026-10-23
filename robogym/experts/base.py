"""Scripted experts: privileged, closed-loop "teleoperators" written as generators over Cartesian
primitives. Each primitive yields one 8-D action per control step for the expert's arm (the other arm,
if any, holds its pose). Phases, events and the current subgoal are exposed for labelling, and the
FailureSpec hooks (drop, weak grip, misgrasp, premature release, bump, stall, + task kinds) live in the
primitives so every task expert gets them for free.

An expert is rebuilt deterministically from (env scenario, failure, seed) by replaying, which is how
tools.expert_at() reaches a recorded step for branching.
"""
from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..env import ACT_DIM, Q_DOWN
from ..failures import FailureSpec


def down_quat(yaw):
    """TCP pointing down with the finger axis rotated by `yaw` about world z (yaw 0: fingers along y)."""
    return G.quat_mul(G.yaw_quat(yaw), Q_DOWN)


def grasp_yaw_for(obj_yaw, cur_yaw):
    """Yaw (mod pi/2 of the object's yaw) closest to the gripper's current yaw: the object's faces are
    reachable in 4 equivalent ways, choose the one with the least wrist rotation."""
    cands = obj_yaw + np.arange(-4, 5) * np.pi / 2
    return float(cands[np.argmin(np.abs(G.wrap(cands - cur_yaw)))])


class Expert:
    name = "expert"

    def __init__(self, env, arm: str | None = None, failure: FailureSpec | None = None, seed: int = 0, noise: bool = True):
        self.env = env
        self.failure = failure
        self.rng = np.random.default_rng([seed, 7919])
        self.noise = noise
        self.arm = arm or self.pick_arm()
        self.i_arm = env.arms.index(self.arm)
        r = self.rng
        self.speed = float(r.uniform(0.85, 1.2)) if noise else 1.0
        self.grasp_jitter = r.normal(0, 0.0015, 2) if noise else np.zeros(2)
        self.place_noise = r.normal(0, 0.0015, 2) if noise else np.zeros(2)
        self.approach_h = float(r.uniform(0.06, 0.10)) if noise else 0.08
        self.home_pose = {s: env.targets[s] for s in env.arms}
        self.tp, self.tq = env.targets[self.arm]
        self.grip = 0.0
        self.holding = False  # intends to hold an object (for drop labelling; stays True through an injected drop)
        self.compliant = False
        self.phase = "start"
        self.subgoal = 0
        self.events = []
        self.retries = 0
        self._gen = None
        self._done = False
        self._stalled = False
        self._bumped = False
        self._dropped = False

    # ------------------------------------------------------------------ framework
    def pick_arm(self):
        return self.env.arms[0]

    def plan(self):
        """Generator yielding actions; subclasses implement the task here."""
        raise NotImplementedError

    @property
    def done(self):
        return self._done

    def _f(self, kind):
        return self.failure is not None and self.failure.kind == kind

    def _f_here(self, kind, phase=None):
        """Failure `kind` is injected on the current subgoal (and phase, if the failure has one)."""
        if not self._f(kind):
            return False
        p = self.failure.params
        if p.get("subgoal", 0) != self.subgoal:
            return False
        return phase is None or p.get("phase", phase) == phase

    def event(self, name):
        self.events.append((name, round(self.env.time, 4)))

    def start_phase(self, name):
        self.phase = name
        self.event(name)

    def act(self):
        if self._gen is None:
            self._gen = self.plan()
        try:
            return next(self._gen)
        except StopIteration:
            self._done = True
            self.phase = "done"
            return self._action()

    def _action(self):
        """Action vector that moves the expert's arm target to (tp, tq) with the current grip/compliance."""
        env = self.env
        a = np.zeros((len(env.arms), ACT_DIM))
        p_cur, q_cur = env.pending_targets(self.arm)  # == env.targets without actuation latency
        a[self.i_arm, :3] = self.tp - p_cur
        a[self.i_arm, 3:6] = G.quat_error(self.tq, q_cur)
        a[self.i_arm, 6] = self.grip
        a[self.i_arm, 7] = float(self.compliant)
        for j, s in enumerate(env.arms):  # the other arm holds its grip command
            if j != self.i_arm:
                a[j, 6] = env.grip_cmd[s]
        return a.reshape(-1)

    # ------------------------------------------------------------------ primitives
    def move(self, p1, q1=None, T=None, speed=0.25, rot_speed=1.2, min_T=0.3, compliant=False, phase=None, lead=None,
             push_axis=None, max_lead=0.004, stop=None):
        """Min-jerk Cartesian segment of the TCP target to (p1, q1) over T seconds (T from distance / speed
        unless given). `lead`: a callable (frac) -> extra offset added to the target (servoing hooks).
        `push_axis` (unit, world): force-limited push, the target may lead the measured TCP by at most
        `max_lead` along that axis (bounds the contact force to servo stiffness x max_lead). `stop`: callable
        -> True ends the segment early (e.g. a snap / latch)."""
        env = self.env
        if phase:
            self.start_phase(phase)
        p0, q0 = self.tp.copy(), self.tq.copy()
        p1 = np.asarray(p1, float)
        q1 = q0 if q1 is None else np.asarray(q1, float)
        if T is None:
            T = max(min_T, np.linalg.norm(p1 - p0) / speed, 0.3 + G.quat_angle(q1, q0) / rot_speed)
        T /= self.speed
        n = max(1, int(round(T / env.dt)))
        self.compliant = compliant
        for i in range(1, n + 1):
            u = G.min_jerk(i / n)
            self.tp = p0 + u * (p1 - p0)
            self.tq = G.slerp(q0, q1, u)
            if lead is not None:
                self.tp = self.tp + lead(i / n)
            if push_axis is not None:
                p_tcp, _ = self.tcp()
                ld = (self.tp - p_tcp) @ push_axis
                if ld > max_lead:
                    self.tp = self.tp - (ld - max_lead) * np.asarray(push_axis, float)
            yield from self._maybe_failure_step(i / n)
            yield self._action()
            if stop is not None and stop():
                return

    def hold(self, T, phase=None, compliant=None):
        if phase:
            self.start_phase(phase)
        if compliant is not None:
            self.compliant = compliant
        n = max(1, int(round(T / self.env.dt)))
        for i in range(n):
            yield from self._maybe_failure_step(i / n)
            yield self._action()

    def grasp_dz(self, width, clearance=0.003):
        """TCP height above the support surface for a top-down grasp of an object of `width`."""
        return self.env.grasp_tcp_height(width, clearance)

    def grasp(self, T_max=1.5, phase="grasp", settle_steps=3):
        """Close the gripper and wait until the jaw stops moving (or T_max)."""
        if self._f_here("weak_grip"):
            self.env.set_regime({"grip_force": self.failure.params["force_frac"]})
            self.event("weak_grip")
        self.start_phase(phase)
        self.grip = 1.0
        self.holding = True
        w0 = self.env.grip_width(self.arm)
        hist = []
        for _i in range(max(1, int(T_max / self.env.dt))):
            yield self._action()
            hist.append(self.env.grip_width(self.arm))
            # settled = the jaw has visibly started closing (robust to actuation latency) and then stopped moving
            if len(hist) > settle_steps and hist[-1] < w0 - 0.004 and abs(hist[-1] - hist[-1 - settle_steps]) < 2e-4:
                break

    def release(self, T_max=1.2, phase="release", min_open=0.008):
        """Open the gripper and wait until the jaw has opened by `min_open` and stopped moving (robust to
        actuation latency and slow / weak grippers), or T_max."""
        self.start_phase(phase)
        self.grip = 0.0
        self.holding = False
        w0 = self.env.grip_width(self.arm)
        hist = []
        for _i in range(max(1, int(T_max / self.env.dt))):
            yield self._action()
            hist.append(self.env.grip_width(self.arm))
            if len(hist) > 3 and hist[-1] > w0 + min_open and abs(hist[-1] - hist[-4]) < 3e-4:
                break

    def settle(self, tol=0.002, rot_tol=0.03, T_max=0.8, phase=None):
        """Hold the target until the measured TCP is within `tol` / `rot_tol` of it (the arm has arrived),
        or T_max: makes grasps and insertions robust to slow / weakened servos and latency."""
        if phase:
            self.start_phase(phase)
        for _ in range(max(1, int(T_max / self.env.dt))):
            p, q = self.tcp()
            if np.linalg.norm(self.tp - p) < tol and G.quat_angle(self.tq, q) < rot_tol:
                break
            yield self._action()

    def set_grip(self, g, T=0.5, phase=None):
        """Command the gripper and hold the pose while it moves."""
        if self._f_here("weak_grip") and g > 0.5:
            self.env.set_regime({"grip_force": self.failure.params["force_frac"]})
            self.event("weak_grip")
        self.grip = float(g)
        self.holding = g > 0.5
        yield from self.hold(T, phase=phase)

    def _maybe_failure_step(self, frac):
        """Phase-local failure hooks: bump (external knock) and stall (freeze) at a fraction of the phase."""
        if self._f_here("bump", self.phase) and not self._bumped and frac >= self.failure.params["at_frac"]:
            p = self.failure.params
            self.env.bump(self.arm, p["force"], p["torque"], p["duration_s"])
            self._bumped = True
            self.event("bump")
        if self._f_here("stall", self.phase) and not self._stalled and frac >= self.failure.params["at_frac"]:
            self._stalled = True
            self.event("stall")
            n = int(self.failure.params["duration_s"] / self.env.dt)
            for _ in range(n):
                yield self._action()
            self.event("stall_end")

    def carry(self, p1, q1=None, **kw):
        """Transport with the `drop` failure hook (gripper opens for open_s at at_frac of the segment)."""
        env = self.env
        if not self._f_here("drop") or self._dropped:
            yield from self.move(p1, q1, phase="carry", **kw)
            return
        self.start_phase("carry")
        p0, q0 = self.tp.copy(), self.tq.copy()
        p1 = np.asarray(p1, float)
        q1 = q0 if q1 is None else np.asarray(q1, float)
        T = max(kw.get("min_T", 0.3), np.linalg.norm(p1 - p0) / kw.get("speed", 0.25), 0.3 + G.quat_angle(q1, q0) / 1.2) / self.speed
        n = max(1, int(round(T / env.dt)))
        k0 = int(self.failure.params["at_frac"] * n)
        k1 = k0 + int(self.failure.params["open_s"] / env.dt)
        g_hold = self.grip
        self.compliant = False
        for i in range(1, n + 1):
            u = G.min_jerk(i / n)
            self.tp = p0 + u * (p1 - p0)
            self.tq = G.slerp(q0, q1, u)
            if k0 <= i < k1:
                if i == k0:
                    self.event("drop_open")
                    self._dropped = True
                self.grip = 0.0
            else:
                self.grip = g_hold
            yield self._action()
        # the opening window may outlast the segment: keep the jaw open for the rest of it (the real
        # gripper needs ~0.6 s to open fully), then close again
        for _ in range(max(0, k1 - max(n, k0))):
            self.grip = 0.0
            yield self._action()
        self.grip = g_hold

    def release_height(self):
        """Extra height at which the `premature_release` failure lets go."""
        return self.failure.params["height"] if self._f_here("premature_release") else 0.0

    def misgrasp_offset(self):
        return self.failure.params["offset"] if self._f_here("misgrasp") else 0.0

    def go_home(self, speed=0.3):
        p, q = self.home_pose[self.arm]
        yield from self.move(p, q, speed=speed, phase="home")

    def lift(self, dz, **kw):
        yield from self.move(self.tp + np.array([0, 0, dz]), phase=kw.pop("phase", "lift"), **kw)

    # ------------------------------------------------------------------ measurement helpers
    def tcp(self):
        return self.env.tcp_pose(self.arm)

    def object_in_hand(self, obj_pos, obj_quat):
        """TCP pose expressed in the object's frame (measured): lets the expert place the *object* exactly."""
        p_t, q_t = self.tcp()
        pi, qi = G.pose_inv(obj_pos, obj_quat)
        return G.pose_mul(pi, qi, p_t, q_t)

    def tcp_for_object_pose(self, obj_p, obj_q, rel):
        """TCP pose that puts an object held with hand-object transform `rel` at (obj_p, obj_q)."""
        return G.pose_mul(np.asarray(obj_p, float), np.asarray(obj_q, float), rel[0], rel[1])
