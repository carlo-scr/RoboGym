"""Bimanual hand-over: the *giver* picks each cube on its side, carries it to a staging pose, and when
the *receiver* is ready, presents it in mid-air with its tool axis horizontal (pointing at the receiver,
fingers along world x). The receiver grasps the presented cube top-down with its fingers along world y
(its pads land on the cube's former top and bottom faces, between the giver's finger links), the giver
lets go and backs out along its tool axis, and the receiver places the cube on the stack. The giver
pre-picks the next cube while the receiver places.

Both arms are CubeStackExperts driven by a tiny shared protocol (Handoff) and merged like DualArmExpert.
Geometry: the receiver's finger has to fit between the giver's finger links (cube width + 2 pad
thicknesses apart) and under the giver's palm (~4 cm above the cube's former top face), which the xArm
gripper does for cubes >= 3.5 cm."""
from __future__ import annotations

import numpy as np

from .. import geometry as G
from ..env import ACT_DIM
from .base import down_quat
from .cube_stack import CubeStackExpert

READY, PRESENTED, TAKEN, RELEASED = 1, 2, 3, 4
GIVER_FAILURES = ("misgrasp", "weak_grip", "drop", "stall")  # the rest hit the receiver (placing arm)


class Handoff:
    def __init__(self):
        self.state = {}

    def get(self, k):
        return self.state.get(k, 0)

    def set(self, k, v):
        self.state[k] = max(self.state.get(k, 0), v)


class Giver(CubeStackExpert):
    name = "handover_giver"

    def __init__(self, env, arm, handoff, point, moves, **kw):
        super().__init__(env, arm=arm, **kw)
        self.handoff, self.point, self.moves = handoff, np.asarray(point, float), moves
        self.side = 1.0 if arm == "left" else -1.0  # world-y sign of this arm's side

    def plan(self):
        env = self.env
        for k in range(len(self.moves)):
            c = self.moves[k][0]
            self.subgoal = k
            yield from self._pick(c)
            pr = env.privileged()
            yield from self.move(self.tp + [0, 0, max(0.0, self.point[2] + 0.05 - pr["cube_pos"][c][2])], speed=0.2, min_T=0.4, phase="lift")
            stage_p = np.array([self.point[0], self.side * 0.16, self.point[2] + 0.05])
            yield from self.carry(stage_p, speed=0.25, min_T=0.6)
            yield from self.wait_until(lambda k=k: self.handoff.get(k) >= READY, phase="stage")
            # present: cube centre at the hand-over point, tool axis horizontal towards the receiver, fingers along x
            pr = env.privileged()
            rel = self.object_in_hand(pr["cube_pos"][c], pr["cube_quat"][c])
            psi = G.nearest_angle([np.pi / 2, -np.pi / 2], G.yaw_of_quat(self.tq) + np.pi)
            q_present = G.quat_mul(G.quat_from_axis_angle([1, 0, 0], -self.side * np.pi / 2), down_quat(psi))
            q_cube = G.quat_mul(q_present, G.quat_conj(rel[1]))
            p_t, q_t = self.tcp_for_object_pose(self.point, q_cube, rel)
            yield from self.carry(p_t, q_t, speed=0.2, min_T=1.0)
            yield from self.settle(tol=0.003, T_max=0.6, phase="present")
            self.handoff.set(k, PRESENTED)
            yield from self.wait_until(lambda k=k: self.handoff.get(k) >= TAKEN, phase="present")
            yield from self.release()
            yield from self.move(self.tp + np.array([0, self.side * 0.12, 0.0]), speed=0.2, min_T=0.5, phase="retreat")
            self.handoff.set(k, RELEASED)
            yield from self.move(self.tp + [0, 0, 0.05], down_quat(psi), speed=0.25, min_T=0.5, phase="retreat")
        self.subgoal = len(self.moves)
        yield from self.go_home()


class Receiver(CubeStackExpert):
    name = "handover_receiver"

    def __init__(self, env, arm, handoff, point, moves, **kw):
        super().__init__(env, arm=arm, **kw)
        self.handoff, self.point, self.moves = handoff, np.asarray(point, float), moves
        self.side = 1.0 if arm == "left" else -1.0

    def plan(self):
        for k in range(len(self.moves)):
            yield from self.subgoal_with_recovery(k, self._receive_and_place, retry=lambda kk: self._do_move(kk, self.moves))
        self.subgoal = len(self.moves)
        yield from self.go_home()

    def _receive_and_place(self, k):
        env = self.env
        c = self.moves[k][0]
        self.subgoal = k
        yaw = G.nearest_angle([0.0, np.pi], G.yaw_of_quat(self.tq) + np.pi)  # fingers along world y
        q = down_quat(yaw)
        ready_p = self.point + np.array([0.0, self.side * 0.12, 0.10])
        yield from self.move(ready_p, q, speed=0.3, phase="ready")
        self.handoff.set(k, READY)
        yield from self.wait_until(lambda: self.handoff.get(k) >= PRESENTED, phase="wait")
        pr = env.privileged()
        cp = pr["cube_pos"][c]
        yield from self.move([cp[0], cp[1], cp[2] + 0.08], q, speed=0.25, min_T=0.4, phase="reach")
        yield from self.move([cp[0], cp[1], cp[2] + 0.004], q, speed=0.1, min_T=0.5, phase="descend")
        yield from self.settle(tol=0.002, T_max=0.5)
        yield from self.grasp()
        self.handoff.set(k, TAKEN)
        yield from self.wait_until(lambda: self.handoff.get(k) >= RELEASED, phase="grasp")
        yield from self._place_held(k, self.moves)


class HandoverExpert:
    """Two experts on one Handoff; phases / events / retries of both arms are exposed for labelling."""

    name = "handover_expert"

    def __init__(self, env, failure=None, seed=0, noise=True, recover=False):
        sc = env.scenario.task
        self.env = env
        self.arms = tuple(env.arms)
        ho = sc["handover"]
        self.giver_arm, self.receiver_arm = ho["giver"], ho["receiver"]
        self.handoff = Handoff()
        self.failure = failure
        planner = CubeStackExpert(env, arm=self.receiver_arm, failure=failure, seed=seed, noise=False)
        moves = planner.plan_moves()  # one shared move list (wrong_cube swaps it for both arms)
        f_g = failure if (failure is not None and failure.kind in GIVER_FAILURES) else None
        f_r = failure if (failure is not None and failure.kind not in GIVER_FAILURES) else None
        self.subs = {
            self.giver_arm: Giver(env, self.giver_arm, self.handoff, ho["point"], moves, failure=f_g, seed=seed, noise=noise),
            self.receiver_arm: Receiver(env, self.receiver_arm, self.handoff, ho["point"], moves, failure=f_r, seed=seed + 31, noise=noise, recover=recover),
        }
        self.subs[self.receiver_arm].events += planner.events  # the wrong_cube event
        self.arm = "both"

    @property
    def done(self):
        return all(ex.done for ex in self.subs.values())

    @property
    def phase(self):
        return "|".join(f"{a[0]}:{self.subs[a].phase}" for a in self.arms)

    @property
    def subgoal(self):
        return self.subs[self.receiver_arm].subgoal

    @property
    def events(self):
        ev = [(f"{a}:{n}", t) for a, ex in self.subs.items() for n, t in ex.events]
        return sorted(ev, key=lambda e: e[1])

    @property
    def retries(self):
        return sum(ex.retries for ex in self.subs.values())

    @property
    def holding(self):
        return any(ex.holding for ex in self.subs.values())

    def holding_of(self, arm):
        return self.subs[arm].holding

    def grip_of(self, arm):
        return self.subs[arm].grip

    def act(self):
        a = np.zeros((len(self.arms), ACT_DIM))
        for i, arm in enumerate(self.arms):
            ex = self.subs[arm]
            a[i] = ex.act().reshape(len(self.arms), ACT_DIM)[ex.i_arm]
        return a.reshape(-1)
