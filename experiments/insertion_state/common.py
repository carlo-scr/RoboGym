"""Shared pieces for the insertion contact-state experiments (see notes/INSERTION_STATE.md).

Everything is expressed in the *believed* port frame: x along the insertion axis (into the panel),
y / z lateral. The believed port is the true port shifted by a perception error `err_yz`; the robot
only knows the believed frame, its own kinematics (commanded and measured TCP), the wrist wrench and
the joint torques. Privileged labels (depth, lateral error, contacts) are logged next to them.
"""
from __future__ import annotations

import mujoco
import numpy as np

from robogym import geometry as G
from robogym.env import ACT_DIM, RoboGymEnv
from robogym.scene import box
from robogym.tasks.cable_plug import CablePlug

MAX_LEAD = 0.005  # force-limited push, as the expert (~servo stiffness x 5 mm)
CATS = ("plug_face", "plug_wedge", "plug_back", "plug_occ", "grip_occ", "grip_panel", "cable_panel")
HR_EVERY = 5  # wrench sample every 5 physics steps of 2 ms: 100 Hz (assumed F/T stream rate)
JACKS = {  # jack variants (task config overrides); "tight" is closer to a real RJ45 jack
    "nominal": {},
    "tight": {"task.clearance": 0.0006, "task.chamfer": 0.001},
}

# Sim fix (experiment-local): the face strips only cover +-rows*row_pitch/2 and the housing behind them is
# visual, so a plug that skids off the strips can pass *under* the panel and register as inserted.
# Close the face below the bottom row with a collision cap. Not above the top row: there the visual housing
# is what the top-down gripper body passes through when plugging the bottom row (a real clearance problem
# of the cell geometry, see notes/INSERTION_STATE.md), and a cap there stops bottom-row plugs 0.4 mm short.
_panel_xml_orig = CablePlug._panel_xml


def _panel_xml_solid(self):
    xml, assets = _panel_xml_orig(self)
    tc = self.tc
    t, hw = tc.panel_thickness, tc.panel_half_width
    face_h = tc.rows * tc.row_pitch / 2
    cap = ((tc.rows * tc.row_pitch + 0.02) / 2 - face_h) / 2
    return xml + "\n" + box("panel_cap_bot", (t / 2, 0, -(face_h + cap)), (t / 2, hw, cap), cls="port"), assets


CablePlug._panel_xml = _panel_xml_solid


def make_env(overrides=None):
    return RoboGymEnv("cable_plug", render=False, **(overrides or {}))


def prepare(env, seed, variant=None):
    """Run the expert until it starts the insertion; return (expert, snapshot) or None if it never gets there."""
    env.reset(seed=seed, variant=dict({"n": 1}, **(variant or {})))
    ex = env.task.make_expert(env, seed=seed)
    while not ex.done:
        a = ex.act()
        if ex.phase == "insert":
            return ex, env.get_state()
        env.step(a)
    return None


class Probe:
    """Geometry + contact bookkeeping for one env (geom-id sets are built once)."""

    def __init__(self, env, c=0):
        self.env, self.c = env, c
        m, task = env.model, env.task
        name = lambda g: m.geom(g).name or ""  # noqa: E731
        self.plug = set(task.plug_geoms[c])
        self.cable = {g for g in range(m.ngeom) if name(g).startswith(f"cable{c}_")}
        self.occ = {g for g in range(m.ngeom) if name(g).startswith("occ")}
        self.face = {g for g in task.panel_geoms if name(g).startswith("panel")}
        self.wedge = {g for g in task.panel_geoms if name(g).startswith("port") and not name(g).endswith("_back")}
        self.back = {g for g in task.panel_geoms if name(g).endswith("_back")}
        self.panel = set(task.panel_geoms)
        arm = env.arms[0]
        pre = arm[0] + "_"
        self.grip = {g for g in range(m.ngeom) if name(g).startswith(pre) and g not in env.arm[arm]["all_pads"]}
        self.pads = set(env.arm[arm]["all_pads"])
        self.arm = arm
        self.ft_site = env.arm[arm]["ft"]
        self._f6 = np.zeros(6)
        # high-rate wrench stream: sample inside env.step's physics loop
        self.hr, self._n_phys = [], 0
        physics_step = env._physics_step

        def logged(control=None):
            physics_step(control)
            self._n_phys += 1
            if self._n_phys % HR_EVERY == 0:
                self.hr.append(self.wrench_world())

        env._physics_step = logged

    def contact_forces(self):
        """Summed normal force per contact category."""
        m, d = self.env.model, self.env.data
        out = dict.fromkeys(CATS, 0.0)
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = con.geom1, con.geom2
            mujoco.mj_contactForce(m, d, i, self._f6)
            f = abs(self._f6[0])
            for a, b in ((g1, g2), (g2, g1)):
                if a in self.plug:
                    if b in self.face:
                        out["plug_face"] += f
                    elif b in self.wedge:
                        out["plug_wedge"] += f
                    elif b in self.back:
                        out["plug_back"] += f
                    elif b in self.occ:
                        out["plug_occ"] += f
                elif a in self.grip:
                    if b in self.occ:
                        out["grip_occ"] += f
                    elif b in self.panel:
                        out["grip_panel"] += f
                elif a in self.cable and (b in self.panel or b in self.occ):
                    out["cable_panel"] += f
        return out

    def wrench_world(self):
        w = self.env.wrench(self.arm)
        R = self.env.data.site_xmat[self.ft_site].reshape(3, 3)
        return np.concatenate([R @ w[:3], R @ w[3:]])

    def joint_torques(self):
        return self.env.data.actuator_force[self.env.arm[self.arm]["act"]].copy()


def true_port_frame(env, port):
    p, q = env.task.port_pose(env, port)
    return p, G.quat_to_mat(q)


class Attempt:
    """Non-privileged blind insertion from a snapshot, logging signals and labels at every control step.

    Phases: 'pre' (stiff: put the believed tip on the believed axis, `pre_x` in front of the believed
    face), 'push' (compliant: drive the tip target to seat depth + overshoot with a force-limited
    lead), 'hold'. Stops early on the latch click (observable on a real cable).
    """

    def __init__(self, env, probe, snap, port, err=(0.0, 0.0, 0.0), pre_x=-0.006, speed=0.02, overshoot=0.003, hold_s=0.5):
        self.env, self.pb, self.snap = env, probe, snap
        self.err = np.asarray(err, float)  # believed port - true port (port frame: axial, y, z)
        self.port = port
        self.pre_x, self.speed, self.overshoot, self.hold_s = pre_x, speed, overshoot, hold_s
        self.rows = []

    # -- frames
    def setup(self):
        env = self.env
        env.set_state(self.snap)
        p_port, R = true_port_frame(env, self.port)
        self.R = R  # orientation assumed known
        self.p_bel = p_port + R @ self.err
        pr = env.privileged()["plugs"][self.pb.c]
        self.tip0 = pr["tip"].copy()  # in-hand estimate (re-measured before insertion, like the expert)
        self.tgt0, self.q0 = env.pending_targets(self.pb.arm)
        self.grip = env.grip_cmd[self.pb.arm]
        self.tare = self.pb.wrench_world()
        self.tc = env.task.tc
        self.n_hr = env.n_sub // HR_EVERY
        self.pb.hr.clear()
        self.pb._n_phys = 0

    def to_bel(self, p):
        return self.R.T @ (np.asarray(p) - self.p_bel)

    def from_bel(self, x):
        return self.p_bel + self.R @ np.asarray(x)

    def tcp_target_for_tip(self, tip_bel):
        """TCP target that would put the tip at `tip_bel` (believed frame) if tracking were perfect."""
        return self.tgt0 + (self.from_bel(tip_bel) - self.tip0)

    def action(self, tp, compliant):
        env = self.env
        a = np.zeros(ACT_DIM * len(env.arms))
        p_cur, q_cur = env.pending_targets(self.pb.arm)
        a[:3] = tp - p_cur
        a[3:6] = G.quat_error(self.q0, q_cur)
        a[6] = self.grip
        a[7] = float(compliant)
        return a

    def step(self, tp, compliant, phase):
        self.env.step(self.action(tp, compliant))
        self.log(phase, tp)

    def log(self, phase, tp_cmd):
        env, pb = self.env, self.pb
        pr = env.privileged()["plugs"][pb.c]
        p_tcp, _ = env.tcp_pose(pb.arm)
        w = pb.wrench_world() - self.tare
        k = pr["nearest_port"]
        p_port, R = true_port_frame(env, self.port)
        rel = R.T @ (pr["tip"] - p_port)
        hr = np.array(pb.hr[-self.n_hr :]) if pb.hr else np.tile(w + self.tare, (self.n_hr, 1))
        hr = np.concatenate([np.repeat(hr[:1], self.n_hr - len(hr), 0), hr]) if len(hr) < self.n_hr else hr
        hr = hr - self.tare
        pb.hr.clear()
        self.rows.append(dict(
            t=env.time, phase=phase,
            cmd=self.to_bel(tp_cmd + (self.tip0 - self.tgt0)),  # commanded tip (believed frame)
            tcp=self.to_bel(p_tcp + (self.tip0 - self.tgt0)),  # measured TCP mapped to the tip (believed frame)
            F=self.R.T @ w[:3], M=self.R.T @ w[3:], tau=pb.joint_torques(),
            Fh=np.concatenate([hr[:, :3] @ self.R, hr[:, 3:] @ self.R], 1),  # 100 Hz wrench in the believed frame
            # privileged labels
            tip_true=rel, port_k=k, on_target=k == self.port, angle=pr["angle"], latched=pr["latched"],
            grasped=pr["grasped_by"] is not None, **{f"f_{c}": v for c, v in pb.contact_forces().items()},
        ))

    def run(self):
        self.setup()
        env, dt = self.env, self.env.dt
        # pre: stiff lateral correction onto the believed axis, pre_x in front of the believed face
        start = self.to_bel(self.tip0)
        goal = np.array([self.pre_x, 0.0, 0.0])
        n = max(4, int(round(0.6 / dt)))
        for i in range(1, n + 1):
            u = G.min_jerk(i / n)
            self.step(self.tcp_target_for_tip(start + u * (goal - start)), False, "pre")
        # push: compliant, force-limited, to seat depth + overshoot
        x1 = self.tc.seat_depth + self.overshoot
        n = max(1, int(round((x1 - self.pre_x) / self.speed / dt)))
        nh = int(round(self.hold_s / dt))
        for i in range(1, n + nh + 1):
            u = G.min_jerk(min(1.0, i / n))
            tp = self.tcp_target_for_tip(goal + u * np.array([x1 - self.pre_x, 0, 0]))
            p_tcp, _ = env.tcp_pose(self.pb.arm)
            lead = (tp - p_tcp) @ self.R[:, 0]
            if lead > MAX_LEAD:
                tp = tp - (lead - MAX_LEAD) * self.R[:, 0]
            self.step(tp, True, "push" if i <= n else "hold")
            if env.task.latched[self.pb.c] >= 0:
                break
        return self.rows


def outcome(rows, tc):
    """Privileged end-state class of an attempt. The port site is on the face plane (depth 0); the chamfer
    occupies depths (0, tc.chamfer]."""
    r = rows[-1]
    if r["latched"] >= 0:
        return "seated" if r["on_target"] else "wrong_port"
    if not r["grasped"]:
        return "dropped"
    depth = r["tip_true"][0]
    if r["on_target"] and depth > tc.chamfer + 0.0005:
        return "in_channel"
    if r["on_target"] and depth > 0.0003:
        return "chamfer"
    blocked = max(x["f_grip_occ"] + x["f_grip_panel"] + x["f_plug_occ"] for x in rows[-5:])
    return "blocked" if blocked > 5.0 else "face"


def stack(rows, key):
    return np.array([r[key] for r in rows])
