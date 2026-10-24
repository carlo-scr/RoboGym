"""RoboGymEnv: one or two xArm 7s on a lab bench, running a registered Task.

Action (8-D per arm, in ARM order, at `cfg.sim.control_hz`):
    [dx, dy, dz, drx, dry, drz, grip, compliant]
  - TCP translation delta (world, m) and rotation delta (world rotation vector, rad), applied to the
    arm's internal TCP *target* (not the measured pose), so replaying recorded actions reproduces the
    same target trajectory.
  - grip in [0, 1]: commanded closure of the position-controlled, force-limited gripper (0 open, 1 closed).
  - compliant > 0.5: insertion mode, soft laterally / in rotation about `cfg.robot.gripper.tool_tip`.
Low level: damped-least-squares IK at every physics step -> joint position servos (Menagerie gains,
gravity compensated), so contact forces are bounded by the servo stiffness.

Observation `state` (28-D per arm): q(7), qd(7), tcp_pos(3), tcp_quat(4), grip_width(1), wrist wrench(6).
`privileged()` adds the task's ground truth (object poses, grasp, insertion depth, ...).
`info()` adds stage / progress / success / regime labels for monitors.

Snapshots: `get_state()` / `set_state()` capture physics + controller + task latch state + action-delay
queue + regime, and `set_state` followed by `step(a)` is bit-exact w.r.t. the original rollout, which is
what counterfactual branching (research.py) relies on.
"""
from __future__ import annotations

import zlib

import mujoco
import numpy as np

from . import geometry as G
from .config import EnvCfg
from .robot import xarm7
from .scenario import NOMINAL_REGIME, Scenario, sample_scenario
from .scene import LIGHTS, PARK, build_mjcf
from .tasks.base import get_task_class

Q_DOWN = np.array([0.0, 0.0, 1.0, 0.0])  # TCP z down, TCP x along world -x, finger axis along world y
ACTION_FIELDS = ("dx", "dy", "dz", "drx", "dry", "drz", "grip", "compliant")
ACT_DIM = len(ACTION_FIELDS)
STATE_PER_ARM = 28
MAX_DELAY = 8


def apply_image_fx(img: np.ndarray, fx: dict, seed: int) -> np.ndarray:
    """Camera degradations: gain/gamma (exposure), motion blur, Gaussian + salt noise, lens occlusion."""
    if not fx:
        return img
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32) / 255.0
    if fx.get("blur", 0) > 1:
        n, a = int(fx["blur"]), fx.get("blur_angle", 0.0)
        acc = np.zeros_like(x)
        for i in range(n):
            s = i - (n - 1) / 2
            acc += np.roll(np.roll(x, int(round(s * np.cos(a))), 1), int(round(s * np.sin(a))), 0)
        x = acc / n
    if "occlusion" in fx:
        cx, cy, r = fx["occlusion"]
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.sqrt(((xx / w - cx) / r) ** 2 + ((yy / h - cy) / (r * 1.3)) ** 2)
        m = np.clip(1.3 - d, 0, 1)[..., None] ** 0.7
        x = x * (1 - 0.93 * m) + 0.10 * 0.93 * m
    x = x * fx.get("gain", 1.0)
    if fx.get("gamma", 1.0) != 1.0:
        x = np.clip(x, 0, None) ** fx["gamma"]
    if fx.get("noise", 0) > 0:
        x = x + rng.normal(0, fx["noise"] / 255.0, x.shape).astype(np.float32)
    if fx.get("salt", 0) > 0:
        m = rng.random(x.shape[:2]) < fx["salt"]
        x[m] = rng.choice([0.0, 1.0], size=(int(m.sum()), 1))
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


class RoboGymEnv:
    STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

    def __init__(self, cfg: EnvCfg | str | None = None, render: bool = True, bimanual: bool = False, **overrides):
        """`cfg` is an EnvCfg or a task name; `bimanual=True` mounts two xArm 7 (the experts then use both);
        dotted `overrides` go into the config (e.g. RoboGymEnv("lego", **{"task.bricks": ["2x2", "2x2"]}))."""
        if cfg is None or isinstance(cfg, str):
            from .config import make_cfg

            cfg = make_cfg(cfg or "cube_stack", bimanual=bimanual, **overrides)
        elif bimanual or overrides:
            cfg = cfg.with_overrides({**({"robot.n_arms": 2} if bimanual else {}), **overrides})
        self.cfg = cfg
        self.arms = cfg.robot.arms
        self.task = get_task_class(cfg.task_name)(cfg)
        tx, ta, te, tcon, tdef = self.task.xml()
        self.xml = build_mjcf(cfg, tx, ta, te, tcon, tdef)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.n_sub = int(round(1.0 / (cfg.sim.control_hz * cfg.sim.timestep)))
        self.dt = self.n_sub * cfg.sim.timestep
        self._renderer = None
        self._render_enabled = render
        g = cfg.robot.gripper
        # ---- robot indices
        self.arm = {}
        for side in self.arms:
            p = side[0]
            jids = [m.joint(f"{p}_j{i + 1}").id for i in range(7)]
            fingers = {"left": [], "right": []}
            for s in fingers:
                for k in (1, 2):
                    fingers[s].append(m.geom(f"{p}_{s}_pad{k}").id)
            self.arm[side] = dict(
                qadr=np.array([m.jnt_qposadr[j] for j in jids]),
                dadr=np.array([m.jnt_dofadr[j] for j in jids]),
                lo=m.jnt_range[jids, 0].copy(),
                hi=m.jnt_range[jids, 1].copy(),
                act=np.array([m.actuator(f"{p}_j{i + 1}").id for i in range(7)]),
                grip_act=m.actuator(f"{p}_grip").id,
                tcp=m.site(f"{p}_tcp").id,
                tip=m.site(f"{p}_tip").id,
                ft=m.site(f"{p}_ft").id,
                hand_body=m.body(f"{p}_gripper_base").id,
                pads={"left": set(fingers["left"]), "right": set(fingers["right"])},
                all_pads=set(fingers["left"]) | set(fingers["right"]),
                pad_ref=(m.geom(f"{p}_left_pad1").id, m.geom(f"{p}_right_pad1").id),
                force_sensor=m.sensor(f"{p}_force").adr[0],
                torque_sensor=m.sensor(f"{p}_torque").adr[0],
                driver_qadr=[m.jnt_qposadr[m.joint(f"{p}_{s}_driver").id] for s in ("left", "right")],
            )
        self.cam_ids = {c: m.camera(c).id for c in cfg.camera.all_cameras(cfg.robot)}
        self.light_ids = {nm: m.light(nm).id for nm in LIGHTS}
        self.distractor_qadr = [m.jnt_qposadr[m.joint(f"distractor{i}").id] for i in range(cfg.bench.n_distractors)]
        self.bench_geom = m.geom("bench").id
        self.pad_thickness = 2 * g.pad_half[1]
        self.task.bind(self)
        self.tool_tip = np.asarray(getattr(self.task, "tool_tip", None) or cfg.robot.gripper.tool_tip, float)
        self.compliant_gains = np.asarray(getattr(self.task, "compliant_gains", None) or cfg.robot.gripper.compliant_gains, float)
        self._bad = [mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADCTRL]
        self._orig = self._snapshot_model()
        self._head_orig = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
        self._grav_orig = m.opt.gravity.copy()
        self._kv_over_kp = np.array(cfg.robot.arm.kv, float) / np.array(cfg.robot.arm.kp, float)
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self.home_q = {s: np.array(cfg.robot.arm.home_q) for s in self.arms}
        self.scenario: Scenario | None = None
        self.regime = dict(NOMINAL_REGIME)
        self._light_base = m.light_diffuse.copy()
        self._head_base = self._head_orig
        self.reset(sample_scenario(cfg, 0, task=self.task))

    def _snapshot_model(self):
        m = self.model
        return {k: getattr(m, k).copy() for k in (
            "geom_rgba", "geom_friction", "geom_pos", "geom_quat", "geom_size", "body_mass", "body_inertia", "body_pos", "body_quat",
            "jnt_stiffness", "light_diffuse", "light_pos", "cam_pos", "cam_quat", "cam_fovy", "geom_matid", "mat_rgba", "actuator_gainprm",
            "actuator_biasprm", "actuator_forcerange", "eq_data", "body_invweight0", "dof_invweight0", "dof_M0", "body_subtreemass")}

    # ------------------------------------------------------------------ reset / randomisation
    def _apply_scenario(self, sc: Scenario):
        m, o, _cfg = self.model, self._orig, self.cfg
        for k, v in o.items():
            getattr(m, k)[:] = v
        m.mat_rgba[m.material("bench").id] = sc.bench_rgba
        m.mat_rgba[m.material("mat").id] = sc.mat_rgba
        for nm, lid in self.light_ids.items():
            if nm in sc.lights:
                m.light_diffuse[lid] = sc.lights[nm]
            m.light_active[lid] = float(np.sum(m.light_diffuse[lid])) > 1e-6
        self._light_base = m.light_diffuse.copy()
        hs, *ht = sc.headlight
        m.vis.headlight.ambient[:] = self._head_orig[0] * hs * np.asarray(ht)
        m.vis.headlight.diffuse[:] = self._head_orig[1] * hs * np.asarray(ht)
        self._head_base = (m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy())
        for c, jit in sc.cam_jitter.items():
            if c in self.cam_ids:
                cid = self.cam_ids[c]
                m.cam_pos[cid] = o["cam_pos"][cid] + np.asarray(jit[:3])
                m.cam_quat[cid] = G.quat_mul(G.quat_from_rotvec(jit[3:6]), o["cam_quat"][cid])
                if len(jit) > 6:
                    m.cam_fovy[cid] = o["cam_fovy"][cid] * (1 + jit[6])  # intrinsics jitter
        for nm, dp in (sc.light_jitter or {}).items():
            if nm in self.light_ids:
                m.light_pos[self.light_ids[nm]] = o["light_pos"][self.light_ids[nm]] + np.asarray(dp)
        for g, mat in (sc.materials or {}).items():  # bench / mat texture variants
            gid = m.geom(g).id
            m.geom_matid[gid] = m.material(mat).id
        self.task.apply(self, sc.task)
        self._apply_regime(sc.regime)

    def _apply_regime(self, regime: dict):
        """Physics knobs relative to the compiled model (idempotent; also used online by set_regime)."""
        m, o, _cfg = self.model, self._orig, self.cfg
        r = dict(NOMINAL_REGIME) | dict(regime)
        self.regime = r
        ag = r["arm_gain"]
        ag = list(ag) if isinstance(ag, (list, tuple)) else [ag] * len(self.arms)
        for side, gs in zip(self.arms, ag):
            a = self.arm[side]
            for act in a["act"]:
                m.actuator_gainprm[act, 0] = o["actuator_gainprm"][act, 0] * gs
                m.actuator_biasprm[act, 1] = o["actuator_biasprm"][act, 1] * gs
            ga = a["grip_act"]
            m.actuator_forcerange[ga] = o["actuator_forcerange"][ga] * r["grip_force"]
            for gid in a["all_pads"]:
                m.geom_friction[gid, 0] = o["geom_friction"][gid, 0] * r["pad_friction"]
        m.geom_friction[self.bench_geom, 0] = o["geom_friction"][self.bench_geom, 0] * r["bench_friction"]
        m.opt.gravity[:] = self._grav_orig * r["gravity"]
        self.delay = int(min(MAX_DELAY, max(0, r["action_delay"])))
        self.obs_noise = float(r["obs_noise"])
        self.task.apply_regime(self, r)
        self._refresh_mass_constants()

    def _refresh_mass_constants(self):
        """Keep MuJoCo's precomputed mass constants consistent after body_mass / body_inertia edits.
        The constraint solver scales contact impedance by body_invweight0 / dof_invweight0 (computed at
        compile time), so a body made heavier than compiled gets a light body's impedance and heavy
        grasps explode. For every body whose mass changed by a factor s: invweights /= s, dof_M0 *= s,
        subtree masses updated up the tree. (mj_setConst would do this too but also re-derives other
        constants in a way that breaks the gripper linkage.)"""
        m, o = self.model, self._orig
        massive = o["body_mass"] > 1e-9  # visual-only / static bodies have no mass and no inverse weight to scale
        ratio = np.where(massive, m.body_mass / np.where(massive, o["body_mass"], 1.0), 1.0)
        changed = np.where(np.abs(ratio - 1.0) > 1e-9)[0]
        m.body_invweight0[:] = o["body_invweight0"]
        m.dof_invweight0[:] = o["dof_invweight0"]
        m.dof_M0[:] = o["dof_M0"]
        m.body_subtreemass[:] = o["body_subtreemass"]
        for b in changed:
            s_ = ratio[b]
            m.body_invweight0[b] /= s_
            dofs = range(m.body_dofadr[b], m.body_dofadr[b] + m.body_dofnum[b])
            for i in dofs:
                m.dof_invweight0[i] /= s_
                m.dof_M0[i] *= s_
            dm = m.body_mass[b] - o["body_mass"][b]
            p = b
            while p >= 0:
                m.body_subtreemass[p] += dm
                if p == 0:
                    break
                p = m.body_parentid[p]

    def set_regime(self, regime: dict):
        """Change the physics regime *now* (test-time shift). Keys: see scenario.REGIME_KEYS."""
        self._apply_regime(dict(self.regime) | dict(regime))

    def reset(self, scenario: Scenario | None = None, seed: int | None = None, shifts=(), severity=0.5, variant=None):
        if scenario is None:
            scenario = sample_scenario(self.cfg, seed if seed is not None else int(np.random.randint(1 << 30)), shifts, severity, variant, task=self.task)
        self.scenario = scenario
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self._apply_scenario(scenario)
        d.eq_active[:] = 1
        # Arms at home (+ noise), grippers open.
        hn = np.asarray(scenario.home_noise) if scenario.home_noise else np.zeros(7 * len(self.arms))
        for i, side in enumerate(self.arms):
            a = self.arm[side]
            q = np.clip(self.home_q[side] + hn[7 * i : 7 * i + 7], a["lo"], a["hi"])
            d.qpos[a["qadr"]] = q
            d.ctrl[a["act"]] = q
            d.ctrl[a["grip_act"]] = self._grip_ctrl(0.0)
        # Distractors: park off-scene unless placed.
        placed = {int(c[0]): c[1:] for c in scenario.distractors}
        for i, qa in enumerate(self.distractor_qadr):
            if i in placed:
                x, y, yaw = placed[i]
                d.qpos[qa : qa + 3] = (x, y, self.cfg.bench.top_z + 0.03)
                d.qpos[qa + 3 : qa + 7] = G.yaw_quat(yaw)
            else:
                d.qpos[qa : qa + 3] = PARK + (0.2 * i, 0.3, 0.05)
                d.qpos[qa + 3 : qa + 7] = (1, 0, 0, 0)
        self.task.reset_state(self, scenario.task)
        mujoco.mj_forward(m, d)
        self.targets = {s: self._tcp_pose(s) for s in self.arms}
        self.grip_cmd = {s: 0.0 for s in self.arms}
        self.grip_now = {s: 0.0 for s in self.arms}  # rate-limited closure actually sent to the servo
        self.compliance = {s: np.ones(6) for s in self.arms}
        self.compliance_center = {s: None for s in self.arms}
        self.ext_force = None
        self.unstable = False
        self.time = 0.0
        self.step_count = 0
        self.last_action = np.zeros(ACT_DIM * len(self.arms))
        self._obs_hist = []
        self._queue = np.zeros((MAX_DELAY, ACT_DIM * len(self.arms)))
        self._queue_n = 0
        self._sched_i = 0
        for _ in range(int(self.cfg.sim.settle_s / self.cfg.sim.timestep)):
            self._physics_step()
        mujoco.mj_forward(m, d)
        self.time = 0.0
        self.targets = {s: self._tcp_pose(s) for s in self.arms}
        self.task.after_settle(self)
        self._update_lights()
        return self.observe()

    @property
    def episode_max_s(self) -> float:
        if self.cfg.sim.episode_max_s:
            return self.cfg.sim.episode_max_s
        t = self.task.default_episode_s(self.scenario.task)
        return t * 1.35 if len(self.arms) == 2 else t  # bimanual: turn-taking at the shared structure

    # ------------------------------------------------------------------ control
    def pending_targets(self, side):
        """The TCP target the arm will hold once the actuation-delay queue has drained (== targets[side]
        without latency). A target-tracking controller / expert must compute its deltas against this,
        otherwise re-sent deltas pile up in the queue and overshoot."""
        p, q = self.targets[side]
        i = self.arms.index(side)
        for k in range(min(self.delay, self._queue_n) - 1, -1, -1):  # oldest pending first
            a = self._queue[k].reshape(len(self.arms), ACT_DIM)[i]
            p = p + a[:3]
            q = G.quat_mul(G.quat_from_rotvec(a[3:6]), q)
        return p.copy(), q.copy()

    def _tcp_pose(self, side):
        sid = self.arm[side]["tcp"]
        return self.data.site_xpos[sid].copy(), G.mat_to_quat(self.data.site_xmat[sid])

    def tcp_pose(self, side):
        return self._tcp_pose(side)

    def tip_pos(self, side):
        return self.data.site_xpos[self.arm[side]["tip"]].copy()

    def _grip_ctrl(self, grip):
        g = self.cfg.robot.gripper
        grip = float(np.clip(grip, 0.0, 1.0))
        return 255.0 * grip if g.kind == "xarm" else (1.0 - grip) * g.stroke / 2

    def grasp_tcp_height(self, width: float, clearance: float = 0.003) -> float:
        """TCP height above the surface an object of `width` rests on such that, once the fingers have
        closed on it, the fingertips stay `clearance` above that surface (accounts for the finger arc)."""
        g = self.cfg.robot.gripper
        drop = xarm7.finger_drop(xarm7.angle_for_opening(width)) if g.kind == "xarm" else 0.0
        return g.tip_z - g.tcp_z + drop + clearance

    def grip_width(self, side):
        a, d = self.arm[side], self.data
        pl, pr = d.geom_xpos[a["pad_ref"][0]], d.geom_xpos[a["pad_ref"][1]]
        return max(0.0, float(np.linalg.norm(pl - pr)) - self.pad_thickness)

    def wrench(self, side):
        """Wrist force/torque sensor (site frame of the gripper base)."""
        a, d = self.arm[side], self.data
        return np.concatenate([d.sensordata[a["force_sensor"] : a["force_sensor"] + 3], d.sensordata[a["torque_sensor"] : a["torque_sensor"] + 3]])

    def _ik(self, side, tgt_p, tgt_q, tgt_v=None):
        m, d, a, ac = self.model, self.data, self.arm[side], self.cfg.robot.arm
        p, q = self._tcp_pose(side)
        off = self.compliance_center[side]
        if off is None:
            mujoco.mj_jacSite(m, d, self._jacp, self._jacr, a["tcp"])
        else:
            p = p + G.quat_rotate(q, off)
            tgt_p = tgt_p + G.quat_rotate(tgt_q, off)
            mujoco.mj_jac(m, d, self._jacp, self._jacr, p, a["hand_body"])
        err = np.concatenate([tgt_p - p, G.quat_error(tgt_q, q)])
        gains = self.compliance[side]
        if not np.all(gains == 1.0):
            R = d.site_xmat[a["tcp"]].reshape(3, 3)
            err = np.concatenate([R @ (gains[:3] * (R.T @ err[:3])), R @ (gains[3:] * (R.T @ err[3:]))])
        J = np.vstack([self._jacp[:, a["dadr"]], self._jacr[:, a["dadr"]]])
        JJt = J @ J.T + ac.ik_damping * np.eye(6)
        Jpinv = J.T @ np.linalg.inv(JJt)
        dq = Jpinv @ err
        qcur = d.qpos[a["qadr"]]
        dq += (np.eye(7) - Jpinv @ J) @ (ac.ik_null_gain * (self.home_q[side] - qcur))
        if tgt_v is not None:
            dq += self._kv_over_kp * (Jpinv @ tgt_v)
        dq = np.clip(dq, -ac.ik_max_dq, ac.ik_max_dq)
        d.ctrl[a["act"]] = np.clip(qcur + dq, a["lo"], a["hi"])

    def _physics_step(self, control=None):
        """mj_step1 (fresh kinematics) -> applied forces, task detents, controller -> mj_step2."""
        m, d = self.model, self.data
        n_bad = sum(d.warning[w].number for w in self._bad)
        mujoco.mj_step1(m, d)
        d.xfrc_applied[:] = 0
        if self.ext_force is not None:
            b, w, n = self.ext_force
            d.xfrc_applied[b] += w
            self.ext_force = (b, w, n - 1) if n > 1 else None
        self.task.pre_step(self)
        self._ramp_grip()
        if control is not None:
            control()
        mujoco.mj_step2(m, d)
        if sum(d.warning[w].number for w in self._bad) > n_bad:
            self.unstable = True  # MuJoCo auto-reset the state; the episode must be discarded
        self.task.post_step(self)

    def _ramp_grip(self):
        """Move the servo's closure target toward the commanded grip at most `close_rate` per second."""
        step = self.cfg.robot.gripper.close_rate * self.cfg.sim.timestep
        for side in self.arms:
            g = self.grip_now[side]
            g = g + float(np.clip(self.grip_cmd[side] - g, -step, step))
            self.grip_now[side] = g
            self.data.ctrl[self.arm[side]["grip_act"]] = self._grip_ctrl(g)

    def clip_action(self, action):
        a = np.asarray(action, float).reshape(len(self.arms), ACT_DIM).copy()
        for i in range(len(self.arms)):
            for sl, lim in ((slice(0, 3), self.cfg.sim.max_dpos), (slice(3, 6), self.cfg.sim.max_drot)):
                n = np.linalg.norm(a[i, sl])
                if n > lim:
                    a[i, sl] *= lim / n
        return a.reshape(-1)

    def _delayed(self, action):
        """Actuation latency: the executed action is the one commanded `delay` control steps ago."""
        if self.delay == 0:
            return action
        self._queue[1:] = self._queue[:-1]
        self._queue[0] = action
        self._queue_n = min(MAX_DELAY, self._queue_n + 1)
        if self._queue_n <= self.delay:
            return np.zeros_like(action)  # nothing commanded yet: hold
        return self._queue[self.delay].copy()

    def step(self, action):
        action = self.clip_action(action)
        self.last_action = action.copy()
        action = self._delayed(action).reshape(len(self.arms), ACT_DIM)
        start = {s: self.targets[s] for s in self.arms}
        goal = {}
        for i, side in enumerate(self.arms):
            dp, dr = action[i, :3], action[i, 3:6]
            p0, q0 = start[side]
            goal[side] = (p0 + dp, G.quat_mul(G.quat_from_rotvec(dr), q0))
            self.grip_cmd[side] = float(np.clip(action[i, 6], 0.0, 1.0))
            if action[i, 7] > 0.5:
                self.set_compliance(side, self.compliant_gains, center=self.tool_tip)
            else:
                self.set_compliance(side, None)
        vel = {s: np.concatenate([goal[s][0] - start[s][0], G.quat_error(goal[s][1], start[s][1])]) / self.dt for s in self.arms}
        for k in range(self.n_sub):
            t = (k + 1) / self.n_sub

            def control(t=t):
                for side in self.arms:
                    (p0, q0), (p1, q1) = start[side], goal[side]
                    self._ik(side, p0 + t * (p1 - p0), G.slerp(q0, q1, t), vel[side])

            self._physics_step(control)
        mujoco.mj_forward(self.model, self.data)
        self.targets = goal
        self.time += self.dt
        self.step_count += 1
        self._update_lights()
        self._tick_schedule()
        return self.observe()

    def _tick_schedule(self):
        sch = self.scenario.regime_schedule if self.scenario else []
        while self._sched_i < len(sch) and self.time >= sch[self._sched_i][0] - 1e-9:
            self.set_regime(sch[self._sched_i][1])
            self._sched_i += 1

    def _update_lights(self):
        fl = self.scenario.flicker if self.scenario else None
        if fl:
            amp, f, ph = fl
            dip = (0.5 + 0.5 * np.sin(2 * np.pi * f * self.time + ph)) ** 6
            self.model.light_diffuse[:] = self._light_base * (1 - amp * dip)
            self.model.vis.headlight.ambient[:] = self._head_base[0] * (1 - amp * dip)
            self.model.vis.headlight.diffuse[:] = self._head_base[1] * (1 - amp * dip)

    def set_compliance(self, side, scale=None, center=None):
        self.compliance[side] = np.ones(6) if scale is None else np.asarray(scale, float)
        self.compliance_center[side] = None if center is None else np.asarray(center, float)

    def apply_force(self, body: str | int, force, torque=(0, 0, 0), duration_s=0.1):
        """External wrench (world frame) on a body for a duration, e.g. a bump on the gripper."""
        b = self.model.body(body).id if isinstance(body, str) else body
        self.ext_force = (b, np.concatenate([force, torque]), max(1, int(duration_s / self.cfg.sim.timestep)))

    def bump(self, side, force, torque=(0, 0, 0), duration_s=0.1):
        self.apply_force(self.arm[side]["hand_body"], force, torque, duration_s)

    # ------------------------------------------------------------------ contacts / grasp
    def touching(self, geoms: set) -> dict:
        """arm -> set of finger names ('left'/'right') whose pads touch any of `geoms`."""
        d = self.data
        out = {s: set() for s in self.arms}
        for c in d.contact[: d.ncon]:
            g1, g2 = c.geom1, c.geom2
            for a, b in ((g1, g2), (g2, g1)):
                if a in geoms:
                    for s in self.arms:
                        for f, pads in self.arm[s]["pads"].items():
                            if b in pads:
                                out[s].add(f)
        return out

    def grasped_by(self, geoms: set):
        """Arm whose two fingers both touch the object made of `geoms`, or None."""
        t = self.touching(geoms)
        for s in self.arms:
            if len(t[s]) == 2:
                return s
        return None

    def contact_force(self, geoms_a: set, geoms_b: set | None = None) -> np.ndarray:
        """Total contact force (world) on geoms_a from geoms_b (or from everything)."""
        m, d = self.model, self.data
        f, f6 = np.zeros(3), np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if g1 in geoms_a and (geoms_b is None or g2 in geoms_b):
                sgn = -1.0
            elif g2 in geoms_a and (geoms_b is None or g1 in geoms_b):
                sgn = 1.0
            else:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            f += sgn * (c.frame.reshape(3, 3).T @ f6[:3])
        return f

    # ------------------------------------------------------------------ observations / labels
    def observe(self):
        d = self.data
        parts = []
        for side in self.arms:
            a = self.arm[side]
            p, q = self._tcp_pose(side)
            parts += [d.qpos[a["qadr"]], d.qvel[a["dadr"]], p, q, [self.grip_width(side)], self.wrench(side)]
        x = np.concatenate(parts).astype(np.float32)
        r = self.regime
        if self.obs_noise > 0 or r.get("ft_noise", 0) > 0 or r.get("ft_bias", 0) != 0:
            rng = np.random.default_rng(zlib.crc32(f"{self.scenario.seed}/obs/{self.step_count}".encode()))
            if self.obs_noise > 0:
                x = x + rng.normal(0, self.obs_noise, x.shape).astype(np.float32)
            for i in range(len(self.arms)):  # wrench dims 22..28 of each 28-block: Gaussian noise [N] + constant bias [N]
                sl = slice(STATE_PER_ARM * i + 22, STATE_PER_ARM * i + 28)
                x[sl] = x[sl] + rng.normal(0, r.get("ft_noise", 0.0), 6).astype(np.float32) + np.float32(r.get("ft_bias", 0.0))
        d_obs = int(min(MAX_DELAY, max(0, r.get("obs_delay", 0))))
        if d_obs > 0:  # sensing latency: report the observation from d_obs control steps ago
            self._obs_hist.append(x)
            self._obs_hist = self._obs_hist[-(MAX_DELAY + 1) :]
            x = self._obs_hist[max(0, len(self._obs_hist) - 1 - d_obs)]
        return x

    def privileged(self) -> dict:
        pr = self.task.privileged(self)
        pr["time"] = self.time
        pr["tcp"] = {s: self._tcp_pose(s) for s in self.arms}
        return pr

    def info(self, pr: dict | None = None) -> dict:
        pr = pr or self.privileged()
        st = self.task.stage(self, pr)
        return dict(stage=st, stage_name=self.task.stages()[st], progress=float(self.task.progress(self, pr)),
                    success=bool(self.task.success(self, pr)), unstable=self.unstable, regime=dict(self.regime), time=self.time)

    def task_features(self, pr: dict | None = None) -> np.ndarray:
        return self.task.task_features(self, pr or self.privileged())

    # ------------------------------------------------------------------ rendering
    def _get_renderer(self, w, h):
        if self._renderer is None or (self._renderer.width, self._renderer.height) != (w, h):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, h, w)
            self._vopt = mujoco.MjvOption()
        return self._renderer

    def render(self, camera: str, width=None, height=None, fx=True, ghost=False, mode="rgb"):
        """Robot camera image incl. the scenario's sensor effects. `ghost` also draws goal ghosts (group 5).
        mode: "rgb" (uint8 HxWx3) | "depth" (float32 metres, HxW) | "seg" (int32 HxW geom ids, -1 = background)."""
        r = self._get_renderer(width or self.cfg.camera.width, height or self.cfg.camera.height)
        self._vopt.geomgroup[5] = 1 if ghost else 0
        if mode == "depth":
            r.enable_depth_rendering()
            r.update_scene(self.data, camera=camera, scene_option=self._vopt)
            out = r.render().copy()
            r.disable_depth_rendering()
            return out.astype(np.float32)
        if mode == "seg":
            r.enable_segmentation_rendering()
            r.update_scene(self.data, camera=camera, scene_option=self._vopt)
            out = r.render().copy()
            r.disable_segmentation_rendering()
            return out[..., 0].astype(np.int32)  # geom id (objtype in channel 1)
        r.update_scene(self.data, camera=camera, scene_option=self._vopt)
        img = r.render().copy()
        if fx and self.scenario is not None and self.scenario.image_fx:
            seed = zlib.crc32(f"{self.scenario.seed}/{camera}/{self.step_count}".encode())
            img = apply_image_fx(img, self.scenario.image_fx, seed)
        return img

    def segmentation_labels(self):
        """geom id -> (body name, geom name) for interpreting segmentation renders."""
        m = self.model
        return {g: (m.body(m.geom_bodyid[g]).name, m.geom(g).name) for g in range(m.ngeom)}

    def render_free(self, lookat=(0.45, 0, 0.1), distance=1.4, azimuth=140, elevation=-25, width=640, height=400, ghost=True):
        r = self._get_renderer(width, height)
        self._vopt.geomgroup[5] = 1 if ghost else 0
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        r.update_scene(self.data, camera=cam, scene_option=self._vopt)
        return r.render().copy()

    # ------------------------------------------------------------------ save / restore
    def get_state(self):
        m, d = self.model, self.data
        phys = np.zeros(mujoco.mj_stateSize(m, self.STATE_SPEC))
        mujoco.mj_getState(m, d, phys, self.STATE_SPEC)
        parts = []
        for s in self.arms:
            cc = self.compliance_center[s]
            parts.append(np.concatenate([self.targets[s][0], self.targets[s][1], [self.grip_cmd[s], self.grip_now[s]], self.compliance[s], np.full(3, np.nan) if cc is None else cc]))
        parts.append([self.time, self.step_count, self._sched_i, self._queue_n, self.delay, self.obs_noise])
        parts.append([-1] + [0] * 7 if self.ext_force is None else [self.ext_force[0], *self.ext_force[1], self.ext_force[2]])
        parts.append(m.eq_data.reshape(-1))
        parts.append(self._queue.reshape(-1))
        parts.append(self.last_action)
        parts.append(self.task.get_aux(self))
        return dict(physics=phys, aux=np.concatenate([np.asarray(p, float).reshape(-1) for p in parts]), regime=dict(self.regime))

    def set_state(self, st):
        m, d = self.model, self.data
        if "regime" in st and st["regime"] != self.regime:
            self._apply_regime(st["regime"])
        mujoco.mj_setState(m, d, np.asarray(st["physics"], float), self.STATE_SPEC)
        aux = np.asarray(st["aux"], float)
        o = 0
        for s in self.arms:
            a = aux[o : o + 18]
            self.targets[s] = (a[:3].copy(), a[3:7].copy())
            self.grip_cmd[s] = float(a[7])
            self.grip_now[s] = float(a[8])
            self.compliance[s] = a[9:15].copy()
            self.compliance_center[s] = None if np.isnan(a[15]) else a[15:18].copy()
            o += 18
        self.time, self.step_count, self._sched_i, self._queue_n = float(aux[o]), int(aux[o + 1]), int(aux[o + 2]), int(aux[o + 3])
        self.delay, self.obs_noise = int(aux[o + 4]), float(aux[o + 5])
        o += 6
        ef = aux[o : o + 8]
        self.ext_force = None if ef[0] < 0 else (int(ef[0]), ef[1:7].copy(), int(ef[7]))
        o += 8
        ne = m.eq_data.size
        m.eq_data[:] = aux[o : o + ne].reshape(m.eq_data.shape)
        o += ne
        nq = self._queue.size
        self._queue = aux[o : o + nq].reshape(self._queue.shape).copy()
        o += nq
        na = ACT_DIM * len(self.arms)
        self.last_action = aux[o : o + na].copy()
        o += na
        self.task.set_aux(self, aux[o : o + self.task.aux_size()])
        mujoco.mj_forward(m, d)
        self._update_lights()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def state_names(cfg: EnvCfg):
    names = []
    for side in cfg.robot.arms:
        names += [f"{side}.q{i + 1}" for i in range(7)] + [f"{side}.qd{i + 1}" for i in range(7)]
        names += [f"{side}.tcp_{a}" for a in "xyz"] + [f"{side}.tcp_q{a}" for a in "wxyz"] + [f"{side}.grip_width"]
        names += [f"{side}.f{a}" for a in "xyz"] + [f"{side}.t{a}" for a in "xyz"]
    return names


def action_names(cfg: EnvCfg):
    return [f"{side}.{a}" for side in cfg.robot.arms for a in ACTION_FIELDS]
