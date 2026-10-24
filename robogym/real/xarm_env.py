"""RealXArmEnv: RoboGymEnv's contract on one or two real xArm 7.

    env = RealXArmEnv(arms=("right",), cameras={"right_wrist": "<realsense serial>", "front": "<serial>"},
                      ips={"right": "192.168.1.xxx"}, cfg=make_cfg("cube_stack"))
    obs = env.reset()                                    # moves to the home pose, opens the gripper
    obs = env.step(action)                               # 8-D per arm Cartesian delta, 20 Hz, blocking
    img = env.render("right_wrist")
    env.record(episode_dir, policy_fn, max_s=20)         # same folder layout as the simulator

Contract (identical to the simulator):
  action  [dx, dy, dz, drx, dry, drz, grip, compliant] per arm; deltas applied to the *commanded* target
  state   28-D per arm: q(7), qd(7), tcp_pos(3), tcp_quat(4), grip_width(1), wrench(6)
The compliant flag maps to the xArm's impedance / force mode when enabled, else to servo position mode.

NOT TESTED ON HARDWARE YET. Written against xarm-python-sdk 1.13 (`XArmAPI`) and pyrealsense2 2.55;
every SDK call sits in the `XArmBackend` class so that the mapping can be fixed in one place, and the
`DryBackend` reproduces the interface with a kinematic stand-in for development and CI.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .. import geometry as G
from ..config import EnvCfg, make_cfg
from ..env import ACT_DIM, STATE_PER_ARM, action_names, state_names

HOME_TCP = {"right": ((0.39, 0.0, 0.29), (0.0, 0.0, 1.0, 0.0))}  # from the simulator's home pose (TCP z down)
GRIPPER_OPEN_MM, GRIPPER_CLOSED_MM = 850.0, 0.0  # xArm gripper position units (0.1 mm)


class DryBackend:
    """Kinematic stand-in: targets are reached instantly, the gripper closes at 1.5 strokes / s."""

    def __init__(self, arm: str, cfg: EnvCfg):
        self.arm, self.cfg = arm, cfg
        p, q = HOME_TCP.get(arm, HOME_TCP["right"])
        self.p, self.q = np.asarray(p, float), np.asarray(q, float)
        self.grip, self.t = 0.0, 0.0
        self.q_joints = np.asarray(cfg.robot.arm.home_q, float)

    def connect(self):
        return True

    def go_home(self):
        p, q = HOME_TCP.get(self.arm, HOME_TCP["right"])
        self.p, self.q = np.asarray(p, float), np.asarray(q, float)

    def servo_cartesian(self, p, q, compliant):
        self.p, self.q = np.asarray(p, float), np.asarray(q, float)

    def set_gripper(self, closure, dt):
        self.grip += float(np.clip(closure - self.grip, -1.5 * dt, 1.5 * dt))

    def joint_state(self):
        return self.q_joints.copy(), np.zeros(7)

    def tcp_pose(self):
        return self.p.copy(), self.q.copy()

    def grip_width(self):
        return self.cfg.robot.gripper.stroke * (1 - self.grip)

    def wrench(self):
        return np.zeros(6)

    def disconnect(self):
        pass


class XArmBackend:
    """UFACTORY xArm Python SDK mapping (one arm)."""

    def __init__(self, arm: str, cfg: EnvCfg, ip: str, speed_mm_s: float = 150.0, ft_sensor: bool = True):
        from xarm.wrapper import XArmAPI  # noqa: lazy, hardware only

        self.arm, self.cfg, self.ip, self.ft = arm, cfg, ip, ft_sensor
        self.api = XArmAPI(ip, do_not_open=True)
        self.speed = speed_mm_s

    def connect(self):
        a = self.api
        a.connect()
        a.motion_enable(enable=True)
        a.clean_error()
        a.set_mode(1)  # servo motion mode (set_servo_cartesian at >= 20 Hz)
        a.set_state(0)
        a.set_gripper_enable(True)
        a.set_gripper_mode(0)
        a.set_gripper_speed(5000)
        if self.ft:
            a.ft_sensor_enable(1)
            a.ft_sensor_set_zero()
        time.sleep(0.5)
        return True

    def go_home(self):
        a = self.api
        a.set_mode(0)
        a.set_state(0)
        a.set_servo_angle(angle=np.degrees(self.cfg.robot.arm.home_q).tolist(), speed=30, wait=True)
        a.set_mode(1)
        a.set_state(0)

    def servo_cartesian(self, p, q, compliant):
        # xArm: mm and degrees (roll, pitch, yaw) in the base frame; our TCP frame is the gripper base + tcp_z
        rpy = _quat_to_rpy_deg(q)
        self.api.set_servo_cartesian([*(np.asarray(p) * 1000.0), *rpy], speed=self.speed, mvacc=2000)
        # compliant: the xArm offers an impedance mode (set_impedance / set_mode(4)) that must be configured
        # per task; left as a no-op here on purpose so that the mapping is set deliberately on the real cell.

    def set_gripper(self, closure, dt):
        pos = GRIPPER_OPEN_MM + (GRIPPER_CLOSED_MM - GRIPPER_OPEN_MM) * float(np.clip(closure, 0, 1))
        self.api.set_gripper_position(pos, wait=False)

    def joint_state(self):
        _, q = self.api.get_servo_angle(is_radian=True)
        _, (q2, qd, _tau) = self.api.get_joint_states(is_radian=True)
        return np.asarray(q[:7], float), np.asarray(qd[:7], float)

    def tcp_pose(self):
        _, pose = self.api.get_position(is_radian=True)
        p = np.asarray(pose[:3], float) / 1000.0
        q = _rpy_to_quat(np.asarray(pose[3:6], float))
        return p, q

    def grip_width(self):
        _, pos = self.api.get_gripper_position()
        return self.cfg.robot.gripper.stroke * float(pos) / GRIPPER_OPEN_MM

    def wrench(self):
        if not self.ft:
            return np.zeros(6)
        _, w = self.api.get_ft_sensor_data()
        return np.asarray(w[:6], float)

    def disconnect(self):
        self.api.disconnect()


class RealSenseCamera:
    def __init__(self, serial: str, width=640, height=480, fps=30):
        import pyrealsense2 as rs  # noqa: lazy

        self.pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self.pipe.start(cfg)

    def read(self):
        f = self.pipe.wait_for_frames().get_color_frame()
        return np.asanyarray(f.get_data()).copy()

    def close(self):
        self.pipe.stop()


class DryCamera:
    def __init__(self, serial=None, width=320, height=240):
        self.w, self.h = width, height

    def read(self):
        return np.zeros((self.h, self.w, 3), np.uint8)

    def close(self):
        pass


class RealXArmEnv:
    def __init__(self, arms=("right",), ips=None, cameras=None, cfg: EnvCfg | None = None, backend="dry"):
        self.cfg = cfg or make_cfg("cube_stack", bimanual=len(arms) == 2)
        self.arms = tuple(arms)
        self.dt = 1.0 / self.cfg.sim.control_hz
        self.backend = backend
        Backend = DryBackend if backend == "dry" else XArmBackend
        self.b = {a: (Backend(a, self.cfg) if backend == "dry" else Backend(a, self.cfg, ips[a])) for a in self.arms}
        Cam = DryCamera if backend == "dry" else RealSenseCamera
        self.cams = {name: Cam(serial) for name, serial in (cameras or {}).items()}
        self.targets, self.grip_cmd = {}, {}
        self.time, self.step_count = 0.0, 0

    # ---- contract
    def reset(self):
        for a in self.arms:
            self.b[a].connect()
            self.b[a].go_home()
            self.b[a].set_gripper(0.0, 1.0)
            self.targets[a] = self.b[a].tcp_pose()
            self.grip_cmd[a] = 0.0
        self.time, self.step_count = 0.0, 0
        self._t_wall = time.time()
        return self.observe()

    def step(self, action):
        a = np.asarray(action, float).reshape(len(self.arms), ACT_DIM)
        for i, arm in enumerate(self.arms):
            for sl, lim in ((slice(0, 3), self.cfg.sim.max_dpos), (slice(3, 6), self.cfg.sim.max_drot)):
                n = np.linalg.norm(a[i, sl])
                if n > lim:
                    a[i, sl] *= lim / n
            p0, q0 = self.targets[arm]
            p1, q1 = p0 + a[i, :3], G.quat_mul(G.quat_from_rotvec(a[i, 3:6]), q0)
            self.targets[arm] = (p1, q1)
            self.grip_cmd[arm] = float(np.clip(a[i, 6], 0, 1))
            self.b[arm].servo_cartesian(p1, q1, a[i, 7] > 0.5)
            self.b[arm].set_gripper(self.grip_cmd[arm], self.dt)
        # keep the 20 Hz cadence
        self._t_wall += self.dt
        lag = self._t_wall - time.time()
        if lag > 0 and self.backend != "dry":
            time.sleep(lag)
        self.time += self.dt
        self.step_count += 1
        return self.observe()

    def observe(self):
        parts = []
        for a in self.arms:
            q, qd = self.b[a].joint_state()
            p, quat = self.b[a].tcp_pose()
            parts += [q, qd, p, quat, [self.b[a].grip_width()], self.b[a].wrench()]
        return np.concatenate(parts).astype(np.float32)

    def render(self, camera):
        return self.cams[camera].read()

    def pending_targets(self, arm):
        return self.targets[arm]

    def tcp_pose(self, arm):
        return self.b[arm].tcp_pose()

    def close(self):
        for b in self.b.values():
            b.disconnect()
        for c in self.cams.values():
            c.close()

    # ---- recording in the simulator's layout
    def record(self, out_dir, policy_fn, max_s=20.0, cameras=None, notes=None):
        """Run `policy_fn(obs) -> action` and write episode.parquet / <cam>.mp4 / annotations.json /
        feature_report.json like robogym.datagen.write_episode (labels left for post-hoc annotation)."""
        import imageio.v2 as imageio
        import pandas as pd

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cameras = list(cameras or self.cams)
        obs = self.reset()
        rec = dict(t=[], state=[], action=[], ee_target=[])
        frames = {c: [] for c in cameras}
        while self.time < max_s:
            for c in cameras:
                frames[c].append(self.render(c))
            a = np.asarray(policy_fn(obs), float)
            rec["t"].append(self.time)
            rec["state"].append(obs)
            rec["action"].append(a.astype(np.float32))
            rec["ee_target"].append(np.concatenate([np.concatenate([self.targets[s][0], self.targets[s][1], [self.grip_cmd[s]]]) for s in self.arms]).astype(np.float32))
            obs = self.step(a)
        n = len(rec["t"])
        pd.DataFrame({"timestamp": np.round(np.asarray(rec["t"]) * 1e9).astype(np.int64), "frame_index": np.arange(n), "observation.state": rec["state"],
                      "action": rec["action"], "action.ee_target": rec["ee_target"]}).to_parquet(out / "episode.parquet")
        for c, fr in frames.items():
            if fr:
                w = imageio.get_writer(out / f"{c}.mp4", fps=self.cfg.sim.control_hz, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
                for f in fr:
                    w.append_data(f)
                w.close()
        json.dump(dict(sim_version="real", task=self.cfg.task_name, category="real", outcome=None, failure_mode=None, notes=notes or "", n_steps=n,
                       duration_s=self.time, arms=list(self.arms)), open(out / "annotations.json", "w"), indent=1)
        json.dump(dict(sim_version="real", fps=self.cfg.sim.control_hz, observation_state_names=state_names(self.cfg), action_names=action_names(self.cfg),
                       cameras=cameras, env_cfg=self.cfg.to_dict()), open(out / "feature_report.json", "w"), indent=1, default=str)
        return out


def _quat_to_rpy_deg(q):
    R = G.quat_to_mat(q)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    roll = np.arctan2(R[2, 1], R[2, 2]) if sy > 1e-6 else np.arctan2(-R[1, 2], R[1, 1])
    pitch = np.arctan2(-R[2, 0], sy)
    yaw = np.arctan2(R[1, 0], R[0, 0]) if sy > 1e-6 else 0.0
    return np.degrees([roll, pitch, yaw]).tolist()


def _rpy_to_quat(rpy):
    r, p, y = rpy
    return G.quat_mul(G.yaw_quat(y), G.quat_mul(G.quat_from_axis_angle([0, 1, 0], p), G.quat_from_axis_angle([1, 0, 0], r)))


def _unused():
    return STATE_PER_ARM
