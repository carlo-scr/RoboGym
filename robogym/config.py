"""Every modelling assumption of RoboGym, as dataclass fields with defaults.

Robot numbers (xArm 7 kinematics, inertia, servo gains, gripper linkage) come from the MuJoCo Menagerie
model of the UFACTORY xArm 7 and are therefore *measured*. Everything else (mounting, cameras, bench,
object dimensions, contact numerics, task tolerances) is a deliberately strong placeholder assumption
(see ASSUMPTIONS.md) so the whole pipeline runs today and single numbers can be swapped later.
All lengths in metres, angles in radians, masses in kg, forces in N.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

ARM_NAMES = ("right", "left")  # y < 0 is "right" when the robot faces +x


# ---------------------------------------------------------------------- robot
@dataclass
class ArmCfg:
    # Position servos exactly as in Menagerie (kp, kv per joint; force limits per joint).
    kp: tuple = (1500, 1500, 1000, 1000, 1000, 800, 800)
    kv: tuple = (150, 150, 100, 100, 100, 80, 80)
    force: tuple = (50, 50, 30, 30, 30, 20, 20)
    # Joint configuration at reset: TCP ~0.42 m in front of the base, 0.25 m above the table, pointing
    # down with the finger axis along world y. (Solved once with the env's own IK, see robot/xarm7.py.)
    home_q: tuple = (0.0, -0.3, 0.0, 0.95, 0.0, 1.25, 0.0)
    gravcomp: bool = True  # the real controller compensates gravity; without it the servos sag a few mm
    joint_damping: tuple = (10, 10, 5, 5, 5, 2, 2)
    armature: float = 0.1
    # Resolved-rate IK (runs at the physics rate): damping, null-space pull to home, max joint step.
    ik_damping: float = 1e-3
    ik_null_gain: float = 0.05
    ik_max_dq: float = 0.25


@dataclass
class GripperCfg:
    # "xarm": Menagerie's UFACTORY gripper (linkage, tendon-driven position servo, 86 mm stroke).
    # "parallel": a simple two-slider jaw with the same stroke and pad geometry (faster, MJX-friendly).
    kind: str = "xarm"
    # Menagerie's gripper servo (kp 100 on the tendon, +-50 force) closes in ~3 s and squeezes with 300+ N
    # at the pads (the knuckle linkage amplifies the actuator force ~7x). We keep its kinematics but use a
    # faster, force-limited servo: ~0.8 s to close, ~40-60 N pad normal force (xArm gripper spec: 20-50 N).
    max_force: float = 5.0  # actuator force range (tendon, xarm) / slider force (parallel)
    servo_kp: float = 300.0
    servo_kv: float = 20.0
    # The commanded closure is rate-limited inside the env (full stroke in 1/close_rate s), like a gripper
    # speed setting: the servo target is never far ahead of the pads when they meet an object, so a heavy
    # object is squeezed, not launched into the opposite pad.
    close_rate: float = 1.5
    joint_frictionloss: float = 0.05  # Menagerie's 1.0 per gripper joint eats most of a force-limited actuator
    joint_armature: float = 0.02
    pad_friction: float = 1.0
    # Rubber-like pads: soft contacts (12 ms time constant, ~1-2 mm squash under 50 N) so that closing on a
    # heavy object is an absorbed impact, not a 500 N spike (pads have contact priority over every object).
    pad_solref: float = 0.004
    pad_solimp: str = "0.95 0.99 0.001"
    pad_condim: int = 4
    linkage_solref: float = 0.005  # four-bar connect / coupling equality constraints
    finger_damping: float = 0.1  # driver joint damping (follower / inner knuckle get the same)
    pad_half: tuple = (0.015, 0.002, 0.0095)  # Menagerie pad boxes (half sizes, finger frame)
    stroke: float = 0.086  # jaw opening, fully open
    # Grasp centre ("TCP") along the tool axis, from the gripper base; fingertips are at 0.172.
    tcp_z: float = 0.151  # centre of the lower pad pair
    tip_z: float = 0.162  # measured from the finger mesh (open)
    # Compliant insertion mode (action dim "compliant" = 1): per-axis IK gains in the TCP frame
    # (x, y = finger axis, z = approach, then rotations), acting about `tool_tip` (TCP frame).
    compliant_gains: tuple = (0.15, 0.15, 1.0, 0.3, 0.3, 0.3)
    tool_tip: tuple = (0.0, 0.0, 0.014)  # default remote-compliance point (tasks override via Task.tool_tip)


@dataclass
class RobotCfg:
    n_arms: int = 1
    # Base plate centres on the bench (x, y) for ("right", "left"). A single arm sits at y = 0.
    base_xy_dual: tuple = ((0.0, -0.45), (0.0, 0.45))
    base_xy_single: tuple = ((0.0, 0.0),)
    base_z: float = 0.0  # bolted straight onto the bench top (table top = z 0)
    base_yaw: float = 0.0  # both arms face +x
    arm: ArmCfg = field(default_factory=ArmCfg)
    gripper: GripperCfg = field(default_factory=GripperCfg)

    @property
    def arms(self) -> tuple:
        return ARM_NAMES[: self.n_arms]

    @property
    def base_xy(self) -> tuple:
        return self.base_xy_dual[: self.n_arms] if self.n_arms == 2 else self.base_xy_single


# ---------------------------------------------------------------------- sensing
@dataclass
class CameraCfg:
    width: int = 320
    height: int = 240  # 4:3 like a RealSense D435 colour stream (640x480), downscaled
    fovy_wrist: float = 58.0
    fovy_fixed: float = 50.0
    # Wrist camera on the gripper base (RealSense on the xArm camera stand): offset (gripper frame) and the
    # point it looks at, on the tool axis a little past the fingertips.
    wrist_offset: tuple = (0.085, 0.0, 0.0)
    wrist_lookat_z: float = 0.21
    # Fixed cameras (world frame): position, look-at.
    fixed: dict = field(default_factory=lambda: {
        "front": ((1.15, 0.0, 0.55), (0.40, 0.0, 0.08)),
        "side": ((0.45, -0.95, 0.50), (0.45, 0.0, 0.08)),
        "top": ((0.45, 0.0, 1.05), (0.45, 0.0, 0.0)),
    })
    record: tuple = ("right_wrist", "front")  # cameras written to the dataset by default

    def all_cameras(self, robot: RobotCfg) -> tuple:
        return tuple(f"{a}_wrist" for a in robot.arms) + tuple(self.fixed)


# ---------------------------------------------------------------------- bench / scene
@dataclass
class BenchCfg:
    x_min: float = -0.30
    x_max: float = 0.95
    y_half: float = 0.60  # widened automatically for two arms
    top_z: float = 0.0
    thickness: float = 0.04
    height: float = 0.75  # floor at -height
    friction: float = 0.9
    n_distractors: int = 6  # pre-allocated loose objects, parked off-scene unless a shift places them
    background: bool = True  # lab walls, floor, ceiling lights, shelves, second table


@dataclass
class SimCfg:
    timestep: float = 0.002
    control_hz: float = 20.0
    max_dpos: float = 0.02  # max TCP translation per control step (0.4 m/s)
    max_drot: float = 0.15  # max TCP rotation per control step
    settle_s: float = 0.4
    episode_max_s: float | None = None  # None: task default (scaled with horizon)
    memory: str = "128M"
    cone: str = "elliptic"
    impratio: float = 10.0
    multiccd: bool = True


# ---------------------------------------------------------------------- tasks
@dataclass
class CubeStackCfg:
    name: str = "cube_stack"
    n_cubes: int = 4  # built into the model; a scenario may use fewer (the rest are parked)
    sizes: tuple = (0.050, 0.045, 0.040, 0.035)  # edge lengths, largest first ("by_size" order)
    density: float = 600.0  # wood
    friction: float = 0.8
    place_tol_xy: float = 0.012  # cube centre offset from the one below that still counts as stacked
    variant: str = "by_size"  # "by_size": largest at the bottom; "pattern": random order, shown as a ghost
    show_goal_ghost: bool = False
    spawn: tuple = (0.30, 0.62, -0.26, 0.26)  # x_min, x_max, y_min, y_max of the spawn area
    min_gap: float = 0.03  # between cubes at spawn
    episode_s_per_cube: float = 7.0


@dataclass
class LegoCfg:
    name: str = "lego"
    # Oversized LEGO-style bricks: `scale` x standard LEGO (8 mm pitch, 9.6 mm brick, 4.8 mm studs). At 4x a
    # 2x2 brick is 63.6 mm wide (the xArm gripper opens 86 mm), 38 mm tall, with 19 mm studs and a 3 mm
    # fit clearance: chunky, forgiving press-fits. Set scale=2 for DUPLO size (tighter tolerances needed).
    scale: float = 4.0
    base_pitch: float = 0.008
    base_brick_h: float = 0.0096
    base_stud_r: float = 0.0024
    gap: float = 0.0025  # brick footprint = n * pitch - gap (2.5 mm between neighbours: placing next to a brick tolerates ~1 mm error)
    stud_cyl_h: float = 0.0  # cylindrical part under the hemispherical stud cap (0: pure sphere cap)
    cavity_extra: float = 0.003  # cavity depth = stud protrusion + extra
    fit_clearance: float = 0.003  # cavity width - stud span (real LEGO: ~0.1 mm)
    density: float = 450.0  # hollow ABS
    friction: float = 0.5
    bricks: tuple = ("2x2", "2x2", "2x2", "2x4")  # built bricks; a scenario uses the first n
    plate_studs: tuple = (6, 6)
    plate_h: float = 0.010
    # Snap model: resistance ramping to snap_force over the last snap_engage of travel, then a weld once
    # seated and aligned (xy / yaw tolerances). Snapped bricks cannot be pulled off again.
    snap_force: float = 6.0
    snap_engage: float = 0.006
    snap_tol_xy: float = 0.004
    snap_tol_yaw: float = 0.12
    snap_tol_z: float = 0.002
    show_goal_ghost: bool = True
    spawn: tuple = (0.22, 0.66, -0.36, 0.36)
    plate_xy: tuple = (0.46, 0.0)  # nominal baseplate centre, randomised per episode
    min_gap: float = 0.03
    episode_s_per_brick: float = 8.0

    @property
    def pitch(self) -> float:
        return self.base_pitch * self.scale

    @property
    def brick_h(self) -> float:
        return self.base_brick_h * self.scale

    @property
    def stud_r(self) -> float:
        return self.base_stud_r * self.scale

    @property
    def cavity_depth(self) -> float:
        return self.stud_r + self.stud_cyl_h + self.cavity_extra


@dataclass
class CableCfg:
    name: str = "cable_plug"
    # RJ45 8P8C plug: body 11.7 x 8.0 x 21 mm + 16 mm boot (grasped), always latch-up.
    body_len: float = 0.021
    body_w: float = 0.0117
    body_h: float = 0.0080
    boot_len: float = 0.016
    boot_w: float = 0.0100
    boot_h: float = 0.0076
    plug_mass: float = 0.012
    grasp_from_tip: float = 0.037  # pad front edge (15 mm half-width) stays behind the 21 mm plug body
    plug_friction: float = 0.3  # MuJoCo takes max(friction) per pair; a grippy plug self-locks on the chamfer
    plug_joint_stiffness: float = 0.02
    plug_joint_damping: float = 0.002
    # Jack: channel = plug + clearance, lead-in chamfer, detent latch over the last mm, weld when seated.
    # Loose by design ("easy mode"): 1.6 mm total clearance + a 2.5 mm lead-in means the expert's 0.3 mm
    # perception noise rarely touches the chamfer. Real RJ45: ~0.2-0.5 mm clearance, ~1 mm lead-in.
    clearance: float = 0.0016
    depth: float = 0.020
    seat_depth: float = 0.0150
    chamfer: float = 0.0025
    wall: float = 0.0025
    port_friction: float = 0.3
    solref: float = 0.006
    solimp: str = "0.98 0.998 0.0005"
    latch_force: float = 10.0
    latch_engage: float = 0.010
    latch_max_lateral: float = 0.0015
    latch_max_angle: float = 0.09
    # Desktop patch panel standing on the bench, facing the robot (insertion along +x).
    rows: int = 2
    cols: int = 12
    port_pitch: float = 0.0185
    row_pitch: float = 0.04445  # 1U: the xArm fingers (37 mm of pads) must clear the plugs in the row above
    face_x: float = 0.62
    row_z: float = 0.095  # centre between the rows, above the bench
    panel_half_width: float = 0.135
    panel_thickness: float = 0.003
    # Cat6 patch cable as a capsule chain with ball joints; far end anchored in a "switch" on the bench.
    n_cables: int = 2  # built; a scenario uses the first n
    n_links: int = 18
    link_len: float = 0.035  # 0.63 m (slack of >= 0.15 m at every port: a taut cable drags the plug off axis)
    radius: float = 0.003
    mass_per_m: float = 0.045
    bend_stiffness: float = 0.004
    damping: float = 0.0008
    armature: float = 2e-6
    cable_friction: float = 0.8
    anchor_x: float = 0.50
    anchor_abs_y: float = 0.37
    spawn: tuple = (0.17, 0.58, -0.41, 0.41)
    episode_s_per_cable: float = 14.0

    @property
    def n_ports(self) -> int:
        return self.rows * self.cols


TASK_CFGS = {"cube_stack": CubeStackCfg, "lego": LegoCfg, "cable_plug": CableCfg}


# ---------------------------------------------------------------------- everything
@dataclass
class EnvCfg:
    task: Any = field(default_factory=CubeStackCfg)
    robot: RobotCfg = field(default_factory=RobotCfg)
    camera: CameraCfg = field(default_factory=CameraCfg)
    bench: BenchCfg = field(default_factory=BenchCfg)
    sim: SimCfg = field(default_factory=SimCfg)

    def __post_init__(self):
        if isinstance(self.task, str):
            self.task = TASK_CFGS[self.task]()
        elif isinstance(self.task, dict):
            d = dict(self.task)
            self.task = TASK_CFGS[d.pop("name")](**d)
        if self.robot.n_arms == 2 and self.bench.y_half < 0.85:
            self.bench.y_half = 0.85
        if self.robot.n_arms == 1 and self.camera.record and any(c.startswith("left_") for c in self.camera.record):
            self.camera.record = tuple(c for c in self.camera.record if not c.startswith("left_"))

    @property
    def task_name(self) -> str:
        return self.task.name

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> EnvCfg:
        return EnvCfg(task=d["task"], robot=_from(RobotCfg, d["robot"], arm=ArmCfg, gripper=GripperCfg),
                      camera=CameraCfg(**d["camera"]), bench=BenchCfg(**d["bench"]), sim=SimCfg(**d["sim"]))

    def with_overrides(self, overrides: dict) -> EnvCfg:
        """New cfg with dotted overrides, e.g. {"task.n_cubes": 3, "robot.n_arms": 2}."""
        d = self.to_dict()
        for k, v in overrides.items():
            cur = d
            parts = k.split(".")
            for p in parts[:-1]:
                cur = cur[p]
            cur[parts[-1]] = v
        return EnvCfg.from_dict(d)


def _from(cls, d, **sub):
    d = dict(d)
    for k, c in sub.items():
        d[k] = c(**d[k])
    return cls(**d)


def make_cfg(task: str = "cube_stack", **overrides) -> EnvCfg:
    """EnvCfg for a task with dotted overrides: make_cfg("lego", **{"task.bricks": ("2x2", "2x2"), "robot.n_arms": 2})."""
    cfg = EnvCfg(task=task)
    return cfg.with_overrides(overrides) if overrides else cfg
