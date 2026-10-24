"""Procedural MJCF for the lab bench scene and the top-level model assembler.

World frame: bench top = z 0, robot base plate(s) at x = 0, workspace towards +x, +z up.
Collision groups (contype / conaffinity):
  bench, objects, gripper   1 / 7   collide with everything
  cables                    2 / 1   collide with bench/objects/gripper, not with themselves or arm links
  arm links                 4 / 1   collide with bench/objects/gripper, not with each other
  visual-only               0 / 0
Tasks contribute (worldbody xml, asset xml, equality xml, contact xml) fragments via Task.xml().
"""
from __future__ import annotations

import numpy as np

from . import textures as T
from .config import EnvCfg
from .geometry import fmt, lookat_xyaxes, mat_to_quat
from .robot import xarm7

LIGHTS = ("key", "fill", "ceil_a", "ceil_b", "spot", "back")
FRONT_QUAT = mat_to_quat(T.FRONT_MAT)
PARK = np.array([3.0, 3.0, -0.60])  # hidden parking shelf for inactive free bodies (far off every camera)


def box(name, pos, size, cls="world", rgba=None, euler=None, quat=None, extra=""):
    s = f'<geom name="{name}" type="box" class="{cls}" pos="{fmt(pos)}" size="{fmt(size)}"'
    if euler is not None:
        s += f' euler="{fmt(euler)}"'
    if quat is not None:
        s += f' quat="{fmt(quat)}"'
    if rgba is not None:
        s += f' rgba="{fmt(rgba, 3)}"'
    return s + f" {extra}/>"


def front_plane(name, pos, half_wh, material):
    """Visual textured plane facing the robot (-x normal). half_wh = (half width along y, half height)."""
    return (f'<geom name="{name}" type="plane" class="visual" pos="{fmt(pos)}" size="{fmt(half_wh)} {2 * max(half_wh):.3f}" '
            f'quat="{fmt(FRONT_QUAT)}" material="{material}"/>')


DISTRACTOR_SHAPES = [
    # screwdriver: handle + shaft
    '<geom type="capsule" size="0.011 0.035" euler="0 1.5708 0" rgba="0.85 0.15 0.1 1" mass="0.05"/>'
    '<geom type="cylinder" size="0.003 0.04" pos="0.075 0 0" euler="0 1.5708 0" rgba="0.7 0.7 0.72 1" mass="0.02"/>',
    '<geom type="cylinder" size="0.03 0.012" rgba="0.1 0.1 0.1 1" mass="0.08"/>',  # tape roll
    '<geom type="box" size="0.05 0.03 0.015" rgba="0.95 0.85 0.2 1" mass="0.15"/>',  # label printer
    '<geom type="box" size="0.04 0.025 0.004" rgba="0.9 0.9 0.92 0.8" mass="0.02"/>',  # bag of cable ties
    '<geom type="sphere" size="0.02" rgba="0.2 0.6 0.9 1" mass="0.03"/>',  # ball
    '<geom type="cylinder" size="0.035 0.045" rgba="0.95 0.95 0.95 1" mass="0.25"/>',  # mug
]


def _bench_xml(cfg: EnvCfg):
    b = cfg.bench
    cx, hx = (b.x_min + b.x_max) / 2, (b.x_max - b.x_min) / 2
    out = [
        box("bench", (cx, 0, b.top_z - b.thickness / 2), (hx, b.y_half, b.thickness / 2), extra=f'material="bench" friction="{b.friction} 0.01 0.001"'),
        box("bench_mat", (cx + 0.05, 0, b.top_z + 0.0004), (hx - 0.1, b.y_half - 0.04, 0.0005), cls="visual", extra='material="mat"'),
    ]
    for sx in (b.x_min + 0.05, b.x_max - 0.05):
        for sy in (-b.y_half + 0.05, b.y_half - 0.05):
            out.append(box(f"leg_{len(out)}", (sx, sy, (b.top_z - b.thickness - b.height) / 2 + 0.0), (0.025, 0.025, (b.height - b.thickness) / 2), cls="visual", rgba=(0.25, 0.26, 0.28, 1)))
    # hidden parking shelf for inactive objects
    out.append(box("parking", PARK + (0.5, 0, -0.05), (1.0, 0.5, 0.02), rgba=(0.3, 0.3, 0.3, 1)))
    return "\n".join(out)


def _background_xml(cfg: EnvCfg):
    if not cfg.bench.background:
        return "", []
    b = cfg.bench
    fz = -b.height
    out, assets = [], []
    assets.append(f'<texture name="pegboard" type="2d" file="{T.pegboard()}"/><material name="pegboard" texture="pegboard" texrepeat="6 3"/>')
    out.append(box("wall_back", (-0.9, 0, fz + 1.5), (0.02, 3.0, 1.5), cls="visual", rgba=(0.86, 0.87, 0.88, 1)))
    out.append(f'<geom name="pegboard" type="plane" class="visual" pos="-0.878 0 {fz + 1.35}" size="1.2 0.6 2.4" quat="{fmt(mat_to_quat(-T.FRONT_MAT @ np.diag([1, -1, 1])))}" material="pegboard"/>')
    out.append(box("wall_side", (0.5, -(b.y_half + 1.2), fz + 1.5), (2.0, 0.02, 1.5), cls="visual", rgba=(0.84, 0.85, 0.87, 1)))
    out.append(box("ceiling", (0.3, 0, fz + 3.0), (3, 3, 0.02), cls="visual", rgba=(0.9, 0.9, 0.92, 1)))
    for i, x in enumerate((-0.3, 0.7)):
        out.append(box(f"fixture{i}", (x, 0, fz + 2.97), (0.15, 0.6, 0.02), cls="visual", rgba=(1, 1, 0.97, 1)))
    # a second bench with a monitor + a tool cabinet
    out.append(box("bench2", (0.3, b.y_half + 0.9, -0.03), (0.6, 0.35, 0.03), cls="visual", extra='material="bench"'))
    out.append(box("monitor", (0.1, b.y_half + 1.1, 0.25), (0.02, 0.28, 0.17), cls="visual", rgba=(0.05, 0.05, 0.06, 1)))
    out.append(box("cabinet", (b.x_max + 0.45, -0.1, fz + 0.45), (0.3, 0.45, 0.45), cls="visual", rgba=(0.55, 0.12, 0.12, 1)))
    assets.append(f'<texture name="logo" type="2d" file="{T.logo_plate()}"/><material name="logo" texture="logo"/>')
    out.append(f'<geom name="logo" type="plane" class="visual" pos="-0.876 0.0 {fz + 2.2}" size="0.3 0.075 0.6" quat="{fmt(mat_to_quat(-T.FRONT_MAT @ np.diag([1, -1, 1])))}" material="logo"/>')
    return "\n".join(out), assets


def _cameras_xml(cfg: EnvCfg):
    out = []
    for name, (pos, tgt) in cfg.camera.fixed.items():
        out.append(f'<camera name="{name}" pos="{fmt(pos)}" xyaxes="{fmt(lookat_xyaxes(pos, tgt))}" fovy="{cfg.camera.fovy_fixed}"/>')
        # a little tripod-mounted camera body so the sensors are visible in third-person renders
        out.append(f'<geom name="cam_{name}" type="box" class="visual" pos="{fmt(np.asarray(pos) + (0, 0, -0.02))}" size="0.03 0.045 0.015" rgba="0.1 0.1 0.11 1"/>')
    return "\n".join(out)


def _distractors_xml(cfg: EnvCfg):
    out = []
    for i in range(cfg.bench.n_distractors):
        parts = DISTRACTOR_SHAPES[i % len(DISTRACTOR_SHAPES)].split("<geom ")[1:]
        geoms = "".join(f'<geom name="distractor{i}_g{j}" class="world" ' + g for j, g in enumerate(parts))
        out.append(f'<body name="distractor{i}" pos="{fmt(PARK + (0.2 * i, 0.3, 0.05))}"><freejoint name="distractor{i}"/>{geoms}</body>')
    return "\n".join(out)


def build_mjcf(cfg: EnvCfg, task_xml: str = "", task_assets=(), task_equality: str = "", task_contact: str = "", task_defaults: str = "") -> str:
    r, cam = cfg.robot, cfg.camera
    bg, a_bg = _background_xml(cfg)
    arms, acts, tends, eqs, cons, sens = [], [], [], [], [], []
    for arm, (bx, by) in zip(r.arms, r.base_xy):
        arms.append(xarm7.arm_xml(r, arm, (bx, by, r.base_z), r.base_yaw))
        acts.append(xarm7.actuators_xml(r, arm))
        tends.append(xarm7.tendons_xml(r, arm))
        eqs.append(xarm7.equalities_xml(r, arm))
        cons.append(xarm7.contacts_xml(r, arm))
        sens.append(xarm7.sensors_xml(r, arm))
    w, h = cam.width, cam.height
    return f"""<mujoco model="robogym_{cfg.task_name}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{cfg.sim.timestep}" integrator="implicitfast" cone="{cfg.sim.cone}" impratio="{cfg.sim.impratio}">
    <flag multiccd="{'enable' if cfg.sim.multiccd else 'disable'}"/>
  </option>
  <size memory="{cfg.sim.memory}"/>
  <visual>
    <global offwidth="{max(w, 1920)}" offheight="{max(h, 1080)}"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.25 0.25 0.25" specular="0 0 0"/>
    <map znear="0.005" zfar="30"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.82 0.86 0.9" rgb2="0.5 0.55 0.6" width="256" height="256"/>
    <texture name="floor" type="2d" file="{T.lab_floor()}"/>
    <material name="floor" texture="floor" texrepeat="2 2" texuniform="true" reflectance="0.05"/>
    <texture name="bench" type="2d" file="{T.wood()}"/>
    <material name="bench" texture="bench" texrepeat="1 1" texuniform="true" specular="0.2"/>
    <texture name="bench_v1" type="2d" file="{T.wood(seed=1, base=(150, 105, 70))}"/>
    <material name="bench_v1" texture="bench_v1" texrepeat="1 1" texuniform="true" specular="0.25"/>
    <texture name="bench_v2" type="2d" file="{T.wood(seed=2, base=(190, 160, 120))}"/>
    <material name="bench_v2" texture="bench_v2" texrepeat="1 1" texuniform="true" specular="0.15"/>
    <texture name="bench_dark" type="2d" file="{T.wood(seed=3, base=(95, 70, 50))}"/>
    <material name="bench_dark" texture="bench_dark" texrepeat="1 1" texuniform="true" specular="0.3"/>
    <texture name="mat" type="2d" file="{T.rubber_mat()}"/>
    <material name="mat" texture="mat" texrepeat="4 4" texuniform="true"/>
    <texture name="mat_grey" type="2d" file="{T.rubber_mat(color=(120, 122, 126), seed=2)}"/>
    <material name="mat_grey" texture="mat_grey" texrepeat="4 4" texuniform="true"/>
    <texture name="mat_green" type="2d" file="{T.rubber_mat(color=(60, 90, 70), seed=3)}"/>
    <material name="mat_green" texture="mat_green" texrepeat="4 4" texuniform="true"/>
    <texture name="mat_blue" type="2d" file="{T.rubber_mat(color=(55, 70, 110), seed=4)}"/>
    <material name="mat_blue" texture="mat_blue" texrepeat="4 4" texuniform="true"/>
    {xarm7.materials_xml()}
    {xarm7.mesh_assets()}
    {"".join(a_bg)}
    {"".join(task_assets)}
  </asset>
  <default>
    <joint axis="0 0 1"/>
    <geom solref="0.004 1" solimp="0.95 0.99 0.001"/>
    <default class="world"><geom contype="1" conaffinity="7" friction="0.8 0.01 0.001"/></default>
    <default class="object"><geom contype="1" conaffinity="7" condim="4" friction="0.8 0.005 0.0005"/></default>
    <default class="visual"><geom contype="0" conaffinity="0" group="2" density="0"/></default>
    <default class="ghost"><geom contype="0" conaffinity="0" group="5" density="0"/></default>
    {xarm7.defaults_xml(r)}
    {task_defaults}
  </default>
  <worldbody>
    <light name="key" pos="0.5 0.3 2.2" dir="-0.1 -0.1 -1" diffuse="0.6 0.6 0.6" specular="0.25 0.25 0.25" castshadow="true"/>
    <light name="fill" pos="-0.3 -1.0 1.6" dir="0.4 0.6 -0.6" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <light name="ceil_a" pos="-0.3 0 2.2" dir="0.2 0 -1" diffuse="0.2 0.2 0.2" castshadow="false"/>
    <light name="ceil_b" pos="0.7 0 2.2" dir="-0.1 0 -1" diffuse="0.2 0.2 0.2" castshadow="false"/>
    <light name="spot" pos="0.9 -0.5 1.2" dir="-0.5 0.5 -0.8" diffuse="0 0 0" cutoff="20" exponent="20" castshadow="true"/>
    <light name="back" pos="1.3 0 0.8" dir="-1 0 -0.4" diffuse="0 0 0" castshadow="false"/>
    <geom name="floor" type="plane" pos="0 0 {-cfg.bench.height}" size="0 0 1" material="floor" class="world"/>
    <body name="latch_anchor" pos="0 0 0"/>
    {_bench_xml(cfg)}
    {bg}
    {_cameras_xml(cfg)}
    {"".join(arms)}
    {task_xml}
    {_distractors_xml(cfg)}
  </worldbody>
  <contact>
    {"".join(cons)}
    {task_contact}
  </contact>
  <tendon>
    {"".join(tends)}
  </tendon>
  <equality>
    {"".join(eqs)}
    {task_equality}
  </equality>
  <actuator>
    {"".join(acts)}
  </actuator>
  <sensor>
    {"".join(sens)}
  </sensor>
</mujoco>
"""
