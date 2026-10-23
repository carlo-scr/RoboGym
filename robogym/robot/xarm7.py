"""UFACTORY xArm 7 (+ xArm gripper) as prefixed MJCF fragments.

Kinematics, inertials, joint limits, servo gains and the gripper linkage are the MuJoCo Menagerie numbers
(`assets/xarm7/menagerie_xarm7.xml`, BSD-3). We emit the XML ourselves so that one or two arms can be
placed with a prefix, gravity compensation switched on, collision classes set, and our own TCP / tip /
force-torque sites and a wrist camera added.

Arm frame conventions (per arm, prefix p = "r" / "l"):
  {p}_tcp   grasp centre between the finger pads: z = approach axis (out of the fingers), y = finger
            axis (the fingers move along +-y), x = the remaining axis. Pointing the TCP z down with
            x along world +x is `Q_DOWN` in env.py.
  {p}_tip   fingertip plane on the tool axis
  {p}_ft    wrist force/torque sensor site (gripper base), Menagerie gripper-base frame
Camera "{arm}_wrist" sits on the gripper base (RealSense-on-the-xArm-camera-stand assumption).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import RobotCfg
from ..geometry import fmt, lookat_xyaxes

MESH_DIR = Path(__file__).resolve().parent.parent / "assets" / "xarm7"
MESHES = ("link_base", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "end_tool", "base_link",
          "left_outer_knuckle", "left_finger", "left_inner_knuckle", "right_outer_knuckle", "right_finger", "right_inner_knuckle")

# (name, pos, quat, inertial(pos, quat, mass, diag), joint range or None (= +-2pi), size class, mesh, material)
_CHAIN = [
    ("link1", (0, 0, 0.267), (1, 0, 0, 0), ((-0.0002, 0.02905, -0.01233), (0.978953, -0.202769, -0.00441617, -0.0227264), 2.382, (0.00569127, 0.00533384, 0.00293865)), None, 1, "link1", "white"),
    ("link2", (0, 0, 0), (1, -1, 0, 0), ((0.00022, -0.12856, 0.01735), (0.50198, 0.86483, -0.00778841, 0.00483285), 1.869, (0.00959898, 0.00937717, 0.00201315)), (-2.059, 2.0944), 1, "link2", "white"),
    ("link3", (0, -0.293, 0), (1, 1, 0, 0), ((0.0466, -0.02463, -0.00768), (0.913819, 0.289775, 0.281481, -0.0416455), 1.6383, (0.00351721, 0.00294089, 0.00195868)), None, 2, "link3", "white"),
    ("link4", (0.0525, 0, 0), (1, 1, 0, 0), ((0.07047, -0.11575, 0.012), (0.422108, 0.852026, -0.126025, 0.282832), 1.7269, (0.00657137, 0.00647948, 0.00186763)), (-0.19198, 3.927), 2, "link4", "white"),
    ("link5", (0.0775, -0.3425, 0), (1, 1, 0, 0), ((-0.00032, 0.01604, -0.026), (0.999311, -0.0304457, 0.000577067, 0.0212082), 1.3203, (0.00534729, 0.00499076, 0.0013489)), None, 2, "link5", "white"),
    ("link6", (0, 0, 0), (1, 1, 0, 0), ((0.06469, 0.03278, 0.02141), (-0.217672, 0.772419, 0.16258, 0.574069), 1.325, (0.00245421, 0.00221646, 0.00107273)), (-1.69297, 3.14159), 3, "link6", "white"),
    ("link7", (0.076, 0.097, 0), (1, -1, 0, 0), ((0, -0.00677, -0.01098), (0.487612, 0.512088, -0.512088, 0.487612), 0.17, (0.000132176, 9.3e-05, 5.85236e-05)), None, 3, "end_tool", "gray"),
]
_BASE_INERTIAL = ((-0.021131, -0.0016302, 0.056488), (0.696843, 0.20176, 0.10388, 0.680376), 0.88556, (0.00382023, 0.00335282, 0.00167725))
_GRIPPER_BASE_INERTIAL = ((-0.00065489, -0.0018497, 0.048028), (0.997403, -0.0717512, -0.0061836, 0.000477479), 0.54156, (0.000471093, 0.000332307, 0.000254799))
JOINT_RANGE = [(-6.28319, 6.28319) if c[4] is None else c[4] for c in _CHAIN]
TWO_PI = 6.28319


# Finger four-bar geometry (gripper-base frame): knuckle pivot at y 0.035, finger offset (0.035465, 0.042039)
# from the pivot, pad inner face 0.024003 + 0.002 inside the finger origin. Closing by theta moves the pads
# inward AND along the tool axis (further from the palm) - a top-down grasp must be planned at the closed angle.
_K_Y, _F_Y, _F_Z, _PAD_IN = 0.035, 0.035465, 0.042039, 0.024003 + 0.002


def opening_for_angle(theta: float) -> float:
    """Jaw opening (pad inner faces) at driver angle theta (xarm gripper)."""
    return 2 * (_K_Y + _F_Y * np.cos(theta) - _F_Z * np.sin(theta) - _PAD_IN)


def angle_for_opening(width: float) -> float:
    """Driver angle at which the pads touch an object of `width` (xarm gripper), clipped to [0, 0.85]."""
    lo, hi = 0.0, 0.85
    if width >= opening_for_angle(lo):
        return lo
    if width <= opening_for_angle(hi):
        return hi
    for _ in range(40):
        mid = (lo + hi) / 2
        if opening_for_angle(mid) > width:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def finger_drop(theta: float) -> float:
    """How much further along the tool axis the pads sit at driver angle theta than when open (xarm gripper)."""
    return _F_Y * np.sin(theta) + _F_Z * (np.cos(theta) - 1.0)


def _inertial(i):
    pos, quat, mass, diag = i
    return f'<inertial pos="{fmt(pos)}" quat="{fmt(quat)}" mass="{mass}" diaginertia="{fmt(diag)}"/>'


def mesh_assets() -> str:
    return "\n    ".join(f'<mesh name="xarm_{m}" file="{MESH_DIR / (m + ".stl")}"/>' for m in MESHES)


def defaults_xml(cfg: RobotCfg) -> str:
    a, g = cfg.arm, cfg.gripper
    return f"""
    <default class="xarm">
      <geom type="mesh" material="xarm_white" contype="4" conaffinity="1" group="1"/>
      <joint axis="0 0 1" armature="{a.armature}" range="{-TWO_PI} {TWO_PI}" frictionloss="1"/>
      <site size="0.001" rgba="1 0 0 1" group="4"/>
      <default class="xarm_hand"><geom contype="1" conaffinity="7" material="xarm_black"/></default>
      <default class="xarm_spring_link"><joint axis="1 0 0" range="0 0.85" stiffness="0.05" springref="2.62" damping="{g.finger_damping}" armature="{g.joint_armature}" frictionloss="{g.joint_frictionloss}"/></default>
      <default class="xarm_driver"><joint axis="1 0 0" range="0 0.85" armature="{g.joint_armature}" damping="{g.finger_damping}" solreflimit="0.005 1" frictionloss="{g.joint_frictionloss}"/></default>
      <default class="xarm_follower"><joint axis="1 0 0" range="0 0.85" solreflimit="0.005 1" armature="{g.joint_armature}" damping="{g.finger_damping}" frictionloss="{g.joint_frictionloss}"/></default>
      <default class="xarm_visual"><geom type="mesh" contype="0" conaffinity="0" group="2" density="0"/></default>
      <default class="xarm_pad"><geom type="box" contype="1" conaffinity="7" condim="{g.pad_condim}" friction="{g.pad_friction} 0.05 0.002" solimp="{g.pad_solimp}" solref="{g.pad_solref} 1" mass="0" priority="1" size="{fmt(g.pad_half)}" rgba="0.05 0.1 0.4 1" group="3"/></default>
    </default>"""


def _xarm_gripper_xml(p: str, arm: str, cfg: RobotCfg) -> str:
    """Menagerie linkage gripper, prefixed. Fingers move along the gripper base's y axis."""
    g = cfg.gripper
    out = [f'<body name="{p}_gripper_base" quat="0 0 0 1" gravcomp="1">', _inertial(_GRIPPER_BASE_INERTIAL),
           f'<geom name="{p}_gripper_base" class="xarm_hand" material="xarm_white" mesh="xarm_base_link"/>']
    for side, s in (("left", 1), ("right", -1)):
        ko = (0, 0.021559 * s, 0.015181)
        kq = (0.47789, 0.87842, 0, 0) if side == "left" else (0.87842, 0.47789, 0, 0)
        out.append(f'<body name="{p}_{side}_outer_knuckle" pos="0 {0.035 * s} 0.059098" gravcomp="1">')
        out.append(_inertial((ko, kq, 0.033618, (1.9111e-05, 1.79089e-05, 1.90167e-06))))
        out.append(f'<joint name="{p}_{side}_driver" class="xarm_driver" axis="{s} 0 0"/>')
        out.append(f'<geom name="{p}_{side}_outer_knuckle" class="xarm_hand" mesh="xarm_{side}_outer_knuckle"/>')
        fq = (0.697634, 0.115353 * s, -0.115353 * s, 0.697634)
        out.append(f'<body name="{p}_{side}_finger" pos="0 {0.035465 * s} 0.042039" gravcomp="1">')
        out.append(_inertial(((0, -0.016413 * s, 0.029258), fq, 0.048304, (1.88037e-05, 1.7493e-05, 3.56792e-06))))
        out.append(f'<joint name="{p}_{side}_finger" class="xarm_follower" axis="{-s} 0 0"/>')
        out.append(f'<geom name="{p}_{side}_finger" class="xarm_visual" material="xarm_black" mesh="xarm_{side}_finger"/>')
        pad_y = -(_PAD_IN - g.pad_half[1]) * s  # inner face at Menagerie's 0.026003, thickness grows outwards
        out.append(f'<geom name="{p}_{side}_pad1" class="xarm_pad" pos="0 {pad_y} 0.032"/>')
        out.append(f'<geom name="{p}_{side}_pad2" class="xarm_pad" pos="0 {pad_y} 0.050"/>')
        out.append("</body></body>")
        iq = (0.664139, -0.242732 * s, 0.242713 * s, 0.664146)
        out.append(f'<body name="{p}_{side}_inner_knuckle" pos="0 {0.02 * s} 0.074098" gravcomp="1">')
        out.append(_inertial(((1.866e-06, 0.0220468 * s, 0.0261335), iq, 0.0230126, (8.34216e-06, 6.0949e-06, 2.75601e-06))))
        out.append(f'<joint name="{p}_{side}_inner_knuckle" class="xarm_spring_link" axis="{s} 0 0"/>')
        out.append(f'<geom name="{p}_{side}_inner_knuckle" class="xarm_hand" mesh="xarm_{side}_inner_knuckle"/>')
        out.append("</body>")
    out.append(_tool_sites_xml(p, arm, cfg))
    out.append("</body>")
    return "\n".join(out)


def _parallel_gripper_xml(p: str, arm: str, cfg: RobotCfg) -> str:
    """Simple two-slider jaw with the xArm gripper's stroke and pad layout (no linkage, MJX-friendly)."""
    g = cfg.gripper
    half = g.stroke / 2 + g.pad_half[1]
    out = [f'<body name="{p}_gripper_base" quat="0 0 0 1" gravcomp="1">',
           f'<geom name="{p}_gripper_base" type="cylinder" class="xarm_hand" material="xarm_white" pos="0 0 0.04" size="0.038 0.04" mass="0.55"/>',
           f'<geom name="{p}_palm" type="box" class="xarm_hand" pos="0 0 0.09" size="0.012 0.05 0.012" mass="0.1"/>']
    for side, s in (("left", 1), ("right", -1)):
        out.append(f'<body name="{p}_{side}_finger" pos="0 {s * half} 0.10" gravcomp="1">')
        out.append(f'<joint name="{p}_{side}_driver" type="slide" axis="0 {-s} 0" range="0 {g.stroke / 2}" damping="5" armature="0.01"/>')
        out.append(f'<geom name="{p}_{side}_finger" type="box" class="xarm_hand" pos="0 {s * 0.004} 0.02" size="0.012 0.004 0.025" mass="0.05"/>')
        out.append(f'<geom name="{p}_{side}_pad1" class="xarm_pad" pos="0 {s * (g.pad_half[1] - 0.002)} 0.033" quat="0.7071 0 0 0.7071"/>')
        out.append(f'<geom name="{p}_{side}_pad2" class="xarm_pad" pos="0 {s * (g.pad_half[1] - 0.002)} 0.051" quat="0.7071 0 0 0.7071"/>')
        out.append("</body>")
    out.append(_tool_sites_xml(p, arm, cfg))
    out.append("</body>")
    return "\n".join(out)


def _tool_sites_xml(p: str, arm: str, cfg: RobotCfg) -> str:
    from ..config import CameraCfg

    g = cfg.gripper
    cam = CameraCfg()  # camera geometry is robot-side; the env's CameraCfg only sets resolution / recording
    cp = np.asarray(cam.wrist_offset)
    xy = lookat_xyaxes(cp, (0, 0, cam.wrist_lookat_z), up=(1, 0, 0))
    return "\n".join([
        f'<site name="{p}_tcp" pos="0 0 {g.tcp_z}" size="0.004" rgba="1 0.5 0 1" group="4"/>',
        f'<site name="{p}_tip" pos="0 0 {g.tip_z}" size="0.003" rgba="1 0 0 1" group="4"/>',
        f'<site name="{p}_ft" pos="0 0 0.0" size="0.003" rgba="0 1 0 1" group="4"/>',
        f'<camera name="{arm}_wrist" pos="{fmt(cp)}" xyaxes="{fmt(xy)}" fovy="{cam.fovy_wrist}"/>',
        f'<geom name="{p}_cam_body" type="box" class="xarm_visual" pos="{fmt(cp + np.array([0.0, 0, -0.012]))}" size="0.013 0.045 0.012" rgba="0.12 0.12 0.13 1"/>',
        f'<geom name="{p}_cam_mount" type="box" class="xarm_visual" pos="{fmt((cp[0] / 2 + 0.02, 0, 0.012))}" size="{cp[0] / 2 - 0.005} 0.01 0.004" rgba="0.3 0.3 0.32 1"/>',
    ])


def arm_xml(cfg: RobotCfg, arm: str, base_xyz, yaw: float = 0.0) -> str:
    """One xArm 7 with gripper rooted at `base_xyz` (world), rotated by `yaw` about z."""
    p = arm[0]
    a = cfg.arm
    gc = ' gravcomp="1"' if a.gravcomp else ""
    out = [f'<body name="{p}_link_base" pos="{fmt(base_xyz)}" quat="{fmt([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])}" childclass="xarm">',
           _inertial(_BASE_INERTIAL),
           f'<geom name="{p}_link_base" mesh="xarm_link_base"/>',
           f'<geom name="{p}_base_plate" type="cylinder" class="xarm_visual" pos="0 0 0.003" size="0.085 0.003" rgba="0.2 0.2 0.22 1"/>']
    close = ["</body>"]
    for i, (name, pos, quat, inert, _rng, _size, mesh, mat) in enumerate(_CHAIN):
        lo, hi = JOINT_RANGE[i]
        out.append(f'<body name="{p}_{name}" pos="{fmt(pos)}" quat="{fmt(quat)}"{gc}>')
        out.append(_inertial(inert))
        out.append(f'<joint name="{p}_j{i + 1}" range="{lo} {hi}" damping="{a.joint_damping[i]}"/>')
        out.append(f'<geom name="{p}_{name}" mesh="xarm_{mesh}" material="xarm_{mat}"/>')
        close.append("</body>")
    out.append(_xarm_gripper_xml(p, arm, cfg) if cfg.gripper.kind == "xarm" else _parallel_gripper_xml(p, arm, cfg))
    out.extend(close)
    return "\n".join(out)


def actuators_xml(cfg: RobotCfg, arm: str) -> str:
    p, a, g = arm[0], cfg.arm, cfg.gripper
    out = []
    for i in range(7):
        lo, hi = JOINT_RANGE[i]
        out.append(f'<general name="{p}_j{i + 1}" joint="{p}_j{i + 1}" biastype="affine" ctrlrange="{lo} {hi}" '
                   f'gainprm="{a.kp[i]}" biasprm="0 {-a.kp[i]} {-a.kv[i]}" forcerange="{-a.force[i]} {a.force[i]}"/>')
    if g.kind == "xarm":
        # Position servo on the mean driver angle (ctrl 0 = open ... 255 = closed = 0.85 rad), force-limited.
        out.append(f'<general name="{p}_grip" tendon="{p}_split" forcerange="{-g.max_force} {g.max_force}" ctrlrange="0 255" '
                   f'biastype="affine" gainprm="{g.servo_kp * 0.85 / 255:.6g}" biasprm="0 {-g.servo_kp} {-g.servo_kv}"/>')
    else:
        # Position servo on the finger slide (ctrl = opening per finger), force-limited.
        out.append(f'<general name="{p}_grip" joint="{p}_left_driver" forcerange="{-g.max_force} {g.max_force}" ctrlrange="0 {g.stroke / 2}" '
                   f'biastype="affine" gainprm="2000" biasprm="0 -2000 -40"/>')
    return "\n".join(out)


def tendons_xml(cfg: RobotCfg, arm: str) -> str:
    p = arm[0]
    if cfg.gripper.kind != "xarm":
        return ""
    return f'<fixed name="{p}_split"><joint joint="{p}_right_driver" coef="0.5"/><joint joint="{p}_left_driver" coef="0.5"/></fixed>'


def equalities_xml(cfg: RobotCfg, arm: str) -> str:
    p = arm[0]
    if cfg.gripper.kind == "xarm":
        return "\n".join([
            f'<connect name="{p}_rlink" anchor="0 0.015 0.015" body1="{p}_right_finger" body2="{p}_right_inner_knuckle" solref="{cfg.gripper.linkage_solref} 1"/>',
            f'<connect name="{p}_llink" anchor="0 -0.015 0.015" body1="{p}_left_finger" body2="{p}_left_inner_knuckle" solref="{cfg.gripper.linkage_solref} 1"/>',
            f'<joint name="{p}_couple" joint1="{p}_left_driver" joint2="{p}_right_driver" polycoef="0 1 0 0 0" solref="{cfg.gripper.linkage_solref} 1"/>',
        ])
    return f'<joint name="{p}_couple" joint1="{p}_left_driver" joint2="{p}_right_driver" polycoef="0 1 0 0 0" solref="0.005 1"/>'


def contacts_xml(cfg: RobotCfg, arm: str) -> str:
    p = arm[0]
    ex = [f'<exclude body1="{p}_gripper_base" body2="{p}_link6"/>', f'<exclude body1="{p}_gripper_base" body2="{p}_link5"/>']
    if cfg.gripper.kind == "xarm":
        for s in ("left", "right"):
            ex += [f'<exclude body1="{p}_{s}_inner_knuckle" body2="{p}_{s}_outer_knuckle"/>',
                   f'<exclude body1="{p}_{s}_inner_knuckle" body2="{p}_{s}_finger"/>',
                   f'<exclude body1="{p}_{s}_inner_knuckle" body2="{p}_gripper_base"/>',
                   f'<exclude body1="{p}_{s}_outer_knuckle" body2="{p}_gripper_base"/>']
    return "\n".join(ex)


def sensors_xml(cfg: RobotCfg, arm: str) -> str:
    p = arm[0]
    return f'<force name="{p}_force" site="{p}_ft"/><torque name="{p}_torque" site="{p}_ft"/>'


def materials_xml() -> str:
    return ('<material name="xarm_white" rgba="0.93 0.93 0.94 1" specular="0.3"/>'
            '<material name="xarm_gray" rgba="0.753 0.753 0.753 1"/>'
            '<material name="xarm_black" rgba="0.149 0.149 0.149 1"/>')
