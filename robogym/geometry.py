"""Small rigid-body math helpers. Quaternions are MuJoCo order (w, x, y, z)."""
from __future__ import annotations

import mujoco
import numpy as np


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def quat_conj(q):
    q = np.asarray(q, float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_rotate(q, v):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(q, float))
    return out


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def quat_from_rotvec(rv):
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < 1e-12:
        return np.array([1.0, 0, 0, 0])
    return quat_from_axis_angle(rv / th, th)


def rotvec_from_quat(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    s = np.linalg.norm(q[1:])
    if s < 1e-12:
        return np.zeros(3)
    return 2 * np.arctan2(s, q[0]) * q[1:] / s


def quat_error(q_target, q_current):
    """World-frame rotation vector that takes q_current to q_target."""
    return rotvec_from_quat(quat_mul(q_target, quat_conj(q_current)))


def quat_angle(q_a, q_b):
    return float(np.linalg.norm(quat_error(q_a, q_b)))


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def mat_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def yaw_quat(yaw):
    return quat_from_axis_angle([0, 0, 1], yaw)


def yaw_of_quat(q):
    R = quat_to_mat(q)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap_to(a, period):
    return (a + period / 2) % period - period / 2


def slerp(q0, q1, t):
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    rv = rotvec_from_quat(quat_mul(q1, quat_conj(q0)))
    return quat_mul(quat_from_rotvec(t * rv), q0)


def level_quat(q):
    """Nearest *level* orientation of a box-like object: the smallest rotation about a horizontal axis that
    makes the body axis closest to world +z exactly vertical (yaw is preserved). For an upright cube this
    is its yaw quaternion; for a cube handed over sideways it is the orientation it will rest in."""
    q = np.asarray(q, float)
    R = quat_to_mat(q)
    i = int(np.argmax(np.abs(R[2])))
    z_b = R[:, i] * np.sign(R[2, i])  # world direction of the body axis closest to up
    axis = np.cross(z_b, [0.0, 0.0, 1.0])
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return q
    return quat_mul(quat_from_axis_angle(axis / s, float(np.arctan2(s, z_b[2]))), q)


def nearest_angle(cands, cur):
    """The candidate angle closest to `cur` (mod 2 pi)."""
    cands = np.asarray(cands, float)
    return float(cands[np.argmin(np.abs(wrap(cands - cur)))])


def pose_mul(p1, q1, p2, q2):
    """(p1,q1) * (p2,q2): compose transforms."""
    return p1 + quat_rotate(q1, p2), quat_mul(q1, q2)


def pose_inv(p, q):
    qi = quat_conj(q)
    return -quat_rotate(qi, p), qi


def lookat_xyaxes(pos, target, up=(0, 0, 1)):
    """MJCF camera `xyaxes` for a camera at `pos` looking at `target` (cameras look along -z)."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, np.asarray(up, float))
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(f, [1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.concatenate([x, y])


def frame_z(z):
    """Rotation matrix whose z axis is `z` (for orienting capsules / cylinders)."""
    z = np.asarray(z, float)
    z = z / (np.linalg.norm(z) + 1e-12)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def min_jerk(t):
    t = np.clip(t, 0.0, 1.0)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def box_inertia(m, full):
    a, b, c = full
    return np.array([b * b + c * c, a * a + c * c, a * a + b * b]) * m / 12


def fmt(v, nd=6):
    return " ".join(f"{float(x):.{nd}g}" for x in np.atleast_1d(v))
