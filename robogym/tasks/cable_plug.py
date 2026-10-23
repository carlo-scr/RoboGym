"""Ethernet (RJ45) cable plugging on the bench: pick a patch cable's plug off the table and plug it into
the LED-marked port of a desktop patch panel facing the robot; with n_cables > 1, plug every cable into
its own marked port in order (long horizon). Ported from RobotPluggingSim (jack wedges with a lead-in
chamfer, capsule-chain cable, detent latch + weld), adapted to the xArm bench.
"""
from __future__ import annotations

import numpy as np

from .. import geometry as G
from .. import textures as T
from ..scene import PARK, box, front_plane
from .base import Task, register

CABLE_COLORS = {
    "blue": (0.15, 0.35, 0.85, 1), "grey": (0.55, 0.57, 0.6, 1), "yellow": (0.95, 0.8, 0.15, 1), "green": (0.2, 0.65, 0.3, 1),
    "white": (0.92, 0.92, 0.9, 1), "black": (0.08, 0.08, 0.09, 1), "red": (0.85, 0.12, 0.12, 1), "orange": (1.0, 0.45, 0.05, 1),
    "purple": (0.5, 0.2, 0.7, 1), "pink": (0.95, 0.45, 0.65, 1), "cyan": (0.1, 0.75, 0.85, 1),
}
NOMINAL_CABLE_COLORS = ("blue", "grey", "yellow", "green", "white")
OOD_CABLE_COLORS = ("red", "orange", "purple", "pink", "cyan", "black")
TARGET_LED = (0.15, 0.45, 1.0, 1)
LED_OFF = (0.06, 0.07, 0.08, 1)
SUB = ("reach", "grasp", "carry", "align", "insert")
_STUB = np.array([[-0.022, 0, 0], [-0.040, 0, -0.004], [-0.054, 0, -0.024], [-0.061, 0, -0.068], [-0.063, 0, -0.17]])


@register
class CablePlug(Task):
    name = "cable_plug"
    failure_types = ("misalign", "partial_insert", "wrong_port")
    shift_types = ("cable_color", "boot_color", "panel_color", "dense_ports", "led_distractors", "heavy_cable", "stiff_cable", "panel_offset", "distractor_cable")
    outcome_modes = ("none", "dropped", "wrong_port", "not_seated", "jammed", "grasp_missed", "incomplete", "sim_unstable", "other")

    def __init__(self, cfg):
        super().__init__(cfg)
        tc = self.tc
        self.tip_len = tc.boot_len + tc.body_len
        # remote-compliance point: the plug tip, in the TCP frame (plug axis along TCP x, grasp centre at the TCP)
        self.tool_tip = (tc.grasp_from_tip, 0.0, cfg.robot.gripper.tip_z - cfg.robot.gripper.tcp_z - 0.004)
        self.compliant_gains = (1.0, 0.15, 0.15, 0.3, 0.3, 0.3)  # stiff along the plug axis (TCP x), soft laterally

    # ---- panel layout
    def port_rc(self, k):
        return divmod(k, self.tc.cols)

    def port_yz(self, k):
        r, c = self.port_rc(k)
        tc = self.tc
        return ((tc.cols - 1) / 2 - c) * tc.port_pitch, ((tc.rows - 1) / 2 - r) * tc.row_pitch

    # ---- model
    def _jack_assets(self):
        tc = self.tc
        W, H = tc.body_w + tc.clearance, tc.body_h + tc.clearance
        c, D = tc.chamfer, tc.depth
        th = max(tc.wall, c + 0.001)
        out = []
        for name, L in (("jack_wedge_h", W + 2 * (c + 0.001)), ("jack_wedge_v", H + 2 * (c + 0.001))):
            v = []
            for yy in (-L / 2, L / 2):
                v += [(-c, yy, -th), (-c, yy, -c), (0, yy, 0), (D, yy, 0), (D, yy, -th)]
            out.append(f'<mesh name="{name}" vertex="{" ".join(G.fmt(x) for x in v)}"/>')
        return out

    def _panel_xml(self):
        tc = self.tc
        W, H = tc.body_w + tc.clearance, tc.body_h + tc.clearance
        t, c, w, D = tc.panel_thickness, tc.chamfer, tc.wall, tc.depth
        hw = tc.panel_half_width
        s2 = np.sqrt(2) / 2
        out, assets = [], []
        rng = np.random.default_rng(0)
        total_h = tc.rows * tc.row_pitch + 0.02
        # housing behind the face (visual) and the face strips between jacks
        out.append(box("panel_housing", (0.03, 0, 0), (0.03, hw, total_h / 2), cls="visual", extra='material="panel"'))
        out.append(box("panel_foot", (0.03, 0, -total_h / 2 - 0.005), (0.04, hw + 0.01, 0.005), cls="world", extra='material="panel"'))
        for r in range(tc.rows):
            zr = ((tc.rows - 1) / 2 - r) * tc.row_pitch
            strip_h = (tc.row_pitch / 2 - H / 2) / 2
            out.append(box(f"panel{r}_top", (t / 2, 0, zr + H / 2 + strip_h), (t / 2, hw, strip_h), cls="port"))
            out.append(box(f"panel{r}_bot", (t / 2, 0, zr - H / 2 - strip_h), (t / 2, hw, strip_h), cls="port"))
            ys = sorted(self.port_yz(r * tc.cols + cc)[0] for cc in range(tc.cols))
            edges = [-hw] + sum([[y - W / 2, y + W / 2] for y in ys], []) + [hw]
            for i in range(0, len(edges), 2):
                y0, y1 = edges[i], edges[i + 1]
                out.append(box(f"panel{r}_strip{i // 2}", (t / 2, (y0 + y1) / 2, zr), (t / 2, (y1 - y0) / 2, H / 2), cls="port"))
            lab = T.port_labels(tc.cols, r * tc.cols + 1)
            assets.append(f'<texture name="labels{r}" type="2d" file="{lab}"/><material name="labels{r}" texture="labels{r}" specular="0.1"/>')
            out.append(front_plane(f"labels{r}", (-0.0004, 0, zr + H / 2 + 0.0085), (tc.cols * tc.port_pitch / 2, 0.003), f"labels{r}"))
            for cc in range(tc.cols):
                k = r * tc.cols + cc
                y, z = self.port_yz(k)
                g = f"port{k}"
                zt, zb, yl, yr = z + H / 2, z - H / 2, y - W / 2, y + W / 2
                out.append(f'<geom name="{g}_bot" type="mesh" mesh="jack_wedge_h" class="port" pos="{G.fmt((0, y, zb))}"/>')
                out.append(f'<geom name="{g}_top" type="mesh" mesh="jack_wedge_h" class="port" pos="{G.fmt((0, y, zt))}" quat="0 1 0 0"/>')
                out.append(f'<geom name="{g}_r" type="mesh" mesh="jack_wedge_v" class="port" pos="{G.fmt((0, yr, z))}" quat="{s2} {s2} 0 0"/>')
                out.append(f'<geom name="{g}_l" type="mesh" mesh="jack_wedge_v" class="port" pos="{G.fmt((0, yl, z))}" quat="{s2} {-s2} 0 0"/>')
                out.append(box(f"{g}_back", (D + w / 2, y, z), (w / 2, W / 2 + w + c, H / 2 + w + c), cls="port"))
                out.append(box(f"{g}_pins", (D - 0.0005, y, z + H / 2 - 0.0015), (0.0004, W * 0.35, 0.001), cls="visual", rgba=(0.8, 0.65, 0.2, 1)))
                out.append(box(f"{g}_led", (-0.0006, y + 0.004, zt + 0.003), (0.0006, 0.0015, 0.0011), cls="visual", rgba=LED_OFF))
                out.append(f'<site name="{g}" pos="{G.fmt((0, y, z))}" size="0.002" rgba="0 1 0 0.3" group="4"/>')
                out.append(f'<body name="occ{k}" pos="{G.fmt((0, y, z))}">')
                seat = tc.seat_depth
                out.append(box(f"occ{k}_body", (seat - tc.body_len / 2, 0, 0), (tc.body_len / 2, tc.body_w / 2, tc.body_h / 2), cls="occ", rgba=(0.85, 0.9, 0.95, 0.85)))
                out.append(box(f"occ{k}_boot", (seat - tc.body_len - tc.boot_len / 2, 0, 0), (tc.boot_len / 2, tc.boot_w / 2, tc.boot_h / 2), cls="occ", rgba=(0.2, 0.4, 0.8, 1)))
                stub = _STUB + np.array([seat - self.tip_len + 0.022, 0, 0]) + rng.normal(0, 0.003, _STUB.shape)
                stub[0] = [seat - self.tip_len, 0, 0]
                for i in range(len(stub) - 1):
                    out.append(f'<geom name="occ{k}_c{i}" type="capsule" class="visual" fromto="{G.fmt(stub[i])} {G.fmt(stub[i + 1])}" size="{tc.radius}" rgba="0.2 0.4 0.8 1"/>')
                out.append("</body>")
        return "\n".join(out), assets

    def _cable_xml(self, c):
        tc = self.tc
        m_link = tc.mass_per_m * tc.link_len
        L = tc.link_len
        j = f'type="ball" stiffness="{tc.bend_stiffness}" damping="{tc.damping}" armature="{tc.armature}"'
        xml, close = [], []
        for i in range(tc.n_links):
            pos = "0 0 0" if i == 0 else f"{L} 0 0"
            xml.append(f'<body name="cable{c}_{i}" pos="{pos}"><joint name="cable{c}_{i}" {j}/>'
                       f'<geom name="cable{c}_{i}" class="cable" type="capsule" fromto="0 0 0 {L} 0 0" size="{tc.radius}" mass="{m_link:.6g}"/>')
            close.append("</body>")
        bl, bw, bh = tc.boot_len, tc.boot_w, tc.boot_h
        tip = self.tip_len
        xml.append(f'<body name="plug{c}" pos="{L} 0 0"><joint name="plug{c}" type="ball" stiffness="{tc.plug_joint_stiffness}" damping="{tc.plug_joint_damping}" armature="{tc.armature}"/>')
        xml.append(f'<inertial pos="{tip / 2} 0 0" mass="{tc.plug_mass}" diaginertia="{G.fmt(G.box_inertia(tc.plug_mass, (tip, tc.body_w, tc.body_h)))}"/>')
        xml.append(box(f"plug{c}_boot", (bl / 2, 0, 0), (bl / 2, bw / 2, bh / 2), cls="plug", rgba=(0.15, 0.35, 0.85, 1)))
        xml.append(box(f"plug{c}_body", (bl + tc.body_len / 2, 0, 0), (tc.body_len / 2, tc.body_w / 2, tc.body_h / 2), cls="plug", extra='material="plugbody"'))
        xml.append(box(f"plug{c}_latch", (bl + 0.009, 0, tc.body_h / 2 + 0.0012), (0.008, 0.003, 0.0006), cls="visual", euler=(0, -0.1, 0), extra='material="plugbody"'))
        xml.append(box(f"plug{c}_pins", (tip - 0.002, 0, -tc.body_h / 2 + 0.0014), (0.0021, tc.body_w * 0.35, 0.0012), cls="visual", rgba=(0.85, 0.7, 0.2, 1)))
        for i in range(3):
            xml.append(box(f"plug{c}_rib{i}", (0.003 + i * 0.004, 0, 0), (0.0008, bw / 2 + 0.0004, bh / 2 + 0.0004), cls="visual", rgba=(0.15, 0.35, 0.85, 1)))
        xml.append(f'<site name="plug{c}_tip" pos="{tip} 0 0" size="0.0015" rgba="1 0 0 1" group="4"/>')
        xml.append(f'<site name="plug{c}_grasp" pos="{tip - tc.grasp_from_tip} 0 0" size="0.0015" rgba="0 0 1 1" group="4"/>')
        xml.append("</body>")
        xml.extend(close)
        return "\n".join(xml)

    def xml(self):
        tc = self.tc
        panel, assets = self._panel_xml()
        assets = list(assets) + self._jack_assets()
        assets.append('<material name="panel" rgba="0.08 0.08 0.09 1" specular="0.3"/><material name="plugbody" rgba="0.85 0.9 0.95 0.85" specular="0.6"/>')
        sw = T.switch_front(8, 0)
        assets.append(f'<texture name="switchtex" type="2d" file="{sw}"/><material name="switchtex" texture="switchtex" specular="0.3"/>')
        face_x = tc.face_x - (0.08 if self.cfg.robot.n_arms == 2 else 0.0)  # two side-mounted arms: panel within reach
        out = [f'<body name="panel" pos="{face_x} 0 {self.cfg.bench.top_z + tc.row_z}">{panel}</body>']
        anchor_z = self.cfg.bench.top_z + tc.body_h / 2
        for c in range(tc.n_cables):
            out.append(f'<body name="switch{c}" pos="{tc.anchor_x} {(-1) ** c * tc.anchor_abs_y} {anchor_z}">'
                       f'<geom name="switch{c}" type="box" class="world" pos="-0.032 0 0.011" size="0.03 0.05 0.015" rgba="0.12 0.12 0.14 1"/>'
                       f'<geom name="switch{c}_led" type="box" class="visual" pos="-0.0015 0.02 0.018" size="0.001 0.003 0.002" rgba="0.1 0.9 0.2 1"/>'
                       f'{self._cable_xml(c)}</body>')
        # a visual-only loose distractor cable
        out.append('<body name="loose_cable" pos="0 0 -5">' + "".join(
            f'<geom name="loose{i}" type="capsule" class="visual" size="{tc.radius} 0.015" rgba="0.6 0.6 0.6 1"/>' for i in range(16)) + "</body>")
        eq = "".join(f'<weld name="latch{c}" body1="latch_anchor" body2="plug{c}" active="false" solref="0.01 1"/>' for c in range(tc.n_cables))
        # the first links leave the switch housing: never let them collide with it (a 3 mm capsule at the
        # housing face otherwise wedges itself in and loads the whole cable with hundreds of N)
        con = "".join(f'<exclude body1="switch{c}" body2="cable{c}_{i}"/>' for c in range(tc.n_cables) for i in range(3))
        defaults = (f'<default class="port"><geom type="box" contype="1" conaffinity="7" friction="{tc.port_friction} 0.005 0.0005" material="panel" solimp="{tc.solimp}" solref="{tc.solref} 1"/></default>'
                    f'<default class="occ"><geom type="box" contype="1" conaffinity="7" friction="{tc.plug_friction} 0.005 0.0005"/></default>'
                    f'<default class="plug"><geom contype="1" conaffinity="7" condim="4" friction="{tc.plug_friction} 0.005 0.0005" solimp="{tc.solimp}" solref="{tc.solref} 1"/></default>'
                    f'<default class="cable"><geom contype="2" conaffinity="1" condim="3" friction="{tc.cable_friction} 0.005 0.0005" rgba="0.15 0.35 0.85 1"/></default>')
        return "\n".join(out), assets, eq, con, defaults

    def bind(self, env):
        m, tc = env.model, self.tc
        n = tc.n_ports
        self.panel = m.body("panel").id
        self.port_sites = [m.site(f"port{k}").id for k in range(n)]
        self.port_leds = [m.geom(f"port{k}_led").id for k in range(n)]
        self.occ_bodies = [m.body(f"occ{k}").id for k in range(n)]
        self.occ_color_geoms = [[m.geom(f"occ{k}_boot").id] + [m.geom(f"occ{k}_c{i}").id for i in range(4)] for k in range(n)]
        jack = ("_bot", "_top", "_l", "_r", "_back")
        self.panel_geoms = {i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("panel") or ((m.geom(i).name or "").startswith("port") and m.geom(i).name.endswith(jack))}
        self.plug_body = [m.body(f"plug{c}").id for c in range(tc.n_cables)]
        self.plug_geoms = [{m.geom(f"plug{c}_boot").id, m.geom(f"plug{c}_body").id} for c in range(tc.n_cables)]
        self.boot_geoms = [[m.geom(f"plug{c}_boot").id] + [m.geom(f"plug{c}_rib{i}").id for i in range(3)] for c in range(tc.n_cables)]
        self.tip_site = [m.site(f"plug{c}_tip").id for c in range(tc.n_cables)]
        self.grasp_site = [m.site(f"plug{c}_grasp").id for c in range(tc.n_cables)]
        self.switch = [m.body(f"switch{c}").id for c in range(tc.n_cables)]
        self.cable_joints = [[m.joint(f"cable{c}_{i}").id for i in range(tc.n_links)] + [m.joint(f"plug{c}").id] for c in range(tc.n_cables)]
        self.cable_geoms = [[m.geom(f"cable{c}_{i}").id for i in range(tc.n_links)] for c in range(tc.n_cables)]
        self.cable_bodies = [[m.body(f"cable{c}_{i}").id for i in range(tc.n_links)] for c in range(tc.n_cables)]
        self.latch_eq = [m.equality(f"latch{c}").id for c in range(tc.n_cables)]
        self.loose_body = m.body("loose_cable").id
        self.loose_geoms = [m.geom(f"loose{i}").id for i in range(16)]
        self.latched = [-1] * tc.n_cables
        self.latch_time = [None] * tc.n_cables
        # remote-compliance point = the held plug's tip: `grasp_from_tip` ahead along TCP x, and below the
        # TCP by the grasp height minus the boot's half height (the fingers drop as they close, see xarm7)
        self.tool_tip = (tc.grasp_from_tip, 0.0, env.grasp_tcp_height(tc.boot_w, clearance=0.0015) - tc.boot_h / 2)

    # ---- scenario
    def sample(self, rng, shifts, variant=None):
        tc = self.tc
        variant = variant or {}
        n = int(variant.get("n", rng.integers(1, tc.n_cables + 1)))
        n = max(1, min(n, tc.n_cables))
        if self.cfg.robot.n_arms == 2:
            # each cable's port on the same side as its anchor (cable c lies on side (-1)^c): y < 0 <-> right arm
            ports = []
            for c in range(n):
                cols = [k for k in range(tc.n_ports) if (self.port_yz(k)[0] < 0) == (c % 2 == 0) and k not in ports]
                ports.append(int(rng.choice(cols)))
        else:
            ports = [int(k) for k in rng.choice(tc.n_ports, size=n, replace=False)]
        if "targets" in variant:
            ports = [int(k) for k in variant["targets"]]
        others = [k for k in range(tc.n_ports) if k not in ports]
        occ_frac = rng.uniform(0.0, 0.5)
        if "dense_ports" in shifts:
            occ_frac = rng.uniform(0.85, 1.0)
        occupied = [[int(k), *CABLE_COLORS[rng.choice(("blue", "grey", "yellow", "green", "white", "black"))]] for k in others if rng.random() < occ_frac]
        cables = []
        used = []
        for c in range(n):
            col = CABLE_COLORS[rng.choice(OOD_CABLE_COLORS if "cable_color" in shifts else NOMINAL_CABLE_COLORS)]
            boot = col if ("cable_color" in shifts or rng.random() < 0.7) else CABLE_COLORS[rng.choice(["black", "grey", "white"])]
            if "boot_color" in shifts:
                boot = tuple(float(x) for x in rng.choice([[1.0, 0.1, 0.8, 1], [0.6, 1.0, 0.1, 1], [1.0, 0.55, 0.0, 1]]))
            side = (-1.0) ** c
            lay = self._sample_layout(rng, side, used)
            used.append(lay)
            cables.append(dict(port=ports[c], color=list(col), boot=list(boot), **lay))
        g = rng.uniform(0.06, 0.12)
        panel_rgba = [g, g, g + 0.01, 1]
        if "panel_color" in shifts:
            panel_rgba = [float(x) for x in rng.choice([[0.85, 0.85, 0.86, 1], [0.15, 0.25, 0.6, 1], [0.6, 0.6, 0.55, 1]])]
        panel_pose = [float(rng.uniform(-0.005, 0.01)), float(rng.uniform(-0.01, 0.01)), float(rng.uniform(-0.005, 0.005)), float(rng.uniform(-0.02, 0.02))]
        if "panel_offset" in shifts:
            sgn = rng.choice([-1, 1], 2)
            panel_pose = [float(rng.uniform(0.02, 0.045)), float(sgn[0] * rng.uniform(0.02, 0.04)), float(rng.uniform(-0.01, 0.01)), float(sgn[1] * rng.uniform(0.05, 0.09))]
        leds = []
        if "led_distractors" in shifts:
            free = [k for k in others if k not in {o[0] for o in occupied}]
            pick = rng.choice(free, size=min(len(free), int(rng.integers(3, 7))), replace=False) if free else []
            leds = [[int(k), *TARGET_LED] for k in pick]
        loose = {}
        if "distractor_cable" in shifts or rng.random() < 0.2:
            pts = self._loose_cable(rng, [lay["plug_xy"] for lay in used])
            loose = {"points": pts, "color": list(cables[0]["color"]) if "distractor_cable" in shifts else list(CABLE_COLORS[rng.choice(NOMINAL_CABLE_COLORS)])}
        return dict(n=n, cables=cables, occupied=occupied, panel_rgba=panel_rgba, panel_pose=panel_pose, led_distractors=leds, loose_cable=loose,
                    cable_mass_scale=float(rng.uniform(2.5, 4.0)) if "heavy_cable" in shifts else float(rng.uniform(0.9, 1.1)),
                    cable_stiffness_scale=float(rng.uniform(3.0, 6.0)) if "stiff_cable" in shifts else float(rng.uniform(0.8, 1.25)))

    def cable_points(self, anchor_xy, theta0, deltas, plug_delta):
        L = self.tc.link_len
        phi = theta0 + np.cumsum(deltas)
        steps = L * np.stack([np.cos(phi), np.sin(phi)], 1)
        pts = np.vstack([anchor_xy, anchor_xy + np.cumsum(steps, 0)])
        return pts, phi[-1] + plug_delta

    def _sample_layout(self, rng, side, used, max_tries=8000):
        """Smooth, non-self-intersecting planar cable layout with a reachable plug on the robot's side."""
        tc = self.tc
        x0, x1, y0, y1 = tc.spawn
        anchor_x = tc.anchor_x
        if self.cfg.robot.n_arms == 2:  # panel 8 cm closer: plug and anchor move back with it, room towards the robot
            x0, x1, anchor_x = 0.10, min(x1, 0.50), tc.anchor_x - 0.08
        n, L = tc.n_links, tc.link_len
        lo, hi = np.array([x0, y0]), np.array([x1, y1])
        centre = (lo + hi) / 2
        for _ in range(max_tries):
            ax = anchor_x + rng.uniform(-0.05, 0.05)
            anchor = np.array([ax, side * tc.anchor_abs_y])
            theta0 = -side * np.pi / 2 + rng.uniform(-0.7, 0.7)
            sv = (np.arange(n) + 0.5) * L
            kappa = rng.uniform(-3, 3) + rng.uniform(0, 5) * np.sin(np.pi * rng.integers(1, 4) * sv / (n * L) + rng.uniform(0, 2 * np.pi))
            deltas = np.zeros(n)
            pos, h = anchor.copy(), theta0
            for i in range(n):
                d = kappa[i] * L if i > 0 else 0.0
                if i > 0:
                    ahead = pos + 0.09 * np.array([np.cos(h + d), np.sin(h + d)])
                    gap = min(np.min(ahead - lo), np.min(hi - ahead))
                    if gap < 0.05:
                        to_c = np.arctan2(*(centre - pos)[::-1])
                        d += np.clip(G.wrap(to_c - h), -0.3, 0.3) * np.clip((0.05 - gap) / 0.05, 0, 1)
                    d = float(np.clip(d, -0.35, 0.35))
                deltas[i] = d
                h += d
                pos = pos + L * np.array([np.cos(h), np.sin(h)])
            plug_delta = rng.uniform(-0.15, 0.15)
            pts, heading = self.cable_points(anchor, theta0, deltas, plug_delta)
            u = np.array([np.cos(heading), np.sin(heading)])
            tip = pts[-1] + self.tip_len * u
            grasp = tip - tc.grasp_from_tip * u
            allp = np.vstack([pts[2:], tip])
            if not (np.all(allp > lo + 0.01) and np.all(allp < hi - 0.01)):
                continue
            if not (x0 + 0.04 < grasp[0] < x1 - 0.04 and abs(grasp[1]) < 0.26):
                continue
            if abs(G.wrap(heading)) > 2.3:
                continue
            D = np.linalg.norm(allp[:, None] - allp[None], axis=-1)
            idx = np.arange(len(allp))
            if np.any(D[np.abs(idx[:, None] - idx[None]) >= 3] < 0.016):
                continue
            if any(np.min(np.linalg.norm(np.asarray(u2["points"]) [:, None] - allp[None], axis=-1)) < 0.03 for u2 in used):
                continue
            if any(np.linalg.norm(np.asarray(u2["plug_xy"]) - grasp) < 0.12 for u2 in used):
                continue
            return dict(anchor_x=float(ax), anchor_side=float(side), theta0=float(theta0), deltas=deltas.tolist(), plug_delta=float(plug_delta),
                        points=allp.tolist(), plug_xy=grasp.tolist())
        raise RuntimeError("could not sample a cable layout")

    def _loose_cable(self, rng, avoid):
        x0, x1, y0, y1 = self.tc.spawn
        for _ in range(200):
            p = np.array([rng.uniform(x0 + 0.04, x1 - 0.04), rng.uniform(y0 + 0.05, y1 - 0.05)])
            h, k0, k1 = rng.uniform(-np.pi, np.pi), rng.uniform(-6, 6), rng.uniform(0, 8)
            pts = [p.copy()]
            for i in range(16):
                h += (k0 + k1 * np.sin(i / 16 * np.pi * 2)) * 0.03
                p = p + 0.03 * np.array([np.cos(h), np.sin(h)])
                pts.append(p.copy())
            pts = np.array(pts)
            if np.all((pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)):
                if all(np.min(np.linalg.norm(pts - np.asarray(a), axis=1)) > 0.06 for a in avoid):
                    return pts.tolist()
        return []

    def footprint_xy(self, sc):
        pts = []
        for c in sc["cables"]:
            pts += c["points"]
        return pts

    def apply(self, env, sc):
        m, o, tc = env.model, env._orig, self.tc
        for c in range(tc.n_cables):
            active = c < sc["n"]
            cab = sc["cables"][c] if active else None
            for g in self.cable_geoms[c]:
                m.geom_rgba[g] = cab["color"] if active else (0.5, 0.5, 0.5, 1)
            for g in self.boot_geoms[c]:
                m.geom_rgba[g] = cab["boot"] if active else (0.5, 0.5, 0.5, 1)
            if active:
                m.body_pos[self.switch[c], :2] = (cab["anchor_x"], cab["anchor_side"] * tc.anchor_abs_y)
                m.body_quat[self.switch[c]] = G.yaw_quat(cab["theta0"])
            else:
                m.body_pos[self.switch[c]] = PARK + (0.5 * c, 0.0, 0.0)
        for g in self.panel_geoms:
            m.geom_rgba[g] = sc["panel_rgba"]
        m.mat_rgba[m.material("panel").id] = sc["panel_rgba"]
        for led in self.port_leds:
            m.geom_rgba[led] = LED_OFF
        for k, *rgba in sc["led_distractors"]:
            m.geom_rgba[self.port_leds[k]] = rgba
        for c in range(sc["n"]):
            m.geom_rgba[self.port_leds[sc["cables"][c]["port"]]] = TARGET_LED
        occ = {int(e[0]): e[1:] for e in sc["occupied"]}
        for k, b in enumerate(self.occ_bodies):
            if k in occ:
                for g in self.occ_color_geoms[k]:
                    m.geom_rgba[g] = occ[k]
            else:
                m.body_pos[b, 2] = -20.0
        dx, dy, dz, yaw = sc["panel_pose"]
        m.body_pos[self.panel] = o["body_pos"][self.panel] + (dx, dy, dz)
        m.body_quat[self.panel] = G.yaw_quat(yaw)
        pts = np.asarray(sc["loose_cable"].get("points", [])) if sc["loose_cable"] else np.zeros((0, 2))
        if len(pts) >= 2:
            m.body_pos[self.loose_body] = (0, 0, env.cfg.bench.top_z + tc.radius)
            for i, g in enumerate(self.loose_geoms):
                a, b = pts[min(i, len(pts) - 2)], pts[min(i + 1, len(pts) - 1)]
                d = np.append(b - a, 0.0)
                L = np.linalg.norm(d) + 1e-9
                m.geom_pos[g] = np.append((a + b) / 2, 0.0)
                m.geom_quat[g] = G.mat_to_quat(G.frame_z(d / L))
                m.geom_size[g, 1] = L / 2
                m.geom_rgba[g] = sc["loose_cable"]["color"]
        else:
            m.body_pos[self.loose_body] = (0, 0, -5)
        self._regime(env, env.regime)

    def apply_regime(self, env, regime):
        self._regime(env, regime)

    def _regime(self, env, r):
        m, o = env.model, env._orig
        sc = env.scenario.task if env.scenario else {}
        ms = sc.get("cable_mass_scale", 1.0) * r.get("cable_mass", 1.0) * r.get("object_mass", 1.0)
        ks = sc.get("cable_stiffness_scale", 1.0) * r.get("cable_stiffness", 1.0)
        for c in range(self.tc.n_cables):
            for b in self.cable_bodies[c]:
                m.body_mass[b] = o["body_mass"][b] * ms
                m.body_inertia[b] = o["body_inertia"][b] * ms
            for j in self.cable_joints[c][:-1]:
                m.jnt_stiffness[j] = o["jnt_stiffness"][j] * ks
            for g in self.plug_geoms[c]:
                m.geom_friction[g, 0] = o["geom_friction"][g, 0] * r.get("object_friction", 1.0)

    def reset_state(self, env, sc):
        m, d = env.model, env.data
        for c in range(self.tc.n_cables):
            if c < sc["n"]:
                cab = sc["cables"][c]
                for j, dl in zip(self.cable_joints[c], list(cab["deltas"]) + [cab["plug_delta"]]):
                    d.qpos[m.jnt_qposadr[j] : m.jnt_qposadr[j] + 4] = G.yaw_quat(dl)
            else:
                for j in self.cable_joints[c]:
                    d.qpos[m.jnt_qposadr[j] : m.jnt_qposadr[j] + 4] = (1, 0, 0, 0)
            d.eq_active[self.latch_eq[c]] = 0
        self.latched = [-1] * self.tc.n_cables
        self.latch_time = [None] * self.tc.n_cables

    def default_episode_s(self, sc):
        return self.tc.episode_s_per_cable * sc["n"] + 2.0

    def horizon(self, sc):
        return sc["n"]

    def goal_description(self, sc):
        return "plug " + ", ".join(f"cable {c} into port {cab['port'] + 1}" for c, cab in enumerate(sc["cables"]))

    # ---- geometry / latch
    def port_pose(self, env, k):
        s = self.port_sites[k]
        return env.data.site_xpos[s].copy(), G.mat_to_quat(env.data.site_xmat[s])

    def plug_geometry(self, env, c):
        """Nearest port to plug c's tip, with depth / lateral error / angle in the panel frame."""
        d = env.data
        tip = d.site_xpos[self.tip_site[c]]
        ax = d.xmat[self.plug_body[c]].reshape(3, 3)[:, 0]
        R = d.xmat[self.panel].reshape(3, 3)
        rel = (tip - d.site_xpos[self.port_sites]) @ R
        i = int(np.argmin(rel[:, 1] ** 2 + rel[:, 2] ** 2 + 0.01 * np.minimum(rel[:, 0], 0) ** 2))
        return i, float(rel[i, 0]), float(np.linalg.norm(rel[i, 1:])), float(np.arccos(np.clip(ax @ R[:, 0], -1, 1)))

    def pre_step(self, env):
        sc = env.scenario.task if env.scenario else None
        if sc is None:
            return
        tc, d = self.tc, env.data
        for c in range(sc["n"]):
            if self.latched[c] >= 0:
                continue
            k, depth, lateral, _ = self.plug_geometry(env, c)
            if depth > tc.latch_engage and lateral < 0.002:
                f = tc.latch_force * min(1.0, (depth - tc.latch_engage) / (tc.seat_depth - tc.latch_engage))
                d.xfrc_applied[self.plug_body[c], :3] -= f * d.xmat[self.panel].reshape(3, 3)[:, 0]

    def post_step(self, env):
        sc = env.scenario.task if env.scenario else None
        if sc is None:
            return
        tc, d, m = self.tc, env.data, env.model
        for c in range(sc["n"]):
            if self.latched[c] >= 0:
                continue
            k, depth, lateral, angle = self.plug_geometry(env, c)
            if depth >= tc.seat_depth and lateral <= tc.latch_max_lateral and angle <= tc.latch_max_angle:
                eq = self.latch_eq[c]
                m.eq_data[eq, 3:6] = d.xpos[self.plug_body[c]]
                m.eq_data[eq, 6:10] = d.xquat[self.plug_body[c]]
                d.eq_active[eq] = 1
                self.latched[c] = k
                self.latch_time[c] = env.time

    def aux_size(self):
        return 2 * self.tc.n_cables

    def get_aux(self, env):
        return np.asarray([v for c in range(self.tc.n_cables) for v in (self.latched[c], -1.0 if self.latch_time[c] is None else self.latch_time[c])], float)

    def set_aux(self, env, aux):
        for c in range(self.tc.n_cables):
            self.latched[c] = int(aux[2 * c])
            self.latch_time[c] = None if aux[2 * c + 1] < 0 else float(aux[2 * c + 1])

    # ---- labels
    def privileged(self, env):
        sc = env.scenario.task
        d = env.data
        n = sc["n"]
        plugs = []
        for c in range(n):
            k, depth, lateral, angle = self.plug_geometry(env, c)
            plugs.append(dict(pos=d.xpos[self.plug_body[c]].copy(), quat=d.xquat[self.plug_body[c]].copy(), tip=d.site_xpos[self.tip_site[c]].copy(),
                              grasp=d.site_xpos[self.grasp_site[c]].copy(), nearest_port=k, depth=depth, lateral=lateral, angle=angle,
                              latched=self.latched[c], target=sc["cables"][c]["port"], grasped_by=env.grasped_by(self.plug_geoms[c])))
        done = [p["latched"] == p["target"] for p in plugs]
        kcur = n
        for c in range(n):
            if not done[c]:
                kcur = c
                break
        return dict(plugs=plugs, done=done, n_done=int(sum(done)), k=kcur, grasped=[p["grasped_by"] for p in plugs])

    def stages(self):
        names = []
        for k in range(self.tc.n_cables):
            names += [f"{s}_{k}" for s in SUB]
        return tuple(names) + ("done",)

    def _substage(self, env, pr, c):
        p = pr["plugs"][c]
        if p["grasped_by"] is None:
            return 0
        pp, _ = self.port_pose(env, p["target"])
        if p["depth"] > 0.0005 and p["nearest_port"] == p["target"]:
            return 4
        if np.linalg.norm(p["tip"] - pp) < 0.03:
            return 3
        if p["pos"][2] > env.cfg.bench.top_z + 0.03:
            return 2
        return 1

    def stage(self, env, pr):
        k = pr["k"]
        if k >= env.scenario.task["n"]:
            return len(self.stages()) - 1
        return len(SUB) * k + self._substage(env, pr, k)

    def progress(self, env, pr):
        n = env.scenario.task["n"]
        k = pr["k"]
        if k >= n:
            return 1.0
        sub = self._substage(env, pr, k)
        return (pr["n_done"] + (0.0, 0.2, 0.45, 0.7, 0.85)[sub]) / n

    def success(self, env, pr):
        return pr["n_done"] == env.scenario.task["n"] and all(g is None for g in pr["grasped"])

    def task_features(self, env, pr):
        """[n, k, progress, grasped, plug pos(3) quat(4), tip(3), target port pos(3) quat(4), tip & axis in port frame(6), grasp - tcp per arm(6), depth lateral angle(3)]."""
        sc = env.scenario.task
        n = sc["n"]
        k = min(pr["k"], n - 1)
        p = pr["plugs"][k]
        pp, pq = self.port_pose(env, p["target"])
        R = G.quat_to_mat(pq)
        rel_tip = R.T @ (p["tip"] - pp)
        rel_ax = R.T @ G.quat_rotate(p["quat"], [1, 0, 0])
        rel = [p["grasp"] - pr["tcp"][s][0] for s in env.arms] + [np.zeros(3)] * (2 - len(env.arms))
        return np.concatenate([[n, k, self.progress(env, pr), float(p["grasped_by"] is not None)], p["pos"], p["quat"], p["tip"], pp, pq, rel_tip, rel_ax,
                               *rel, [p["depth"], p["lateral"], p["angle"]]]).astype(np.float32)

    def feature_names(self):
        return (["n", "k", "progress", "grasped"] + [f"plug_{a}" for a in "xyz"] + [f"plug_q{a}" for a in "wxyz"] + [f"tip_{a}" for a in "xyz"]
                + [f"port_{a}" for a in "xyz"] + [f"port_q{a}" for a in "wxyz"] + [f"tip_rel_{a}" for a in "xyz"] + [f"axis_rel_{a}" for a in "xyz"]
                + [f"grasp_rel_{s}_{a}" for s in ("right", "left") for a in "xyz"] + ["depth", "lateral", "angle"])

    def record_columns(self, env, pr):
        out = dict(n_done=float(pr["n_done"]))
        for c, p in enumerate(pr["plugs"]):
            out[f"depth_{c}"] = p["depth"]
            out[f"lateral_{c}"] = p["lateral"]
            out[f"angle_{c}"] = p["angle"]
            out[f"latched_{c}"] = float(p["latched"])
        return out

    def outcome(self, env, record, expert):
        pr = self.privileged(env)
        if env.unstable:
            return "sim_unstable"
        if self.success(env, pr):
            return "none"
        k = pr["k"]
        p = pr["plugs"][k]
        if p["latched"] >= 0 and p["latched"] != p["target"]:
            return "wrong_port"
        if not any(g == k for g in record.get("grasped_k", [])):
            return "grasp_missed"
        if record.get("drops", 0) > 0:
            return "dropped"
        if p["grasped_by"] is not None:
            return "incomplete"
        if record.get("max_depth", {}).get(k, -1) > 0.001:
            return "not_seated"
        if any(e[0].split(":")[-1] in ("insert", "jam") for e in expert.events):  # (bimanual events are arm-prefixed)
            return "jammed"
        return "other"

    def make_expert(self, env, failure=None, seed=0, noise=True, bimanual=None):
        """Scripted expert. With two arms (and bimanual not False) both arms work: each takes the objects on
        its side of the bench and they take turns at the shared structure."""
        from ..experts.cable_plug import CablePlugExpert

        if bimanual is None:
            bimanual = len(env.arms) == 2
        if bimanual and len(env.arms) == 2:
            from ..experts.dual import DualArmExpert

            sc = env.scenario.task
            return DualArmExpert(env, CablePlugExpert, {c: sc["cables"][c]["plug_xy"] for c in range(sc["n"])}, failure=failure, seed=seed, noise=noise, ordered=False)
        return CablePlugExpert(env, failure=failure, seed=seed, noise=noise)

    def sample_failure(self, kind, s, rng, sc):
        if kind == "misalign":
            ang = rng.uniform(0, 2 * np.pi)
            r = 0.001 + 0.004 * s
            rv = rng.normal(0, 1, 3)
            rv[0] = 0.0
            return dict(dyz=[float(r * np.cos(ang)), float(r * np.sin(ang))], drot=(rv / (np.linalg.norm(rv) + 1e-9) * (0.03 + 0.15 * s)).tolist())
        if kind == "partial_insert":
            return dict(depth_frac=float(0.85 - 0.6 * s))
        if kind == "wrong_port":
            c = 0
            occ = {int(e[0]) for e in sc["occupied"]} | {cab["port"] for cab in sc["cables"]}
            free = [k for k in range(self.tc.n_ports) if k not in occ]
            y0, z0 = self.port_yz(sc["cables"][c]["port"])
            dist = [abs(self.port_yz(k)[0] - y0) + 2.0 * abs(self.port_yz(k)[1] - z0) for k in free]
            near = [k for k, dd in sorted(zip(free, dist), key=lambda t: t[1])][:2]
            return dict(port=int(rng.choice(near)) if near else int(free[0]))
        return {}
