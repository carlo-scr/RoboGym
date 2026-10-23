# Modelling assumptions (v0.1)

The robot is the UFACTORY xArm 7 with its standard gripper, taken from MuJoCo Menagerie (measured
kinematics, inertia, joint limits, servo gains and the gripper linkage). Everything else is a
deliberately strong placeholder that lives in one `robogym/config.py` field, so it can be replaced
when the lab's real set-up is measured. The right-hand column says what would replace each guess.

## Robot and bench

| Assumption | Value | Config | Replace with |
|---|---|---|---|
| Arms | 1 (default) or 2 xArm 7, bolted on the bench top at x = 0, facing +x | `robot.n_arms`, `base_xy_*`, `base_z` | the lab's mounting plate positions |
| Dual-arm spacing | bases at y = +-0.45 m | `robot.base_xy_dual` | measured |
| Joint servos | Menagerie position servos (kp 1500/1000/800, kv 150/100/80), force limits 50/30/20 N m, gravity compensated | `robot.arm.*` | xArm controller gains / torque limits |
| Home pose | TCP ~0.39 m ahead, 0.29 m up, pointing down | `robot.arm.home_q` | the lab's ready pose |
| Gripper | Menagerie linkage, but a force-limited servo (~50 N pad normal force; Menagerie's: 300+ N) whose closure command is rate-limited like a gripper speed setting (full stroke in ~0.9 s) | `robot.gripper.max_force / servo_kp / servo_kv / close_rate / joint_frictionloss` | measured xArm gripper speed / force setting |
| Pads | Menagerie's two 30 x 4 x 19 mm boxes per finger, friction 1.0, stiff contacts with priority over every object | `robot.gripper.pad_*` | measured pad material |
| Finger arc | the pads swing 11-13 mm along the tool axis while closing; grasp heights are planned from the object width (`env.grasp_tcp_height`) | `robot/xarm7.py` | - |
| Grasp centre | lower pad pair centre (0.151 m from the gripper base); fingertips at 0.162 m | `robot.gripper.tcp_z / tip_z` | - |
| Wrist camera | 85 mm off the tool axis on the robot side, tilted to look past the fingertips, 58 deg vertical FOV, 320x240 | `camera.wrist_*`, `camera.width/height` | RealSense mount pose and intrinsics |
| Fixed cameras | front (1.15, 0, 0.55), side (0.45, -0.95, 0.5), top (0.45, 0, 1.05) looking at the workspace | `camera.fixed` | tripod poses |
| Recorded cameras | right wrist + front | `camera.record` | - |
| Bench | 1.25 x 1.2 m top at z = 0 (1.7 m wide for two arms), rubber mat, wood frame | `bench.*` | lab bench |
| Control | 20 Hz Cartesian deltas (<= 2 cm, 0.15 rad per step), resolved-rate IK at 500 Hz | `sim.*` | - |
| Compliant mode | per-axis IK gains in the TCP frame about a task-specific remote centre (plug tip / brick centre) | `robot.gripper.compliant_gains`, `Task.tool_tip / compliant_gains` | - |
| Wrist F/T | ideal force/torque sensor on the gripper base | `robot/xarm7.sensors_xml` | real sensor noise / bias |

## Cube stacking

| Assumption | Value | Config |
|---|---|---|
| Cubes | 4 wooden cubes 50/45/40/35 mm (600 kg/m3), friction 0.8 | `task.sizes`, `density`, `friction` |
| Goal | stack on the bottom cube where it lies; `by_size` (largest first) or a random `pattern` shown as a ghost | `task.variant`, `show_goal_ghost` |
| Spawn | x 0.30-0.62, |y| <= 0.26, >= 16 cm from the base cube | `task.spawn` |
| Seated | centre within 12 mm of the cube below, correct height, level, still | `task.place_tol_xy` |

## Oversized LEGO

| Assumption | Value | Config |
|---|---|---|
| Scale | 4x standard LEGO: 32 mm pitch, 2x2 = 61.5 mm, 38.4 mm tall, 19 mm hemispherical studs | `task.scale`, `base_*`, `gap` |
| Fit | 3 mm total clearance between studs and cavity walls (real: ~0.1 mm) | `task.fit_clearance` |
| Snap | 6 N detent over the last 6 mm, weld once seated within 4 mm / 7 deg; snapped bricks cannot be removed | `task.snap_*` |
| Baseplate | 6 x 6 studs, pose randomised +-3 cm / +-15 deg | `task.plate_studs`, `plate_xy` |
| Goal | random stable structure: top-down insertable, >= 75 % supported, reachable by the gripper pads; shown as a ghost | `tasks/lego.py::_sample_goals` |
| Bricks | 2x2 x3 + 2x4 (first n used) | `task.bricks` |

## Ethernet plugging

| Assumption | Value | Config |
|---|---|---|
| Plug | RJ45 body 11.7 x 8.0 x 21 mm + 16 mm boot, 12 g, grasped 37 mm behind the tip, always latch-up | `task.body_*`, `boot_*`, `grasp_from_tip` |
| Jack | 1.6 mm clearance, 2.5 mm lead-in, 20 mm deep, latch at 15 mm (10 N detent), weld when seated within 1.5 mm / 5 deg | `task.clearance`, `chamfer`, `latch_*` |
| Panel | 2 x 12 ports, 18.5 mm pitch, 1U (44.45 mm) row pitch, face at x = 0.62, 95 mm above the bench, pose jitter +-1 cm | `task.rows/cols/*_pitch`, `face_x`, `row_z` |
| Cable | 0.63 m Cat6 as 18 capsules with ball-joint springs, anchored in a switch housing at the bench edge | `task.n_links`, `link_len`, `anchor_*` |
| Occupancy | 0-50 % of the other ports hold colour-coded cables; `dense_ports` 85-100 % | `tasks/cable_plug.py` |
| Perception | expert sees the port with 0.3 mm / 0.3 deg noise | `experts/cable_plug.py` |

## Simulation numerics (found while tuning, keep in mind)

* **Mass changes need consistent solver constants.** Regimes scale `body_mass` / `body_inertia` at run time;
  MuJoCo's constraint solver scales contact impedance with the compile-time `body_invweight0` /
  `dof_invweight0`, so a heavier-than-compiled body gets a light body's impedance and a fast grasp
  launches it into the opposite pad (explosion). `env._refresh_mass_constants` rescales those constants
  per body. Do **not** call `mj_setConst` on this model: it leaves the gripper four-bar in its other
  assembly branch (fingers lock shut).
* **Visual-only geoms have density 0.** Otherwise a visual box silently adds default-density mass.
* **Thin pads tunnel.** Menagerie's pads are 4 mm boxes; penetration beyond 2 mm flips the contact normal.
  Keep pad contacts stiff (solref 4 ms) and the closing speed limited instead of thickening the pads (which
  collide with neighbouring bricks).
* **Finger arc.** The four-bar fingers move 11-13 mm along the tool axis while closing; grasp heights are
  planned at the closed angle (`env.grasp_tcp_height`), otherwise fingertips hit the bench.
* **Gravity compensation** must include the gripper's knuckles and fingers; otherwise compliant mode sags
  ~0.6 mm and the plug lands on the jack's lower chamfer.
* **Actuation latency**: the experts compute deltas against `env.pending_targets` (the target after the
  delay queue drains); against the lagging executed target, repeated deltas pile up and overshoot.
* Constraint time constants must stay >= 2 x timestep (4 ms at the 2 ms step): a 3 ms linkage `solref`
  blew up everything.

## Known limits (v0.1)

* The finger meshes are visual only; the pads (two 30 x 4 x 19 mm boxes per finger) carry all contacts.
* Snapped bricks and latched plugs are welded in place: no disassembly / unplugging tasks yet.
* Dual-arm scenes compile and run, but the experts are single-arm (the arm closer to the objects).
* No MJX / Warp batch stepping yet; the model uses only primitives, convex meshes and welds, so it should port.
