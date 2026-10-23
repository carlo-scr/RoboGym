# xArm 7 meshes

Vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/ufactory_xarm7)
(`ufactory_xarm7`, commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`, 2026-07-04), BSD-3-Clause
(UFACTORY Inc., see `LICENSE`). `menagerie_xarm7.xml` is the untouched upstream MJCF, kept for reference;
RoboGym emits its own per-arm MJCF from the same kinematic/inertial numbers (`robogym/robot/xarm7.py`)
so that two prefixed arms, gravity compensation, the TCP/camera sites and collision classes can be set
programmatically.
