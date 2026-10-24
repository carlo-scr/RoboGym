"""Real-cell interface: the same action / observation / episode contract as RoboGymEnv, implemented over
the UFACTORY xArm Python SDK and Intel RealSense cameras. Hardware code is imported lazily; the `dry`
backend lets the control loop, recording and calibration scripts run without a robot."""
