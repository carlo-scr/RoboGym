"""Digital-twin checks: replay a recorded episode's commanded targets in the simulator from the same
initial robot pose and report the divergence of proprioception over time. Used (a) to measure the
sim-to-real gap once real recordings exist and (b) as a regression test that the simulator reproduces
its own recordings (bit-exact for sim episodes)."""
from __future__ import annotations

import numpy as np

from ..env import ACT_DIM, STATE_PER_ARM, RoboGymEnv


def replay_actions(env: RoboGymEnv, actions, scenario=None, seed=None):
    """Apply recorded actions open loop; returns the simulated state trace [T, 28 n]."""
    env.reset(scenario, seed=seed)
    out = [env.observe()]
    for a in actions[:-1]:
        out.append(env.step(np.asarray(a, float).reshape(-1)[: ACT_DIM * len(env.arms)]))
    return np.stack(out)


def divergence(sim_states, real_states, n_arms=1):
    """Per-step TCP position error (m) and grip-width error between two state traces of equal length."""
    T = min(len(sim_states), len(real_states))
    s, r = np.asarray(sim_states[:T]), np.asarray(real_states[:T])
    rows = {}
    for i in range(n_arms):
        o = STATE_PER_ARM * i
        rows[f"arm{i}_tcp_err_m"] = np.linalg.norm(s[:, o + 14 : o + 17] - r[:, o + 14 : o + 17], axis=1)
        rows[f"arm{i}_grip_err_m"] = np.abs(s[:, o + 21] - r[:, o + 21])
        rows[f"arm{i}_wrench_err_N"] = np.linalg.norm(s[:, o + 22 : o + 25] - r[:, o + 22 : o + 25], axis=1)
    return rows


def summarize_divergence(rows):
    return {k: dict(mean=float(v.mean()), max=float(v.max()), final=float(v[-1])) for k, v in rows.items()}
