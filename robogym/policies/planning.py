"""The planning half of a long-horizon task: subgoals, per-task adapters and planners.

A *plan* is an ordered list of `Subgoal(obj, goal)`: cube c onto support s / spot j (cube_stack), brick k
at its lattice goal (lego), cable c into port p (cable_plug). The oracle plan is what the task sampled
(`scenario.task["moves"]`, build order, target ports); every other planner is judged against it.

`TaskAdapter` is the glue the hierarchical policy and the evaluation need per task:
  oracle / heuristic / candidates    plans and the subgoal space
  achieved(env, sg, pr)              is this planned subgoal physically done (object at *its* goal, released)
  oracle_flags / oracle_next         the task's own achievement bookkeeping (sticky, in oracle order)
  valid(env, plan)                   is a plan acceptable (dependencies, goals), with a reason
  at_other_goal(env, pr, obj, k)     the object was put at a *different* valid goal (a planning error)
  expert_run(ex, k, sg, plan)        the expert's generator for one planned subgoal (scripted executor)
  features(env, pr, k, sg, plan)     task features re-targeted to the planned subgoal (learned executor)
  scene_text / llm_format / parse_plan   text scene description and JSON plan format (LLM planner)
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import geometry as G
from ..failures import FailureSpec


@dataclass(frozen=True)
class Subgoal:
    obj: int
    goal: object = None  # task-specific: cube (support, spot) | lego None | cable port

    def key(self):
        g = self.goal
        return (int(self.obj), tuple(int(x) for x in g) if isinstance(g, (list, tuple)) else (None if g is None else int(g)))

    def to_json(self):
        return dict(obj=int(self.obj), goal=None if self.goal is None else (list(self.goal) if isinstance(self.goal, (list, tuple)) else int(self.goal)))

    @staticmethod
    def from_json(d):
        g = d.get("goal")
        return Subgoal(int(d["obj"]), tuple(g) if isinstance(g, list) else g)


def plan_keys(plan):
    return [s.key() for s in plan]


def plan_diff(plan, oracle, ordered=True):
    """How a plan deviates from the oracle: missing / extra subgoals, order, matching prefix length."""
    p, o = plan_keys(plan), plan_keys(oracle)
    prefix = 0
    for a, b in zip(p, o):
        if a != b:
            break
        prefix += 1
    missing = [k for k in o if k not in p]
    extra = [k for k in p if k not in o]
    same_set = not missing and not extra and len(p) == len(o)
    return dict(exact=p == o, same_set=same_set, order_ok=(p == o) if ordered else same_set, prefix=prefix, missing=len(missing), extra=len(extra), n=len(p))


# ---------------------------------------------------------------------- adapters
class TaskAdapter:
    name = ""
    ordered = True  # subgoals depend on earlier ones (a different order is a different plan)

    @classmethod
    def oracle(cls, env):
        raise NotImplementedError

    @classmethod
    def heuristic(cls, env):
        return cls.oracle(env)

    @classmethod
    def candidates(cls, env):
        return cls.oracle(env)

    @classmethod
    def achieved(cls, env, sg, pr=None) -> bool:
        raise NotImplementedError

    @classmethod
    def oracle_flags(cls, pr) -> list:
        raise NotImplementedError

    @classmethod
    def oracle_next(cls, pr) -> int:
        return int(pr.get("k", pr.get("n_done", 0)))

    @classmethod
    def valid(cls, env, plan):
        d = plan_diff(plan, cls.oracle(env), cls.ordered)
        return (d["exact"] if cls.ordered else d["same_set"]), ("" if d["exact"] else ("missing" if d["missing"] else "extra" if d["extra"] else "order"))

    @classmethod
    def at_other_goal(cls, env, pr, obj, k) -> bool:
        return False

    @classmethod
    def object_geoms(cls, env, obj) -> set:
        raise NotImplementedError

    @classmethod
    def expert_subgoal_id(cls, k, sg) -> int:
        return int(sg.obj)

    @classmethod
    def expert_run(cls, ex, k, sg, plan):
        raise NotImplementedError

    @classmethod
    def features(cls, env, pr, k, sg, plan) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def describe(cls, env, sg) -> str:
        return f"{sg.obj} -> {sg.goal}"

    @classmethod
    def scene_text(cls, env) -> str:
        raise NotImplementedError

    @classmethod
    def llm_format(cls, env) -> str:
        raise NotImplementedError

    @classmethod
    def parse_plan(cls, env, data) -> list:
        raise NotImplementedError


def _rel_tcp(env, pr, p):
    rel = [p - pr["tcp"][s][0] for s in env.arms] + [np.zeros(3)] * (2 - len(env.arms))
    return np.concatenate(rel)


class CubeAdapter(TaskAdapter):
    name = "cube_stack"

    @classmethod
    def oracle(cls, env):
        return [Subgoal(int(c), (int(s), int(spot))) for c, s, spot in env.scenario.task["moves"]]

    @classmethod
    def heuristic(cls, env):
        """From what a perception module would give: cube sizes / poses, the goal ghost (final heights),
        the marked spots. Stack bottom-up by final height; unstack top-down onto the spots in order."""
        sc, pr = env.scenario.task, env.privileged()
        n, mode, moves = sc["n"], sc["mode"], sc["moves"]
        last = {}
        for k, (c, _s, _spot) in enumerate(moves):
            last[c] = k
        z_final = [sc["goals"][last[c]][2] if c in last else float(pr["cube_pos"][c][2]) for c in range(n)]
        bottom_up = [int(i) for i in np.argsort(z_final, kind="stable")]
        stack = [Subgoal(bottom_up[k], (bottom_up[k - 1], -1)) for k in range(1, n)]
        if mode in ("stack", "handover"):
            return stack
        top_down = [int(i) for i in np.argsort(-pr["cube_pos"][:n, 2], kind="stable")]
        unstack = [Subgoal(top_down[i], (-1, i)) for i in range(n - 1)]
        if mode == "unstack":
            return unstack
        return unstack + [Subgoal(top_down[-1], (-1, n - 1))] + stack

    @classmethod
    def candidates(cls, env):
        sc = env.scenario.task
        n = sc["n"]
        out = [Subgoal(c, (s, -1)) for c in range(n) for s in range(n) if s != c]
        out += [Subgoal(c, (-1, j)) for c in range(n) for j in range(len(sc["spots"]))]
        return out

    @classmethod
    def _goal(cls, env, pr, sg):
        sc = env.scenario.task
        pose = {i: (pr["cube_pos"][i], pr["cube_quat"][i]) for i in range(sc["n"])}
        return env.task._goal_from(dict(sc, moves=[[int(sg.obj), int(sg.goal[0]), int(sg.goal[1])]]), 0, pose)

    @classmethod
    def achieved(cls, env, sg, pr=None):
        from ..tasks.cube_stack import SPOT_TOL_XY

        pr = pr or env.privileged()
        c = int(sg.obj)
        if pr["grasped"][c] is not None:
            return False
        gp, _gq = cls._goal(env, pr, sg)
        p = pr["cube_pos"][c]
        level = float(np.max(np.abs(G.quat_to_mat(pr["cube_quat"][c])[2])))
        tol = env.task.tc.place_tol_xy if sg.goal[0] >= 0 else SPOT_TOL_XY
        return bool(np.linalg.norm(p[:2] - gp[:2]) < tol and abs(p[2] - gp[2]) < 0.006 and level > 0.97 and np.linalg.norm(pr["cube_vel"][c][:3]) < 0.05)

    @classmethod
    def oracle_flags(cls, pr):
        return [bool(x) for x in pr["achieved"]]

    @classmethod
    def oracle_next(cls, pr):
        return int(pr["n_done"])

    @classmethod
    def resting_on(cls, env, pr, obj):
        """('cube', j) | ('spot', j) | ('bench', None) | None (in the air / tilted): what the cube rests on."""
        from ..tasks.cube_stack import SPOT_TOL_XY

        sc = env.scenario.task
        p = pr["cube_pos"][obj]
        if float(np.max(np.abs(G.quat_to_mat(pr["cube_quat"][obj])[2]))) < 0.97:
            return None
        for j in range(sc["n"]):
            if j == obj:
                continue
            q = pr["cube_pos"][j]
            dz = p[2] - q[2]
            if 0 < dz < (sc["sizes"][j] + sc["sizes"][obj]) / 2 + 0.012 and np.linalg.norm(p[:2] - q[:2]) < 0.75 * (sc["sizes"][j] + sc["sizes"][obj]) / 2:
                return ("cube", j)
        if p[2] > env.cfg.bench.top_z + sc["sizes"][obj] / 2 + 0.012:
            return None
        for j, s in enumerate(sc["spots"]):
            if np.linalg.norm(p[:2] - np.asarray(s[:2])) < SPOT_TOL_XY:
                return ("spot", j)
        return ("bench", None)

    @classmethod
    def at_other_goal(cls, env, pr, obj, k):
        """The cube rests level on a cube / spot that is *not* the goal of oracle move k (a planning error);
        resting on the right support but off-centre, tilted or in the air is an execution error."""
        if pr["achieved"][k] or pr["grasped"][obj] is not None:
            return False
        on = cls.resting_on(env, pr, obj)
        if on is None or on[0] == "bench":
            return False
        _c, s_goal, spot_goal = env.scenario.task["moves"][k]
        return bool(on != (("cube", s_goal) if s_goal >= 0 else ("spot", spot_goal)))

    @classmethod
    def object_geoms(cls, env, obj):
        return env.task.geom_sets[obj]

    @classmethod
    def expert_subgoal_id(cls, k, sg):
        return int(k)

    @classmethod
    def expert_run(cls, ex, k, sg, plan):
        moves = [[int(s.obj), int(s.goal[0]), int(s.goal[1])] for s in plan]
        return ex._do_move(k, moves)

    @classmethod
    def features(cls, env, pr, k, sg, plan):
        sc = env.scenario.task
        n, ci = sc["n"], int(sg.obj)
        p, q = pr["cube_pos"][ci], pr["cube_quat"][ci]
        goal = cls._goal(env, pr, sg)[0]
        allp = np.zeros(3 * env.task.tc.n_cubes)
        allp[: 3 * n] = pr["cube_pos"].reshape(-1)
        return np.concatenate([[n, k, env.task.progress(env, pr), float(pr["grasped"][ci] is not None)], p, q, goal, goal - p, _rel_tcp(env, pr, p), allp]).astype(np.float32)

    @classmethod
    def describe(cls, env, sg):
        cols = env.scenario.task["colors"]
        s, spot = sg.goal
        return f"{cols[sg.obj]} -> " + (f"on {cols[s]}" if s >= 0 else f"spot {spot}")

    @classmethod
    def scene_text(cls, env):
        sc, pr = env.scenario.task, env.privileged()
        n, top = sc["n"], env.cfg.bench.top_z
        lines = [f"Task: {env.task.goal_description(sc)}.", f"Mode: {sc['mode']}.", "Cubes (colour: side, position x,y on the bench, where it rests):"]
        for i in range(n):
            p = pr["cube_pos"][i]
            on = "the bench"
            if p[2] > top + sc["sizes"][i] / 2 + 0.01:
                below = [j for j in range(n) if j != i and pr["cube_pos"][j][2] < p[2] and np.linalg.norm(pr["cube_pos"][j][:2] - p[:2]) < 0.03]
                on = f"top of the {sc['colors'][below[0]]} cube" if below else "something"
            lines.append(f"  - {sc['colors'][i]}: {sc['sizes'][i] * 1000:.0f} mm, at ({p[0] * 100:.1f}, {p[1] * 100:.1f}) cm, resting on {on}")
        if sc["spots"]:
            lines.append("Marked spots on the bench (index: position x,y):")
            lines += [f"  - spot {j}: ({x * 100:.1f}, {y * 100:.1f}) cm" for j, (x, y, _yaw) in enumerate(sc["spots"])]
        lines.append("A cube can only be picked when nothing rests on it, and only placed on a cube that is not held.")
        return "\n".join(lines)

    @classmethod
    def llm_format(cls, env):
        return ('Answer with only a JSON array of moves in execution order. Each move is {"cube": "<colour>", "on": "<colour>"} '
                '(place the cube on top of that cube) or {"cube": "<colour>", "spot": <index>} (put it down on that marked spot).')

    @classmethod
    def parse_plan(cls, env, data):
        cols = env.scenario.task["colors"]
        idx = {c: i for i, c in enumerate(cols)}
        plan = []
        for m in data:
            c = idx[str(m["cube"]).strip().lower()]
            if "spot" in m and m["spot"] is not None:
                plan.append(Subgoal(c, (-1, int(m["spot"]))))
            else:
                plan.append(Subgoal(c, (idx[str(m["on"]).strip().lower()], -1)))
        return plan


class LegoAdapter(TaskAdapter):
    name = "lego"

    @classmethod
    def oracle(cls, env):
        return [Subgoal(k, None) for k in range(env.scenario.task["n"])]

    @classmethod
    def heuristic(cls, env):
        """Build order from the goal model (the ghost): level by level, ties in brick order."""
        sc = env.scenario.task
        return [Subgoal(int(k), None) for k in sorted(range(sc["n"]), key=lambda k: sc["goals"][k][2])]

    @classmethod
    def candidates(cls, env):
        return cls.oracle(env)

    @classmethod
    def achieved(cls, env, sg, pr=None):
        pr = pr or env.privileged()
        return bool(pr["at_goal"][sg.obj]) and pr["grasped"][sg.obj] is None

    @classmethod
    def oracle_flags(cls, pr):
        return [bool(x) for x in pr["at_goal"]]

    @classmethod
    def _cells(cls, env, k):
        task, sc = env.task, env.scenario.task
        i, j, lvl, yq = sc["goals"][k]
        nx0, ny0 = task.kinds[k]
        nx, ny = (nx0, ny0) if yq % 2 == 0 else (ny0, nx0)
        return task._cells(i, j, nx, ny), lvl

    @classmethod
    def valid(cls, env, plan):
        """Every brick once; a brick's supports (level below, overlapping cells) before it; nothing that
        would cover its cells (same level or above, overlapping) before it (top-down insertion)."""
        sc = env.scenario.task
        n = sc["n"]
        objs = [int(s.obj) for s in plan]
        if sorted(objs) != list(range(n)):
            return False, "missing" if len(set(objs)) < n else "extra"
        cells = {k: cls._cells(env, k) for k in range(n)}
        placed = []
        for k in objs:
            ck, lk = cells[k]
            for j in placed:
                cj, lj = cells[j]
                if ck & cj and lj >= lk:
                    return False, "blocked"
            if lk > 0:
                below = {j for j in range(n) if j != k and cells[j][1] == lk - 1 and cells[j][0] & ck}
                if not below <= set(placed):
                    return False, "unsupported"
            placed.append(k)
        return True, ""

    @classmethod
    def at_other_goal(cls, env, pr, obj, k):
        return bool(env.task._snapped[obj]) and not bool(pr["at_goal"][obj])

    @classmethod
    def object_geoms(cls, env, obj):
        return env.task.geom_sets[obj]

    @classmethod
    def expert_run(cls, ex, k, sg, plan):
        return ex._place_brick(int(sg.obj))

    @classmethod
    def features(cls, env, pr, k, sg, plan):
        task, sc = env.task, env.scenario.task
        n, kb = sc["n"], int(sg.obj)
        p, q = pr["brick_pos"][kb], pr["brick_quat"][kb]
        gp = pr["goals"][kb, :3]
        gyaw = G.yaw_of_quat(pr["goals"][kb, 3:])
        fit = pr["fits"][kb]
        allp = np.zeros(3 * task.n_bricks)
        allp[: 3 * n] = pr["brick_pos"].reshape(-1)
        return np.concatenate([[n, kb, task.progress(env, pr), float(pr["grasped"][kb] is not None)], p, q, gp, [gyaw], gp - p,
                               [fit["r_xy"], fit["r_z"], fit["r_yaw"]], _rel_tcp(env, pr, p), allp]).astype(np.float32)

    @classmethod
    def _label(cls, env, k):
        sc = env.scenario.task
        return f"{sc['colors'][k]} {sc['kinds'][k]}"

    @classmethod
    def describe(cls, env, sg):
        return cls._label(env, sg.obj)

    @classmethod
    def scene_text(cls, env):
        sc, pr = env.scenario.task, env.privileged()
        lines = ["Task: assemble bricks on the baseplate to match the shown target model.",
                 f"Baseplate at ({sc['plate'][0] * 100:.1f}, {sc['plate'][1] * 100:.1f}) cm.", "Bricks (name: size, position, target cell and level in the model):"]
        for k in range(sc["n"]):
            i, j, lvl, _yq = sc["goals"][k]
            p = pr["brick_pos"][k]
            lines.append(f"  - {cls._label(env, k)}: at ({p[0] * 100:.1f}, {p[1] * 100:.1f}) cm; target cell ({i}, {j}), level {lvl}")
        lines.append("Bricks are pressed on from above: a brick at level L needs the bricks under it (level L-1) in place first, "
                     "and nothing at its level or above may already cover its cells.")
        return "\n".join(lines)

    @classmethod
    def llm_format(cls, env):
        return 'Answer with only a JSON array of brick names (e.g. ["red 2x2", "blue 2x4"]) in build order.'

    @classmethod
    def parse_plan(cls, env, data):
        labels = {cls._label(env, k): k for k in range(env.scenario.task["n"])}
        cols = {env.scenario.task["colors"][k]: k for k in range(env.scenario.task["n"])}
        out = []
        for x in data:
            s = str(x["brick"] if isinstance(x, dict) else x).strip().lower()
            out.append(Subgoal(labels[s] if s in labels else cols[s.split()[0]], None))
        return out


class CableAdapter(TaskAdapter):
    name = "cable_plug"
    ordered = False

    @classmethod
    def oracle(cls, env):
        return [Subgoal(c, int(cab["port"])) for c, cab in enumerate(env.scenario.task["cables"])]

    @classmethod
    def heuristic(cls, env):
        """Lit port per cable (observable), nearest plug first."""
        pr = env.privileged()
        plan = cls.oracle(env)
        d = [np.linalg.norm(pr["plugs"][s.obj]["pos"] - env.task.port_pose(env, s.goal)[0]) for s in plan]
        return [plan[i] for i in np.argsort(d, kind="stable")]

    @classmethod
    def candidates(cls, env):
        return [Subgoal(c, p) for c in range(env.scenario.task["n"]) for p in range(env.task.tc.n_ports)]

    @classmethod
    def achieved(cls, env, sg, pr=None):
        pr = pr or env.privileged()
        pl = pr["plugs"][sg.obj]
        return pl["latched"] == int(sg.goal) and pl["grasped_by"] is None

    @classmethod
    def oracle_flags(cls, pr):
        return [bool(x) for x in pr["done"]]

    @classmethod
    def at_other_goal(cls, env, pr, obj, k):
        pl = pr["plugs"][obj]
        return pl["latched"] >= 0 and pl["latched"] != pl["target"]

    @classmethod
    def object_geoms(cls, env, obj):
        return env.task.plug_geoms[obj]

    @classmethod
    def expert_run(cls, ex, k, sg, plan):
        c, port = int(sg.obj), int(sg.goal)
        if port != int(ex.env.scenario.task["cables"][c]["port"]):
            ex.failure = FailureSpec("wrong_port", 1.0, {"subgoal": c, "port": port})  # the expert's own "wrong port" hook
        return ex._plug_cable(c)

    @classmethod
    def features(cls, env, pr, k, sg, plan):
        task, sc = env.task, env.scenario.task
        n, c = sc["n"], int(sg.obj)
        p = pr["plugs"][c]
        pp, pq = task.port_pose(env, int(sg.goal))
        R = G.quat_to_mat(pq)
        rel_tip = R.T @ (p["tip"] - pp)
        rel_ax = R.T @ G.quat_rotate(p["quat"], [1, 0, 0])
        return np.concatenate([[n, c, task.progress(env, pr), float(p["grasped_by"] is not None)], p["pos"], p["quat"], p["tip"], pp, pq, rel_tip, rel_ax,
                               _rel_tcp(env, pr, p["grasp"]), [p["depth"], p["lateral"], p["angle"]]]).astype(np.float32)

    @classmethod
    def describe(cls, env, sg):
        return f"cable {sg.obj} -> port {int(sg.goal) + 1}"

    @classmethod
    def scene_text(cls, env):
        sc, pr, task = env.scenario.task, env.privileged(), env.task
        lines = ["Task: plug each loose Ethernet cable into the port whose LED is lit for it.", "Cables (index: plug position on the bench, lit target port):"]
        for c, cab in enumerate(sc["cables"]):
            p = pr["plugs"][c]["pos"]
            r, col = task.port_rc(cab["port"])
            lines.append(f"  - cable {c}: plug at ({p[0] * 100:.1f}, {p[1] * 100:.1f}) cm; lit port {cab['port'] + 1} (row {r + 1}, column {col + 1})")
        occ = [o[0] + 1 for o in sc["occupied"]]
        lines.append(f"Ports already occupied by other cables: {occ if occ else 'none'}.")
        return "\n".join(lines)

    @classmethod
    def llm_format(cls, env):
        return 'Answer with only a JSON array in execution order, one object per cable: {"cable": <index>, "port": <port number as listed>}.'

    @classmethod
    def parse_plan(cls, env, data):
        return [Subgoal(int(m["cable"]), int(m["port"]) - 1) for m in data]


ADAPTERS = {"cube_stack": CubeAdapter, "lego": LegoAdapter, "cable_plug": CableAdapter}


def adapter_for(env) -> type[TaskAdapter]:
    return ADAPTERS[env.task.name]


# ---------------------------------------------------------------------- planners
class Planner:
    name = "planner"
    privileged = False

    def reset(self, env, seed=None):
        pass

    def plan(self, env, seed=None) -> list:
        raise NotImplementedError

    def info(self) -> dict:
        return {}


class OraclePlanner(Planner):
    name = "oracle"
    privileged = True

    def plan(self, env, seed=None):
        return adapter_for(env).oracle(env)


class HeuristicPlanner(Planner):
    """Rule-based plan from the observable scene (object poses, sizes, goal display): what a perception
    module plus hand-written task logic gives. Reads simulator poses as the perception stand-in."""

    name = "heuristic"
    privileged = True

    def plan(self, env, seed=None):
        return adapter_for(env).heuristic(env)


class PermutedPlanner(Planner):
    """Deterministic bad orders: the oracle plan reversed or shuffled (per-episode seed)."""

    privileged = True

    def __init__(self, kind="reverse", seed=0):
        self.kind, self.seed0 = kind, seed
        self.name = kind

    def plan(self, env, seed=None):
        plan = adapter_for(env).oracle(env)
        if self.kind == "reverse":
            return plan[::-1]
        rng = np.random.default_rng([self.seed0, int(seed or 0)])
        return [plan[i] for i in rng.permutation(len(plan))]


class CorruptedPlanner(Planner):
    """A base planner with controlled planning errors: with probability p_swap two adjacent subgoals are
    swapped, with p_goal one subgoal gets a different (valid-looking) goal, with p_drop one subgoal is left
    out. The knob for "how good must the planner be" curves."""

    def __init__(self, base=None, p_swap=0.0, p_goal=0.0, p_drop=0.0, seed=0):
        self.base = base or OraclePlanner()
        self.p_swap, self.p_goal, self.p_drop, self.seed0 = float(p_swap), float(p_goal), float(p_drop), seed
        self.privileged = self.base.privileged
        self.name = f"corrupt({self.base.name};swap{p_swap},goal{p_goal},drop{p_drop})"
        self.last = {}

    def reset(self, env, seed=None):
        self.base.reset(env, seed)

    def plan(self, env, seed=None):
        ad = adapter_for(env)
        plan = list(self.base.plan(env, seed))
        rng = np.random.default_rng([self.seed0, int(seed or 0), 5])
        ops = []
        if len(plan) > 1 and rng.random() < self.p_swap:
            i = int(rng.integers(len(plan) - 1))
            plan[i], plan[i + 1] = plan[i + 1], plan[i]
            ops.append(f"swap{i}")
        if plan and rng.random() < self.p_goal:
            i = int(rng.integers(len(plan)))
            alt = [c for c in ad.candidates(env) if c.obj == plan[i].obj and c.key() != plan[i].key()]
            if alt:
                plan[i] = alt[int(rng.integers(len(alt)))]
                ops.append(f"goal{i}")
        if len(plan) > 1 and rng.random() < self.p_drop:
            i = int(rng.integers(len(plan)))
            del plan[i]
            ops.append(f"drop{i}")
        self.last = dict(corruptions=ops)
        return plan

    def info(self):
        return dict(self.last, **self.base.info())


LLM_SYSTEM = ("You plan for a robot arm that picks and places one object at a time on a lab bench. "
              "Given the scene and the goal, output the sequence of subgoals that achieves the goal; respect the stated physical "
              "constraints (what can be picked or placed when). Think it through, then answer with only the JSON asked for, no prose.")


def extract_json(text: str):
    """The first JSON array in a response (tolerates ```json fences and surrounding prose)."""
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.S)
    s = m.group(1) if m else text[text.find("[") : text.rfind("]") + 1]
    return json.loads(s)


class LLMPlanner(Planner):
    """Off-the-shelf planner: a language model reads a text description of the scene and the goal and
    returns the subgoal sequence. `ask(system, user) -> str` is the model call (default: Claude through the
    Anthropic SDK, with Anthropic's default refusal fallbacks enabled); responses are cached on disk by
    (model, prompt) so re-running an evaluation is free and deterministic. A response that cannot be
    parsed gives an *empty* plan (counted as a planning failure), with the error in `info()`."""

    name = "llm"

    def __init__(self, model="claude-opus-5-5", effort="medium", ask=None, cache_dir=None, max_tokens=16000):
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.ask = ask or self._ask_claude
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._client = None
        self.name = f"llm({model})" if ask is None else "llm(custom)"
        self.last = {}

    def _ask_claude(self, system, user):
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic()
        r = self._client.beta.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system, messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort}, betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        if r.stop_reason == "refusal":
            raise RuntimeError(f"model declined: {getattr(r.stop_details, 'category', None)}")
        return "".join(b.text for b in r.content if b.type == "text")

    def plan(self, env, seed=None):
        ad = adapter_for(env)
        user = ad.scene_text(env) + "\n\n" + ad.llm_format(env)
        key = hashlib.sha1(f"{self.model}|{LLM_SYSTEM}|{user}".encode()).hexdigest()
        text, cached = None, False
        if self.cache_dir and (self.cache_dir / f"{key}.json").exists():
            text, cached = json.load(open(self.cache_dir / f"{key}.json"))["text"], True
        self.last = dict(llm_cached=cached, llm_error=None, llm_prompt_chars=len(user))
        try:
            if text is None:
                text = self.ask(LLM_SYSTEM, user)
                if self.cache_dir:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    json.dump(dict(model=self.model, prompt=user, text=text), open(self.cache_dir / f"{key}.json", "w"), indent=1)
            self.last["llm_text"] = text[-1500:]
            return ad.parse_plan(env, extract_json(text))
        except Exception as e:  # noqa: BLE001  (a bad plan is a benchmark outcome, not a crash)
            self.last["llm_error"] = f"{type(e).__name__}: {e}"[:300]
            return []

    def info(self):
        return dict(self.last)
