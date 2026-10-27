"""E0: why does the privileged expert fail? Runs the expert (n=1 cable, nominal) and records, for every
insert attempt, the peak contact force per category and the expert's verdict (latched / jam)."""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from common import CATS, Probe, make_env  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "runs" / "insertion_state"


def run_seed(seed):
    env = make_env()
    env.reset(seed=seed, variant={"n": 1})
    ex = env.task.make_expert(env, seed=seed)
    pb = Probe(env)
    attempts, cur, n_ev = [], None, 0
    while not ex.done:
        env.step(ex.act())
        new = ex.events[n_ev:]
        n_ev = len(ex.events)
        for name, _t in new:
            if name == "insert":
                cur = dict.fromkeys(CATS, 0.0)
                cur["max_depth"] = -1.0
                attempts.append(cur)
            elif name in ("jam", "latched") and cur is not None:
                cur["verdict"] = name
        if ex.phase == "insert" and cur is not None:
            for k, v in pb.contact_forces().items():
                cur[k] = max(cur[k], v)
            cur["max_depth"] = max(cur["max_depth"], env.privileged()["plugs"][0]["depth"])
    for a in attempts:
        a.setdefault("verdict", "other")
    info = env.info()
    return dict(seed=seed, success=bool(info["success"]), port=int(env.scenario.task["cables"][0]["port"]),
                n_occupied=len(env.scenario.task.get("occupied", [])), attempts=attempts, unstable=bool(env.unstable))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as p:
        res = p.map(run_seed, range(args.n), chunksize=1)
    (OUT / "e0_expert.json").write_text(json.dumps(res, indent=1, default=float))

    succ = np.mean([r["success"] for r in res])
    att = [a for r in res for a in r["attempts"]]
    jams = [a for a in att if a["verdict"] == "jam"]

    def cause(a):
        grip = a["grip_occ"] + a["grip_panel"]
        plug = a["plug_face"] + a["plug_wedge"] + a["plug_back"] + a["plug_occ"]
        if grip > 5 and grip >= plug:
            return "gripper_collision"
        if a["plug_face"] > 5 and a["plug_face"] >= a["plug_wedge"]:
            return "plug_on_face"
        if a["plug_wedge"] > 5:
            return "plug_on_jack_wedge"
        if a["plug_occ"] > 5:
            return "plug_on_neighbour"
        return "no_contact_found"

    causes = {}
    for a in jams:
        causes[cause(a)] = causes.get(cause(a), 0) + 1
    fail_first = {}
    for r in res:
        if not r["success"] and r["attempts"]:
            c = cause(r["attempts"][0]) if r["attempts"][0]["verdict"] == "jam" else "first_attempt_" + r["attempts"][0]["verdict"]
            fail_first[c] = fail_first.get(c, 0) + 1
    print(f"expert success {succ:.2f} over {len(res)} seeds; unstable {sum(r['unstable'] for r in res)}")
    print(f"attempts {len(att)}, jam verdicts {len(jams)}; jam causes: {causes}")
    print(f"failed episodes by first-attempt cause: {fail_first}")
    print("median peak force per category on jams:", {k: round(float(np.median([a[k] for a in jams])), 1) for k in CATS} if jams else {})


if __name__ == "__main__":
    main()
