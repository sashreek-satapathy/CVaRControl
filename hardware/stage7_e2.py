"""
stage7_e2.py -- E2 (PRIMARY): the offset step from E1's OWN measured d_alpha.

Two subcommands, kept separate on purpose so you look at the number before
spending any more device time:

    python3 stage7_e2.py predict --raw raw_mean_1790502945.json
        -- free, local, no hardware. Prints |d_alpha|, t*, |dc|, R_pred, and
           the discriminator (mean_loss_change), and writes
           stage7_e2_step.json with the control ready to submit.

    python3 stage7_e2.py submit --step stage7_e2_step.json --backend ibm_fez \
        --max-sets 8 --out raw_offset_live_$(date +%s).json
        -- submits c_live at the SAME reps/shots and the SAME phi draws as
           the mean arm (phi_draws() is seeded and frozen), so E2 is a paired
           comparison against E1, not two independent samples.

This mirrors stage7_ibm.dryrun()'s E2 block exactly, just fed from real
hardware counts instead of local sampling. The one step that is easy to get
backwards: DE-AMPLIFY FIRST. The device returns the error of the reps-fold
sequence; the control being moved is the SINGLE gate. Feeding the amplified
epsilon through the reps=1 Jacobian overshoots by ~reps x -- verified in the
dry run's own comments (|dc|=12, R=-11 when done wrong; |dc|=0.006, R=+0.17
when eps is divided by reps first).
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage7 import (CompositeX, W_METRIC, predict, reachable_projector,
                    offset_control, OPERATING_POINT)
from stage7_ibm import analyse_arm, run_arm_ibm, phi_draws, get_backend, _validate_raw


def cmd_predict(a):
    op = OPERATING_POINT
    d = json.loads(Path(a.raw).read_text())
    if d["arm"] != "mean":
        sys.exit(f"--raw must be the E1 'mean' arm's file (got arm={d['arm']!r})")
    reps = d["reps"]
    shots = d["shots"]
    if reps != op["reps"]:
        print(f"WARNING: raw file reps={reps} != frozen op reps={op['reps']}; "
              f"proceeding with the file's own reps, but this departs from "
              f"the pre-registration.")

    a_mean = analyse_arm(d["raw"], shots, op["alpha"], a.p_ro)

    # DE-AMPLIFY FIRST. arms["mean"]["eps"] is the AMPLIFIED (reps-fold)
    # epsilon per record; divide by reps to get back to single-gate space
    # before doing anything else with it. No extra debias here -- this
    # matches dryrun()'s own E2 block exactly (it recomputes L raw from the
    # de-amplified eps rather than re-deriving a debias term at this scale).
    E = np.array(a_mean["eps"], float) / float(reps)
    L = np.einsum("ki,ij,kj->k", E, W_METRIC, E)

    ctrl_mean = np.array(d["control"], float)
    m1 = CompositeX(op["n"], reps=1)
    P_B, rank, _ = reachable_projector(m1.jacobian(ctrl_mean, 0.0))
    pr = predict(E, L, op["alpha"], P_B=P_B)
    c_live, dc_live = offset_control(m1, ctrl_mean, pr["step"], 0.0)

    print(f"E1 (de-amplified, single gate): mean_L={L.mean():.6f}  "
          f"CVaR={pr['cvar_0']:.6f}")
    print(f"reachable rank   = {rank} / {m1.dim}  "
          f"({'full rank, good' if rank == m1.dim else 'RANK DEFICIT -- ansatz cannot reach the requested step'})")
    print(f"|d_alpha|        = {np.linalg.norm(pr['d']):.5f}")
    print(f"t*               = {pr['t_best']:.3f}  "
          f"(grid edge hit: {pr['at_grid_edge']})")
    print(f"R_pred           = {pr['R_pred']:+.4f}")
    print(f"mean_loss_change = {pr['mean_loss_change']:+.4f}  -- "
          + ("genuine tail trade, discriminator PASSES"
             if pr["tail_specific"] else
             "DISCRIMINATOR FAILS: mean arm not converged, gain would be "
             "baseline slack, not a tail trade"))
    print(f"|dc| (control step, single-gate space) = "
          f"{np.linalg.norm(dc_live):.5f}")

    out = {
        "source_raw": str(a.raw), "reps": reps, "shots": shots,
        "operating_point": op,
        "c_mean": ctrl_mean.tolist(), "c_live": c_live.tolist(),
        "dc_live": dc_live.tolist(),
        "predictor": pr, "reachable_rank": rank,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {a.out}")

    if not pr["tail_specific"]:
        print("\nSTOP: do not run `submit` on this step. The discriminator "
              "already failed on the measured E1 data -- submitting anyway "
              "spends window time on a result the pre-registration says to "
              "discard.")


def cmd_submit(a):
    op = OPERATING_POINT
    step = json.loads(Path(a.step).read_text())
    c_live = np.array(step["c_live"], float)
    reps = step["reps"]
    shots = step["shots"]
    pr = step["predictor"]
    if not pr.get("tail_specific", True) and not a.force:
        sys.exit("this step's discriminator failed at `predict` time -- "
                 "refusing to submit. Re-run `predict` on a converged mean "
                 "arm, or pass --force if you really mean to spend the "
                 "window on it anyway.")

    mR = CompositeX(op["n"], reps=reps)
    phis = phi_draws(op)          # SAME frozen seed as every other arm --
                                   # this is what makes E2 a PAIRED comparison
    backend = get_backend(a.backend, a.instance)
    print(f"backend {backend.name}, arm offset_live, reps={reps}, "
          f"K={len(phis)}, shots={shots}")
    raw, usage = run_arm_ibm(mR, c_live, phis, shots, backend, a.max_sets,
                             tag="offset_live")
    out = a.out or f"raw_offset_live_{int(time.time())}.json"
    Path(out).write_text(json.dumps(
        {"arm": "offset_live", "backend": backend.name, "shots": shots,
         "reps": reps, "operating_point": op,
         "usage": [str(u) for u in usage],
         "control": c_live.tolist(), "raw": raw}, indent=1, default=float))
    print(f"wrote {out}  (raw counts saved: re-analysis never needs a re-run)")
    _validate_raw(raw, shots, tag="offset_live")
    print(f"\nnext: python3 stage7_ibm.py analyse --raw raw_mean_*.json "
          f"{out}")


def main():
    ap = argparse.ArgumentParser(
        description="Stage 7 E2 -- offset step from E1's live measured d_alpha")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("predict", help="free, local: compute c_live from a "
                                        "mean-arm raw file")
    p1.add_argument("--raw", required=True,
                    help="the E1 (arm=mean) raw json file, e.g. "
                         "raw_mean_1790502945.json")
    p1.add_argument("--p-ro", type=float, default=0.01)
    p1.add_argument("--out", default="stage7_e2_step.json")
    p1.set_defaults(func=cmd_predict)

    p2 = sub.add_parser("submit", help="spends device time: submit c_live "
                                       "to hardware")
    p2.add_argument("--step", required=True, help="output of `predict`")
    p2.add_argument("--backend", required=True)
    p2.add_argument("--instance", default=None)
    p2.add_argument("--max-sets", type=int, default=None)
    p2.add_argument("--out", default=None)
    p2.add_argument("--force", action="store_true",
                    help="submit even if the discriminator already failed "
                         "at predict time")
    p2.set_defaults(func=cmd_submit)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
