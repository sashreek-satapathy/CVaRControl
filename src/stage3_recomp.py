"""
stage3_recomp.py -- recompute Stage 3's R_tail with the paper's comparator.

WHY
============================================================================
The R_tail stored in each point's JSON (and copied into *_geo.json) is the
closed-system value against the MULTISTART mean control.  The paper does not
report that number: the fairness protocol compares the CVaR control with the
best mean-objective control found by ANY route -- multistart, continuation or
reverse warm start -- chosen on the validation ensemble (pitfalls 6 and 7).
For Stage 0/1/2 the correction happens inside `fast_viper.py validate`, which
also re-scores everything in the full Lindblad model.  Part C cannot be
validated that way (its targets and anharmonicities are not the ones compiled
into the Lindblad worker), so its numbers were left uncorrected -- which is
why, for example, the two-level X_pi/2 17-mode point reports R_tail = +0.31 at
a setting where Stage 0 reports about zero.

This script applies the comparator rule to the saved closed-system test
losses.  It re-optimizes nothing: it reads the per-draw test losses already in
each point's .npz and the validation numbers already in its .json, so it runs
in seconds.

USAGE   python stage3_recomp.py [results_stage3]
OUTPUT  results_stage3/stage3_recomp.json   (small; send this back)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np


def cvar(L, a):
    L = np.asarray(L, float)
    n = len(L)
    m = max(int(round(a * n)), 1)
    return float(np.partition(L, n - m)[n - m:].mean())


def main():
    base = Path(sys.argv[1] if len(sys.argv) > 1 else "results_stage3")
    out = []
    for jf in sorted(base.glob("stage3*/*.json")):
        if jf.name.endswith("_geo.json"):
            continue
        pt = json.loads(jf.read_text())
        npz = jf.with_suffix(".npz")
        if "cells" not in pt or not npz.exists():
            continue
        arr = np.load(npz)
        L = {k[6:]: arr[k] for k in arr.files if k.startswith("testL_")}
        if "mean" not in L:
            continue
        # the fairness rule: best mean-objective control by VALIDATION mean loss
        cands = []
        for c in pt["cells"]:
            if "reverse_val_mean" in c and f"rev_{c['alpha']}" in L:
                cands.append(("mean", c["reverse_val_mean"] - c["reverse_vs_mean_ctrl"]))
                cands.append((f"rev_{c['alpha']}", c["reverse_val_mean"]))
        best = min(cands, key=lambda t: t[1])[0] if cands else "mean"
        row = {"file": f"{jf.parent.name}/{jf.name}", "stem": jf.stem,
               "comparator": best,
               "mean_loss_multistart": float(L["mean"].mean()),
               "mean_loss_best": float(L[best].mean()),
               "nominal_share": None}
        for k in ("kappa_star", "kappa_calib", "rho_da", "theta", "theta_name", "axis",
                  "anharm_MHz", "basis", "n_basis", "noise_scale", "sigma_d_MHz", "sigma_a",
                  "sigma_phi_sq", "closed_system_only", "stage"):
            if k in pt:
                row[k] = pt[k]
        row["Tg"] = pt["grid"]["Tg"]
        row["n_levels"] = pt["grid"].get("n_levels")
        for c in pt["cells"]:
            a = c["alpha"]
            key = f"cvar_{a}"
            if key not in L:
                continue
            cm_raw, cm_best, cc_ = cvar(L["mean"], a), cvar(L[best], a), cvar(L[key], a)
            row[f"R_raw@{a}"] = float(1 - cc_ / cm_raw)
            row[f"R_best@{a}"] = float(1 - cc_ / cm_best)
            row[f"cvar_mean@{a}"], row[f"cvar_cvar@{a}"] = cm_best, cc_
            row[f"reverse_class@{a}"] = c.get("reverse_class")
            row[f"rel_pulse_change@{a}"] = c.get("rel_pulse_change")
        out.append(row)
    (base / "stage3_recomp.json").write_text(json.dumps({"rows": out}, indent=1))
    n_fixed = sum(1 for r in out if r["comparator"] != "mean")
    print(f"{len(out)} points, comparator upgraded at {n_fixed}")
    print(f"{'point':52s} {'cmp':>9} {'R_raw':>8} {'R_best':>8}")
    for r in out:
        if "R_raw@0.05" in r:
            flag = "  <-- artifact" if r["R_raw@0.05"] - r["R_best@0.05"] > 0.05 else ""
            print(f"{r['file'][:52]:52s} {r['comparator']:>9} {r['R_raw@0.05']:+8.3f} {r['R_best@0.05']:+8.3f}{flag}")
    print(f"\n-> {base/'stage3_recomp.json'}")


if __name__ == "__main__":
    main()