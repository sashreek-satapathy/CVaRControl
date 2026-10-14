#!/usr/bin/env python3
"""Final local forensic audit for the Stage 7 IBM dataset.

Run from the stage7_code directory:
    python3 audit_stage7.py

No IBM/Qiskit connection is used. No hardware jobs are submitted.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from stage7 import CompositeX, W_METRIC, cvar

ROOT = Path(__file__).resolve().parent

FILES = {
    "E0": "raw_hard_E0.json",
    "E1": "raw_mean_1790502945.json",
    "E2": "raw_e2_live_1790502945.json",
    "E3": "raw_cvar_E3_1790509726.json",
    "E4M": "raw_e4_mean_s03_1790511613.json",
    "E4C": "raw_e4_cvar_s03_1790512144.json",
}

EXPECTED_KEYS = ["0|X", "0|Y", "0|Z", "+|X", "+|Y", "+|Z"]


def expval(counts: dict[str, int]) -> float:
    n0 = int(counts.get("0", 0))
    n1 = int(counts.get("1", 0))
    tot = n0 + n1
    if tot <= 0:
        raise ValueError("empty count dictionary")
    return (n0 - n1) / tot


def eps_from_record(rec: dict) -> np.ndarray:
    c = rec["counts"]
    vals0 = [expval(c[k]) for k in ("0|X", "0|Y", "0|Z")]
    vals1 = [expval(c[k]) for k in ("+|X", "+|Y", "+|Z")]
    # Same reconstruction used by stage7_ibm.py: two rotated Bloch vectors,
    # then nearest rotation + logarithm are implemented in stage7.eps_from_counts.
    from stage7 import eps_from_counts
    return np.asarray(eps_from_counts(np.asarray(vals0), np.asarray(vals1)), float)


def analyse(d: dict) -> dict:
    shots = int(d["shots"])
    E = np.array([eps_from_record(r) for r in d["raw"]], float)
    L_raw = np.einsum("ki,ij,kj->k", E, W_METRIC, E)
    bias = (3.0 / shots) / 6.0
    L = np.maximum(L_raw - bias, 0.0)
    return {
        "E": E,
        "L_raw": L_raw,
        "L": L,
        "mean_L": float(L.mean()),
        "cvar": float(cvar(L, 0.2)),
        "bias": bias,
        "n_clipped": int(np.count_nonzero(L_raw < bias)),
        "shots": shots,
        "phis": np.array([float(r["phi"]) for r in d["raw"]]),
        "control": np.asarray(d.get("control", []), float),
    }


def corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def tail_indices(L: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    k = max(1, int(np.ceil(alpha * len(L))))
    return np.argsort(L)[-k:]


def paired(name_a: str, a: dict, name_b: str, b: dict) -> None:
    dL = b["L"] - a["L"]
    R = 1.0 - b["cvar"] / a["cvar"]
    ratio = b["mean_L"] / a["mean_L"]
    delta_mean = float(dL.mean())
    ti_a = set(tail_indices(a["L"]))
    ti_b = set(tail_indices(b["L"]))
    overlap = len(ti_a & ti_b)
    union = len(ti_a | ti_b)
    print(f"\n{name_a} -> {name_b}")
    print(f"  mean_L:      {a['mean_L']:.8f} -> {b['mean_L']:.8f}  ratio={ratio:.6f}")
    print(f"  CVaR_0.2:    {a['cvar']:.8f} -> {b['cvar']:.8f}  R={R:+.6f}")
    print(f"  delta mean:  {delta_mean:+.8f}")
    print(f"  paired corr: {corr(a['L'], b['L']):+.6f}")
    print(f"  tail overlap:{overlap}/52  Jaccard={overlap/union:.6f}")
    print(f"  dL median:   {np.median(dL):+.8f}")
    print(f"  dL min/max:  {dL.min():+.8f} / {dL.max():+.8f}")


def main() -> None:
    print("STAGE 7 FINAL LOCAL FORENSIC AUDIT")
    print("No IBM jobs are submitted.\n")

    data = {}
    results = {}
    for label, fn in FILES.items():
        p = ROOT / fn
        if not p.exists():
            raise FileNotFoundError(f"Missing {fn}")
        d = json.loads(p.read_text())
        data[label] = d
        results[label] = analyse(d)

        raw = d["raw"]
        bad_shots = 0
        bad_keys = 0
        for r in raw:
            if sorted(r["counts"].keys()) != sorted(EXPECTED_KEYS):
                bad_keys += 1
            for k in EXPECTED_KEYS:
                if sum(int(v) for v in r["counts"][k].values()) != int(d["shots"]):
                    bad_shots += 1
        print(f"{label}: {fn}")
        print(f"  records={len(raw)}/256  keys_bad={bad_keys}  shot_totals_bad={bad_shots}")
        print(f"  shots={d['shots']} reps={d.get('reps')} backend={d.get('backend')}")
        print(f"  mean_L={results[label]['mean_L']:.8f} CVaR={results[label]['cvar']:.8f}")
        print(f"  shot-bias={results[label]['bias']:.8f} clipped={results[label]['n_clipped']}")

    # Phase pairing checks
    for a, b in [("E1", "E2"), ("E1", "E3"), ("E4M", "E4C")]:
        diff = np.max(np.abs(results[a]["phis"] - results[b]["phis"]))
        print(f"\nPhase pairing {a} vs {b}: max |dphi| = {diff:.3e}")

    paired("E1", results["E1"], "E2", results["E2"])
    paired("E1", results["E1"], "E3", results["E3"])
    paired("E4M", results["E4M"], "E4C", results["E4C"])

    # Ideal-model check for E4 using the exact hardware controls and exact same phis.
    print("\nE4 ideal-model replay")
    model = CompositeX(3, reps=int(data["E4M"]["reps"]))
    for label in ("E4M", "E4C"):
        Lideal = model.loss_batch(results[label]["control"], results[label]["phis"])
        print(f"  {label}: mean_L={np.mean(Lideal):.8f} CVaR={cvar(Lideal,0.2):.8f}")
        results[label]["Lideal"] = np.asarray(Lideal)
    R_ideal = 1.0 - cvar(results["E4C"]["Lideal"],0.2) / cvar(results["E4M"]["Lideal"],0.2)
    ratio_ideal = np.mean(results["E4C"]["Lideal"]) / np.mean(results["E4M"]["Lideal"])
    R_hw = 1.0 - results["E4C"]["cvar"] / results["E4M"]["cvar"]
    ratio_hw = results["E4C"]["mean_L"] / results["E4M"]["mean_L"]
    print(f"  ideal:    R={R_ideal:+.6f} mean-ratio={ratio_ideal:.6f}")
    print(f"  hardware: R={R_hw:+.6f} mean-ratio={ratio_hw:.6f}")
    print(f"  ideal-vs-hardware sign reversal: {np.sign(R_ideal) != np.sign(R_hw)}")

    # E4 per-phi ideal/hardware correlation within each arm.
    for label in ("E4M", "E4C"):
        print(f"  {label}: corr(hardware, ideal)={corr(results[label]['L'], results[label]['Lideal']):+.6f}")

    print("\nAUDIT COMPLETE")


if __name__ == "__main__":
    main()
