#!/usr/bin/env python3
"""
Stage 7 deep forensic audit.

LOCAL ONLY:
- No IBM/Qiskit Runtime connection
- No hardware jobs
- Uses the six completed Stage 7 raw JSON datasets
- Preserves the Stage 7 reconstruction/debiasing/CVaR definitions
- Adds supplementary diagnostics:
    * per-phase extraction
    * phi / |phi| dependence
    * paired delta-L
    * tail membership
    * bootstrap uncertainty
    * E0 background comparison
    * ideal-vs-hardware E4 comparison
    * publication-quality diagnostic figures
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stage7 import W_METRIC, cvar, eps_from_counts, CompositeX


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "stage7_deep"
OUT.mkdir(exist_ok=True)

FILES = {
    "E0": "raw_hard_E0.json",
    "E1": "raw_mean_1790502945.json",
    "E2": "raw_e2_live_1790502945.json",
    "E3": "raw_cvar_E3_1790509726.json",
    "E4M": "raw_e4_mean_s03_1790511613.json",
    "E4C": "raw_e4_cvar_s03_1790512144.json",
}

EXPECTED_KEYS = [
    "0|X", "0|Y", "0|Z",
    "+|X", "+|Y", "+|Z",
]

ALPHA = 0.20
N_BOOT = 10000
BOOT_SEED = 20260927


# ============================================================
# Basic utilities
# ============================================================

def expval(counts: dict[str, int]) -> float:
    n0 = int(counts.get("0", 0))
    n1 = int(counts.get("1", 0))
    n = n0 + n1

    if n <= 0:
        raise ValueError("Empty count dictionary")

    return (n0 - n1) / n


def eps_from_record(rec: dict) -> np.ndarray:
    c = rec["counts"]

    vals0 = np.array([
        expval(c["0|X"]),
        expval(c["0|Y"]),
        expval(c["0|Z"]),
    ])

    vals1 = np.array([
        expval(c["+|X"]),
        expval(c["+|Y"]),
        expval(c["+|Z"]),
    ])

    return np.asarray(
        eps_from_counts(vals0, vals1),
        dtype=float,
    )


def analyse_dataset(d: dict) -> dict:
    shots = int(d["shots"])

    E = np.array(
        [eps_from_record(r) for r in d["raw"]],
        dtype=float,
    )

    L_raw = np.einsum(
        "ki,ij,kj->k",
        E,
        W_METRIC,
        E,
    )

    # EXACT Stage 7 finite-shot debiasing convention.
    bias = (3.0 / shots) / 6.0

    L = np.maximum(L_raw - bias, 0.0)

    phis = np.array(
        [float(r["phi"]) for r in d["raw"]],
        dtype=float,
    )

    return {
        "E": E,
        "L_raw": L_raw,
        "L": L,
        "phis": phis,
        "bias": bias,
        "shots": shots,
        "reps": int(d.get("reps", 0)),
        "mean_L": float(np.mean(L)),
        "cvar": float(cvar(L, ALPHA)),
        "n_clipped": int(np.count_nonzero(L_raw < bias)),
        "control": np.asarray(d.get("control", []), dtype=float),
        "meta": d,
    }


def tail_indices(L: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    k = max(1, int(np.ceil(alpha * len(L))))
    return np.argsort(L)[-k:]


def bootstrap_mean(x, rng, n_boot=N_BOOT):
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    return np.mean(x[idx], axis=1)


def bootstrap_cvar(x, rng, n_boot=N_BOOT):
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = x[idx]

    k = max(1, int(np.ceil(ALPHA * n)))

    # Upper-tail atomic CVaR for each bootstrap sample.
    vals_sorted = np.sort(vals, axis=1)
    return np.mean(vals_sorted[:, -k:], axis=1)


def percentile_ci(x, lo=2.5, hi=97.5):
    return (
        float(np.percentile(x, lo)),
        float(np.percentile(x, hi)),
    )


def corr(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ============================================================
# Load data
# ============================================================

print("=" * 72)
print("STAGE 7 DEEP FORENSIC AUDIT")
print("LOCAL ONLY — NO IBM JOBS")
print("=" * 72)

data = {}
for label, filename in FILES.items():
    path = ROOT / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )

    print(f"Loading {filename}")
    data[label] = json.loads(path.read_text())

results = {
    label: analyse_dataset(d)
    for label, d in data.items()
}


# ============================================================
# Structural validation
# ============================================================

print("\nSTRUCTURAL VALIDATION")

for label, d in data.items():
    raw = d["raw"]
    bad_keys = 0
    bad_shots = 0

    for rec in raw:
        if sorted(rec["counts"].keys()) != sorted(EXPECTED_KEYS):
            bad_keys += 1

        for key in EXPECTED_KEYS:
            total = sum(
                int(v)
                for v in rec["counts"][key].values()
            )

            if total != int(d["shots"]):
                bad_shots += 1

    print(
        f"{label}: "
        f"records={len(raw)}/256, "
        f"bad_keys={bad_keys}, "
        f"bad_shots={bad_shots}"
    )


# ============================================================
# Summary table
# ============================================================

summary_rows = []

for label, r in results.items():
    summary_rows.append({
        "experiment": label,
        "shots": r["shots"],
        "reps": r["reps"],
        "mean_L": r["mean_L"],
        "CVaR_0.2": r["cvar"],
        "bias": r["bias"],
        "clipped": r["n_clipped"],
        "clip_fraction": r["n_clipped"] / 256.0,
    })

write_csv(
    OUT / "summary.csv",
    summary_rows,
    [
        "experiment",
        "shots",
        "reps",
        "mean_L",
        "CVaR_0.2",
        "bias",
        "clipped",
        "clip_fraction",
    ],
)


# ============================================================
# Per-phase extraction
# ============================================================

print("\nPER-PHASE EXTRACTION")

phase_rows = []

for label, r in results.items():
    E = r["E"]
    L_raw = r["L_raw"]
    L = r["L"]
    phi = r["phis"]

    for k in range(len(phi)):
        phase_rows.append({
            "experiment": label,
            "index": k,
            "phi": phi[k],
            "abs_phi": abs(phi[k]),
            "eps_x": E[k, 0],
            "eps_y": E[k, 1],
            "eps_z": E[k, 2],
            "eps_norm": float(np.linalg.norm(E[k])),
            "L_raw": L_raw[k],
            "L": L[k],
            "clipped": int(L_raw[k] < r["bias"]),
        })

write_csv(
    OUT / "per_phase.csv",
    phase_rows,
    [
        "experiment",
        "index",
        "phi",
        "abs_phi",
        "eps_x",
        "eps_y",
        "eps_z",
        "eps_norm",
        "L_raw",
        "L",
        "clipped",
    ],
)


# ============================================================
# Phase pairing
# ============================================================

print("\nPHASE PAIRING")

pairs = [
    ("E1", "E2"),
    ("E1", "E3"),
    ("E4M", "E4C"),
]

for a, b in pairs:
    diff = np.max(
        np.abs(
            results[a]["phis"] -
            results[b]["phis"]
        )
    )

    print(
        f"{a} vs {b}: "
        f"max |Δphi| = {diff:.3e}"
    )


# ============================================================
# Paired loss diagnostics
# ============================================================

paired_rows = []

print("\nPAIRED LOSS DIAGNOSTICS")

for a, b in pairs:
    La = results[a]["L"]
    Lb = results[b]["L"]
    phi = results[a]["phis"]

    dL = Lb - La

    R = 1.0 - (
        results[b]["cvar"] /
        results[a]["cvar"]
    )

    mean_ratio = (
        results[b]["mean_L"] /
        results[a]["mean_L"]
    )

    ia = set(tail_indices(La))
    ib = set(tail_indices(Lb))

    overlap = len(ia & ib)
    union = len(ia | ib)

    print(f"\n{a} -> {b}")
    print(f"  Δmean L       = {np.mean(dL):+.8f}")
    print(f"  median ΔL     = {np.median(dL):+.8f}")
    print(f"  min/max ΔL    = {np.min(dL):+.8f} / {np.max(dL):+.8f}")
    print(f"  mean ratio    = {mean_ratio:.6f}")
    print(f"  CVaR R        = {R:+.6f}")
    print(f"  correlation   = {corr(La, Lb):+.6f}")
    print(f"  tail overlap  = {overlap}/52")
    print(f"  tail Jaccard  = {overlap / union:.6f}")

    for k in range(256):
        paired_rows.append({
            "pair": f"{a}->{b}",
            "index": k,
            "phi": phi[k],
            "abs_phi": abs(phi[k]),
            "L_a": La[k],
            "L_b": Lb[k],
            "delta_L": dL[k],
            "tail_a": int(k in ia),
            "tail_b": int(k in ib),
        })

write_csv(
    OUT / "paired_per_phase.csv",
    paired_rows,
    [
        "pair",
        "index",
        "phi",
        "abs_phi",
        "L_a",
        "L_b",
        "delta_L",
        "tail_a",
        "tail_b",
    ],
)


# ============================================================
# Phi-bin analysis
# Exploratory only — not a replacement for preregistered analysis.
# ============================================================

print("\nPHI-BIN ANALYSIS")
print("Exploratory supplementary diagnostic.")

phi = results["E4M"]["phis"]
abs_phi = np.abs(phi)

# Quartiles give four equal-sized bins.
edges = np.quantile(
    abs_phi,
    [0.0, 0.25, 0.50, 0.75, 1.0],
)

# Ensure unique bin edges.
edges = np.unique(edges)

phi_rows = []

for a, b in pairs:
    La = results[a]["L"]
    Lb = results[b]["L"]
    dL = Lb - La

    # Use the phase ensemble belonging to the pair.
    ph = results[a]["phis"]
    ap = np.abs(ph)

    for j in range(len(edges) - 1):
        lo = edges[j]
        hi = edges[j + 1]

        if j == len(edges) - 2:
            mask = (ap >= lo) & (ap <= hi)
        else:
            mask = (ap >= lo) & (ap < hi)

        if not np.any(mask):
            continue

        phi_rows.append({
            "pair": f"{a}->{b}",
            "bin": j + 1,
            "abs_phi_min": lo,
            "abs_phi_max": hi,
            "n": int(np.sum(mask)),
            "mean_L_a": float(np.mean(La[mask])),
            "mean_L_b": float(np.mean(Lb[mask])),
            "mean_delta_L": float(np.mean(dL[mask])),
            "median_delta_L": float(np.median(dL[mask])),
        })

write_csv(
    OUT / "phi_bins.csv",
    phi_rows,
    [
        "pair",
        "bin",
        "abs_phi_min",
        "abs_phi_max",
        "n",
        "mean_L_a",
        "mean_L_b",
        "mean_delta_L",
        "median_delta_L",
    ],
)


# ============================================================
# Bootstrap paired statistics
# ============================================================

print("\nBOOTSTRAP ANALYSIS")
print(f"Bootstrap replicates: {N_BOOT}")

rng = np.random.default_rng(BOOT_SEED)

bootstrap_rows = []

for a, b in pairs:
    La = results[a]["L"]
    Lb = results[b]["L"]

    n = len(La)

    idx = rng.integers(
        0,
        n,
        size=(N_BOOT, n),
    )

    A = La[idx]
    B = Lb[idx]

    mean_A = np.mean(A, axis=1)
    mean_B = np.mean(B, axis=1)

    k = max(1, int(np.ceil(ALPHA * n)))

    A_sorted = np.sort(A, axis=1)
    B_sorted = np.sort(B, axis=1)

    cvar_A = np.mean(
        A_sorted[:, -k:],
        axis=1,
    )

    cvar_B = np.mean(
        B_sorted[:, -k:],
        axis=1,
    )

    delta_mean = mean_B - mean_A
    mean_ratio = mean_B / mean_A

    R = 1.0 - (
        cvar_B / cvar_A
    )

    def add_bootstrap_stat(name, values):
        lo, hi = percentile_ci(values)

        bootstrap_rows.append({
            "pair": f"{a}->{b}",
            "statistic": name,
            "estimate": float(np.mean(values)),
            "median_bootstrap": float(np.median(values)),
            "ci95_low": lo,
            "ci95_high": hi,
        })

    add_bootstrap_stat(
        "delta_mean_L",
        delta_mean,
    )

    add_bootstrap_stat(
        "mean_ratio",
        mean_ratio,
    )

    add_bootstrap_stat(
        "R_CVaR",
        R,
    )

    print(f"\n{a} -> {b}")

    lo, hi = percentile_ci(delta_mean)
    print(
        f"  Δmean L 95% CI: "
        f"[{lo:+.6f}, {hi:+.6f}]"
    )

    lo, hi = percentile_ci(R)
    print(
        f"  R 95% CI:      "
        f"[{lo:+.6f}, {hi:+.6f}]"
    )

write_csv(
    OUT / "bootstrap.csv",
    bootstrap_rows,
    [
        "pair",
        "statistic",
        "estimate",
        "median_bootstrap",
        "ci95_low",
        "ci95_high",
    ],
)


# ============================================================
# Tail membership
# ============================================================

print("\nTAIL MEMBERSHIP")

tail_rows = []

for label, r in results.items():
    inds = set(tail_indices(r["L"]))

    for k in range(256):
        tail_rows.append({
            "experiment": label,
            "index": k,
            "phi": r["phis"][k],
            "abs_phi": abs(r["phis"][k]),
            "L": r["L"][k],
            "tail": int(k in inds),
        })

write_csv(
    OUT / "tail_membership.csv",
    tail_rows,
    [
        "experiment",
        "index",
        "phi",
        "abs_phi",
        "L",
        "tail",
    ],
)


# ============================================================
# E0 background diagnostic
# ============================================================

print("\nE0 BACKGROUND DIAGNOSTIC")

e0_mean = results["E0"]["mean_L"]
e0_cvar = results["E0"]["cvar"]

background_rows = []

for label in ["E1", "E2", "E3", "E4M", "E4C"]:
    mean_L = results[label]["mean_L"]
    cv = results[label]["cvar"]

    background_rows.append({
        "experiment": label,
        "mean_L": mean_L,
        "CVaR": cv,
        "E0_mean_L": e0_mean,
        "E0_CVaR": e0_cvar,
        "mean_minus_E0": mean_L - e0_mean,
        "CVaR_minus_E0": cv - e0_cvar,
        "mean_fraction_E0": e0_mean / mean_L,
        "cvar_fraction_E0": e0_cvar / cv,
    })

    print(
        f"{label}: "
        f"mean-E0={mean_L-e0_mean:+.8f}, "
        f"CVaR-E0={cv-e0_cvar:+.8f}"
    )

write_csv(
    OUT / "e0_background.csv",
    background_rows,
    [
        "experiment",
        "mean_L",
        "CVaR",
        "E0_mean_L",
        "E0_CVaR",
        "mean_minus_E0",
        "CVaR_minus_E0",
        "mean_fraction_E0",
        "cvar_fraction_E0",
    ],
)


# ============================================================
# E4 ideal replay
# ============================================================

print("\nE4 IDEAL REPLAY")

# CompositeX takes n/target/reps; sigma is represented by the
# actual phi ensemble supplied to model.eps().
#
# E4 uses sigma = 0.3 and the exact same hardware phase draws.
sigma = 0.3
phis_e4 = results["E4M"]["phis"]

model = CompositeX(
    n=3,
    reps=int(results["E4M"]["reps"]),
)

c_mean = results["E4M"]["control"]
c_cvar = results["E4C"]["control"]


def ideal_losses(control):
    vals = []

    for phi_value in phis_e4:
        eps = np.asarray(
            model.eps(
                control,
                phi_value,
            ),
            dtype=float,
        )

        vals.append(
            float(
                eps @ W_METRIC @ eps
            )
        )

    return np.asarray(vals, dtype=float)


try:
    ideal_M = ideal_losses(c_mean)
    ideal_C = ideal_losses(c_cvar)

    ideal_mean_M = float(np.mean(ideal_M))
    ideal_mean_C = float(np.mean(ideal_C))

    ideal_cvar_M = float(
        cvar(ideal_M, ALPHA)
    )

    ideal_cvar_C = float(
        cvar(ideal_C, ALPHA)
    )

    ideal_R = 1.0 - (
        ideal_cvar_C /
        ideal_cvar_M
    )

    ideal_ratio = (
        ideal_mean_C /
        ideal_mean_M
    )

    hardware_R = 1.0 - (
        results["E4C"]["cvar"] /
        results["E4M"]["cvar"]
    )

    print(
        f"  ideal mean: "
        f"{ideal_mean_M:.8f} -> "
        f"{ideal_mean_C:.8f}"
    )

    print(
        f"  ideal CVaR: "
        f"{ideal_cvar_M:.8f} -> "
        f"{ideal_cvar_C:.8f}"
    )

    print(
        f"  ideal R: "
        f"{ideal_R:+.8f}"
    )

    print(
        f"  ideal mean-ratio: "
        f"{ideal_ratio:.8f}"
    )

    print(
        f"  hardware R: "
        f"{hardware_R:+.8f}"
    )

    print(
        f"  hardware mean-ratio: "
        f"{results['E4C']['mean_L'] / results['E4M']['mean_L']:.8f}"
    )

    print(
        f"  sign reversal: "
        f"{np.sign(ideal_R) != np.sign(hardware_R)}"
    )

except Exception as exc:
    print("Ideal replay failed:", exc)
    ideal_M = None
    ideal_C = None

# ============================================================
# FIGURE 1
# Hardware loss versus |phi|
# ============================================================

plt.figure(figsize=(9, 6))

for label in ["E1", "E2", "E3", "E4M", "E4C"]:
    plt.scatter(
        np.abs(results[label]["phis"]),
        results[label]["L"],
        s=14,
        alpha=0.45,
        label=label,
    )

plt.xlabel(r"$|\phi|$")
plt.ylabel(r"Debiased loss $L$")
plt.title("Stage 7 hardware loss versus injected phase")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUT / "figure_1_loss_by_phi.png",
    dpi=250,
)
plt.close()


# ============================================================
# FIGURE 2
# Paired delta-L
# ============================================================

plt.figure(figsize=(9, 6))

for pair in [("E1", "E2"), ("E1", "E3"), ("E4M", "E4C")]:
    a, b = pair

    dL = (
        results[b]["L"] -
        results[a]["L"]
    )

    plt.scatter(
        results[a]["phis"],
        dL,
        s=16,
        alpha=0.55,
        label=f"{a} → {b}",
    )

plt.axhline(0.0, linestyle="--")

plt.xlabel(r"$\phi$")
plt.ylabel(r"$\Delta L = L_B-L_A$")
plt.title("Paired per-phase loss changes")
plt.legend()
plt.tight_layout()
plt.savefig(
    OUT / "figure_2_delta_loss.png",
    dpi=250,
)
plt.close()


# ============================================================
# FIGURE 3
# E4 tail membership
# ============================================================

plt.figure(figsize=(10, 5))

tail_M = tail_indices(results["E4M"]["L"])
tail_C = tail_indices(results["E4C"]["L"])

membership_M = np.zeros(256)
membership_C = np.zeros(256)

membership_M[tail_M] = 1
membership_C[tail_C] = 1

x = np.arange(256)

plt.scatter(
    x,
    membership_M + 0.03,
    s=18,
    label="E4 mean tail",
)

plt.scatter(
    x,
    membership_C - 0.03,
    s=18,
    label="E4 CVaR tail",
)

plt.xlabel("Phase-draw index")
plt.ylabel("Tail membership")
plt.title("E4 upper-tail membership")
plt.yticks([0, 1], ["CVaR arm", "Mean arm"])
plt.legend()
plt.tight_layout()
plt.savefig(
    OUT / "figure_3_tail_membership.png",
    dpi=250,
)
plt.close()


# ============================================================
# FIGURE 4
# Ideal vs hardware E4
# ============================================================

if ideal_M is not None:

    plt.figure(figsize=(9, 6))

    plt.scatter(
        ideal_M,
        results["E4M"]["L"],
        s=20,
        alpha=0.65,
        label="E4 mean",
    )

    plt.scatter(
        ideal_C,
        results["E4C"]["L"],
        s=20,
        alpha=0.65,
        label="E4 CVaR",
    )

    mn = min(
        np.min(ideal_M),
        np.min(ideal_C),
    )

    mx = max(
        np.max(ideal_M),
        np.max(ideal_C),
    )

    plt.plot(
        [mn, mx],
        [mn, mx],
        linestyle="--",
        label="y=x",
    )

    plt.xlabel("Ideal-model loss")
    plt.ylabel("Hardware reconstructed loss")
    plt.title("E4 ideal-model versus hardware loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT / "figure_4_ideal_vs_hardware.png",
        dpi=250,
    )

    plt.close()


# ============================================================
# Final report
# ============================================================

report = []

report.append("STAGE 7 DEEP FORENSIC AUDIT")
report.append("=" * 72)
report.append("LOCAL ANALYSIS ONLY — NO IBM JOBS SUBMITTED")
report.append("")

report.append("DATASET SUMMARY")
report.append("-" * 72)

for label in ["E0", "E1", "E2", "E3", "E4M", "E4C"]:
    r = results[label]

    report.append(
        f"{label}: "
        f"mean_L={r['mean_L']:.8f}, "
        f"CVaR={r['cvar']:.8f}, "
        f"clipped={r['n_clipped']}/256 "
        f"({100*r['n_clipped']/256:.2f}%)"
    )

report.append("")
report.append("PAIRED RESULTS")
report.append("-" * 72)

for a, b in pairs:
    La = results[a]["L"]
    Lb = results[b]["L"]

    R = 1.0 - (
        results[b]["cvar"] /
        results[a]["cvar"]
    )

    ratio = (
        results[b]["mean_L"] /
        results[a]["mean_L"]
    )

    report.append(
        f"{a}->{b}: "
        f"R={R:+.8f}, "
        f"mean_ratio={ratio:.8f}, "
        f"corr={corr(La,Lb):+.6f}"
    )

report.append("")
report.append("E4 IDEAL VS HARDWARE")
report.append("-" * 72)

if ideal_M is not None:
    report.append(
        f"Ideal R={ideal_R:+.8f}"
    )

    report.append(
        f"Hardware R={hardware_R:+.8f}"
    )

    report.append(
        f"Sign reversal="
        f"{np.sign(ideal_R) != np.sign(hardware_R)}"
    )

report.append("")
report.append("IMPORTANT INTERPRETATION")
report.append("-" * 72)
report.append(
    "These supplementary diagnostics do not replace the "
    "preregistered primary analysis."
)
report.append(
    "They are intended to characterize phase dependence, "
    "tail membership, finite-shot effects, and the "
    "ideal-versus-hardware discrepancy."
)

(OUT / "deep_audit.txt").write_text(
    "\n".join(report)
)

print("\n" + "=" * 72)
print("DEEP AUDIT COMPLETE")
print("=" * 72)
print(f"Outputs written to: {OUT}")
print("")
print("Files:")
for p in sorted(OUT.iterdir()):
    print(" ", p.name)

