#!/usr/bin/env python3

"""
STAGE 7 MASTER DATA EXTRACTOR
=============================

Run from stage7_code:

    python3 collect_stage7.py

This does NOT submit anything to IBM.

It collects:
    - all raw IBM JSON metadata
    - all raw 0/1 counts
    - probabilities and expectation values
    - reconstructed epsilon_x/y/z
    - epsilon norm
    - raw loss
    - shot-bias correction
    - debiased/clipped loss
    - per-phase data
    - every paired comparison
    - controls
    - job/usage metadata
    - bootstrap results
    - phi-bin results
    - tail membership
    - E0 background
    - existing deep-audit output
    - existing figures
    - preregistration/protocol files
    - SHA256 hashes of raw datasets

Output:

    stage7_complete/
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "stage7_complete"
DEEP = ROOT / "stage7_deep"

RAW_FILES = {
    "E0": "raw_hard_E0.json",
    "E1": "raw_mean_1790502945.json",
    "E2": "raw_e2_live_1790502945.json",
    "E3": "raw_cvar_E3_1790509726.json",
    "E4M": "raw_e4_mean_s03_1790511613.json",
    "E4C": "raw_e4_cvar_s03_1790512144.json",
}


# ============================================================
# Utilities
# ============================================================

def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_csv(path, rows):
    rows = list(rows)

    if not rows:
        return

    fields = []

    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def expval(counts):
    n0 = int(counts.get("0", 0))
    n1 = int(counts.get("1", 0))

    total = n0 + n1

    if total == 0:
        return np.nan

    return (n0 - n1) / total


def upper_tail_cvar(values, alpha=0.2):
    """
    Atomic-safe upper-tail CVaR.
    """

    values = np.sort(np.asarray(values, dtype=float))

    n = len(values)
    k = alpha * n

    if k <= 0:
        raise ValueError("alpha must be > 0")

    full = int(np.floor(k))
    frac = k - full

    if full == 0:
        return float(values[-1])

    total = float(np.sum(values[-full:]))

    if frac > 1e-12 and full < n:
        total += frac * float(values[-full - 1])
        denominator = k
    else:
        denominator = float(full)

    return total / denominator


def reconstruct_epsilon(record):
    """
    Uses the actual Stage 7 reconstruction implementation.
    """

    from stage7 import eps_from_counts

    counts = record["counts"]

    v0 = np.array([
        expval(counts["0|X"]),
        expval(counts["0|Y"]),
        expval(counts["0|Z"]),
    ])

    vp = np.array([
        expval(counts["+|X"]),
        expval(counts["+|Y"]),
        expval(counts["+|Z"]),
    ])

    return np.asarray(
        eps_from_counts(v0, vp),
        dtype=float,
    )


# ============================================================
# Analyse one raw dataset
# ============================================================

def analyse_dataset(label, data):

    from stage7 import W_METRIC

    shots = int(data["shots"])

    op = data.get("operating_point", {})

    alpha = float(op.get("alpha", 0.2))
    sigma = op.get("sigma", "")
    reps = int(data.get("reps", 0))

    records = data["raw"]

    eps = []
    phis = []

    for record in records:

        eps.append(
            reconstruct_epsilon(record)
        )

        phis.append(
            float(record["phi"])
        )

    eps = np.asarray(eps, dtype=float)
    phis = np.asarray(phis, dtype=float)

    # Raw quadratic loss
    L_raw = np.einsum(
        "ki,ij,kj->k",
        eps,
        W_METRIC,
        eps,
    )

    # Frozen Stage 7 shot-bias correction
    bias = (3.0 / shots) / 6.0

    L = np.maximum(
        L_raw - bias,
        0.0,
    )

    clipped = L_raw < bias

    # --------------------------------------------------------
    # Per-phase records
    # --------------------------------------------------------

    rows = []

    for i, record in enumerate(records):

        row = {
            "experiment": label,
            "index": i,
            "phi": float(record["phi"]),
            "abs_phi": abs(float(record["phi"])),

            "eps_x": float(eps[i, 0]),
            "eps_y": float(eps[i, 1]),
            "eps_z": float(eps[i, 2]),
            "eps_norm": float(np.linalg.norm(eps[i])),

            "L_raw": float(L_raw[i]),
            "shot_bias": float(bias),
            "L": float(L[i]),

            "clipped": bool(clipped[i]),

            "shots": shots,
            "reps": reps,
            "backend": data.get("backend", ""),
            "arm": data.get("arm", ""),
            "sigma": sigma,
        }

        # Every individual measurement
        for key, counts in record["counts"].items():

            n0 = int(counts.get("0", 0))
            n1 = int(counts.get("1", 0))

            total = n0 + n1

            row[f"{key}_n0"] = n0
            row[f"{key}_n1"] = n1
            row[f"{key}_shots"] = total

            row[f"{key}_p0"] = (
                n0 / max(1, total)
            )

            row[f"{key}_p1"] = (
                n1 / max(1, total)
            )

            row[f"{key}_expval"] = (
                (n0 - n1) / max(1, total)
            )

        rows.append(row)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "experiment": label,
        "arm": data.get("arm", ""),
        "backend": data.get("backend", ""),

        "shots": shots,
        "reps": reps,
        "K": len(records),

        "sigma": sigma,
        "alpha": alpha,

        "phi_seed": op.get(
            "phi_seed",
            20260926,
        ),

        "mean_L": float(np.mean(L)),
        "median_L": float(np.median(L)),
        "std_L": float(np.std(L, ddof=1)),

        "min_L": float(np.min(L)),
        "max_L": float(np.max(L)),

        "mean_L_raw": float(np.mean(L_raw)),
        "median_L_raw": float(np.median(L_raw)),

        "CVaR_0.2": float(
            upper_tail_cvar(
                L,
                alpha,
            )
        ),

        "shot_bias": float(bias),

        "clipped": int(
            np.sum(clipped)
        ),

        "clip_fraction": float(
            np.mean(clipped)
        ),

        "control": json.dumps(
            data.get("control", []),
            default=float,
        ),

        "usage": json.dumps(
            data.get("usage", []),
            default=str,
        ),
    }

    return {
        "rows": rows,
        "summary": summary,
        "E": eps,
        "L": L,
        "L_raw": L_raw,
        "phis": phis,
        "data": data,
    }


# ============================================================
# Paired analysis
# ============================================================

def paired_rows(A, B, nameA, nameB):

    rows = []

    for i in range(len(A["L"])):

        dL = (
            B["L"][i]
            -
            A["L"][i]
        )

        rows.append({
            "pair": f"{nameA}->{nameB}",
            "index": i,

            "phi": float(A["phis"][i]),
            "abs_phi": abs(float(A["phis"][i])),

            "L_A": float(A["L"][i]),
            "L_B": float(B["L"][i]),

            "delta_L_B_minus_A": float(dL),

            "epsA_x": float(A["E"][i, 0]),
            "epsA_y": float(A["E"][i, 1]),
            "epsA_z": float(A["E"][i, 2]),

            "epsB_x": float(B["E"][i, 0]),
            "epsB_y": float(B["E"][i, 1]),
            "epsB_z": float(B["E"][i, 2]),

            "delta_eps_x": float(
                B["E"][i, 0] -
                A["E"][i, 0]
            ),

            "delta_eps_y": float(
                B["E"][i, 1] -
                A["E"][i, 1]
            ),

            "delta_eps_z": float(
                B["E"][i, 2] -
                A["E"][i, 2]
            ),
        })

    return rows


# ============================================================
# Main
# ============================================================

def main():

    OUT.mkdir(exist_ok=True)

    source_dir = OUT / "source_files"
    source_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("STAGE 7 MASTER DATA EXTRACTION")
    print("=" * 72)
    print("LOCAL ONLY — NO IBM JOBS WILL BE SUBMITTED\n")

    datasets = {}
    manifest = {
        "created_utc":
            datetime.now(timezone.utc).isoformat(),

        "no_ibm_jobs_submitted":
            True,

        "datasets": {},
    }

    # ========================================================
    # Load every raw hardware dataset
    # ========================================================

    for label, filename in RAW_FILES.items():

        path = ROOT / filename

        if not path.exists():

            print(
                f"[MISSING] {label}: {filename}"
            )

            continue

        print(
            f"[LOAD] {label}: {filename}"
        )

        data = load_json(path)

        datasets[label] = data

        shutil.copy2(
            path,
            source_dir / filename,
        )

        manifest["datasets"][label] = {
            "filename": filename,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    print()

    # ========================================================
    # Analyse datasets
    # ========================================================

    analysed = {}

    all_phase_rows = []

    summaries = []

    for label, data in datasets.items():

        print(
            f"[ANALYSE] {label}"
        )

        result = analyse_dataset(
            label,
            data,
        )

        analysed[label] = result

        all_phase_rows.extend(
            result["rows"]
        )

        summaries.append(
            result["summary"]
        )

    # ========================================================
    # Summary
    # ========================================================

    write_csv(
        OUT / "summary.csv",
        summaries,
    )

    # ========================================================
    # Complete per-phase dataset
    # ========================================================

    write_csv(
        OUT / "per_phase.csv",
        all_phase_rows,
    )

    # ========================================================
    # Raw counts in long format
    # ========================================================

    count_rows = []

    for label, data in datasets.items():

        for i, record in enumerate(
            data["raw"]
        ):

            for key, counts in record[
                "counts"
            ].items():

                prep, measurement = (
                    key.split("|")
                )

                n0 = int(
                    counts.get("0", 0)
                )

                n1 = int(
                    counts.get("1", 0)
                )

                total = n0 + n1

                count_rows.append({

                    "experiment": label,

                    "index": i,

                    "phi": float(
                        record["phi"]
                    ),

                    "prep": prep,

                    "measurement":
                        measurement,

                    "n0": n0,
                    "n1": n1,

                    "shots": total,

                    "p0":
                        n0 / max(1, total),

                    "p1":
                        n1 / max(1, total),

                    "expval":
                        (n0 - n1) /
                        max(1, total),

                })

    write_csv(
        OUT / "raw_counts.csv",
        count_rows,
    )

    # ========================================================
    # EVERY useful paired comparison
    # ========================================================

    pair_definitions = [

        ("E1", "E2"),
        ("E1", "E3"),
        ("E4M", "E4C"),

        ("E0", "E1"),
        ("E0", "E2"),
        ("E0", "E3"),
        ("E0", "E4M"),
        ("E0", "E4C"),

    ]

    all_paired = []

    for A, B in pair_definitions:

        if A not in analysed:
            continue

        if B not in analysed:
            continue

        print(
            f"[PAIR] {A}->{B}"
        )

        all_paired.extend(
            paired_rows(
                analysed[A],
                analysed[B],
                A,
                B,
            )
        )

    write_csv(
        OUT / "paired_per_phase.csv",
        all_paired,
    )

    # ========================================================
    # Controls + metadata
    # ========================================================

    control_rows = []

    for label, data in datasets.items():

        control = data.get(
            "control",
            [],
        )

        row = {

            "experiment": label,

            "arm":
                data.get(
                    "arm",
                    "",
                ),

            "backend":
                data.get(
                    "backend",
                    "",
                ),

            "shots":
                data.get(
                    "shots",
                    "",
                ),

            "reps":
                data.get(
                    "reps",
                    "",
                ),

            "control_json":
                json.dumps(
                    control,
                    default=float,
                ),

            "control_norm":
                float(
                    np.linalg.norm(control)
                )
                if control
                else "",

            "usage_json":
                json.dumps(
                    data.get(
                        "usage",
                        [],
                    ),
                    default=str,
                ),
        }

        for i, value in enumerate(
            control
        ):

            row[f"c{i}"] = value

        control_rows.append(row)

    # Also capture frozen .npy controls
    for filename in (
        "stage7_c_mean.npy",
        "stage7_c_cvar.npy",
    ):

        path = ROOT / filename

        if not path.exists():
            continue

        control = np.load(path)

        control_rows.append({

            "experiment":
                "CONTROL_FILE",

            "arm":
                filename,

            "control_json":
                json.dumps(
                    control.tolist()
                ),

            "control_norm":
                float(
                    np.linalg.norm(
                        control
                    )
                ),

        })

    write_csv(
        OUT / "controls.csv",
        control_rows,
    )

    # ========================================================
    # Archive ALL raw JSON data
    # ========================================================

    with open(
        OUT / "raw_all.json",
        "w",
    ) as f:

        json.dump(
            datasets,
            f,
            indent=2,
            default=float,
        )

    # ========================================================
    # Copy forensic analysis
    # ========================================================

    forensic_files = (

        "summary.csv",
        "bootstrap.csv",
        "phi_bins.csv",
        "tail_membership.csv",
        "e0_background.csv",
        "paired_per_phase.csv",
        "per_phase.csv",
        "deep_audit.txt",

    )

    for filename in forensic_files:

        source = (
            DEEP / filename
        )

        if source.exists():

            shutil.copy2(
                source,
                OUT / (
                    "deep_" + filename
                ),
            )

    # ========================================================
    # Copy figures
    # ========================================================

    figure_dir = (
        OUT / "figures"
    )

    figure_dir.mkdir(
        exist_ok=True
    )

    if DEEP.exists():

        for figure in DEEP.glob(
            "*.png"
        ):

            shutil.copy2(
                figure,
                figure_dir /
                figure.name,
            )

    # ========================================================
    # Copy preregistration / protocol
    # ========================================================

    for pattern in (
        "*prereg*.json",
        "*PREREG*.json",
        "*protocol*.md",
        "*PROTOCOL*.md",
    ):

        for path in ROOT.glob(
            pattern
        ):

            if path.is_file():

                shutil.copy2(
                    path,
                    OUT / path.name,
                )

    # ========================================================
    # Save complete manifest
    # ========================================================

    manifest["total_datasets"] = (
        len(datasets)
    )

    manifest["total_phase_records"] = (
        sum(
            len(x["data"]["raw"])
            for x in analysed.values()
        )
    )

    manifest["total_count_rows"] = (
        len(count_rows)
    )

    manifest["total_paired_rows"] = (
        len(all_paired)
    )

    with open(
        OUT / "manifest.json",
        "w",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # ========================================================
    # Human-readable master report
    # ========================================================

    report = (
        OUT /
        "MASTER_REPORT.txt"
    )

    with open(
        report,
        "w",
    ) as f:

        f.write(
            "STAGE 7 MASTER DATA REPORT\n"
        )

        f.write(
            "=" * 72 +
            "\n\n"
        )

        f.write(
            "LOCAL ANALYSIS ONLY\n"
        )

        f.write(
            "NO IBM JOBS SUBMITTED\n\n"
        )

        f.write(
            f"Created UTC: "
            f"{manifest['created_utc']}\n\n"
        )

        f.write(
            "DATASETS\n"
        )

        f.write(
            "-" * 72 +
            "\n"
        )

        for label in datasets:

            f.write(
                f"{label}\n"
            )

        f.write(
            "\nSUMMARY\n"
        )

        f.write(
            "-" * 72 +
            "\n"
        )

        for summary in summaries:

            f.write(

                f"{summary['experiment']:4s} | "

                f"backend="
                f"{summary['backend']} | "

                f"shots="
                f"{summary['shots']} | "

                f"reps="
                f"{summary['reps']} | "

                f"K="
                f"{summary['K']} | "

                f"mean_L="
                f"{summary['mean_L']:.12f} | "

                f"CVaR="
                f"{summary['CVaR_0.2']:.12f} | "

                f"bias="
                f"{summary['shot_bias']:.12f} | "

                f"clipped="
                f"{summary['clipped']}/"
                f"{summary['K']}\n"

            )

        f.write(
            "\nRAW FILE HASHES\n"
        )

        f.write(
            "-" * 72 +
            "\n"
        )

        for label, info in (
            manifest["datasets"]
            .items()
        ):

            f.write(
                f"{label}: "
                f"{info['filename']}\n"
            )

            f.write(
                f"SHA256: "
                f"{info['sha256']}\n\n"
            )

    # ========================================================
    # Final console
    # ========================================================

    print()
    print("=" * 72)
    print("STAGE 7 MASTER EXTRACTION COMPLETE")
    print("=" * 72)

    print(
        f"Datasets: "
        f"{', '.join(datasets)}"
    )

    print(
        f"Phase records: "
        f"{manifest['total_phase_records']}"
    )

    print(
        f"Raw count rows: "
        f"{manifest['total_count_rows']}"
    )

    print(
        f"Paired rows: "
        f"{manifest['total_paired_rows']}"
    )

    print()
    print(
        f"Everything is in:"
    )

    print(
        f"    {OUT}"
    )

    print()
    print(
        "IMPORTANT: this script does not submit IBM jobs."
    )


if __name__ == "__main__":
    main()
