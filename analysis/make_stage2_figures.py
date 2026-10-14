"""
make_stage2_figures.py -- figures, table and statistics for VIPER Stage 2.

Run from the folder that contains results_stage2/ (after `stage2.py summary`):
    python make_stage2_figures.py [results_stage2]

Reads only saved files (no simulation):
    results_stage2/stage2_summary.json          per-point statistics (stage2.py)
    results_stage2/*/*_pool.json / _pool.npz    candidate scalars + quantile tables
    results_stage2/validate.json                full-Lindblad R_tail vs best mean
    results_stage2/*/pt_*.json                  noise widths and test seeds
    results_stage2/stage1_error_clouds.npz      error rotation vectors (kappa)

Writes into figures/:
    fig_stage2_tests.pdf/png    rank agreement, reversals, dominance, pool regret vs sigma_phi^2
    fig_stage2_shape.pdf/png    CVaR/mean ratio and normalized loss distributions
    fig_stage2_kappa.pdf/png    reversal fraction against kappa
    stage2_stats.json           every number quoted in the paper
    stage2_table.tex            per-point table for the paper

The dominance test here (T3') replaces the one printed by stage2.py: pairs
must differ by >= 10% in mean loss, and crossings are counted only in the
noise-dominated upper half of the distribution (q >= 0.5).  The reason is in
the stats file: at small noise every crossing of the full distribution lies at
q < 0.5 and is caused by the controls' residual coherent (noise-free) error,
which Theorem 1 assumes to be zero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2, norm

TWO_PI = 2 * np.pi
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 1.4,
})
# Okabe-Ito hues (CVD-safe); every series also has its own marker and a legend entry
GROUPS = [  # key, label, colour, marker, linestyle
    ((2, 8.0), "2 levels, 8 ns", "#009E73", "^", "-"),
    ((2, 12.7), "2 levels, 12.7 ns", "#0072B2", "o", "-"),
    ((2, 20.0), "2 levels, 20 ns", "#D55E00", "s", "-"),
    ((4, 12.7), "4 levels, 12.7 ns", "#CC79A7", "D", "--"),
    ((4, 20.8), "4 levels, 20.8 ns", "#E69F00", "v", "--"),
]
Q_TAIL_LO, SEP_MEAN, REL = 0.5, 0.10, 0.02


def dominance(Q, m, q, qlo=Q_TAIL_LO, sep=SEP_MEAN, cap=300):
    """Fraction of mean-separated competitive pairs whose quantile curves cross."""
    comp = np.flatnonzero(m <= 2 * m.min())
    if len(comp) > cap:
        comp = comp[np.argsort(m[comp])][np.round(np.linspace(0, len(comp) - 1, cap)).astype(int)]
    iu = np.triu_indices(len(comp), 1)
    mm = m[comp]
    keep = (np.abs(mm[:, None] - mm[None]) > sep * (mm[:, None] + mm[None]) / 2)[iu]
    mask = (q >= qlo) & (q <= 0.995)
    Qs = Q[comp][:, mask]
    D = (Qs[:, None] - Qs[None])[iu]
    S = (0.5 * (Qs[:, None] + Qs[None]))[iu]
    pos, neg = (D > REL * S).any(1), (D < -REL * S).any(1)
    v = pos & neg
    return (float(v[keep].mean()) if keep.any() else float("nan")), int(keep.sum())


def crossing_diagnostics(Q, m, q, L0):
    """For the full-distribution test: where do crossings sit, and does the
    better (lower-mean) control carry the larger coherent error?"""
    comp = np.flatnonzero(m <= 2 * m.min())
    iu = np.triu_indices(len(comp), 1)
    mm, L0c = m[comp], L0[comp]
    keep = (np.abs(mm[:, None] - mm[None]) > SEP_MEAN * (mm[:, None] + mm[None]) / 2)[iu]
    mask = (q >= 0.05) & (q <= 0.995)
    Qs, qq = Q[comp][:, mask], q[mask]
    D = (Qs[:, None] - Qs[None])[iu]
    S = (0.5 * (Qs[:, None] + Qs[None]))[iu]
    sg = np.sign((mm[:, None] - mm[None])[iu])[:, None]
    rev = D * sg < -REL * S                      # levels where the higher-mean control is better
    v = rev.any(1) & keep
    if not v.any():
        return {"n": 0}
    maxq = np.array([qq[r].max() for r in rev[v]])
    dL0 = (L0c[:, None] - L0c[None])[iu]
    return {"n": int(v.sum()), "frac": float(v.sum() / keep.sum()),
            "max_q_median": float(np.median(maxq)), "max_q_max": float(maxq.max()),
            "better_has_larger_L0": float(np.mean(np.sign(dL0[v]) != sg[v, 0]))}


def kappa(eps, z):
    X = np.column_stack([np.ones_like(z), z, z * z])
    B, *_ = np.linalg.lstsq(X, eps, rcond=None)
    return float(np.linalg.norm(B[2]) / np.linalg.norm(B[1]))


def load(root: Path):
    summ = {r["file"]: r for r in json.loads((root / "stage2_summary.json").read_text())["rows"]}
    val = {}
    for x in json.loads((root / "validate.json").read_text())["rows"]:
        val.setdefault(x["file"], {})[x["alpha"]] = x
    clouds = np.load(root / "stage1_error_clouds.npz")
    pts = []
    for f, r in summ.items():
        sub, stem = f.split("/"); stem = stem[:-5]
        pj = json.loads((root / sub / f"{stem}.json").read_text())
        z = np.load(root / sub / f"{stem}_pool.npz")
        dom, npair = dominance(z["quantiles"].astype(float), z["means"], z["q_levels"])
        dom_full, _ = dominance(z["quantiles"].astype(float), z["means"], z["q_levels"], qlo=0.05)
        diag = crossing_diagnostics(z["quantiles"].astype(float), z["means"], z["q_levels"], z["L0"])
        te = next(e for e in pj["ensembles"] if e["name"] == "test")
        d = np.random.default_rng(te["seed"]).normal(0.0, pj["sigma_d"], te["n"])
        ck = f"{root.name}__{sub}__{stem}__mean"
        kap = kappa(clouds[ck].astype(float), d / pj["sigma_d"]) if ck in clouds.files else float("nan")
        v5, v2 = val.get(f, {}).get(0.05, {}), val.get(f, {}).get(0.2, {})
        pts.append({**r, "levels": r["n_levels"], "kappa": kap, "dom_upper": dom, "dom_pairs": npair,
                    "dom_full_sep": dom_full, "crossing": diag,
                    "R_L05": v5.get("R_tail_vs_best_mean_lindblad"),
                    "R_L05_CI": [v5.get("R_lo_vs_best_mean"), v5.get("R_hi_vs_best_mean")],
                    "R_L20": v2.get("R_tail_vs_best_mean_lindblad"),
                    "same_budget_R_L05": v5.get("R_tail_lindblad"),
                    "upgraded05": v5.get("best_mean_source", "mean") != "mean"})
    return pts


def _series(pts, key, fn):
    g = sorted([p for p in pts if (p["levels"], p["Tg"]) == key], key=lambda p: p["sigma_phi_sq"])
    s0 = [p for p in g if p["seed"] == 0]
    rep = [p for p in g if p["seed"] != 0]
    return ([p["sigma_phi_sq"] for p in s0], [fn(p) for p in s0],
            [p["sigma_phi_sq"] for p in rep], [fn(p) for p in rep])


def fig_tests(pts, out):
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0))
    panels = [
        (axes[0, 0], lambda p: p["kendall_tau_competitive@0.05"], r"Kendall $\tau$ (mean vs CVaR$_{0.05}$)",
         "(a) rank agreement", 1.0),
        (axes[0, 1], lambda p: p["reversal_frac@0.05"], "fraction of pairs reversed",
         "(b) confirmed rank reversals", 0.0),
        (axes[1, 0], lambda p: p["dom_upper"], "fraction of pairs crossing",
         r"(c) dominance violations ($q\geq0.5$)", 0.0),
    ]
    for ax, fn, ylab, title, theory in panels:
        for key, lab, col, mk, ls in GROUPS:
            x, y, xr, yr = _series(pts, key, fn)
            ax.plot(x, y, ls=ls, marker=mk, color=col, ms=5, label=lab)
            if xr:
                ax.plot(xr, yr, ls="none", marker=mk, mfc="none", mec=col, ms=5)
        ax.axhline(theory, color="#555", lw=0.8, ls=":")
        ax.axvspan(1e-3, 0.05, color="#eeeeee", zorder=0)
        ax.set_xscale("log"); ax.set_xlim(1.1e-3, 2.3)
        ax.set_xlabel(r"accumulated phase variance $\sigma_\phi^2$"); ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=9)
    axes[0, 0].text(1.3e-3, 0.02, "Theorem 1 region", fontsize=7, color="#555")
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 1].legend(loc="upper left", fontsize=6.5)
    ax = axes[1, 1]
    for key, lab, col, mk, ls in GROUPS:
        g = [p for p in pts if (p["levels"], p["Tg"]) == key and p["R_L05"] is not None]
        for p in g:
            ax.plot(p["R_L05"], p["R_pool_crossfit@0.05"], marker=mk, color=col, ms=5,
                    mfc=col if p["seed"] == 0 else "none", ls="none")
    lim = [-0.06, 0.4]
    ax.plot(lim, lim, color="#555", lw=0.8, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$R_{\rm tail}(0.05)$, full Lindblad, vs best mean")
    ax.set_ylabel(r"pool regret $R_{\rm pool}(0.05)$")
    ax.set_title("(d) pool regret tracks the tail gain", loc="left", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_stage2_tests.{ext}")
    plt.close(fig)


def fig_shape(pts, root, out):
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.8))
    ax = axes[0]
    for key, lab, col, mk, ls in GROUPS:
        g = sorted([p for p in pts if (p["levels"], p["Tg"]) == key and p["seed"] == 0],
                   key=lambda p: p["sigma_phi_sq"])
        x = [p["sigma_phi_sq"] for p in g]
        y = np.array([p["ratio_median@0.05"] for p in g])
        lo = y - np.array([p["ratio_q25@0.05"] for p in g])
        hi = np.array([p["ratio_q75@0.05"] for p in g]) - y
        ax.errorbar(x, y, yerr=[lo, hi], ls=ls, marker=mk, color=col, ms=4, capsize=2, label=lab)
    ax.axhline(5.582, color="#555", lw=0.8, ls=":")
    ax.text(1.2e-3, 6.2, "Proposition 1: 5.58", fontsize=7, color="#555")
    ax.set_xscale("log"); ax.set_ylim(0, 12)
    ax.set_xlabel(r"$\sigma_\phi^2$"); ax.set_ylabel(r"CVaR$_{0.05}$ / mean (median, IQR)")
    ax.set_title("(a) shape of the loss distribution", loc="left", fontsize=9)
    ax.legend(fontsize=6, loc="upper left", ncol=1)
    ax = axes[1]
    xs = np.geomspace(1e-3, 30, 300)
    for pv, col, lab in ((0.0016, "#0072B2", r"$\sigma_\phi^2=0.0016$"), (1.6, "#D55E00", r"$\sigma_\phi^2=1.6$")):
        z = np.load(root / "stage2_2level" / f"pt_Tg12.7_pv{pv:g}_s0_pool.npz")
        Q, m, q = z["quantiles"].astype(float), z["means"], z["q_levels"]
        comp = np.flatnonzero(m <= 2 * m.min())
        sel = comp[np.round(np.linspace(0, len(comp) - 1, min(40, len(comp)))).astype(int)]
        for j, i in enumerate(sel):
            ax.plot(Q[i] / m[i], 1 - q, color=col, lw=0.9 if pv < 0.01 else 0.6,
                    alpha=0.8 if pv < 0.01 else 0.5, label=lab if j == 0 else None)
    ax.plot(xs, chi2.sf(xs, 1), color="#222", lw=1.0, ls="--", label=r"$\chi^2_1$ (Theorem 1 shape)")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-2, 30); ax.set_ylim(1e-3, 1)
    ax.set_xlabel("loss / mean loss"); ax.set_ylabel("fraction of draws above")
    ax.set_title("(b) normalized loss distributions, 12.7 ns", loc="left", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_stage2_shape.{ext}")
    plt.close(fig)


def fig_kappa(pts, out):
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for key, lab, col, mk, ls in GROUPS:
        g = [p for p in pts if (p["levels"], p["Tg"]) == key]
        ax.plot([p["kappa"] for p in g], [p["reversal_frac@0.05"] for p in g], ls="none", marker=mk,
                color=col, ms=5, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\kappa$ of the mean control"); ax.set_ylabel("fraction of pairs reversed")
    ax.axvspan(1e-3, 0.1, color="#eeeeee", zorder=0)
    ax.set_xlim(left=min(p["kappa"] for p in pts) * 0.7)
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_stage2_kappa.{ext}")
    plt.close(fig)


def stats_and_table(pts, out):
    two = [p for p in pts if p["levels"] == 2]
    small = [p for p in two if p["sigma_phi_sq"] <= 0.0064]
    lin = [p for p in two if p["sigma_phi_sq"] <= 0.04]
    ok = [p for p in pts if p["R_L05"] is not None]
    rho_k = spearmanr([p["kappa"] for p in pts], [p["reversal_frac@0.05"] for p in pts])
    rho_pool = spearmanr([p["R_L05"] for p in ok], [p["R_pool_crossfit@0.05"] for p in ok])
    st = {
        "n_points": len(pts),
        "two_level_small_noise": {
            "n_points": len(small),
            "reversals_total": int(sum(round(p["reversal_frac@0.05"] * p["n_pairs_separated@0.05"]) for p in small)),
            "pairs_total": int(sum(p["n_pairs_separated@0.05"] for p in small)),
            "tau_range": [min(p["kendall_tau_competitive@0.05"] for p in small),
                          max(p["kendall_tau_competitive@0.05"] for p in small)],
            "ratio05_range": [min(p["ratio_median@0.05"] for p in small), max(p["ratio_median@0.05"] for p in small)],
            "ratio20_range": [min(p["ratio_median@0.2"] for p in small), max(p["ratio_median@0.2"] for p in small)],
            "R_pool05_range": [min(p["R_pool_crossfit@0.05"] for p in small), max(p["R_pool_crossfit@0.05"] for p in small)]},
        "two_level_linear_region_dom_upper_max": max(p["dom_upper"] for p in lin),
        "two_level_linear_region_reversal_max": max(p["reversal_frac@0.05"] for p in lin),
        "crossing_diagnostics_small_noise": {p["file"]: p["crossing"] for p in lin},
        "spearman_kappa_vs_reversals": [float(rho_k.statistic), float(rho_k.pvalue)],
        "spearman_Rpool_vs_RL05": [float(rho_pool.statistic), float(rho_pool.pvalue), len(ok)],
        "prop1": {"c_0.05": 5.582, "c_0.2": 3.249},
        "points": [{k: p[k] for k in ("file", "levels", "Tg", "seed", "sigma_phi_sq", "kappa",
                                      "kendall_tau_competitive@0.05", "reversal_frac@0.05",
                                      "n_pairs_separated@0.05", "dom_upper", "dom_full_sep",
                                      "ratio_median@0.05", "ratio_median@0.2", "R_pool_crossfit@0.05",
                                      "R_L05", "R_L05_CI", "R_L20", "upgraded05", "nominal_share_median")}
                   for p in pts],
    }
    (out / "stage2_stats.json").write_text(json.dumps(st, indent=1, default=float))
    rows = []
    for key, lab, *_ in GROUPS:
        for p in sorted([p for p in pts if (p["levels"], p["Tg"]) == key],
                        key=lambda p: (p["sigma_phi_sq"], p["seed"])):
            ci = p["R_L05_CI"]
            rl = (f"${p['R_L05']:+.3f}$" + ("$^{\\circ}$" if p["upgraded05"] else "")) if p["R_L05"] is not None else "--"
            rows.append(f"{p['levels']} & {p['Tg']:g} & {p['seed']} & {p['sigma_phi_sq']:g} & "
                        f"{p['kappa']:.2f} & {p['kendall_tau_competitive@0.05']:.2f} & "
                        f"{p['reversal_frac@0.05']:.3f} & {p['dom_upper']:.3f} & {p['ratio_median@0.05']:.2f} & "
                        f"${p['R_pool_crossfit@0.05']:+.3f}$ & {rl} \\\\")
    (out / "stage2_table.tex").write_text("\n".join(rows) + "\n")
    return st


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results_stage2")
    out = Path("figures"); out.mkdir(exist_ok=True)
    pts = load(root)
    fig_tests(pts, out); fig_shape(pts, root, out); fig_kappa(pts, out)
    st = stats_and_table(pts, out)
    print(json.dumps({k: v for k, v in st.items() if k not in ("points", "crossing_diagnostics_small_noise")},
                     indent=1, default=float))


if __name__ == "__main__":
    main()