"""
make_stage6_figures.py -- figures and re-analysis for Stage 6 (the measurement-only
predictor, its calibration, and the leverage mechanism).

Reads   results_stage6/stage6_summary.json
        results_stage6/stage6_2level/s6_*.json          (predictor + raw ground truth)
        results_stage6/polish_2level/polish_s6_*.json    (shared-basin-pool ground truth)
        results_stage6/retro_results_stage5/retro_s5_*.json   (the T1 mixture arm)
        results_stage6/stage6_rtn/s6rtn_*.json           (the fluctuator arm)
Writes  figures/fig_stage6_calibration.pdf / .png
        figures/fig_stage6_leverage.pdf   / .png
        figures/tab_stage6.tex
        figures/stage6_stats.json

Four things in here differ from `stage6.py summary`, each because the run exposed a
mis-specification, and each is argued in the paper rather than silently applied:

 1. R is taken from the SHARED BASIN POOL (`polish`), not from the main run's
    best-of-{mean, rev_*} comparator.  In 10 of 16 cells a reverse run -- a mean
    optimization warm started from a CVaR solution -- found a strictly better MEAN
    optimum than the mean multistart did, by a factor of four in one cell.  Scoring
    R against that is not a fairness correction; it gives the mean arm five restarts,
    four of them with inside information, against the CVaR arm's one.  Pooling every
    control found at a point and running both objectives from every start cut the
    replicate scatter in R by 10-27x and moved the slope from -1.15 to +0.56.

 2. H3 as pre-registered asked for slope in [0.7, 1.3].  The measured slope is
    0.588 +- 0.025 across four alpha and 0.589 +- 0.033 across eight (alpha, seed)
    cells, with Spearman 0.85-0.97.  That is a near-perfect ORDERING with a constant
    multiplicative offset, not a unit-level prediction.  We report it as a
    calibration constant and give the mechanism: the optimizer travels a median 0.50
    of the predicted step while the geometry co-moves.

 3. H2's min-cosine criterion is replaced by significant anti-alignment.  cos = 0.53
    in three dimensions has null p = 0.47 -- uninformative, not contradictory.  The
    gate keeps points where a local displacement model can apply at all: resolved,
    the CVaR arm did not find a better mean optimum (it jumped basins rather than
    trading), and |R_raw| > 0.005 (there is a displacement to have a direction).

 4. H5's Lambda is NOT a universal suppression factor.  It is exact for a
    control-INERT rare channel -- the T1 jump branch, where R_mixture/R_bulk = Lambda
    to 0.65% over 40 cell-alpha pairs -- and fails by two orders of magnitude for a
    fluctuator that loads the same coordinate as the bulk.  Sweeping the fluctuator's
    RARITY over 25x leaves the measured gain flat to 4%.  What governs is the
    fraction of tail loss the control can reduce; Lambda is its inert limit.

USAGE   python make_stage6_figures.py [results_root]
"""
from __future__ import annotations
import json, sys, glob, math
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT = Path("figures"); OUT.mkdir(exist_ok=True)
C_MEAN, C_CV05, C_CV20, C_GREY = "#0072B2", "#D55E00", "#E69F00", "#555555"
C_GRN, C_PUR = "#009E73", "#CC79A7"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.3,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5})

ALPHAS = ["0.2", "0.1", "0.05", "0.025"]
ACOL = {"0.2": C_MEAN, "0.1": C_GRN, "0.05": C_CV05, "0.025": C_PUR}
AMARK = {"0.2": "o", "0.1": "s", "0.05": "^", "0.025": "D"}
KAPPA_MAX_PERTURBATIVE = 1.0     # the two beta-outlier Stage 5 cells sit at kappa = 7.4


# ------------------------------------------------------------------ loading
def _load(root, pattern):
    out = []
    for p in sorted(Path(root).rglob(pattern)):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


def load_all(root):
    d = {}
    d["main"] = [r for r in _load(root, "s6_Tg*.json") if "predictor" in r and "evals" in r]
    d["polish"] = {r.get("stem"): r for r in _load(root, "polish_s6_*.json")}
    d["t1"] = [r for r in _load(root, "retro_s5_*.json") if "predictor_T1" in r]
    # The fluctuator arm, with the pooled rerun preferred WHOLE where it exists.
    # `polish_s6_*.json` does not match `polish_s6rtn_*.json` -- the glob needs
    # the underscore after s6 -- so the two never mix, and a polished record
    # carries its own ground_truth, predictor and suppression recomputed from
    # the same control.  Substitution is therefore by record, never by field:
    # pairing a pooled R with the source file's Lambda would silently fabricate
    # or destroy the flatness the alignment claim rests on.
    _rtn_src = {r["stem"]: r for r in _load(root, "s6rtn_*.json") if "suppression" in r}
    _rtn_pol = {r["stem"]: r for r in _load(root, "polish_s6rtn_*.json")
                if "suppression" in r and "ground_truth" in r}
    d["rtn"] = []
    for stem, src in sorted(_rtn_src.items()):
        pol = _rtn_pol.get(stem)
        if pol is None:
            src = dict(src); src["_R_source"] = "single multistart (unpooled)"
        else:
            src = dict(pol)
            src.setdefault("rtn_amp_over_Tg", _rtn_src[stem].get("rtn_amp_over_Tg"))
            src.setdefault("rtn_p_requested", _rtn_src[stem].get("rtn_p_requested"))
            src.setdefault("fit_R2_min", _rtn_src[stem].get("fit_R2_min"))
            src["_R_source"] = "shared basin pool"
        d["rtn"].append(src)
    d["n_rtn_polished"] = len(_rtn_pol)
    d["rtn_all_polished"] = bool(_rtn_src) and len(_rtn_pol) >= len(_rtn_src)
    # rglob can find a stale copy at the tree root as well as the real one under
    # results_stage6/; take the newest and say which, because silently reading
    # the wrong summary inverts every panel that reads the boundary grid.
    s = sorted(Path(root).rglob("stage6_summary.json"),
               key=lambda q: q.stat().st_mtime, reverse=True)
    d["summary"] = json.loads(s[0].read_text()) if s else {}
    d["summary_path"] = str(s[0]) if s else None
    if len(s) > 1:
        print(f"NOTE {len(s)} stage6_summary.json found; using the newest: {s[0]}")
    d["main"].sort(key=lambda r: (r["kappa"], r["seed"]))
    return d


def R_of(rec, pol, k):
    """Measured R at alpha=k: shared basin pool when available, else the main run."""
    q = (pol.get(rec.get("stem")) or {}).get("R_pool", {}).get(k)
    if q is not None:
        return float(q["R"]), "pool"
    return float(rec["ground_truth"][k]["R"]), "main"


def through_origin(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float(np.sum(x * y) / np.sum(x ** 2)) if np.sum(x ** 2) > 0 else float("nan")


# ------------------------------------------------------------------ stats
def calibration_stats(d):
    out, src = {}, set()
    for k in ALPHAS:
        rows = [r for r in d["main"] if k in r["predictor"] and k in r.get("ground_truth", {})]
        if not rows:
            continue
        rp = np.array([r["predictor"][k]["R_pred"] for r in rows])
        R = np.empty(len(rows))
        for i, r in enumerate(rows):
            R[i], s = R_of(r, d["polish"], k); src.add(s)
        sl = through_origin(rp, R)
        sp = stats.spearmanr(rp, R)
        # replicate scatter: the two seeds at each phase_var are the replicate pair
        cells = {}
        for r, v in zip(rows, R):
            cells.setdefault(round(r["phase_var"], 8), []).append(v)
        dif = [abs(v[0] - v[1]) for v in cells.values() if len(v) == 2]
        sd_rep = float(np.mean(dif) / math.sqrt(2.0)) if dif else float("nan")
        sd_tot = float(np.std(R, ddof=1))
        ceil = math.sqrt(max(0.0, 1.0 - (sd_rep / sd_tot) ** 2)) if sd_tot > 0 else float("nan")
        per_seed = {}
        for sd in sorted({r["seed"] for r in rows}):
            m = np.array([r["seed"] == sd for r in rows])
            if m.sum() >= 3:
                per_seed[str(sd)] = {"n": int(m.sum()), "slope": through_origin(rp[m], R[m]),
                                     "spearman": list(stats.spearmanr(rp[m], R[m]))}
        out[k] = {"n": len(rows), "slope": sl,
                  "spearman": [float(sp.statistic), float(sp.pvalue)],
                  "replicate_sd_R": sd_rep, "sd_R": sd_tot, "attenuation_ceiling": ceil,
                  "spearman_over_ceiling": float(abs(sp.statistic) / ceil) if ceil > 0 else None,
                  "by_seed": per_seed,
                  "median_cos": float(np.median([r["ground_truth"][k]["cos"] for r in rows])),
                  "median_step_fraction": float(np.median(
                      [r["ground_truth"][k]["de0_norm"]
                       / max(np.linalg.norm(r["predictor"][k]["step"]), 1e-300) for r in rows]))}
    sl = [out[k]["slope"] for k in out]
    per = [v["slope"] for k in out for v in out[k]["by_seed"].values()]
    out["_pooled"] = {"R_source": sorted(src),
                      "slope_mean": float(np.mean(sl)), "slope_sd": float(np.std(sl, ddof=1)),
                      "slope_by_alpha_seed_mean": float(np.mean(per)),
                      "slope_by_alpha_seed_sd": float(np.std(per, ddof=1)),
                      "n_alpha_seed_cells": len(per)}
    return out


def t1_stats(d):
    rows, rec = [], []
    for r in d["t1"]:
        if r["kappa"] > KAPPA_MAX_PERTURBATIVE:
            continue
        for k in ALPHAS:
            pb, pm = r["predictor"].get(k), r["predictor_T1"].get(k)
            if not pb or not pm or pb["R_pred"] <= 0:
                continue
            lam, ratio = pm["Lambda"], pm["R_pred"] / pb["R_pred"]
            rho = pb["cvar"] / r["mean_loss"]
            an = float(k) * rho / (r["mixture"]["beta"] + float(k) * rho)
            rows.append({"stem": r["stem"], "alpha": float(k), "beta": r["mixture"]["beta"],
                         "Lambda": lam, "ratio": ratio, "Lambda_analytic": an,
                         "rho_bulk": rho, "R_bulk": pb["R_pred"], "R_mix": pm["R_pred"]})
            rec.append(r.get("recorded_R"))
    L = np.array([x["Lambda"] for x in rows]); Rt = np.array([x["ratio"] for x in rows])
    A = np.array([x["Lambda_analytic"] for x in rows])
    return {"n_pairs": len(rows),
            "n_cells": len({x["stem"] for x in rows}),
            "ratio_vs_Lambda_median_rel_err": float(np.median(np.abs(Rt / L - 1.0))),
            "ratio_vs_Lambda_max_rel_err": float(np.max(np.abs(Rt / L - 1.0))),
            "Lambda_vs_analytic_median_rel_err": float(np.median(np.abs(L / A - 1.0))),
            "Lambda_vs_analytic_max_rel_err": float(np.max(np.abs(L / A - 1.0))),
            "Lambda_range": [float(L.min()), float(L.max())],
            "beta_range": [float(min(x["beta"] for x in rows)),
                           float(max(x["beta"] for x in rows))],
            "excluded_kappa_gt_1": sorted({r["stem"] for r in d["t1"]
                                           if r["kappa"] > KAPPA_MAX_PERTURBATIVE}),
            "rows": rows}


def rtn_stats(d, k="0.05"):
    rar, amp = [], []
    for r in d["rtn"]:
        su = r["suppression"].get(k)
        if not su:
            continue
        row = {"stem": r["stem"], "amp": r["rtn_amp_over_Tg"],
               "p": r.get("rtn_p_requested"), "p_active": r.get("p_active"),
               "p_over_alpha": su.get("p_over_alpha"),
               "Lambda": su["Lambda"], "ratio": su["ratio"],
               # the cosine was stored through direction_cosine, which negates its
               # first argument for H2's convention; both arguments here are steps.
               "cos": (None if su.get("conflict_cos") is None
                       else abs(float(su["conflict_cos"]))),
               "R_bulk": su["R_pred_bulk_only"], "R_mix": su["R_pred_mixture"],
               "R_meas": r["ground_truth"][k]["R"],
               "R_raw": r["ground_truth"][k]["R_raw"],
               "fit_R2_min": r.get("fit_R2_min"), "kappa": r.get("kappa")}
        (rar if row["p"] is not None else amp).append(row)
    rar.sort(key=lambda x: x["p"]); amp.sort(key=lambda x: x["amp"])
    # the flatness claim: rarity points inside the mixture regime (p/alpha < 1)
    inreg = [x for x in rar if (x["p_over_alpha"] or 9) < 1.0]
    rm = np.array([x["R_meas"] for x in inreg]) if inreg else np.array([])
    lam = np.array([x["Lambda"] for x in inreg]) if inreg else np.array([])
    allp = rar + amp
    rp = np.array([x["R_mix"] for x in allp]); rq = np.array([x["R_meas"] for x in allp])
    return {"rarity": rar, "amplitude": amp,
            "n_in_regime": len(inreg),
            "p_span": ([float(min(x["p"] for x in inreg)), float(max(x["p"] for x in inreg))]
                       if inreg else None),
            "p_span_factor": (float(max(x["p"] for x in inreg) / min(x["p"] for x in inreg))
                              if inreg else None),
            "R_meas_mean": float(rm.mean()) if rm.size else None,
            "R_meas_spread_frac": (float((rm.max() - rm.min()) / abs(rm.mean()))
                                   if rm.size and rm.mean() != 0 else None),
            "Lambda_span_factor": (float(lam.max() / lam.min())
                                   if lam.size and lam.min() > 0 else None),
            "R_mix_span_factor": (float(max(x["R_mix"] for x in inreg)
                                        / min(x["R_mix"] for x in inreg)) if inreg else None),
            "R_meas_over_R_bulk": [float(x["R_meas"] / x["R_bulk"]) for x in inreg],
            "slope_mixture_vs_truth": through_origin(rp, rq),
            "spearman_mixture_vs_truth": list(stats.spearmanr(rp, rq)) if len(allp) > 2 else None,
            "R_source": sorted({x.get("_R_source", "unknown")
                                for x in d["rtn"]}) if d["rtn"] else [],
            "n_polished": d.get("n_rtn_polished", 0),
            "caveat": (
                "the RTN arm's ground truth is the SHARED BASIN POOL, on the same "
                "protocol as the main grid; predictor and Lambda are recomputed from "
                "the polished mean control, so R and Lambda come from one control."
                if d.get("rtn_all_polished") else
                "the RTN arm's ground truth is UNPOLISHED -- one multistart, one "
                "seed, a two-element comparator set -- which is the state the main "
                "grid was in when its slope read -1.15.  The finding rests on the "
                "flatness of R across the rarity sweep, not on any single value.")}


# ------------------------------------------------------------------ figure 1
def fig_calibration(d, cal, path):
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))

    a = ax[0]
    lo, hi = 1e-3, 1.0
    a.plot([lo, hi], [lo, hi], color=C_GREY, lw=0.9, ls="--", zorder=1)
    sbar = cal["_pooled"]["slope_mean"]
    a.plot([lo, hi], [sbar * lo, sbar * hi], color="#222222", lw=1.1, zorder=2)
    ndrop = 0
    for k in ALPHAS:
        rows = [r for r in d["main"] if k in r["predictor"]]
        xy = [(r["predictor"][k]["R_pred"], R_of(r, d["polish"], k)[0]) for r in rows]
        ndrop += sum(1 for _, v in xy if v <= 0)
        xy = [(u, v) for u, v in xy if v > 0]          # log axes cannot show R <= 0
        a.plot([u for u, _ in xy], [v for _, v in xy], AMARK[k], ms=3.4, mfc="none",
               mew=0.9, color=ACOL[k], ls="none", label=rf"$\alpha={k}$", zorder=3)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(3e-3, 0.9); a.set_ylim(1e-3, 0.9)
    a.set_xlabel(r"$R_{\rm pred}$  (measurement only)")
    a.set_ylabel(r"$R$  (measured, shared basin pool)")
    a.text(0.04, 0.95, rf"slope $={sbar:.3f}$", transform=a.transAxes, fontsize=7,
           va="top", color="#222222")
    a.text(0.04, 0.86, "dashed: identity", transform=a.transAxes, fontsize=6.3,
           va="top", color=C_GREY)
    if ndrop:
        a.text(0.04, 0.78, f"{ndrop} points with $R \\leq 0$ omitted",
               transform=a.transAxes, fontsize=6.3, va="top", color=C_GREY)
    a.legend(loc="lower right", handletextpad=0.4, ncol=2, fontsize=6.5,
             columnspacing=0.8)
    a.set_title("(a) level", fontsize=8, loc="left")

    a = ax[1]
    xs = [float(k) for k in ALPHAS if k in cal]
    sl = [cal[k]["slope"] for k in ALPHAS if k in cal]
    a.axhspan(cal["_pooled"]["slope_mean"] - cal["_pooled"]["slope_sd"],
              cal["_pooled"]["slope_mean"] + cal["_pooled"]["slope_sd"],
              color=C_GREY, alpha=0.15, lw=0)
    a.axhline(cal["_pooled"]["slope_mean"], color=C_GREY, lw=0.8)
    for k in ALPHAS:
        if k not in cal:
            continue
        for sd, v in cal[k]["by_seed"].items():
            a.plot([float(k)], [v["slope"]], "_", ms=9, mew=1.1,
                   color=ACOL[k], alpha=0.75, zorder=2)
    a.plot(xs, sl, "o-", ms=4, color="#222222", zorder=3)
    a.set_xscale("log"); a.set_xlabel(r"$\alpha$"); a.set_ylabel("through-origin slope")
    a.set_xticks([float(k) for k in ALPHAS])
    a.set_xticklabels([k for k in ALPHAS])
    a.minorticks_off()
    a.set_ylim(0.0, 1.05)
    a.axhline(1.0, color=C_GREY, lw=0.7, ls=":")
    a.text(0.04, 0.94, rf"$\bar s = {cal['_pooled']['slope_mean']:.3f}"
                       rf" \pm {cal['_pooled']['slope_sd']:.3f}$",
           transform=a.transAxes, fontsize=7, va="top", color="#222222")
    a.text(0.04, 0.06, "ticks: single seeds", transform=a.transAxes, fontsize=6.3,
           color=C_GREY)
    a.set_title("(b) flat in " + r"$\alpha$", fontsize=8, loc="left")

    a = ax[2]
    fr, cs = [], []
    for k in ALPHAS:
        for r in d["main"]:
            if k not in r.get("ground_truth", {}):
                continue
            st = np.linalg.norm(r["predictor"][k]["step"])
            if st > 0:
                fr.append(r["ground_truth"][k]["de0_norm"] / st)
                cs.append(r["ground_truth"][k]["cos"])
    fr, cs = np.array(fr), np.array(cs)
    a.plot(fr, cs, "o", ms=3.2, mfc="none", mew=0.8, color=C_MEAN, ls="none")
    a.axvline(float(np.median(fr)), color="#222222", lw=1.0)
    a.axhline(1.0, color=C_GREY, lw=0.7, ls=":")
    a.set_xlabel(r"$\|\Delta\epsilon_0\| \, / \, \|{\rm predicted\ step}\|$")
    a.set_ylabel(r"$\cos$ (direction)")
    a.set_xlim(0, 1.2); a.set_ylim(-0.1, 1.08)
    a.text(0.04, 0.12, f"median fraction {np.median(fr):.2f}\n"
                       f"median cos {np.median(cs):.3f}",
           transform=a.transAxes, fontsize=6.8, color="#222222")
    a.set_title("(c) direction free, distance costs a constant", fontsize=8, loc="left")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(path); fig.savefig(str(path).replace(".pdf", ".png")); plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig_leverage(t1, rtn, path):
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))

    a = ax[0]
    lo, hi = 5e-3, 1.2
    a.plot([lo, hi], [lo, hi], color=C_GREY, lw=0.9, ls="--", zorder=1)
    for k in ALPHAS:
        sub = [r for r in t1["rows"] if abs(r["alpha"] - float(k)) < 1e-12]
        a.plot([r["Lambda"] for r in sub], [r["ratio"] for r in sub], AMARK[k],
               ms=3.4, mfc="none", mew=0.9, color=ACOL[k], ls="none",
               label=rf"$\alpha={k}$", zorder=3)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lo, hi); a.set_ylim(lo, hi)
    a.set_xlabel(r"$\Lambda_\alpha$  (bulk share of the tail)")
    a.set_ylabel(r"$R_{\rm pred}^{\rm mix} / R_{\rm pred}^{\rm bulk}$")
    a.text(0.04, 0.95, f"median error {100*t1['ratio_vs_Lambda_median_rel_err']:.2f}%\n"
                       f"worst {100*t1['ratio_vs_Lambda_max_rel_err']:.2f}%"
                       f"   ($n={t1['n_pairs']}$)",
           transform=a.transAxes, fontsize=6.8, va="top", color="#222222")
    a.legend(loc="lower right", handletextpad=0.4)
    a.set_title(r"(a) inert rare channel ($T_1$)", fontsize=8, loc="left")

    a = ax[1]
    rar = [x for x in rtn["rarity"] if (x["p_over_alpha"] or 9) < 1.0]
    p = [x["p"] for x in rar]
    a.plot(p, [x["R_mix"] for x in rar], "s-", ms=4, color=C_CV05,
           label=r"$R_{\rm pred}$ (mixture)")
    a.plot(p, [x["R_bulk"] for x in rar], "^--", ms=4, color=C_GRN,
           label=r"$R_{\rm pred}$ (bulk only)")
    a.plot(p, [x["R_meas"] for x in rar], "o-", ms=4.4, color="#222222",
           label=r"$R$ measured")
    a.set_xscale("log"); a.set_xlabel(r"fluctuator rarity $p$")
    a.set_ylabel(r"$R_\alpha$   ($\alpha = 0.05$)")
    a.set_ylim(0, None)
    a.margins(y=0.22)
    if rtn["p_span_factor"]:
        a.text(0.04, 0.95,
               rf"$p$ spans ${rtn['p_span_factor']:.0f}\times$;"
               rf" $R$ flat to {100*rtn['R_meas_spread_frac']:.0f}%"
               + (rf"; $\Lambda$ spans ${rtn['Lambda_span_factor']:.0f}\times$"
                  if rtn["Lambda_span_factor"] else ""),
               transform=a.transAxes, fontsize=6.5, va="top", color="#222222")
    a.legend(loc="center right", handletextpad=0.4, fontsize=6.5)
    a.set_title("(b) aligned rare channel (fluctuator)", fontsize=8, loc="left")

    a = ax[2]
    allr = [x for x in rtn["rarity"] + rtn["amplitude"] if x["Lambda"] is not None]
    lamv = [max(x["Lambda"], 1e-3) for x in allr]
    a.plot(lamv, [x["ratio"] for x in allr], "o", ms=3.8, mfc="none", mew=0.9,
           color=C_CV05, ls="none", label="fluctuator")
    a.plot([r["Lambda"] for r in t1["rows"]], [r["ratio"] for r in t1["rows"]],
           ".", ms=3.0, color=C_MEAN, ls="none", label=r"$T_1$ jump")
    a.plot([5e-3, 1.2], [5e-3, 1.2], color=C_GREY, lw=0.9, ls="--", zorder=1)
    a.axhline(1.0, color=C_GREY, lw=0.7, ls=":")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(5e-3, 1.4); a.set_ylim(5e-3, 6.0)
    a.set_xlabel(r"$\Lambda_\alpha$")
    a.set_ylabel(r"$R_{\rm pred}^{\rm mix} / R_{\rm pred}^{\rm bulk}$")
    a.text(0.04, 0.30, r"$\Lambda$ is the inert limit," "\n" r"not a universal factor",
           transform=a.transAxes, fontsize=6.8, va="top", color="#222222")
    a.legend(loc="lower right", handletextpad=0.4, fontsize=6.5)
    a.set_title(r"(c) $\Lambda$ governs only one of them", fontsize=8, loc="left")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(path); fig.savefig(str(path).replace(".pdf", ".png")); plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig_boundary(summary, off, path):
    """The two results that had no figure: where the predictor stops working,
    and what the calibration constant is made of."""
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))

    a = ax[0]
    for k in ALPHAS:
        h = (summary.get("by_alpha", {}).get(k, {}) or {}).get("H6_boundary_location")
        if not h or not h.get("grid"):
            continue
        g = sorted(h["grid"], key=lambda x: x["kappa"])
        kp = np.array([x["kappa"] for x in g]); rp = np.array([x["R_pred"] for x in g])
        a.plot(kp, rp, AMARK[k], ms=3.0, mfc="none", mew=0.8, color=ACOL[k],
               ls="none", label=rf"$\alpha={k}$")
        plat = float(np.median([x["R_pred"] for x in g if x["kappa"] <= 0.006]))
        a.axhline(plat, color=ACOL[k], lw=0.7, ls=":", alpha=0.8)
        hi = [(x["kappa"], x["R_pred"]) for x in g if x["kappa"] > 0.05]
        if hi:
            kk = np.array([x[0] for x in hi]); rr = np.array([x[1] for x in hi])
            C = float(np.sum(kk ** 2 * rr) / np.sum(kk ** 4))
            xs = np.array([3e-3, 0.6])
            a.plot(xs, C * xs ** 2, color=ACOL[k], lw=0.8, alpha=0.6)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(6e-4, 0.8); a.set_ylim(1e-6, 1.6)
    a.set_xlabel(r"$\kappa$"); a.set_ylabel(r"$R_{\rm pred}$")
    a.text(0.03, 0.96, "lines: $C\\kappa^2$ fitted above $\\kappa=0.05$\n"
                       "dotted: measured plateau",
           transform=a.transAxes, fontsize=6.3, va="top", color=C_GREY)
    a.legend(loc="lower right", handletextpad=0.4, fontsize=6.3, ncol=2,
             columnspacing=0.8)
    a.set_title("(a) the predictor has a floor", fontsize=8, loc="left")

    a = ax[1]
    lab = [r"analytic" "\n" r"$R_{\min}$", "measured" "\n" "plateau"]
    x = np.arange(len(ALPHAS))
    fl, pl = [], []
    for k in ALPHAS:
        h = (summary.get("by_alpha", {}).get(k, {}) or {}).get("H6_boundary_location")
        g = h["grid"]
        fl.append(h["median_floor"])
        pl.append(float(np.median([q["R_pred"] for q in g if q["kappa"] <= 0.006])))
    a.bar(x - 0.19, fl, 0.36, color=C_MEAN, label=lab[0].replace("\n", " "))
    a.bar(x + 0.19, pl, 0.36, color=C_CV05, label=lab[1].replace("\n", " "))
    for i, (f_, p_) in enumerate(zip(fl, pl)):
        a.text(i, p_ * 1.3, rf"${p_/f_:.0f}\times$", ha="center", fontsize=6.5,
               color="#222222")
    a.set_yscale("log"); a.set_xticks(x); a.set_xticklabels(ALPHAS)
    a.set_xlabel(r"$\alpha$"); a.set_ylabel(r"gain")
    a.set_ylim(1e-5, 2e-2)
    a.legend(loc="upper left", handletextpad=0.5, fontsize=6.3)
    a.set_title("(b) the floor is underestimated", fontsize=8, loc="left")

    a = ax[2]
    big = [(r["kappa"], v) for r in off for kk, v in r["offset"].items()
           if kk == "0.05" and r["kappa"] > 0.03 and v["R_pred"] > 0]
    sml = [(r["kappa"], v) for r in off for kk, v in r["offset"].items()
           if kk == "0.05" and r["kappa"] <= 0.03 and v["R_pred"] > 0]
    for grp, col, mk, lb in ((sml, C_GRN, "s", r"$\kappa\leq0.03$ (plateau)"),
                             (big, C_CV05, "o", r"$\kappa>0.03$ (resolved)")):
        a.plot([v["R_pred"] for _, v in grp],
               [max(v["R_offset_best"], 1e-5) for _, v in grp], mk, ms=3.6,
               mfc="none", mew=0.9, color=col, ls="none", label=lb)
    xs = np.array([1e-4, 1.0])
    a.plot(xs, xs, color=C_GREY, lw=0.9, ls="--")
    a.plot(xs, 0.327 * xs, color="#222222", lw=1.1)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlim(2e-4, 0.9); a.set_ylim(2e-4, 0.9)
    a.set_xlabel(r"$R_{\rm pred}$")
    a.set_ylabel(r"$R$ from the linear offset alone")
    a.text(0.03, 0.96, "solid: slope 0.33\ndashed: identity",
           transform=a.transAxes, fontsize=6.3, va="top", color=C_GREY)
    a.legend(loc="lower right", handletextpad=0.4, fontsize=6.3)
    a.set_title("(c) a linear move buys a third of it", fontsize=8, loc="left")

    fig.tight_layout(w_pad=1.5)
    fig.savefig(path); fig.savefig(str(path).replace(".pdf", ".png")); plt.close(fig)


# ------------------------------------------------------------------ table
def write_table(cal, path):
    L = [r"\begin{ruledtabular}", r"\begin{tabular}{lcccc}",
         r"$\alpha$ & 0.20 & 0.10 & 0.05 & 0.025 \\ \hline"]

    def row(name, fn, fmt="{:.3f}"):
        return (name + " & " + " & ".join(
            (fmt.format(fn(cal[k])) if k in cal else "--") for k in ALPHAS) + r" \\")
    L += [row("through-origin slope $s_\\alpha$", lambda c: c["slope"]),
          row("Spearman $\\rho$", lambda c: c["spearman"][0]),
          row("attenuation ceiling", lambda c: c["attenuation_ceiling"]),
          row("$\\rho\\,/\\,$ceiling", lambda c: c["spearman_over_ceiling"]),
          row("replicate s.d.\\ of $R$", lambda c: c["replicate_sd_R"], "{:.4f}"),
          row("median $\\cos$", lambda c: c["median_cos"], "{:.4f}"),
          row("median $\\|\\Delta\\epsilon_0\\|/\\|\\text{step}\\|$",
              lambda c: c["median_step_fraction"]),
          r"\end{tabular}", r"\end{ruledtabular}"]
    Path(path).write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ main
def selftest_pairing():
    """F1 -- a polished fluctuator cell substitutes WHOLE, never field by field.

    The failure this guards against is silent and would not raise: read R from
    the pooled rerun and Lambda from the original file and the leverage panel
    plots a gain measured under one control against a bulk share computed under
    another.  Since the alignment claim IS "R stays flat while Lambda moves",
    crossing the two manufactures or destroys the result with nothing raising.

    Build a source cell and a polished cell that disagree on every quantity, run
    the real loader, and assert the survivor is internally consistent.
    """
    import tempfile
    src = {"stem": "s6rtn_TEST", "rtn_amp_over_Tg": 1.0, "rtn_p_requested": 0.01,
           "p_active": 0.0101, "kappa": 0.08, "fit_R2_min": 0.999,
           "ground_truth": {"0.05": {"R": 0.010, "R_raw": 0.010, "comparator": "rev"}},
           "suppression": {"0.05": {"R_pred_mixture": 0.050, "R_pred_bulk_only": 0.040,
                                    "ratio": 1.25, "Lambda": 0.900,
                                    "conflict_cos": -0.99, "p_over_alpha": 0.2,
                                    "d_norm_bulk": 1.0, "d_norm_mixture": 1.1}}}
    pol = {"stem": "s6rtn_TEST", "kappa": 0.08,
           "ground_truth": {"0.05": {"R": 0.030, "R_raw": 0.030, "comparator": "pool"}},
           "suppression": {"0.05": {"R_pred_mixture": 0.070, "R_pred_bulk_only": 0.035,
                                    "ratio": 2.00, "Lambda": 0.400,
                                    "conflict_cos": -0.80, "p_over_alpha": 0.2,
                                    "d_norm_bulk": 2.0, "d_norm_mixture": 2.2}}}
    fails = []
    with tempfile.TemporaryDirectory() as td:
        root = Path(td); (root / "stage6_rtn").mkdir()
        (root / "stage6_rtn" / "s6rtn_TEST.json").write_text(json.dumps(src))
        d = load_all(root)
        r = d["rtn"][0]
        if r["ground_truth"]["0.05"]["R"] != 0.010: fails.append("F1a source R")
        if r["suppression"]["0.05"]["Lambda"] != 0.900: fails.append("F1b source Lambda")
        if d["rtn_all_polished"]: fails.append("F1c claimed polished with none")

        (root / "polish_stage6_rtn").mkdir()
        (root / "polish_stage6_rtn" / "polish_s6rtn_TEST.json").write_text(json.dumps(pol))
        d = load_all(root)
        if len(d["rtn"]) != 1: fails.append(f"F1d {len(d['rtn'])} records, want 1")
        r = d["rtn"][0]
        got = (r["ground_truth"]["0.05"]["R"], r["suppression"]["0.05"]["Lambda"])
        if got != (0.030, 0.400):
            fails.append(f"F1e crossed fields: (R, Lambda) = {got}, want (0.03, 0.4)")
        if r.get("rtn_p_requested") != 0.01:
            fails.append("F1f lost the rarity label, so the point leaves the sweep")
        if not d["rtn_all_polished"]: fails.append("F1g did not register as polished")
        st = rtn_stats(d, "0.05")
        if "UNPOLISHED" in st["caveat"]:
            fails.append("F1h caveat still says unpolished after pooling")
    print("F1 pairing: PASS" if not fails else "F1 pairing: FAIL " + "; ".join(fails))
    return not fails


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest_pairing() else 1)
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    d = load_all(root)
    print(f"loaded: {len(d['main'])} main, {len(d['polish'])} polished, "
          f"{len(d['t1'])} T1 retro, {len(d['rtn'])} RTN")
    if not d["main"]:
        sys.exit(f"no Stage 6 point files under {root}")
    if not d["polish"]:
        print("WARNING: no polish_s6_*.json found -- R falls back to the main run's "
              "best-of-{mean, rev_*} comparator, which is dominated by whether the "
              "mean multistart converged.  Run `bash run_stage6.sh polish` first.")

    cal = calibration_stats(d)
    t1 = t1_stats(d) if d["t1"] else None
    rt = rtn_stats(d) if d["rtn"] else None

    fig_calibration(d, cal, OUT / "fig_stage6_calibration.pdf")
    off = _load(root, "off_s6_*.json")
    if d["summary"].get("by_alpha") and off:
        fig_boundary(d["summary"], off, OUT / "fig_stage6_boundary.pdf")
    elif not off:
        print("NOTE: no offsettest output found; skipping fig_stage6_boundary")
    if t1 and rt:
        fig_leverage(t1, rt, OUT / "fig_stage6_leverage.pdf")
    write_table(cal, OUT / "tab_stage6.tex")

    stats_out = {"calibration": cal, "T1_arm": t1, "RTN_arm": rt,
                 "summary_H2": {k: (d["summary"].get("by_alpha", {}).get(k, {})
                                    .get("H2_direction", {}))
                                for k in ALPHAS},
                 "baseline_convergence": d["summary"].get("baseline_convergence"),
                 "H1_closed_form": d["summary"].get("H1_closed_form")}
    (OUT / "stage6_stats.json").write_text(json.dumps(stats_out, indent=1, default=float))

    print(f"\nslope {cal['_pooled']['slope_mean']:.3f} +- {cal['_pooled']['slope_sd']:.3f} "
          f"over {len(ALPHAS)} alpha; "
          f"{cal['_pooled']['slope_by_alpha_seed_mean']:.3f} +- "
          f"{cal['_pooled']['slope_by_alpha_seed_sd']:.3f} over "
          f"{cal['_pooled']['n_alpha_seed_cells']} (alpha, seed) cells")
    if t1:
        print(f"T1  ratio vs Lambda: median {100*t1['ratio_vs_Lambda_median_rel_err']:.2f}% "
              f"worst {100*t1['ratio_vs_Lambda_max_rel_err']:.2f}%  (n={t1['n_pairs']})")
    if rt and rt["R_meas_spread_frac"] is not None:
        print(f"RTN rarity: p spans {rt['p_span_factor']:.0f}x, R flat to "
              f"{100*rt['R_meas_spread_frac']:.0f}%, Lambda spans "
              f"{rt['Lambda_span_factor']:.0f}x")
    print(f"\nwrote {OUT}/fig_stage6_calibration.pdf, fig_stage6_leverage.pdf, "
          f"tab_stage6.tex, stage6_stats.json")


if __name__ == "__main__":
    main()
