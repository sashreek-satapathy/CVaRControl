"""
make_stage6_ktrain.py -- the training-tail control experiment (Stage 6C).

Question.  The calibration constant s = 0.588 +- 0.025 says the measurement-only
predictor orders pulses almost perfectly but overstates the level by a fixed
factor.  One candidate explanation is statistical: the CVaR arm optimizes an
alpha-tail estimated from only alpha*K_train training draws, so with a small
training tail it aims at noise and under-delivers.  If that is the whole story,
R/R_pred must climb toward 1 as K_train grows at FIXED alpha.

Design.  alpha is pinned at 0.05 and only K_train moves, over 128 -> 5120, so the
training-tail size alpha*K_train moves 6.4 -> 256 draws while the risk measure,
the test ensemble (50k) and the seed are all held.  Three nominal kappa.

Reads   results_stage6/ktrain_2level/s6kt_*.json   (or any root containing them)
Writes  figures/fig_stage6_ktrain.pdf / .png
        figures/tab_stage6_ktrain.tex
        figures/stage6_ktrain_stats.json

USAGE   python make_stage6_ktrain.py [results_root]
"""
from __future__ import annotations
import json, sys
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

ALPHA = "0.05"
# A series is admitted to the level test only if the reduced quadratic model that
# DEFINES kappa holds across it and kappa itself is actually held fixed.  Both
# thresholds are the ones used everywhere else in the paper.
R2_MIN      = 0.95      # fit_R2_min of the quadratic surrogate
KAPPA_SPREAD_MAX = 0.25 # max/min of realized kappa within a series


def load(root):
    rows = []
    for p in sorted(Path(root).rglob("s6kt_*.json")):
        d = json.loads(p.read_text())
        if ALPHA not in d.get("predictor", {}) or ALPHA not in d.get("ground_truth", {}):
            continue
        pr, gt, ev = d["predictor"][ALPHA], d["ground_truth"][ALPHA], d["evals"]
        comp = gt["comparator"]
        rows.append(dict(
            stem=d["stem"], nominal=str(d["stem"]).split("_")[2],
            K=int(d["K_train"]), aK=float(d.get("alpha_K_train", float(ALPHA) * d["K_train"])),
            kappa=float(d["kappa"]), R2=float(d["fit_R2_min"]),
            R_pred=float(pr["R_pred"]), t_best=float(pr["t_best"]),
            R=float(gt["R"]), R_raw=float(gt["R_raw"]), cos=float(gt["cos"]),
            comparator=comp,
            ml_mean=float(ev["mean"]["mean_loss"]),
            ml_cvar=float(ev["cvar_" + ALPHA]["mean_loss"]),
            seconds=float(d.get("seconds", float("nan")))))
    for r in rows:
        r["ratio"] = r["R"] / r["R_pred"] if r["R_pred"] else float("nan")
        r["ml_ratio"] = r["ml_cvar"] / r["ml_mean"] if r["ml_mean"] else float("nan")
        # the discriminator of Sec. IV L: a genuine mean-for-tail trade RAISES the
        # mean loss of the tail arm; a value below 1 means the mean arm was simply
        # not converged and the "gain" is baseline slack.
        r["trade"] = bool(r["ml_ratio"] > 1.0)
    rows.sort(key=lambda r: (r["nominal"], r["K"]))
    return rows


def series(rows):
    out = {}
    for r in rows:
        out.setdefault(r["nominal"], []).append(r)
    for k, v in out.items():
        ka = np.array([x["kappa"] for x in v])
        spread = float(ka.max() / ka.min() - 1.0)
        r2 = float(min(x["R2"] for x in v))
        rat = np.array([x["ratio"] for x in v], float)
        lk = np.log(np.array([x["aK"] for x in v], float))
        rho, p = stats.spearmanr([x["aK"] for x in v], rat)
        # log-log slope of R/R_pred against the training-tail size.  Starvation
        # predicts a clearly positive slope saturating at ratio = 1.
        sl = float(np.polyfit(lk, np.log(np.clip(rat, 1e-12, None)), 1)[0])
        out[k] = dict(rows=v, kappa_median=float(np.median(ka)),
                      kappa_spread=spread, R2_min=r2,
                      admitted=bool(spread <= KAPPA_SPREAD_MAX and r2 >= R2_MIN),
                      ratio_min=float(rat.min()), ratio_max=float(rat.max()),
                      ratio_span=float(rat.max() / rat.min()),
                      spearman=[float(rho), float(p)], loglog_slope=sl,
                      n_trade=int(sum(x["trade"] for x in v)), n=len(v))
    return out


def figure(ser, path):
    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))
    cols = [C_MEAN, C_CV05, C_GREY]
    marks = ["o", "s", "^"]
    keys = sorted(ser, key=lambda k: ser[k]["kappa_median"])

    # (a) the level test
    a = ax[0]
    for i, k in enumerate(keys):
        s = ser[k]
        x = [r["aK"] for r in s["rows"]]; y = [r["ratio"] for r in s["rows"]]
        lab = (rf"$\kappa\simeq{s['kappa_median']:.2f}$" if s["admitted"]
               else rf"$\kappa$ uncontrolled")
        a.plot(x, y, marker=marks[i], color=cols[i], ms=4.5, mfc="white", mew=1.2,
               lw=1.1, ls=("-" if s["admitted"] else ":"),
               alpha=(1.0 if s["admitted"] else 0.45), label=lab)
    a.axhline(1.0, color="k", lw=0.8, ls="--")
    a.set_ylim(0.02, 1.9)
    a.text(0.02, 0.985, "starvation limit", fontsize=6.5, color="k",
           transform=a.transAxes, va="top",
           bbox=dict(fc="white", ec="none", pad=0.6))
    a.axhspan(0.45, 0.75, color=C_GRN, alpha=0.14, lw=0)
    a.text(0.02, 0.775, r"calibration band $s=0.59$", fontsize=6.5, color=C_GRN,
           transform=a.transAxes, va="top",
           bbox=dict(fc="white", ec="none", pad=0.6))
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel(r"training-tail draws $\alpha K_{\rm train}$")
    a.set_ylabel(r"$R\,/\,R_{\rm pred}$")
    a.set_title(r"(a) level does not climb with the tail", fontsize=8, loc="left")
    a.legend(loc="lower right")

    # (b) the discriminator
    b = ax[1]
    for i, k in enumerate(keys):
        s = ser[k]
        x = [r["aK"] for r in s["rows"]]; y = [r["ml_ratio"] for r in s["rows"]]
        b.plot(x, y, marker=marks[i], color=cols[i], ms=4.5, mfc="white", mew=1.2,
               lw=1.1, ls=("-" if s["admitted"] else ":"),
               alpha=(1.0 if s["admitted"] else 0.45))
    b.axhline(1.0, color="k", lw=0.8, ls="--")
    b.set_ylim(0.6, 4.3)
    b.axhspan(0.6, 1.0, color=C_GREY, alpha=0.10, lw=0)
    b.text(0.03, 0.97, "above the line: genuine mean-for-tail trade", fontsize=6.5,
           transform=b.transAxes, ha="left", va="top")
    b.text(0.03, 0.89, "below: mean arm was not converged", fontsize=6.5, color=C_GREY,
           transform=b.transAxes, ha="left", va="top")
    b.set_xscale("log")
    b.set_xlabel(r"training-tail draws $\alpha K_{\rm train}$")
    b.set_ylabel(r"mean loss, tail arm $/$ mean arm")
    b.set_title(r"(b) the price rises, the gain does not", fontsize=8, loc="left")

    # (c) why the third series is excluded
    c = ax[2]
    for i, k in enumerate(keys):
        s = ser[k]
        x = [r["aK"] for r in s["rows"]]
        c.plot(x, [r["R2"] for r in s["rows"]], marker=marks[i], color=cols[i],
               ms=4.5, mfc="white", mew=1.2, lw=1.1,
               ls=("-" if s["admitted"] else ":"),
               alpha=(1.0 if s["admitted"] else 0.45))
    c.axhline(R2_MIN, color="k", lw=0.8, ls="--")
    c.text(0.02, 0.80, r"admission floor $R^2=0.95$", fontsize=6.5,
           transform=c.transAxes, va="top",
           bbox=dict(fc="white", ec="none", pad=0.6))
    c.set_xscale("log"); c.set_ylim(0.4, 1.04)
    c.set_xlabel(r"training-tail draws $\alpha K_{\rm train}$")
    c.set_ylabel(r"$R^2_{\min}$ of the quadratic surrogate")
    c.set_title(r"(c) the excluded series fails the model", fontsize=8, loc="left")

    fig.tight_layout(pad=0.4)
    fig.savefig(path); fig.savefig(str(path).replace(".pdf", ".png"))
    plt.close(fig)


def table(rows, path):
    L = [r"\begin{ruledtabular}", r"\begin{tabular}{rrrrrrrc}",
         r"$K_{\rm train}$ & $\alpha K_{\rm train}$ & $\kappa$ & $R^2_{\min}$ &"
         r" $R_{\rm pred}$ & $R$ & $R/R_{\rm pred}$ & trade? \\ \hline"]
    last = None
    for r in rows:
        if r["nominal"] != last:
            if last is not None:
                L.append(r"\hline")
            last = r["nominal"]
        L.append(f"{r['K']} & {r['aK']:.0f} & {r['kappa']:.3f} & {r['R2']:.3f} & "
                 f"{r['R_pred']:.3f} & {r['R']:.3f} & {r['ratio']:.3f} & "
                 + ("yes" if r["trade"] else "no") + r" \\")
    L += [r"\end{tabular}", r"\end{ruledtabular}"]
    Path(path).write_text("\n".join(L) + "\n")


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results_stage6"
    rows = load(root)
    if not rows:
        print(f"no s6kt_*.json under {root}"); return
    ser = series(rows)
    figure(ser, OUT / "fig_stage6_ktrain.pdf")
    table(rows, OUT / "tab_stage6_ktrain.tex")
    stats_out = {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in ser.items()}
    stats_out["_rows"] = rows
    stats_out["_total_hours"] = sum(r["seconds"] for r in rows) / 3600.0
    (OUT / "stage6_ktrain_stats.json").write_text(json.dumps(stats_out, indent=1))
    for k in sorted(ser, key=lambda k: ser[k]["kappa_median"]):
        s = ser[k]
        print(f"{k:8s} kappa~{s['kappa_median']:.3f} spread {s['kappa_spread']*100:5.1f}% "
              f"R2min {s['R2_min']:.3f} admitted={s['admitted']} "
              f"ratio {s['ratio_min']:.3f}-{s['ratio_max']:.3f} (span {s['ratio_span']:.1f}x) "
              f"rho={s['spearman'][0]:+.2f} p={s['spearman'][1]:.2f} "
              f"loglog {s['loglog_slope']:+.2f} trades {s['n_trade']}/{s['n']}")
    print(f"total {stats_out['_total_hours']:.2f} h")


if __name__ == "__main__":
    main()
