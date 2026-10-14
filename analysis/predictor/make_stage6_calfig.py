"""
make_stage6_calfig.py -- rebuild fig_stage6_calibration.pdf from the POST-POOL
summary alone.

Why this exists.  The three-panel version in make_stage6_figures.py draws its
first panel as a point-by-point scatter of R against R_pred, and it needs
results_stage6/polish_2level/polish_s6_*.json to do it.  When those files are
absent the script silently falls back to the main run's best-of-comparator R and
draws the PRE-pool relationship -- slope -0.259 -- underneath a caption quoting
the post-pool 0.588.  Rather than ship that, this script builds the three panels
that the summary alone can support correctly.  Everything here is read from
stage6_summary.json written after `polish`.

USAGE  python make_stage6_calfig.py stage6_summary.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
# Replicate scatter under the ORIGINAL best-of-comparator rule, recorded before
# the pool was run (Table S-XIII in the supplement).
SD_PRE = {"0.2": 0.2717, "0.1": 0.2494, "0.05": 0.1783, "0.025": 0.1362}


def main():
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "stage6_summary.json")
    S = json.loads(src.read_text())
    ba = S["by_alpha"]
    a = np.array([float(x) for x in ALPHAS])
    sl = np.array([ba[k]["H3_level"]["slope"] for k in ALPHAS])
    seeds = {s: np.array([ba[k]["H3_level"]["seed_blocked"]["by_seed"][s]["slope"]
                          for k in ALPHAS]) for s in ("0", "1")}
    rho = np.array([ba[k]["H3_level"]["spearman"][0] for k in ALPHAS])
    ceil = np.array([ba[k]["H3_level"]["attenuation_ceiling"] for k in ALPHAS])
    sdpost = np.array([ba[k]["H3_level"]["replicate_sd_R"] for k in ALPHAS])
    sdpre = np.array([SD_PRE[k] for k in ALPHAS])
    mbar, msd = float(sl.mean()), float(sl.std(ddof=1))

    fig, ax = plt.subplots(1, 3, figsize=(7.1, 2.35))

    # (a) the constant
    p = ax[0]
    p.axhspan(mbar - msd, mbar + msd, color=C_MEAN, alpha=0.13, lw=0)
    p.axhline(mbar, color=C_MEAN, lw=1.2)
    for s, mk, c in (("0", "v", C_GREY), ("1", "^", C_GREY)):
        p.plot(a, seeds[s], mk, color=c, ms=4, mfc="none", mew=1.0,
               ls="none", label=("single seeds" if s == "0" else None))
    p.plot(a, sl, "o-", color=C_MEAN, ms=5, mfc="white", mew=1.4,
           label="all points")
    p.axhline(1.0, color="k", lw=0.8, ls="--")
    p.text(0.03, 0.97, "unit slope (perfect level)", fontsize=6.5,
           transform=p.transAxes, va="top")
    p.text(0.03, 0.40, rf"$s={mbar:.3f}\pm{msd:.3f}$", fontsize=7.5,
           color=C_MEAN, transform=p.transAxes, va="top")
    p.set_xscale("log"); p.set_ylim(0.0, 1.25); p.minorticks_off()
    p.set_xticks(a); p.set_xticklabels([f"{x:g}" for x in a])
    p.set_xlabel(r"tail level $\alpha$")
    p.set_ylabel(r"through-origin slope $s_\alpha$")
    p.set_title("(a) one constant, flat in " + r"$\alpha$", fontsize=8, loc="left")
    p.legend(loc="lower right")

    # (b) ordering against the ceiling scatter allows
    q = ax[1]
    q.plot(a, ceil, "s--", color=C_GREY, ms=4.5, mfc="white", mew=1.2,
           label="attainable ceiling")
    q.plot(a, rho, "o-", color=C_CV05, ms=5, mfc="white", mew=1.4,
           label=r"measured Spearman $\rho$")
    for x, r, c in zip(a, rho, ceil):
        q.annotate(f"{r/c:.2f}", (x, r), textcoords="offset points",
                   xytext=(0, -11), ha="center", fontsize=6.5, color=C_CV05)
    q.set_xscale("log"); q.set_ylim(0.70, 1.045); q.minorticks_off()
    q.set_xticks(a); q.set_xticklabels([f"{x:g}" for x in a])
    q.set_xlabel(r"tail level $\alpha$")
    q.set_ylabel(r"rank correlation with measured $R$")
    q.set_title(r"(b) ordering is near the ceiling", fontsize=8, loc="left")
    q.legend(loc="lower left")

    # (c) where the constant came from: the restart-budget repair
    r = ax[2]
    w = 0.34
    o = np.argsort(a)                     # ascending alpha, as in (a) and (b)
    idx = np.arange(len(a), dtype=float)
    r.bar(idx - w / 2 - 0.01, sdpre[o], width=w, color=C_GREY, alpha=0.55,
          label="best-of-comparator")
    r.bar(idx + w / 2 + 0.01, sdpost[o], width=w, color=C_GRN,
          label="shared basin pool")
    for i, (u, v) in enumerate(zip(sdpre[o], sdpost[o])):
        r.text(i, max(u, v) * 1.25, rf"${u/v:.0f}\times$", ha="center",
               fontsize=6.5, color=C_GREY)
    r.set_yscale("log"); r.set_ylim(5e-3, 2.2)
    r.set_xticks(idx); r.set_xticklabels([f"{x:g}" for x in a[o]])
    r.set_xlabel(r"tail level $\alpha$")
    r.set_ylabel("replicate s.d. of $R$ between seeds")
    r.set_title(r"(c) why the slope is a calibration", fontsize=8, loc="left")
    r.legend(loc="upper right")

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_stage6_calibration.pdf")
    fig.savefig(OUT / "fig_stage6_calibration.png")
    plt.close(fig)
    print(f"slopes {np.round(sl,4)}  mean {mbar:.4f} sd {msd:.4f}")
    print(f"rho/ceiling {np.round(rho/ceil,3)}")
    print(f"scatter reduction {np.round(sdpre/sdpost,1)}")


if __name__ == "__main__":
    main()
