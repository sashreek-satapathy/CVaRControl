"""
make_figures.py -- paper figures + one merged table from every validate.json.

    python make_figures.py                 # uses every ./results_*/validate.json
    python make_figures.py results_fast results_strong ...

Only FULL-LINDBLAD numbers are plotted.  The plotted R_tail is always the
conservative one: CVaR control vs the BEST mean-objective control found at that
point (multistart + reverse warm starts, selected on validation data).  The
mean side gets the extra reverse-start search, so a positive value is a LOWER
BOUND on the CVaR advantage.  Nothing is dropped: filled markers = the reverse
starts found no better mean control; open markers = they did (comparator was
upgraded).  Values below the axis are drawn as arrows at the bottom edge.

Writes to figures/:
  fig_collapse.(pdf|png)   R_tail vs sigma_phi^2, 2-level detuning-only, one
                           series per gate time -> tests the mechanism claim
                           (curves collapse iff sigma_phi = sigma_d*Tg is the
                           controlling variable)
  fig_tg_sweep.(pdf|png)   R_tail vs Tg: transmon, 2-level, detuning-only,
                           amplitude-only, strong-search re-run
  fig_audit.(pdf|png)      Eq. 7 audit at the onset, all alpha, every seed
  all_results.csv          every validated row, with its source folder
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TWO_PI = 2 * np.pi
SERIES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "lines.linewidth": 1.4,
    "lines.markersize": 4.5, "legend.frameon": False, "savefig.dpi": 300,
    "savefig.bbox": "tight", "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5})


def load(dirs):
    rows = []
    for d in dirs:
        f = Path(d) / "validate.json"
        if not f.exists():
            print(f"skip {d}: no validate.json")
            continue
        for r in json.loads(f.read_text())["rows"]:
            r["source"] = Path(d).name
            r["kind"] = r["file"].split("/")[0]
            use_best = r.get("R_tail_vs_best_mean_lindblad") is not None and \
                r.get("best_mean_source", "mean") != "mean"
            r["R"] = r["R_tail_vs_best_mean_lindblad"] if use_best else r["R_tail_lindblad"]
            r["R_lo"] = r["R_lo_vs_best_mean"] if use_best else r["R_lo_lindblad"]
            r["R_hi"] = r["R_hi_vs_best_mean"] if use_best else r["R_hi_lindblad"]
            r["sigma_phi_sq"] = (TWO_PI * r["sigma_d_MHz"] * 1e-3 * r["Tg"]) ** 2
            rows.append(r)
    # A POINT (one json file) is "consistent" unless some reverse warm start
    # found a mean control >0.1% better than the multistart winner; then the
    # comparator was upgraded (open marker).  Used for styling only.
    by_point = {}
    for r in rows:
        by_point.setdefault((r["source"], r["file"]), []).append(r)
    for pts in by_point.values():
        ok = all(p.get("reverse_class_fast") != "artifact_mean_missed_basin" for p in pts)
        for p in pts:
            p["consistent"] = ok
    return rows


YMIN = -0.08   # axis floor; cells below it are drawn as arrows


def _err(ax, rr, x, k, label, ls=None):
    rr = sorted(rr, key=lambda r: r[x])
    if not rr:
        return
    c, m = SERIES[k % 6], MARKERS[k % 6]
    xs = np.array([r[x] for r in rr]); y = np.array([r["R"] for r in rr])
    lo = np.array([r["R_lo"] for r in rr]); hi = np.array([r["R_hi"] for r in rr])
    inb = y >= YMIN
    ax.plot(xs, np.clip(y, YMIN, None), color=c, ls=ls if ls is not None else STYLES[k % 6],
            lw=1.2, zorder=1)
    for i in np.where(inb)[0]:
        filled = rr[i].get("consistent", True)
        ax.errorbar(xs[i], y[i], yerr=[[y[i] - lo[i]], [hi[i] - y[i]]], color=c, marker=m,
                    mfc=c if filled else "white", mec=c, ls="none", capsize=2, elinewidth=0.7,
                    zorder=3)
    for i in np.where(~inb)[0]:
        ax.plot(xs[i], YMIN, marker="v", color=c, mfc="white", ms=5, zorder=4, clip_on=False)
        ax.annotate(f"{y[i]:.2f}", (xs[i], YMIN), xytext=(5, 2), textcoords="offset points",
                    ha="left", va="bottom", fontsize=6.5, color=c)
    ax.plot([], [], color=c, marker=m, ls=ls if ls is not None else STYLES[k % 6], label=label)


def _finish(fig, axes, ncol):
    from matplotlib.lines import Line2D
    h, l = axes[0].get_legend_handles_labels()
    h += [Line2D([], [], color="#555", marker="o", ls="none"),
          Line2D([], [], color="#555", marker="o", mfc="white", ls="none")]
    l += ["comparator: mean multistart", "comparator upgraded by reverse start"]
    fig.legend(h, l, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.3, wspace=0.08)
    for ax in axes:
        ax.set_ylim(YMIN - 0.012, None)


def fig_collapse(rows, out):
    rr = [r for r in rows if r["kind"].startswith("sigma") and r["n_levels"] == 2]
    if not rr:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), sharey=True)
    for ax, a in zip(axes, (0.05, 0.2)):
        tgs = sorted({round(r["Tg"], 3) for r in rr})
        for k, tg in enumerate(tgs):
            sel = [r for r in rr if abs(r["alpha"] - a) < 1e-9 and abs(r["Tg"] - tg) < 1e-3]
            by_src = {}
            for r in sel:
                by_src.setdefault(r["source"], []).append(r)
            for j, (src, s_rows) in enumerate(sorted(by_src.items())):
                _err(ax, s_rows, "sigma_phi_sq", k, f"$T_g$ = {tg:g} ns" if j == 0 else None,
                     ls="-" if j == 0 else "none")
        ax.axhline(0, color="#333", lw=0.7)
        ax.axvspan(1e-4, 0.05, color="#999", alpha=0.12, lw=0)
        ax.set_xscale("log"); ax.set_xlim(1e-3, 3)
        ax.set_xlabel(r"accumulated phase variance $\sigma_\phi^2 = (\sigma_\delta T_g)^2$")
        ax.set_title(rf"$\alpha$ = {a:g}", fontsize=9)
    axes[0].set_ylabel(r"$R_{\rm tail}$ vs best mean control")
    _finish(fig, axes, 3)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_collapse.{ext}")
    plt.close(fig)


def fig_tg(rows, out):
    groups = [
        ("transmon (4 levels)", lambda r: r["kind"] == "sweep" and r["n_levels"] == 4
         and r["source"] == "results_fast"),
        ("strong search, 4 levels", lambda r: r["kind"] == "sweep" and r["source"] == "results_strong"),
        ("qubit (2 levels)", lambda r: r["kind"].startswith("sweep") and r["n_levels"] == 2
         and r["source"] != "results_strong_2lvl"),
        ("detuning only, 4 levels", lambda r: r["kind"] == "sweep" and r["source"] == "results_det_only"),
        ("amplitude only, 4 levels", lambda r: r["kind"] == "sweep" and r["source"] == "results_amp_only"),
        ("strong search, 2 levels", lambda r: r["source"] == "results_strong_2lvl"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.4), sharey=True)
    for ax, a in zip(axes, (0.05, 0.2)):
        for k, (lab, f) in enumerate(groups):
            sel = [r for r in rows if f(r) and abs(r["alpha"] - a) < 1e-9]
            if sel:
                _err(ax, sel, "Tg", k, lab)
        ax.axhline(0, color="#333", lw=0.7)
        ax.axvline(12.67, color="#666", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([6, 8, 10, 15, 20, 30]); ax.set_xticks([], minor=True)
        ax.set_xticklabels(["6", "8", "10", "15", "20", "30"])
        ax.set_xlabel(r"gate time $T_g$ (ns)")
        ax.set_title(rf"$\alpha$ = {a:g}", fontsize=9)
    axes[0].set_ylabel(r"$R_{\rm tail}$ vs best mean control")
    axes[0].text(12.67, 0.99, " leakage onset", transform=axes[0].get_xaxis_transform(),
                 fontsize=6.5, color="#666", va="top")
    _finish(fig, axes, 3)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_tg_sweep.{ext}")
    plt.close(fig)


def fig_audit(rows, out):
    rr = [r for r in rows if r["kind"] == "audit"]
    if not rr:
        return
    srcs = sorted({r["source"] for r in rr})
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for k, src in enumerate(srcs):
        sel = sorted([r for r in rr if r["source"] == src], key=lambda r: r["alpha"])
        x = np.arange(len(sel)) + (k - (len(srcs) - 1) / 2) * 0.18
        y = np.array([r["R_tail_lindblad"] for r in sel])
        lo = np.array([r["R_lo_lindblad"] for r in sel]); hi = np.array([r["R_hi_lindblad"] for r in sel])
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt=MARKERS[k % 5], color=SERIES[k % 5],
                    capsize=2, elinewidth=0.7, label=src.replace("results_", ""))
        ax.set_xticks(np.arange(len(sel)), [f"{r['alpha']:g}" for r in sel])
    ax.axhline(0, color="#333", lw=0.7)
    ax.set_xlabel(r"CVaR level $\alpha$")
    ax.set_ylabel(r"$R_{\rm tail}$ at $T_g$ = 12.7 ns")
    ax.legend(loc="best")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_audit.{ext}")
    plt.close(fig)


def fig_leak(out, src=Path("results_fast/characterize/leak_vs_Tg.json")):
    """Nominal Hann-pulse leakage vs Tg with the onset and its threshold band."""
    if not src.exists():
        print(f"skip leak figure: no {src}")
        return
    d = json.loads(src.read_text())
    tg = np.array([r["Tg"] for r in d["rows"]]); lk = np.array([r["leak"] for r in d["rows"]])
    bg, on, rob = d["background"], d["Tg_onset"], d["Tg_onset_robustness"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.axvspan(min(rob.values()), max(rob.values()), color="#999", alpha=0.15, lw=0)
    ax.loglog(tg, lk, color=SERIES[0], marker="o", ms=3.5, label="Hann pulse, nominal")
    ax.axhline(bg, color="#888", lw=0.8, ls=":", label="thermal background")
    ax.axhline(d["onset_multiplier"] * bg, color="#444", lw=0.8, ls="--",
               label=rf"{d['onset_multiplier']:g}$\times$ background")
    ax.axvline(on, color="#444", lw=0.8)
    ax.annotate(rf"$T_g^{{\rm on}}$ = {on:.1f} ns", (on, 3e-3), xytext=(6, 0),
                textcoords="offset points", fontsize=7.5)
    ax.set_xlabel(r"gate time $T_g$ (ns)"); ax.set_ylabel(r"leakage $P_{\geq 2}$")
    ax.legend(loc="upper right", fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"leak_vs_Tg.{ext}")
    plt.close(fig)


def appendix_table(rows, out):
    """LaTeX longtable of every validated cell (goes in the paper appendix)."""
    esc = lambda t: t.replace("_", "\\_")
    lines = ["\\begin{longtable}{@{}llrrrrrl@{}}", "\\toprule",
             "Source & Point & $T_g$ & $\\sigma_\\delta/2\\pi$ & $\\alpha$ & "
             "$R_{\\rm tail}$ vs best mean [95\\% CI] & same-budget & comparator\\\\",
             "\\midrule", "\\endhead"]
    for r in sorted(rows, key=lambda r: (r["source"], r["file"], r["alpha"])):
        st = "multistart" if r["consistent"] else "upgraded"
        src = esc(r["source"].replace("results_", ""))
        ab = {"sweep": "Tg", "sweep_2level": "Tg-2lv", "sigma_2level": "sig-2lv", "audit": "audit"}
        kd, nm = r["file"].replace(".json", "").split("/")
        pt = esc(ab.get(kd, kd) + (" " + nm.split("_")[-1] if kd != "audit" else ""))
        cell = "${:+.3f}$ [{:+.3f}, {:+.3f}]".format(r["R"], r["R_lo"], r["R_hi"])
        lines.append("{} & {} & {:.1f} & {:.2f} & {:g} & {} & ${:+.3f}$ & {}\\\\".format(
            src, pt, r["Tg"], r["sigma_d_MHz"], r["alpha"], cell, r["R_tail_lindblad"], st))
    lines += ["\\bottomrule", "\\end{longtable}"]
    (out / "appendix_table.tex").write_text("\n".join(lines))


def main():
    dirs = sys.argv[1:] or sorted(str(p.parent) for p in Path(".").glob("results_*/validate.json"))
    rows = load(dirs)
    out = Path("figures"); out.mkdir(exist_ok=True)
    fig_collapse(rows, out)
    fig_tg(rows, out)
    fig_audit(rows, out)
    fig_leak(out)
    appendix_table(rows, out)
    n_lim = len({(r["source"], r["file"]) for r in rows if not r["consistent"]})
    print(f"{n_lim} points with an upgraded comparator (open markers)")
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "all_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows from {len(dirs)} folders -> {out}/")


if __name__ == "__main__":
    main()