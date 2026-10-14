"""
make_stage1_figures.py -- the mechanism figures, from saved outputs only (no simulation).

    python make_stage1_figures.py results_stage1 results_fast results_2lvl_long results_strong results_strong_2lvl

Each folder needs validate.json; folders that also have stage1_diagnose.json and
stage1_error_clouds.npz (written by `stage1.py diagnose <folder>`) enter the analysis.

For every point, the MEAN control's per-draw error vector eps (from the saved
clouds) is fitted in whitened noise coordinates z (independent unit normals):

        eps  ~  e0  +  A z  +  (second-order terms Q)

  kappa     = |Q| / |A|  -- second-order error relative to the first-order error
              the control leaves.  Theorem 1's linear-coupling hypothesis is
              kappa = 0; a single channel gives kappa ~ sigma_phi.
  offset/q  = along the dominant noise direction, -(e0 . q) / |q|^2 for each
              control: the coherent offset in units of the second-order
              coefficient.  1-D model eps = e0 + q z^2: the mean optimum is
              offset/q = E[z^2] = 1; the CVaR_alpha optimum is 2.76 (alpha =
              0.05) and 1.70 (alpha = 0.2), computed below by Monte Carlo.

  R_model   zero-parameter prediction (two-level points): keep the mean control's
              fitted A and Q, let BOTH objectives choose only the coherent
              offset e0, loss = eps^T W eps with the 4-ket metric W = diag(3,3,2)/16.
              Within this reduced model R depends on the noise only through
              kappa and the normalised geometry of A and Q (Proposition 4).
  independence test: partial Spearman correlations of R_tail with |rho|, noise
              strength and Tg after controlling for kappa or R_model.

Writes figures/fig_kappa, fig_offset, fig_model (pdf|png), stage1_mechanism.csv,
stage1_independence.json
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import spearmanr, rankdata, pearsonr

W_LOSS = np.diag([3.0, 3.0, 2.0]) / 16.0       # 2-level, 4-ket loss metric in the eps frame

TWO_PI = 2 * np.pi
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.6, "lines.markersize": 5,
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5})
GROUPS = [  # label, colour, marker, predicate(folder, file, n_levels)
    ("Stage 0: qubit, detuning only", "#0072B2", "o", lambda f, fi, L: "sigma" in fi and L == 2),
    ("Stage 0: qubit, both channels", "#56B4E9", "s", lambda f, fi, L: "sweep_2level" in fi),
    ("Stage 0: transmon", "#D55E00", "D", lambda f, fi, L: fi.startswith("sweep/") and L == 4),
    ("Stage 1: amplitude width sweep", "#009E73", "^", lambda f, fi, L: "channels" in fi and "rho+0_" in fi),
    ("Stage 1: correlation sweep", "#CC79A7", "v", lambda f, fi, L: "channels" in fi and "rho+0_" not in fi),
]


def cvar(L, a):
    q = np.quantile(L, 1 - a)
    return L[L >= q].mean()


def toy_offsets():
    z2 = np.random.default_rng(0).standard_normal(2_000_000) ** 2
    return {a: minimize_scalar(lambda c: cvar((z2 - c) ** 2, a), bounds=(0, 8), method="bounded").x
            for a in (0.05, 0.2)}


def test_draws(r, stem):
    sd, sa, rho = TWO_PI * r["sigma_d_MHz"] * 1e-3, r["sigma_a"], r["rho_da"]
    seed = r.get("test_seed", 33 + 1000 * (int(stem.rsplit("_s", 1)[1]) if stem.startswith("ch_") else 0))
    rng = np.random.default_rng(seed)
    d, g = rng.normal(0, sd, 1000), rng.normal(0, sa, 1000)
    if rho and sd > 0 and sa > 0:
        g = rho * sa * d / sd + np.sqrt(1 - rho ** 2) * g
    xs = ([d / sd] if sd > 0 else []) + ([g / sa] if sa > 0 else [])
    if len(xs) == 2:                                    # whiten; dominant direction first
        x1, x2 = xs
        zp, zm = (x1 + x2) / np.sqrt(2 * (1 + rho)), (x1 - x2) / np.sqrt(2 * (1 - rho))
        xs = [zp, zm] if rho >= 0 else [zm, zp]
    return xs, abs(rho) >= 0.9 or len(xs) == 1           # 1-D model meaningful?


def _cvar_fast(L, a):
    n = len(L); m = int(round(a * n))
    return np.partition(L, n - m)[n - m:].mean()


def r_model(B, n, alpha=0.05, N=60_000, seed=1):
    """Reduced model: eps = e0 + (A z + Q(z)) with A, Q = fitted B[1:]; each objective
    picks only e0.  Mean optimum is exact (e0 = -E[noise part]); CVaR by Nelder-Mead."""
    z = list(np.random.default_rng(seed).standard_normal((n, N)))
    mon = z + [z[i] * z[j] for i in range(n) for j in range(i, n)]
    V = np.column_stack(mon) @ B[1:]
    loss = lambda e0: np.einsum("ki,ij,kj->k", V + e0, W_LOSS, V + e0)
    e_mean = -V.mean(0)
    res = minimize(lambda e: _cvar_fast(loss(e), alpha), e_mean, method="Nelder-Mead",
                   options={"xatol": 1e-7, "fatol": 1e-12, "maxiter": 4000})
    return 1.0 - res.fun / _cvar_fast(loss(e_mean), alpha)


def _partial(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    res = lambda u: u - np.polyval(np.polyfit(rc, u, 1), rc)
    return pearsonr(res(ra), res(rb))


def fit(eps, xs):
    n = len(xs)
    X = np.column_stack([np.ones(len(eps))] + xs + [xs[i] * xs[j] for i in range(n) for j in range(i, n)])
    B, *_ = np.linalg.lstsq(X, eps, rcond=None)
    return B, n


def main():
    folders = sys.argv[1:] or sorted(str(p.parent) for p in Path(".").glob("results_*/validate.json"))
    toy = toy_offsets()
    R = {}
    for f in folders:
        vf = Path(f) / "validate.json"
        if vf.exists():
            for r in json.loads(vf.read_text())["rows"]:
                ub = r.get("best_mean_source", "mean") != "mean" and r.get("R_tail_vs_best_mean_lindblad") is not None
                R[(Path(f).name, r["file"], r["alpha"])] = (
                    r["R_tail_vs_best_mean_lindblad"] if ub else r["R_tail_lindblad"],
                    r.get("reverse_class_fast") != "artifact_mean_missed_basin",
                    r["R_lo_vs_best_mean"] if ub else r["R_lo_lindblad"])
    rows = []
    for f in folders:
        dj, cz = Path(f) / "stage1_diagnose.json", Path(f) / "stage1_error_clouds.npz"
        if not (dj.exists() and cz.exists()):
            pass   # its points are still used if another folder's diagnose file covers them
            continue
        C = np.load(cz)
        by_pt = {}
        for r in json.loads(dj.read_text())["rows"]:        # one diagnose file may cover several folders
            if "audit" not in r["file"]:
                by_pt.setdefault((r["folder"], r["file"]), {})[r["control"]] = r
        for (src, fi), ctl in by_pt.items():
            if "mean" not in ctl or (src, fi, 0.05) not in R:
                continue
            sub, stem = fi[:-5].split("/")
            xs, one_d = test_draws(ctl["mean"], stem)
            out = {"folder": src, "file": fi, "n_levels": ctl["mean"]["n_levels"],
                   "sigma_d_MHz": ctl["mean"]["sigma_d_MHz"], "sigma_a": ctl["mean"]["sigma_a"],
                   "rho": ctl["mean"]["rho_da"], "R05": R[(src, fi, 0.05)][0],
                   "comparator_multistart": R[(src, fi, 0.05)][1], "R05_lo": R[(src, fi, 0.05)][2],
                   "R20": R.get((src, fi, 0.2), (None,))[0]}
            for k in ("mean", "cvar_0.05", "cvar_0.2"):
                key = f"{ctl['mean']['folder']}__{sub}__{stem}__{k}"
                if key not in C.files:
                    continue
                B, n = fit(C[key].astype(float), xs)
                if k == "mean":
                    out["kappa"] = float(np.linalg.norm(B[1 + n:]) / np.linalg.norm(B[1:1 + n]))
                    out["Tg"] = ctl["mean"]["Tg"]
                    if ctl["mean"]["n_levels"] == 2:
                        out["R_model"] = float(r_model(B, n))
                if one_d:
                    B1, _ = fit(C[key].astype(float), [xs[0]])
                    q = B1[2]
                    out[f"offset_q_{k}"] = float(-B1[0] @ q / (q @ q))
            rows.append(out)
    for g in rows:
        g["group"] = next((lab for lab, _, _, p in GROUPS if p(g["folder"], g["file"], g["n_levels"])), "other")

    out = Path("figures"); out.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "stage1_mechanism.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)

    # ---- fig_kappa: R_tail vs kappa -------------------------------------------------------
    k = np.array([r["kappa"] for r in rows]); y = np.array([r["R05"] for r in rows])
    rs_all = spearmanr(k, y)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for lab, col, mk, _ in GROUPS:
        sel = [r for r in rows if r["group"] == lab]
        for filled in (True, False):
            ss = [r for r in sel if r["comparator_multistart"] == filled]
            if ss:
                ax.scatter([r["kappa"] for r in ss], [max(r["R05"], -0.08) for r in ss], marker=mk,
                           s=26, color=col, facecolor=col if filled else "white", linewidths=0.9,
                           label=lab if filled or not any(r["comparator_multistart"] for r in sel) else None,
                           zorder=3)
    ax.axhline(0, color="#333", lw=0.7)
    ax.axvspan(1e-2, 0.1, color="#999", alpha=0.12, lw=0)
    ax.set_xscale("log"); ax.set_xlim(1e-2, 8); ax.set_ylim(-0.09, None)
    ax.set_xlabel(r"$\kappa$ = |second-order error| / |first-order error| (mean control)")
    ax.set_ylabel(r"$R_{\rm tail}(0.05)$ vs best mean control")
    ax.text(0.02, 0.97, f"Spearman {rs_all.statistic:.2f} (p = {rs_all.pvalue:.0e}), {len(rows)} points",
            transform=ax.transAxes, fontsize=7, va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.9), fontsize=6.5)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_kappa.{ext}")
    plt.close(fig)

    # ---- fig_offset: coherent pre-compensation vs the 1-D model -----------------------------
    # points with a significant divergence (CI above zero) where a 1-D model applies
    sel = [r for r in rows if "offset_q_cvar_0.05" in r and r["R05_lo"] > 0 and r["n_levels"] == 2]
    sel.sort(key=lambda r: (r["group"], r["rho"], r["sigma_d_MHz"]))
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    xs_ = np.arange(len(sel))
    for key, col, mk, lab in (("offset_q_mean", "#555", "o", "mean control"),
                              ("offset_q_cvar_0.2", "#E69F00", "s", r"CVaR, $\alpha$ = 0.2"),
                              ("offset_q_cvar_0.05", "#D55E00", "D", r"CVaR, $\alpha$ = 0.05")):
        ax.scatter(xs_, [r.get(key, np.nan) for r in sel], color=col, marker=mk, s=24, label=lab, zorder=3)
    for val, col, lab in ((1.0, "#555", "1-D model: mean"), (toy[0.2], "#E69F00", None), (toy[0.05], "#D55E00", None)):
        ax.axhline(val, color=col, lw=0.8, ls="--")
    for val, col in ((toy[0.05], "#D55E00"), (toy[0.2], "#E69F00"), (1.0, "#555")):
        ax.text(len(sel) - 0.45, val, f"model {val:.2f}", va="bottom", ha="left", fontsize=6.5, color=col)

    def lab(r):
        if "sigma" in r["file"]:
            return f"det. {r['sigma_d_MHz']:g} MHz"
        if r["sigma_d_MHz"] == 0:
            return "amp. only"
        s = r["file"].rsplit("_s", 1)[1][:-5]
        return rf"$\rho$={r['rho']:+g} (s{s})"
    ax.set_xticks(xs_, [lab(r) for r in sel], rotation=35, ha="right", fontsize=7)
    ax.set_xlim(-0.5, len(sel) + 0.6)
    ax.set_ylabel("coherent offset / second-order coeff.")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_offset.{ext}")
    plt.close(fig)
    # ---- fig_model + independence test (two-level, multistart comparator) --------------------
    q = [r for r in rows if "R_model" in r and r["comparator_multistart"]]
    ym = np.array([r["R05"] for r in q])
    cov = {"kappa": [r["kappa"] for r in q], "R_model": [r["R_model"] for r in q],
           "|rho|": [abs(r["rho"]) for r in q],
           "detuning phase var": [(TWO_PI * r["sigma_d_MHz"] * 1e-3 * r["Tg"]) ** 2 for r in q],
           "amplitude phase var": [(r["sigma_a"] * np.pi / 2) ** 2 for r in q],
           "total phase var": [(TWO_PI * r["sigma_d_MHz"] * 1e-3 * r["Tg"]) ** 2 + (r["sigma_a"] * np.pi / 2) ** 2
                               for r in q],
           "Tg": [r["Tg"] for r in q]}
    cov = {k: np.array(v) for k, v in cov.items()}
    ind = {"n_points": len(q), "marginal": {}, "given_kappa": {}, "given_R_model": {}}
    for k, v in cov.items():
        m = spearmanr(v, ym); ind["marginal"][k] = [float(m.statistic), float(m.pvalue)]
        for c in ("kappa", "R_model"):
            if k not in ("kappa", "R_model"):
                pc = _partial(v, ym, cov[c]); ind[f"given_{c}"][k] = [float(pc.statistic), float(pc.pvalue)]
    for c in ("kappa", "R_model"):
        ind[f"{c}_given_each"] = {k: [float(x) for x in _partial(cov[c], ym, cov[k])]
                                  for k in cov if k not in ("kappa", "R_model")}
    (out / "stage1_independence.json").write_text(json.dumps(ind, indent=1))
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    for glab, col, mk, _ in GROUPS:
        ss = [r for r in q if r["group"] == glab]
        if ss:
            ax.scatter([r["R_model"] for r in ss], [r["R05"] for r in ss], marker=mk, s=26, color=col,
                       label=glab, zorder=3)
    lim = max(0.45, max(cov["R_model"]) * 1.05)
    ax.plot([0, lim], [0, lim], color="#555", lw=0.8, ls="--")
    ax.axhline(0, color="#333", lw=0.7)
    ax.set_xlim(-0.01, lim); ax.set_ylim(-0.06, lim)
    ax.set_xlabel(r"reduced-model prediction $R_{\rm model}(0.05)$")
    ax.set_ylabel(r"measured $R_{\rm tail}(0.05)$, full Lindblad")
    ms = spearmanr(cov["R_model"], ym)
    ax.text(0.03, 0.97, f"Spearman {ms.statistic:.2f}, {len(q)} points\ndashed: measured = model",
            transform=ax.transAxes, fontsize=7, va="top")
    ax.legend(loc="center right", fontsize=6)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_model.{ext}")
    plt.close(fig)
    print("independence test (partial Spearman, R_tail vs covariate):")
    for k in ind["given_kappa"]:
        print(f"  {k:22s} marginal {ind['marginal'][k][0]:+.2f}   | kappa {ind['given_kappa'][k][0]:+.2f} "
              f"(p={ind['given_kappa'][k][1]:.2f})   | R_model {ind['given_R_model'][k][0]:+.2f} "
              f"(p={ind['given_R_model'][k][1]:.2f})")

    print(f"{len(rows)} points; Spearman(kappa, R) = {rs_all.statistic:.2f}, p = {rs_all.pvalue:.1e}; "
          f"1-D model offsets: alpha 0.05 -> {toy[0.05]:.2f}, 0.2 -> {toy[0.2]:.2f}")
    for r in sel:
        print(f"  {lab(r):22s} mean {r.get('offset_q_mean', np.nan):.2f}  cvar0.2 {r.get('offset_q_cvar_0.2', np.nan):.2f}"
              f"  cvar0.05 {r.get('offset_q_cvar_0.05', np.nan):.2f}   R05 {r['R05']:+.3f}")


if __name__ == "__main__":
    main()