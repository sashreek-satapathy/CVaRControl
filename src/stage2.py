"""
stage2.py -- VIPER Stage 2: a direct test of Theorem 1's premise.

THE QUESTION
============================================================================
Theorem 1 does not say "R_tail = 0".  It says something stronger: under
(A1) Gaussian noise and (A2) a phase linear in the noise, the loss
distribution of EVERY control depends on the control only through one
scalar sigma_f, so the whole family of loss distributions is totally ordered
by first-order stochastic dominance, and every monotone risk measure ranks
every pair of controls the same way.  Stage 0 (E4) tested one consequence:
the two optimizers' winners agree at small sigma_phi^2.  Stage 2 tests the
premise itself, on a large pool of candidate controls, and watches it break
as sigma_phi^2 grows:

  T1  RANK AGREEMENT.  Over every candidate the optimizers visit (L-BFGS-B
      iterates of all mean, CVaR and reverse runs), Kendall tau between the
      ranking by mean loss and by CVaR_alpha.  Theorem: tau = 1.
  T2  CONFIRMED RANK REVERSALS.  Fraction of well-separated pairs (u, v)
      where u has the lower mean but v the lower CVaR, confirmed on BOTH
      independent halves of a fresh 4000-draw pool ensemble.  Theorem: 0.
  T3  DOMINANCE VIOLATIONS.  Fraction of pairs whose quantile functions
      cross (u better somewhere, v better elsewhere, by > 2%), again on both
      halves.  Theorem: 0.  This is stricter than T2.
  T4  SHAPE INVARIANCE.  CVaR_alpha / mean for every competitive candidate.
      Theorem + Proposition 1: all equal to 1 + 2 z phi(z)/alpha
      (3.25 at alpha = 0.2, 5.58 at 0.05, 8.45 at 0.01), with zero spread.
  T5  POOL REGRET.  Pick the best-mean and the best-CVaR candidate on one
      half, score both by CVaR on the other half (cross-fitted, so no
      winner's curse): R_pool = 1 - CVaR(best-CVaR)/CVaR(best-mean).
      Theorem: 0 up to noise.

Every point also runs the standard Stage-0 comparison unchanged (same
fairness protocol, JSON + NPZ in the standard layout), so
`fast_viper.py validate` and `stage1.py diagnose` work on the output folder
as they do for every earlier stage (Lindblad R_tail and kappa per point).

USAGE (from the iris folder; needs the Stage-1 versions of stage0_v2.py and
fast_viper.py, which you already have)
============================================================================
  python stage2.py selftest                      # maths only, seconds, no simulator
  bash run_stage2.sh 2>&1 | tee stage2_log.txt   # the whole overnight batch
or step by step:
  python stage2.py run --levels 2 --Tg 8 12.7 20
  python stage2.py run --levels 2 --Tg 12.7 --seed 1
  python stage2.py run --levels 4 --Tg 12.7 20.8
  python fast_viper.py validate --outdir results_stage2
  python stage1.py diagnose results_stage2
  python stage2.py summary

WHAT TO SEND BACK
  results_stage2/stage2_summary.json
  results_stage2/*/*_pool.json   and   results_stage2/*/*_pool.npz   (small)
  results_stage2/validate.json   and   results_stage2/stage1_diagnose.json
  stage2_log.txt

NOT executed on Claude's side (per your preference): `selftest` (pure numpy,
synthetic data) was run; the simulator path is syntax-checked only.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

PHASE_VARS = (0.0016, 0.0064, 0.04, 0.16, 0.41, 1.6)   # = E4's sigma_phi^2 values
ALPHAS_POOL = (0.2, 0.05, 0.01)
GEOM_ITERS = (0, 1, 2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512, 724, 1000)
Q_GRID = np.r_[np.linspace(0.05, 0.95, 19), 0.975, 0.99, 0.995]      # dominance test
Q_FIG = np.r_[np.linspace(0.01, 0.99, 99), 0.995, 0.999]              # figures (CCDFs)
REL_SEP = 0.02            # pairs must differ by > 2% to count (T2, T3)
COMP_FACTOR = 2.0         # competitive = mean loss within 2x of the pool's best
MAX_PAIRS_SET = 300       # cap on candidates entering the pairwise tests


# ============================================================================
# 1. PURE MATHS (no simulator; checked by `selftest`)
# ============================================================================

def cvar_rows(L, alpha):
    """Exact finite-sample Rockafellar-Uryasev CVaR of every row of L (C, n);
    the same estimator as stage0_v2.tail_weights."""
    L = np.atleast_2d(np.asarray(L, float))
    n = L.shape[1]
    t = alpha * n
    m = min(int(math.floor(t)), n)
    S = -np.sort(-L, axis=1)
    out = S[:, :m].sum(axis=1)
    if m < n and t > m:
        out = out + (t - m) * S[:, m]
    return out / t


def c_alpha(alpha):
    """Proposition 1: CVaR_alpha(Z^2) / E[Z^2] = 1 + 2 z phi(z) / alpha."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return float(1 + 2 * z * norm.pdf(z) / alpha)


def theorem_residual(sigma_phi_sq, alpha):
    """r_alpha = P(|phi| > pi) / alpha at phase variance sigma_phi^2 (Theorem 1)."""
    from scipy.stats import norm
    if sigma_phi_sq <= 0:
        return 0.0
    return float(2 * norm.sf(math.pi / math.sqrt(sigma_phi_sq)) / alpha)


def _kendall(a, b):
    from scipy.stats import kendalltau
    if len(a) < 3:
        return float("nan")
    return float(kendalltau(a, b).statistic)


def _pick_set(means, idx, cap):
    """At most `cap` candidates from idx, evenly spaced in mean rank."""
    idx = np.asarray(idx)
    if len(idx) <= cap:
        return idx
    order = idx[np.argsort(means[idx])]
    return order[np.round(np.linspace(0, len(order) - 1, cap)).astype(int)]


def _pairs_upper(x):
    d = x[:, None] - x[None, :]
    iu = np.triu_indices(len(x), 1)
    return d[iu]


def reversal_stats(mA, mB, cA, cB, rel=REL_SEP):
    """T2: confirmed rank reversals between mean (m) and a tail measure (c),
    each computed on the two halves A, B.  A pair counts only if it is
    separated by > rel (relative) in both measures on both halves."""
    out = {}
    dmA, dmB, dcA, dcB = (_pairs_upper(v) for v in (mA, mB, cA, cB))
    sm = (mA[:, None] + mA[None, :])[np.triu_indices(len(mA), 1)] / 2
    sc = (cA[:, None] + cA[None, :])[np.triu_indices(len(cA), 1)] / 2
    sep = ((np.abs(dmA) > rel * sm) & (np.abs(dmB) > rel * sm)
           & (np.abs(dcA) > rel * sc) & (np.abs(dcB) > rel * sc))
    consistent = (np.sign(dmA) == np.sign(dmB)) & (np.sign(dcA) == np.sign(dcB))
    rev = sep & consistent & (np.sign(dmA) != np.sign(dcA))
    n_sep = int((sep & consistent).sum())
    out["n_pairs_separated"] = n_sep
    out["n_reversals"] = int(rev.sum())
    out["reversal_frac"] = float(rev.sum() / n_sep) if n_sep else float("nan")
    return out


def dominance_violation_stats(QA, QB, rel=REL_SEP):
    """T3: pairs whose quantile functions cross by more than rel (relative to
    the pair's mean quantile at that level), on both halves."""
    C = QA.shape[0]
    iu = np.triu_indices(C, 1)
    res = []
    for Q in (QA, QB):
        D = Q[:, None, :] - Q[None, :, :]                   # (C, C, nq)
        S = 0.5 * (Q[:, None, :] + Q[None, :, :])
        thr = rel * np.maximum(S, 1e-300)
        pos = (D > thr).any(axis=2)[iu]
        neg = (D < -thr).any(axis=2)[iu]
        res.append((pos, neg))
    (pA, nA), (pB, nB) = res
    viol = pA & nA & pB & nB
    ordered = (pA ^ nA) & (pB ^ nB) & (pA == pB)            # strictly ordered both halves
    n_eval = int((viol | ordered).sum())
    return {"n_pairs_decided": n_eval, "n_violations": int(viol.sum()),
            "violation_frac": float(viol.sum() / n_eval) if n_eval else float("nan")}


def pool_metrics(L, L0=None, comp_factor=COMP_FACTOR, cap=MAX_PAIRS_SET):
    """All Stage-2 statistics for one point.  L: (C, n) pool losses, C
    candidates on n fresh draws.  L0: nominal (noise-free) loss per candidate."""
    L = np.asarray(L, float)
    A, B = L[:, 0::2], L[:, 1::2]
    means = L.mean(1)
    mA, mB = A.mean(1), B.mean(1)
    comp = np.flatnonzero(means <= comp_factor * means.min())
    S = _pick_set(means, comp, cap)
    QA = np.quantile(A[S], Q_GRID, axis=1).T
    QB = np.quantile(B[S], Q_GRID, axis=1).T
    out = {"n_candidates": int(len(L)), "n_competitive": int(len(comp)), "n_pairs_set": int(len(S)),
           "best_mean": float(means.min()),
           "dominance": dominance_violation_stats(QA, QB), "alpha": {}}
    for a in ALPHAS_POOL:
        cv = cvar_rows(L, a)
        cA, cB = cvar_rows(A, a), cvar_rows(B, a)
        ratio = cv[comp] / means[comp]
        # T5, cross-fitted
        r = []
        for (mX, cX, cY) in ((mA, cA, cB), (mB, cB, cA)):
            i_m, i_c = int(np.argmin(mX)), int(np.argmin(cX))
            r.append(1.0 - cY[i_c] / cY[i_m])
        i_m, i_c = int(np.argmin(means)), int(np.argmin(cv))
        out["alpha"][str(a)] = {
            "kendall_tau_all": _kendall(means, cv),
            "kendall_tau_competitive": _kendall(means[comp], cv[comp]),
            **reversal_stats(mA[S], mB[S], cA[S], cB[S]),
            "ratio_pred": c_alpha(a),
            "ratio_median": float(np.median(ratio)),
            "ratio_q25": float(np.quantile(ratio, 0.25)), "ratio_q75": float(np.quantile(ratio, 0.75)),
            "ratio_min": float(ratio.min()), "ratio_max": float(ratio.max()),
            "R_pool_crossfit": float(np.mean(r)), "R_pool_halves": [float(x) for x in r],
            "R_pool_insample": float(1.0 - cv[i_c] / cv[i_m]),
            "same_winner": bool(i_m == i_c),
        }
    if L0 is not None:
        sh = np.asarray(L0, float)[comp] / means[comp]
        out["nominal_share_median"] = float(np.median(sh))
        out["nominal_share_q25"] = float(np.quantile(sh, 0.25))
        out["nominal_share_q75"] = float(np.quantile(sh, 0.75))
    return out


def selftest():
    """Synthetic checks of every statistic: an exactly ordered family must
    give tau = 1, no reversals, no violations, ratio = c_alpha, R_pool ~ 0;
    a pre-compensation family (z^2 - c)^2 must show all of them."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal(4000)
    ok = True
    # (i) Theorem-1 family: L = sin^2(sigma Z / 2), 80 controls
    sig = np.geomspace(0.02, 0.1, 80)                    # small sigma: Prop. 1 regime
    L = np.sin(np.outer(sig, z) / 2) ** 2
    m = pool_metrics(L, L0=np.zeros(len(sig)), comp_factor=1e9)
    a05 = m["alpha"]["0.05"]
    checks = [("ordered: tau = 1", abs(a05["kendall_tau_all"] - 1) < 1e-12),
              ("ordered: no reversals", a05["n_reversals"] == 0),
              ("ordered: no dominance violations", m["dominance"]["n_violations"] == 0),
              ("ordered: ratio = c_0.05 (5.58) within 5%",
               abs(a05["ratio_median"] / c_alpha(0.05) - 1) < 0.05),
              ("ordered: ratio spread < 1%", (a05["ratio_max"] - a05["ratio_min"]) / a05["ratio_median"] < 0.01),
              ("ordered: R_pool = 0", abs(a05["R_pool_crossfit"]) < 1e-12),
              ("ordered: same winner", a05["same_winner"])]
    # (ii) pre-compensation family: L = (z^2 - c)^2, c in [0, 3.5]
    cs = np.linspace(0, 3.5, 60)
    L2 = (z[None] ** 2 - cs[:, None]) ** 2
    m2 = pool_metrics(L2, comp_factor=1e9)
    b05 = m2["alpha"]["0.05"]
    checks += [("crossing: tau < 0.9", b05["kendall_tau_all"] < 0.9),
               ("crossing: reversals found", b05["n_reversals"] > 0),
               ("crossing: dominance violations found", m2["dominance"]["n_violations"] > 0),
               ("crossing: R_pool > 0.2", b05["R_pool_crossfit"] > 0.2),
               ("crossing: winners differ", not b05["same_winner"])]
    # (iii) estimator matches stage0_v2's exact weights on a hand example
    x = np.arange(1.0, 129.0)
    checks += [("cvar_rows exact on 1..128 at alpha=0.05",
                abs(cvar_rows(x, 0.05)[0] - (sum(range(123, 129)) + 0.4 * 122) / 6.4) < 1e-12),
               ("c_alpha(0.05) = 5.582", abs(c_alpha(0.05) - 5.582) < 1e-3),
               ("residual at sigma_phi^2 = 0.05 ~ 1.5e-43",
                1.4e-43 < theorem_residual(0.05, 0.05) < 1.7e-43)]
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok &= bool(passed)
    print(f"  ordered family:  tau={a05['kendall_tau_all']:.3f}  rev={a05['reversal_frac']:.3f}  "
          f"viol={m['dominance']['violation_frac']:.3f}  ratio={a05['ratio_median']:.3f}  "
          f"R_pool={a05['R_pool_crossfit']:+.3f}")
    print(f"  crossing family: tau={b05['kendall_tau_all']:.3f}  rev={b05['reversal_frac']:.3f}  "
          f"viol={m2['dominance']['violation_frac']:.3f}  R_pool={b05['R_pool_crossfit']:+.3f}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


# ============================================================================
# 2. SIMULATION (imports the simulator only here, so selftest runs anywhere)
# ============================================================================

REC: list[dict] = []


def _install_recorder(v2):
    """Replace Problem.optimize by an identical L-BFGS-B call that also
    records every iterate (same options, same return dict)."""
    from scipy.optimize import minimize

    def optimize(self, c0, mode, alpha, lam, tag=""):
        t0 = time.monotonic()
        path = [np.array(c0, float).copy()]

        def f(x):
            v, g, _ = self.objective(x, mode, alpha, lam)
            return v, g

        v0 = f(c0)[0]
        res = minimize(f, c0, method="L-BFGS-B", jac=True, callback=lambda xk: path.append(xk.copy()),
                       options={"maxiter": self.c.maxiter, "ftol": self.c.ftol, "gtol": self.c.gtol})
        REC.append({"tag": tag, "mode": mode, "alpha": float(alpha), "path": path, "final": res.x.copy()})
        return {"coef": res.x, "obj_start": v0, "obj_end": float(res.fun),
                "nit": int(res.nit), "nfev": int(res.nfev), "success": bool(res.success),
                "message": str(res.message), "seconds": time.monotonic() - t0}

    v2.Problem.optimize = optimize


def _candidates():
    """(label, mode, alpha, iteration, coef) for the recorded runs, thinned to
    the geometric iteration set plus each run's final point; duplicates removed."""
    out, seen = [], set()
    for r in REC:
        n = len(r["path"]) - 1
        its = sorted({i for i in GEOM_ITERS if i <= n} | {n})
        for i in its:
            c = r["final"] if i == n else r["path"][i]
            key = np.round(c, 10).tobytes()
            if key in seen:
                continue
            seen.add(key)
            out.append((r["tag"], r["mode"], r["alpha"], int(i), np.array(c, float)))
    return out


def run(a):
    import fast_viper as fv                 # patches stage0_v2 with the fast simulator
    import stage0_v2 as v2
    import fade_realistic as fr
    from fade_realistic import cfg, TWO_PI
    import inspect as _inspect
    if "rho" not in _inspect.signature(v2.Ensemble.__init__).parameters:
        raise SystemExit("stage2.py needs the Stage-1 versions of stage0_v2.py and fast_viper.py.")
    _install_recorder(v2)

    base = v2.V2Config()
    base.n_levels = a.levels
    base.n_restarts = a.restarts
    base.maxiter = a.maxiter
    base.sigma_a = 0.0                        # detuning only: the setting of Theorem 1
    if a.seed:
        base.seed_train += 1000 * a.seed
        base.seed_val += 1000 * a.seed
        base.seed_test += 1000 * a.seed
    d = Path(a.outdir) / f"stage2_{a.levels}level"
    d.mkdir(parents=True, exist_ok=True)
    grid = [(Tg, pv) for Tg in a.Tg for pv in a.phase_var]
    try:
        for i, (Tg, pv) in enumerate(grid):
            stem = f"pt_Tg{Tg:g}_pv{pv:g}_s{a.seed}"
            if (d / f"{stem}_pool.json").exists():
                v2.log(f"  {stem} already done, skipping")
                continue
            sd = math.sqrt(pv) / Tg                                   # rad/ns
            cc = v2.V2Config(**asdict(base))
            cc.sigma_d = float(sd)
            v2.log(f"=== Stage 2 point {i+1}/{len(grid)}: Tg={Tg} ns, sigma_phi^2={pv}, "
                   f"sigma_d={sd / TWO_PI * 1e3:.3f} MHz, levels={a.levels}, seed={a.seed} ===")
            t0 = time.monotonic()
            REC.clear()
            P = v2.Problem(Tg, cc)
            pt, controls, test_L = v2.compare_mean_vs_cvar(P, cc.alpha_sweep,
                                                           20000 + 100 * i + 10000 * a.seed, stem)
            pt.update(sigma_d_MHz=float(sd / TWO_PI * 1e3), sigma_phi_sq=float(pv), stage="2-dominance")
            v2.write_json(d / f"{stem}.json", pt)
            v2.save_arrays(d / f"{stem}.npz", P, controls, test_L)
            t_opt = time.monotonic() - t0

            # ---- the pool: every thinned iterate, on a FRESH ensemble ----
            pool = v2.Ensemble(a.n_pool, cc.sigma_d, 0.0, 44 + 1000 * a.seed, "pool")
            cands = _candidates()
            Lp = np.empty((len(cands), a.n_pool))
            L0 = np.empty(len(cands))
            def pool_losses(uI_f, uQ_f):          # chunked: keeps memory low at 4 levels
                return np.concatenate([1.0 - fv.fast_gather(uI_f, uQ_f, ch, np.zeros_like(ch))[0]
                                       for ch in np.array_split(pool.deltas, max(1, a.n_pool // 1000))])

            for j, (_, _, _, _, coef) in enumerate(cands):
                uI_f, uQ_f = P.filtered(coef)
                Lp[j] = pool_losses(uI_f, uQ_f)
                L0[j] = 1.0 - fv.fast_gather(uI_f, uQ_f, np.zeros(1), np.zeros(1))[0][0]
            met = pool_metrics(Lp, L0)
            # where do the two pipeline winners sit in the pool?
            win = {}
            for k in ["mean"] + [f"cvar_{x}" for x in cc.alpha_sweep]:
                uI_f, uQ_f = P.filtered(controls[k])
                Lw = pool_losses(uI_f, uQ_f)
                win[k] = {"mean": float(Lw.mean()),
                          **{f"cvar_{al}": float(cvar_rows(Lw, al)[0]) for al in ALPHAS_POOL}}
            summ = {
                "stem": stem, "Tg": float(Tg), "n_levels": a.levels, "seed": a.seed,
                "sigma_phi_sq": float(pv), "sigma_d_MHz": float(sd / TWO_PI * 1e3),
                "r_alpha_theorem_0.05": theorem_residual(pv, 0.05),
                "n_pool": a.n_pool, "pool_seed": pool.seed, "restarts": a.restarts,
                "R_tail_fast": {str(x["alpha"]): x["R_tail"] for x in pt["cells"]},
                "R_tail_fast_CI": {str(x["alpha"]): [x["R_lo"], x["R_hi"]] for x in pt["cells"]},
                "reverse_class": {str(x["alpha"]): x.get("reverse_class") for x in pt["cells"]},
                "winners_on_pool": win, "metrics": met,
                "seconds_opt": t_opt, "seconds_pool": time.monotonic() - t0 - t_opt,
                "candidates": [{"run": c[0], "mode": c[1], "alpha": c[2], "iter": c[3],
                                "mean": float(Lp[j].mean()), "L0": float(L0[j]),
                                **{f"cvar_{al}": float(cvar_rows(Lp[j], al)[0]) for al in ALPHAS_POOL}}
                               for j, c in enumerate(cands)],
            }
            v2.write_json(d / f"{stem}_pool.json", summ)
            np.savez_compressed(d / f"{stem}_pool.npz", q_levels=Q_FIG,
                                quantiles=np.quantile(Lp, Q_FIG, axis=1).T.astype(np.float32),
                                means=Lp.mean(1), L0=L0)
            if a.save_raw:
                np.savez_compressed(d / f"{stem}_pool_raw.npz", L=Lp.astype(np.float32))
            m5 = met["alpha"]["0.05"]
            v2.log(f"  -> {stem}: {met['n_candidates']} candidates ({met['n_competitive']} competitive) | "
                   f"tau(mean,CVaR05)={m5['kendall_tau_competitive']:+.3f}  "
                   f"reversals={m5['n_reversals']}/{m5['n_pairs_separated']}  "
                   f"dominance viol={met['dominance']['n_violations']}/{met['dominance']['n_pairs_decided']}  "
                   f"CVaR05/mean={m5['ratio_median']:.2f} (pred {m5['ratio_pred']:.2f}, "
                   f"IQR {m5['ratio_q25']:.2f}-{m5['ratio_q75']:.2f})  R_pool={m5['R_pool_crossfit']:+.3f}  "
                   f"R_tail(fast)={summ['R_tail_fast']['0.05']:+.3f}  "
                   f"[{summ['seconds_opt']:.0f}+{summ['seconds_pool']:.0f}s]")
    finally:
        fr._shutdown_pool()


def summary(outdir):
    rows = []
    for pf in sorted(Path(outdir).glob("*/*_pool.json")):
        s = json.loads(pf.read_text())
        met = s["metrics"]
        s["file"] = f"{pf.parent.name}/{s['stem']}.json"      # unique across 2- and 4-level folders
        r = {k: s[k] for k in ("file", "stem", "Tg", "n_levels", "seed", "sigma_phi_sq", "sigma_d_MHz",
                               "r_alpha_theorem_0.05")}
        r.update(n_candidates=met["n_candidates"], n_competitive=met["n_competitive"],
                 dominance_violation_frac=met["dominance"]["violation_frac"],
                 dominance_pairs=met["dominance"]["n_pairs_decided"],
                 nominal_share_median=met.get("nominal_share_median"))
        for al, m in met["alpha"].items():
            for k in ("kendall_tau_competitive", "kendall_tau_all", "reversal_frac", "n_pairs_separated",
                      "ratio_median", "ratio_q25", "ratio_q75", "ratio_pred", "R_pool_crossfit", "same_winner"):
                r[f"{k}@{al}"] = m[k]
        for al, v in s["R_tail_fast"].items():
            r[f"R_tail_fast@{al}"] = v
        rows.append(r)
    vf = Path(outdir) / "validate.json"
    if vf.exists():                               # attach full-Lindblad R_tail vs best mean
        vr = json.loads(vf.read_text())["rows"]
        for r in rows:
            for x in vr:
                if x["file"] == r["file"]:
                    r[f"R_tail_lindblad@{x['alpha']}"] = x["R_tail_vs_best_mean_lindblad"]
                    r[f"R_tail_lindblad_CI@{x['alpha']}"] = [x["R_lo_vs_best_mean"], x["R_hi_vs_best_mean"]]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "stage2_summary.json").write_text(json.dumps({"rows": rows}, indent=1, default=float))
    keys = sorted({k for r in rows for k in r})
    with open(Path(outdir) / "stage2_summary.csv", "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"{'point':34s} {'tau05':>6s} {'rev05':>6s} {'viol':>6s} {'ratio05':>8s} {'Rpool05':>8s} {'Rtail':>7s}")
    for r in rows:
        print(f"{r['file'][7:]:40s} {r['kendall_tau_competitive@0.05']:+6.3f} {r['reversal_frac@0.05']:6.3f} "
              f"{r['dominance_violation_frac']:6.3f} {r['ratio_median@0.05']:8.2f} "
              f"{r['R_pool_crossfit@0.05']:+8.3f} {r.get('R_tail_lindblad@0.05', r['R_tail_fast@0.05']):+7.3f}")
    print(f"-> {Path(outdir) / 'stage2_summary.json'} ({len(rows)} points)")


def main():
    ap = argparse.ArgumentParser(description="VIPER Stage 2: direct test of Theorem 1's dominance premise")
    sp = ap.add_subparsers(dest="step", required=True)
    sp.add_parser("selftest")
    r = sp.add_parser("run")
    r.add_argument("--levels", type=int, choices=[2, 3, 4], default=2)
    r.add_argument("--Tg", type=float, nargs="+", default=[12.7])
    r.add_argument("--phase-var", type=float, nargs="+", default=list(PHASE_VARS))
    r.add_argument("--restarts", type=int, default=8)
    r.add_argument("--maxiter", type=int, default=500)
    r.add_argument("--n-pool", type=int, default=4000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--save-raw", action="store_true", help="also keep all pool losses (large)")
    r.add_argument("--outdir", default="results_stage2")
    s = sp.add_parser("summary")
    s.add_argument("--outdir", default="results_stage2")
    a = ap.parse_args()
    if a.step == "selftest":
        raise SystemExit(0 if selftest() else 1)
    if a.step == "summary":
        summary(a.outdir)
    else:
        run(a)


if __name__ == "__main__":
    main()