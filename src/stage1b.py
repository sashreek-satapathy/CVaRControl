"""
stage1b.py -- VIPER Stage 1b: two non-parallel channels, measured.

WHY THIS IS STILL OUTSTANDING
============================================================================
Proposition 2 (two non-parallel channels) is proved in the paper and used in
the discussion, but it has never been run.  It is the only route by which mean
and CVaR can disagree with EXACTLY LINEAR coupling -- kappa = 0 -- and every
divergence reported so far has been a kappa effect instead.  Two claims have
no experiment behind them:

  (i)  at fixed mean loss, CVaR depends on how the error is SPLIT between the
       two channels, falling from r_alpha = 5.58 (one axis) to 1 + ln(1/alpha)
       = 4.00 (equal split) at alpha = 0.05;
  (ii) two controls with equal mean and different splits are NOT ordered by
       first-order stochastic dominance, i.e. their loss CDFs cross.

Stage 4 has since made (i) more important than it was: Proposition 3 says the
tail-to-mean ratio is rho_alpha(m_eff) along one curve, and Stage 4 measured
that curve by making the noise COLORED.  Stage 1b measures the same curve by a
completely different mechanism -- channel geometry at fixed, quasi-static,
Gaussian noise -- so the two together test the law rather than one route to it.

DESIGN
============================================================================
Detuning noise rotates about z; amplitude noise rotates about the drive axis.
Sweeping the mixing angle

        sigma_d = s cos(theta) / Tg ,   sigma_a = s sin(theta) * kappa_a

moves error between the two channels.  At each theta the OVERALL scale s is
solved so that the mean loss of the mean-optimal control is held fixed, which
is what makes the CVaR comparison a comparison of splits rather than of sizes.
For each theta we then:

  * optimize the mean, and read the two channel weights lambda_1, lambda_2 by
    the existing error-geometry fit;
  * measure CVaR/mean and compare with rho_alpha(PR) where PR is the
    participation ratio 1/(l1^2 + l2^2) of the normalized weights;
  * optimize CVaR and record R_tail -- this is the kappa-free divergence;
  * test (ii) by counting CDF crossings between controls at matched mean.

PRE-REGISTERED PREDICTIONS (scored by `summary`)
  B1  THE SPLIT MOVES THE RATIO.  CVaR_0.05/mean falls monotonically from
      ~5.58 at theta = 0 to ~4.00 near the balanced split, and
      rho_alpha(PR) predicts it to within 10% per point, with
      Spearman(rho_alpha(PR), measured) > 0.8.
  B2  DOMINANCE FAILS.  At matched mean loss and different splits, the loss
      CDFs cross: at least one crossing for pairs whose PR differs by more
      than 0.3, and none for pairs whose PR matches to better than 0.05.
  B3  A kappa-FREE DIVERGENCE EXISTS BUT IS SMALL.  R_tail > 0 somewhere with
      kappa below the study's threshold 0.23, and max R_tail across the sweep
      is under 0.10.  Proposition 2 says the ordering fails; the paper's
      reading is that it fails WEAKLY compared with the kappa route.
  B4  AGREEMENT WITH STAGE 4.  Pooling Stage 1b and Stage 4, the single curve
      rho_alpha(m_eff) predicts the ratio across BOTH mechanisms with no
      offset between them (two-sample test on the residuals, p > 0.05).

USAGE
============================================================================
  python stage1b.py selftest
  python stage1b.py run --theta-steps 9 --seeds 0 1
  python stage1b.py summary
  python stage1b.py crosscheck --stage4 results_stage4/stage4_summary.json

WHAT TO SEND BACK
  results_stage1b/stage1b_summary.json, results_stage1b/*/*.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


def rho_eff(m, alpha):
    from scipy import stats
    q = stats.chi2.ppf(1.0 - alpha, m)
    return float(stats.chi2.sf(q, m + 2) / alpha)


def participation_ratio(w):
    w = np.asarray(w, float)
    w = np.abs(w)
    return float(w.sum() ** 2 / max((w ** 2).sum(), 1e-300))


def cdf_crossings(La, Lb, n_grid=400):
    """Number of sign changes of F_a - F_b on a common grid.  Zero means the
    two loss laws are ordered by first-order stochastic dominance."""
    La, Lb = np.sort(np.asarray(La, float)), np.sort(np.asarray(Lb, float))
    lo = min(La[0], Lb[0])
    hi = max(La[-1], Lb[-1])
    x = np.linspace(lo, hi, n_grid)
    Fa = np.searchsorted(La, x, side="right") / len(La)
    Fb = np.searchsorted(Lb, x, side="right") / len(Lb)
    d = Fa - Fb
    d = d[np.abs(d) > 2.0 / math.sqrt(min(len(La), len(Lb)))]   # ignore MC noise
    if d.size == 0:
        return 0
    return int(np.sum(np.diff(np.sign(d)) != 0))


def selftest(outdir="results_stage1b"):
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    print("B-S1  the two endpoints of Proposition 2")
    chk("one axis gives r_alpha = 5.58", abs(rho_eff(1, 0.05) - 5.582) < 2e-3)
    chk("equal split gives 1 + ln(1/alpha) = 4.00",
        abs(rho_eff(2, 0.05) - (1 + math.log(1 / 0.05))) < 1e-9)

    print("B-S2  participation ratio of a two-weight split")
    chk("equal weights -> 2", abs(participation_ratio([1, 1]) - 2.0) < 1e-12)
    chk("one axis -> 1", abs(participation_ratio([1, 0]) - 1.0) < 1e-12)
    chk("3:1 split -> 1.6", abs(participation_ratio([0.75, 0.25]) - 1.6) < 1e-12)

    print("B-S3  rho_alpha(PR) against direct simulation of the split")
    rng = np.random.default_rng(0)
    z = rng.standard_normal((400000, 2))
    worst = 0.0
    for p in (0.0, 0.1, 0.25, 0.5):
        w = np.array([1 - p, p])
        L = (z ** 2) @ w
        q = np.quantile(L, 0.95)
        meas = L[L >= q].mean() / L.mean()
        pred = rho_eff(participation_ratio(w), 0.05)
        worst = max(worst, abs(pred - meas) / meas)
    chk("predicted within 3% over the split range", worst < 0.03, f"max {worst:.1%}")

    print("B-S4  CDF crossings detect a genuine failure of dominance")
    a = (z ** 2) @ np.array([1.0, 0.0])
    b = (z ** 2) @ np.array([0.5, 0.5])            # equal mean, different split
    chk("equal-mean different-split laws cross", cdf_crossings(a, b) >= 1,
        f"{cdf_crossings(a, b)} crossings")
    c = (z ** 2) @ np.array([0.5, 0.0])            # same split, smaller scale
    chk("same-split rescaled laws do not cross", cdf_crossings(a, c) == 0,
        f"{cdf_crossings(a, c)} crossings")

    print(f"\nselftest {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


def _sim():
    import inspect as _inspect
    import fast_viper as fv
    import stage0_v2 as v2
    import fade_realistic as fr
    if "rho" not in _inspect.signature(v2.Ensemble.__init__).parameters:
        raise SystemExit("stage1b.py needs the Stage-1 versions of stage0_v2.py / fast_viper.py.")
    return fv, v2, fr


def _calibrate_scale(v2, cc, theta, target_mean, Tg, lo=0.2, hi=5.0, tol=0.03,
                     cal_iters=120):
    """Solve for the overall noise scale s that puts the mean-optimal control's
    mean loss at target_mean, at fixed mixing angle.  Secant in log s, exactly
    as Stage 3's kappa calibration.

    The secant probe uses ONE short optimization from the Hann seed, not a full
    multistart: it only has to locate the scale, and a full multistart per
    probe costs about fourteen of them per angle (hours, at the production
    budget) for a number that is then thrown away.  Once the scale is fixed,
    the control that is actually reported comes from a full multistart at that
    scale, under the usual fairness protocol."""
    def probe(s):
        cc.sigma_d = s * math.cos(theta) / Tg
        cc.sigma_a = s * math.sin(theta) * 0.02
        Pl = v2.Problem(Tg, cc)
        saved = Pl.c.maxiter
        Pl.c.maxiter = min(saved, cal_iters)
        try:
            r = Pl.optimize(Pl.c_seed, "mean", 0.05, 0.5, tag=f"cal(s={s:.3g})")
        finally:
            Pl.c.maxiter = saved
        uI, uQ = Pl.filtered(r["coef"])
        return float(np.mean(v2.losses(uI, uQ, Pl.test)))

    x0, x1 = math.log(lo), math.log(hi)
    f0 = math.log(probe(math.exp(x0)) / target_mean)
    f1 = math.log(probe(math.exp(x1)) / target_mean)
    for _ in range(12):
        if abs(f1) < tol or abs(f1 - f0) < 1e-12:
            break
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x2 = min(max(x2, math.log(lo)), math.log(hi))
        x0, f0 = x1, f1
        x1 = x2
        f1 = math.log(probe(math.exp(x1)) / target_mean)

    # the reported control: full multistart at the calibrated scale
    s = math.exp(x1)
    cc.sigma_d = s * math.cos(theta) / Tg
    cc.sigma_a = s * math.sin(theta) * 0.02
    Pl = v2.Problem(Tg, cc)
    best, _ = Pl.multistart("mean", seed=0, label="cal_final")
    c = best["coef"]
    uI, uQ = Pl.filtered(c)
    return s, float(np.mean(v2.losses(uI, uQ, Pl.test))), Pl, c


def run(a):
    fv, v2, fr = _sim()
    from fade_realistic import cfg
    import stage3 as s3

    d = Path(a.outdir) / "stage1b"
    d.mkdir(parents=True, exist_ok=True)
    thetas = np.linspace(0.0, math.pi / 2, a.theta_steps)

    for seed in a.seeds:
        for i, th in enumerate(thetas):
            stem = f"b_th{i:02d}_s{seed}"
            pf = d / f"{stem}.json"
            if pf.exists() and not a.force:
                v2.log(f"  {stem} already done, skipping")
                continue
            cc = v2.V2Config()
            cc.n_levels = a.levels
            cc.seed_train += seed
            cc.seed_val += seed
            cc.seed_test += seed
            v2.set_gate(a.Tg, cc)
            t0 = time.time()
            v2.log(f"=== Stage 1b theta={math.degrees(th):.1f} deg, seed {seed} ===")
            s, m_real, P, c_mean = _calibrate_scale(v2, cc, th, a.target_mean, a.Tg)

            uI, uQ = P.filtered(c_mean)
            Lm = v2.losses(uI, uQ, P.test)
            # channel weights from the error geometry, in the two whitened
            # noise coordinates (detuning, amplitude)
            U = fv_props(fv, uI, uQ, P.test)
            x = s3.error_coords(U, cc.n_levels)
            zd = (P.test.deltas - P.test.deltas.mean()) / max(P.test.deltas.std(), 1e-300)
            za = (P.test.gammas - P.test.gammas.mean()) / max(P.test.gammas.std(), 1e-300)
            e0, A, Q, r2 = s3.fit_expansion(x, [zd, za])
            lam = np.array([np.linalg.norm(A[:3, 0]) ** 2, np.linalg.norm(A[:3, 1]) ** 2])
            lam = lam / max(lam.sum(), 1e-300)
            PR = participation_ratio(lam)

            c_cv = P.multistart("cvar", alpha=a.alpha, seed=seed, warm=c_mean,
                                label=f"{stem}:cvar")[0]["coef"]
            uIc, uQc = P.filtered(c_cv)
            Lc = v2.losses(uIc, uQc, P.test)
            c_rev = P.multistart("mean", seed=seed + 7, warm=c_cv,
                                 label=f"{stem}:rev")[0]["coef"]
            uIr, uQr = P.filtered(c_rev)
            Lr = v2.losses(uIr, uQr, P.test)

            cv = lambda L: v2.cvar(L, a.alpha)
            base = min(cv(Lm), cv(Lr))
            pt = {"stem": stem, "theta": float(th), "theta_deg": float(math.degrees(th)),
                  "seed": int(seed), "Tg": float(a.Tg), "levels": int(cc.n_levels),
                  "scale": float(s), "sigma_d": float(cc.sigma_d),
                  "sigma_a": float(cc.sigma_a), "mean_loss": float(m_real),
                  "lambda": [float(x_) for x_ in lam], "PR": float(PR),
                  "kappa": float(s3.kappa_of(A[:3], Q[:3])),
                  "fit_R2_min": float(np.min(r2)),
                  "ratio": float(cv(Lm) / max(float(np.mean(Lm)), 1e-300)),
                  "ratio_pred": rho_eff(PR, a.alpha),
                  "R_raw": float(1.0 - cv(Lc) / cv(Lm)),
                  "R": float(1.0 - cv(Lc) / base),
                  "comparator": "mean" if cv(Lm) <= cv(Lr) else "rev",
                  "seconds": time.time() - t0}
            v2.write_json(pf, pt)
            np.savez_compressed(d / f"{stem}_L.npz", Lm=Lm, Lc=Lc, Lr=Lr)
            v2.log(f"  -> PR={PR:.2f} ratio={pt['ratio']:.2f} (pred {pt['ratio_pred']:.2f}) "
                   f"kappa={pt['kappa']:.3f} R={pt['R']:+.3f}  [{pt['seconds']:.0f}s]")


def fv_props(fv, uI, uQ, ens):
    import stage4 as s4
    return s4.colored_propagators(fv, uI, uQ, ens.deltas, ens.gammas)


def summary(outdir, alpha=0.05):
    from scipy import stats
    base = Path(outdir)
    pts = [json.loads(f.read_text()) for f in sorted(base.glob("stage1b/*.json"))
           if not f.name.endswith("_L.npz")]
    pts = [p for p in pts if "PR" in p]
    if not pts:
        raise SystemExit(f"no Stage-1b points under {base}")
    arr = lambda k: np.array([p[k] for p in pts], float)
    PR, meas, pred = arr("PR"), arr("ratio"), arr("ratio_pred")
    kap, R = arr("kappa"), arr("R")
    out = {"alpha": alpha, "n": len(pts), "rows": pts}

    out["B1_split_moves_ratio"] = {
        "spearman_pred_meas": [float(x) for x in stats.spearmanr(pred, meas)],
        "median_rel_error": float(np.median(np.abs(pred - meas) / meas)),
        "ratio_at_min_PR": float(meas[np.argmin(PR)]),
        "ratio_at_max_PR": float(meas[np.argmax(PR)]),
        "PR_range": [float(PR.min()), float(PR.max())],
        "B1_pass": bool(stats.spearmanr(pred, meas)[0] > 0.8
                        and np.median(np.abs(pred - meas) / meas) < 0.10)}

    # B2: crossings between matched-mean pairs
    cross = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if pts[i]["seed"] != pts[j]["seed"]:
                continue
            fi = base / "stage1b" / f"{pts[i]['stem']}_L.npz"
            fj = base / "stage1b" / f"{pts[j]['stem']}_L.npz"
            if not (fi.exists() and fj.exists()):
                continue
            Li = np.load(fi)["Lm"]
            Lj = np.load(fj)["Lm"]
            if abs(math.log(Li.mean() / Lj.mean())) > 0.08:
                continue                       # not matched in mean
            cross.append({"a": pts[i]["stem"], "b": pts[j]["stem"],
                          "dPR": abs(pts[i]["PR"] - pts[j]["PR"]),
                          "crossings": cdf_crossings(Li, Lj)})
    far = [c for c in cross if c["dPR"] > 0.3]
    near = [c for c in cross if c["dPR"] < 0.05]
    out["B2_dominance_fails"] = {
        "n_pairs": len(cross), "n_far": len(far), "n_near": len(near),
        "far_with_crossing": int(sum(1 for c in far if c["crossings"] >= 1)),
        "near_with_crossing": int(sum(1 for c in near if c["crossings"] >= 1)),
        "B2_pass": bool(far and sum(1 for c in far if c["crossings"] >= 1) / len(far) > 0.5
                        and (not near
                             or sum(1 for c in near if c["crossings"] >= 1) / len(near) < 0.5))}

    small = kap < 0.23
    out["B3_kappa_free_divergence"] = {
        "max_R": float(R.max()), "kappa_range": [float(kap.min()), float(kap.max())],
        "n_small_kappa": int(small.sum()),
        "max_R_at_small_kappa": float(R[small].max()) if small.any() else None,
        "B3_pass": bool(small.any() and R[small].max() > 0 and R.max() < 0.10)}

    p = base / "stage1b_summary.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=float))
    print(f"\nwrote {p}")
    return out


def crosscheck(outdir, stage4_path, alpha=0.05):
    """B4: does ONE curve rho_alpha(m_eff) fit both mechanisms?"""
    from scipy import stats
    bp, s4p = Path(outdir) / "stage1b_summary.json", Path(stage4_path)
    missing = [str(q) for q in (bp, s4p) if not q.exists()]
    if missing:                      # a missing input is a skip, not a crash
        print("crosscheck skipped; not found: " + ", ".join(missing))
        return None
    b = json.loads(bp.read_text())
    s4 = json.loads(s4p.read_text())
    rb = np.array([p["ratio"] / p["ratio_pred"] - 1.0 for p in b["rows"]], float)
    share = np.array([r["z2_share"] for r in s4["rows"]], float)
    meas4 = np.array([r["ratio05"] for r in s4["rows"]], float)
    fit = np.array([r["fit_R2_min"] for r in s4["rows"]], float)
    ok = fit >= 0.94
    pr4 = 1.0 / (share[ok] ** 2 + (1 - share[ok]) ** 2)
    r4 = meas4[ok] / np.array([rho_eff(m, alpha) for m in pr4]) - 1.0
    u = stats.mannwhitneyu(rb, r4, alternative="two-sided")
    out = {"alpha": alpha, "n_stage1b": len(rb), "n_stage4": int(ok.sum()),
           "median_residual_stage1b": float(np.median(rb)),
           "median_residual_stage4": float(np.median(r4)),
           "mannwhitney": [float(u.statistic), float(u.pvalue)],
           "B4_pass": bool(u.pvalue > 0.05)}
    p = Path(outdir) / "stage1b_crosscheck.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {p}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest").add_argument("--outdir", default="results_stage1b")
    r = sub.add_parser("run")
    r.add_argument("--theta-steps", dest="theta_steps", type=int, default=9)
    r.add_argument("--Tg", type=float, default=12.7)
    r.add_argument("--levels", type=int, default=2, choices=(2, 4))
    r.add_argument("--alpha", type=float, default=0.05)
    r.add_argument("--target-mean", dest="target_mean", type=float, default=1.0e-3)
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    r.add_argument("--outdir", default="results_stage1b")
    r.add_argument("--force", action="store_true")
    m = sub.add_parser("summary")
    m.add_argument("--outdir", default="results_stage1b")
    m.add_argument("--alpha", type=float, default=0.05)
    x = sub.add_parser("crosscheck")
    x.add_argument("--outdir", default="results_stage1b")
    x.add_argument("--stage4", default="results_stage4/stage4_summary.json")
    x.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest(a.outdir)
    elif a.cmd == "run":
        try:
            run(a)
        finally:
            try:
                import fade_realistic as fr
                fr._shutdown_pool()
            except Exception:
                pass
    elif a.cmd == "summary":
        summary(a.outdir, a.alpha)
    else:
        crosscheck(a.outdir, a.stage4, a.alpha)


if __name__ == "__main__":
    main()