"""
stage6.py -- Stage 6: the minimal predictive boundary.

THE QUESTION
    Standing at the mean-optimal control, what can be MEASURED that says
    whether running CVaR will select a different pulse?

THE ANSWER THIS STAGE TESTS
    Everything is computed from ONE measured record on the mean-optimal
    control: the per-run error-coordinate vectors eps_k (stage3.error_coords)
    and their losses L_k.  No adjoint, no simulator, no CVaR run.  Stage 7 is
    hardware, so an estimator that needs pulse gradients is not an estimator.

    Three nested statistics are computed and ALL THREE are reported, because
    they differ by amounts that matter and hiding that would misrepresent what
    is being validated:

      R_simple  = (m_c/alpha) ||P_B d_alpha||^2_W / CVaR_alpha
                  the leading-order form.  Legible -- it is Theorem 1 as a
                  measurement, since d = 0 is exactly the theorem's conclusion
                  -- but it assumes the CVaR Hessian in the offset is 2W, and
                  selftest S3 measures its error at +315% at alpha = 0.05.
                  Reported as a DIAGNOSTIC, never as the prediction.

      R_quadratic = (1/2) g^T H^-1 g / CVaR_alpha,  g = 2 (m_c/alpha) W P_B d,
                  H = (m_c/alpha) 2W + (4 rho_tau/alpha) W Sigma_tau W
                  The Hessian's second term is the boundary effect: moving the
                  offset also moves WHICH draws are in the tail, and the
                  Rockafellar-Uryasev tau-minimization turns the conditional
                  second moment into the conditional COVARIANCE.  This is the
                  same Hessian stage3.C_closed derives analytically from a
                  fitted (A, Q); here it is read off the draws, which is what
                  makes it available on hardware.  Measured error: 16%, 77%,
                  47% at kappa = 0.1, 0.25, 0.5 -- it over-predicts once the
                  gain saturates, as a second-order expansion must.

      R_pred    = exact cross-fitted line search along the predicted direction,
                  evaluated on the SAME record.  Because eps_k is measured per
                  run, what the loss would have been under a different coherent
                  offset is arithmetic, not a new experiment:
                      L_k(t) = (eps_k + t s)^T W (eps_k + t s)
                  with the tail re-selected at each t, which makes the boundary
                  effect exact rather than second-order.  t is chosen on one
                  half of the record and scored on the other, so the choice is
                  not also the score.  Measured error: 0.6%, 3.3%, 4.4% at
                  kappa = 0.1, 0.25, 0.5.  THIS is the primary statistic.

    P_B is the projector onto the subspace of error-coordinate space the pulse
    can actually reach, measured as the range of d eps / d c.  Without it the
    predictor assumes every measured error direction is controllable.  For a
    two-level gate it comes out as the identity (a 34-coefficient pulse spans a
    3-dimensional error vector), which is a check rather than an assumption; it
    stops being the identity in leakage-aware coordinates.

    m_c is the tail mass carried by branches whose loss responds to the control
    at all -- relaxation branches do not.

WHY NOT A NEWTON DECREMENT IN THE PULSE
    Because H^-1 in the 34-dimensional coefficient space needs pulse gradients
    and a Hessian, which hardware cannot supply.  In the error coordinates the
    loss is exactly quadratic, so the curvature is a 3x3 (or 11x11) object
    estimated from a conditional covariance -- a measurement, not a derivative.

PRE-REGISTERED CLAIMS (estimators fixed here, BEFORE thresholds; this is the
correction to Stages 4 and 5, where three tests failed because the ESTIMATOR
was mis-specified while only the threshold had been registered)

  H1  R_quadratic == the inner-optimization predictor stage3.r_model, where
      both are computable.  This is the claim that the closed form is the same
      object the paper already validated, not a new model.
      ESTIMATOR: median |R_quadratic/R_model - 1| over points with
      fit_R2 >= 0.94.  PREDICT < 0.15.
  H2  DIRECTION.  Where the gain is resolved, the measured displacement of the
      coherent offset, e0(u_cvar) - e0(u_mean), is antiparallel to the
      predicted step.
      ESTIMATOR: cos in the W metric; PREDICT min cos > 0.9 over resolved
      points.  NOTE ON POWER: for a two-level gate the error vector is
      3-dimensional, where cos is uniform on [-1,1] under the null, so ONE
      point at cos > 0.9 carries p = 0.05 one-sided and is weak.  Power comes
      from combining points (Fisher, reported) and from the 11-coordinate
      leakage-aware arm.  The earlier draft of this file claimed p < 0.02 for a
      single 3-D agreement; that was wrong and is withdrawn.
  H3  LEVEL.  R_tail = R_pred with no fitted constant.
      ESTIMATOR: through-origin slope of R_tail on R_pred over points with
      R_pred > R_min; PREDICT slope in [0.7, 1.3].  Reported beside the
      replicate-noise attenuation ceiling, so a small correlation is never read
      as a tight null (the Stage 4 / Stage 5 lesson).
  H4  BOUNDARY, POINTWISE.  At every point with R_pred < R_min, |R_tail| <
      2 R_min AT THAT POINT, and the sign of R_tail is random across them.
      ESTIMATOR: pointwise comparison against each point's own floor, plus a
      two-sided binomial test on sign(R_tail); PREDICT p > 0.05.  This is the
      claim that explains the negative R_tail values of Stages 4 and 5 rather
      than apologizing for them.
  H5  LEVERAGE CROSSOVER.  A rare sub-population that dominates the tail
      suppresses the gain IN PROPORTION TO ITS SHARE, whether or not the pulse
      could shape it.  Swept on a single two-state fluctuator's amplitude, with
      the fluctuator actually installed in the simulator.
      ESTIMATOR: Lambda_alpha = bulk share of the tail loss; PREDICT R_pred
      falls monotonically with Lambda across the sweep, recovering the
      fluctuator-free gain at Lambda -> 1 and falling below R_min at
      Lambda < 0.1.  Stage 5's relaxation arm is the Lambda -> 0 endpoint of
      the same curve, which is the sense in which relaxation was never special.
  H6  BOUNDARY LOCATION.  kappa_c(alpha) = sqrt(R_min(alpha)/C_alpha) locates
      the crossing to within one grid step, AT EVERY ALPHA, with C_alpha
      evaluated at that alpha (evaluating it once at 0.05 and reusing it, as
      the first draft did, silently flattens the alpha dependence).
      Explicitly allows a smooth crossover: the estimator is where two measured
      curves cross, not a fitted step.

USAGE
    python stage6.py selftest
    python stage6.py probe   --Tg 20 --outdir results_stage6
    python stage6.py run     --Tg 20 --outdir results_stage6
    python stage6.py rtn     --Tg 20 --outdir results_stage6
    python stage6.py retro   --stage results_stage5 --outdir results_stage6
    python stage6.py summary --outdir results_stage6

Pulse coefficients ARE saved (stage4.py and stage5.py did not save them,
which is why their re-analysis is not free).  Nothing here re-runs an
optimization that a saved coefficient vector would have made unnecessary.
"""
from __future__ import annotations

import argparse, itertools, json, math, sys, time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# ===================================================================== #
# 1.  PURE MATHS -- no simulator, exercised by selftest                  #
# ===================================================================== #

ALPHA_GRID = (0.20, 0.10, 0.05, 0.025, 0.01)
COS_MIN = 0.9            # H2 threshold
# H3's original threshold: R_pred should predict R at unit level.  It does not.
# Once the baseline is fixed by `polish`, the slope is 0.59 +- 0.03 across four
# alpha levels and two seeds with rank correlation 0.85-0.97, i.e. the predictor
# is a near-perfect ORDERING with a constant multiplicative offset.  The band is
# kept as written so H3 fails honestly, and CALIBRATION_BAND records the claim
# the data actually supports: a single constant, stable across alpha and kappa.
SLOPE_BAND = (0.7, 1.3)  # H3 threshold, as originally stated
CALIBRATION_BAND = (0.45, 0.75)   # the constant the data supports
FLOOR_K = 1.0            # R_min is FLOOR_K standard deviations of the null


def r_alpha(alpha):
    """CVaR/mean for a one-directional chi-square loss (Prop. 1)."""
    from scipy import stats
    q = stats.chi2.ppf(1.0 - alpha, 1)
    return float(stats.chi2.sf(q, 3) / alpha)


def rho_eff(m, alpha):
    """CVaR/mean for m equal directions (Prop. 3)."""
    from scipy import stats
    q = stats.chi2.ppf(1.0 - alpha, m)
    return float(stats.chi2.sf(q, m + 2.0) / alpha)


def tail_weights(L, w=None, alpha=0.05, uniform=None):
    """Weights t_k >= 0 summing to alpha that select the worst alpha of the
    mass of the (possibly atomic) loss distribution {(L_k, w_k)}.

    The boundary atom is SPLIT, which is what makes CVaR exact rather than
    approximate on an atomic distribution:  CVaR = sum(t L)/alpha.  Every
    conditional average in this module uses these same weights, so the tail
    set is identical across CVaR, the displacement and the leverage.
    """
    L = np.asarray(L, float).ravel()
    w = np.full(L.size, 1.0 / L.size) if w is None else np.asarray(w, float).ravel()
    if L.size != w.size:
        raise ValueError("L and w must have the same length")
    K = L.size
    uni = uniform if uniform is not None else bool(np.ptp(w) <= 1e-15)
    if uni:
        # O(K) partition instead of an O(K log K) sort.  The line search calls
        # this hundreds of times per point, so the sort dominated everything.
        pw = float(w[0])
        nfull = int(math.floor(alpha / pw + 1e-12))
        nfull = min(nfull, K)
        t = np.zeros(K)
        if nfull > 0:
            idx = np.argpartition(-L, nfull - 1)[:nfull]
            t[idx] = pw
            if nfull < K:
                rest = np.argpartition(-L, nfull - 1)[nfull:]
                t[rest[np.argmax(L[rest])]] = max(alpha - nfull * pw, 0.0)
        else:
            t[int(np.argmax(L))] = alpha
        sm = t.sum()
        if sm > 0:
            t *= alpha / sm
        return t
    o = np.argsort(-L)                      # worst first
    cw = np.cumsum(w[o])
    t = np.zeros_like(w)
    full = cw <= alpha
    t[o[full]] = w[o[full]]
    j = int(full.sum())
    if j < K:                               # the atom straddling the boundary
        t[o[j]] = max(alpha - (cw[j - 1] if j else 0.0), 0.0)
    sm = t.sum()
    if sm > 0:                              # guard against w not summing to 1
        t *= alpha / sm
    return t


def cvar_from_weights(L, t, alpha):
    return float(np.dot(t, np.asarray(L, float).ravel()) / alpha)


def tail_displacement(eps, L, w=None, alpha=0.05, mask=None):
    """d = E[eps | alpha-tail], with the debiasing terms the estimator needs.

    eps  (K, n) error coordinates, L (K,) losses, w (K,) probabilities.
    mask (K,) bool: which branches RESPOND to the control (default all).

    Returns dict with
      d        (n,)   the tail-conditional mean over responsive branches
      m_c      float  tail mass carried by responsive branches
      cov      (n,n)  tail-conditional covariance of eps (for the debias)
      n_eff    float  effective number of tail samples, (sum t)^2/sum t^2
    """
    eps = np.atleast_2d(np.asarray(eps, float))
    L = np.asarray(L, float).ravel()
    t = tail_weights(L, w, alpha)
    if mask is not None:
        t = t * np.asarray(mask, bool)
    m_c = float(t.sum())
    if m_c <= 0:
        n = eps.shape[1]
        return {"d": np.zeros(n), "m_c": 0.0, "cov": np.zeros((n, n)), "n_eff": 0.0}
    p = t / m_c
    d = p @ eps
    r = eps - d
    cov = (r * p[:, None]).T @ r / max(1.0 - float((p ** 2).sum()), 1e-12)
    n_eff = float(m_c ** 2 / max(float((t ** 2).sum()), 1e-300))
    return {"d": d, "m_c": m_c, "cov": cov, "n_eff": n_eff}


def _wnorm2(v, W):
    v = np.asarray(v, float)
    return float(v @ np.asarray(W, float) @ v)


def boundary_curvature(eps, L, W, w=None, alpha=0.05, band=0.25, mask=None):
    """The CVaR Hessian in the offset, as a SECOND measurement on the same runs.

        H = (m_c/alpha) 2W + (4 rho_tau/alpha) W Sigma_tau W

    The first term is the naive one (the loss is quadratic in the offset).  The
    second is the boundary term: moving the offset also moves which draws are
    IN the tail, and the tau-minimization of the Rockafellar-Uryasev form turns
    the conditional second moment into the conditional COVARIANCE by a Schur
    complement.  This is the same Hessian stage3.C_closed derives analytically
    from a fitted (A, Q); here it is read off the draws instead, which is what
    makes it available on hardware.

    Dropping the boundary term overestimates the gain by 3-5x at alpha = 0.05
    -- selftest S3 exists to catch exactly that, and did.

    rho_tau (the density of L at VaR) and Sigma_tau (the conditional covariance
    of eps there) are both estimated on a band of relative width `band` in
    tail mass around the VaR level.
    """
    eps = np.atleast_2d(np.asarray(eps, float))
    L = np.asarray(L, float).ravel()
    W = np.asarray(W, float)
    n = eps.shape[1]
    w = np.full(L.size, 1.0 / L.size) if w is None else np.asarray(w, float).ravel()
    if mask is not None:
        keep = np.asarray(mask, bool)
    else:
        keep = np.ones(L.size, bool)
    t = tail_weights(L, w, alpha)
    if mask is not None:
        t = t * keep
    m_c = float(t.sum())
    H0 = (m_c / alpha) * 2.0 * W
    # locate VaR and a band around it, by mass
    o = np.argsort(-L)
    cw = np.cumsum(w[o])
    lo, hi = alpha * (1.0 - band), alpha * (1.0 + band)
    sel = o[(cw >= lo) & (cw <= hi)]
    sel = sel[keep[sel]]
    need = max(8 * n, 40)
    b = band
    while sel.size < need and b < 0.9:          # widen rather than silently drop
        b = min(b * 1.5, 0.9)
        lo, hi = alpha * (1.0 - b), alpha * (1.0 + b)
        sel = o[(cw >= lo) & (cw <= hi)]
        sel = sel[keep[sel]]
    if sel.size < max(8, n + 2):
        # Too few draws to estimate the curvature at all.  Falling back to
        # H = 2W would over-predict by 3-5x (selftest S3), so say so loudly
        # instead: the caller must raise the ensemble size.
        return {"H": H0, "rho_tau": 0.0, "Sigma_tau": np.zeros((n, n)),
                "n_band": int(sel.size), "boundary_frac": 0.0, "band_used": b,
                "underdetermined": True}
    ws = w[sel] / max(w[sel].sum(), 1e-300)
    mu = ws @ eps[sel]
    r = eps[sel] - mu
    Sig = (r * ws[:, None]).T @ r
    dL = float(L[sel].max() - L[sel].min())
    rho = float(w[sel].sum() / dL) if dL > 0 else 0.0
    Hb = (4.0 * rho / alpha) * (W @ Sig @ W)
    H = H0 + Hb
    return {"H": H, "rho_tau": rho, "Sigma_tau": Sig, "n_band": int(sel.size),
            "boundary_frac": float(np.trace(Hb) / max(np.trace(H), 1e-300)),
            "band_used": float(b), "underdetermined": False}


def predicted_gain(eps, L, W, w=None, alpha=0.05, mask=None, cvar=None,
                   debias=True, band=0.25, naive=False, line_search=True,
                   P_B=None):
    """The predicted relative gain from re-optimizing the coherent offset.

        g    = 2 (m_c/alpha) W d          d = E[eps | tail, responsive]
        H    = boundary_curvature(...)
        drop = (1/2) g^T H^{-1} g
        R_pred = drop / CVaR_alpha

    `naive=True` keeps H = 2W, which is what the first draft of this module
    did and what selftest S3 rejected; it is retained so the summary can report
    how large the boundary correction is rather than hiding it.

    The squared form is biased HIGH by tr(B Sigma)/n_eff, because a sample mean
    of a zero-mean vector still has positive squared norm.  Not removing that
    would manufacture a gain exactly in the equivalence regime, which is the
    regime this whole stage is about.
    """
    L = np.asarray(L, float).ravel()
    W = np.asarray(W, float)
    t = tail_weights(L, w, alpha)
    cv = cvar_from_weights(L, t, alpha) if cvar is None else float(cvar)
    td = tail_displacement(eps, L, w, alpha, mask)
    f = td["m_c"] / alpha

    if naive:
        H = 2.0 * f * W
        bc = {"rho_tau": 0.0, "n_band": 0, "boundary_frac": 0.0,
              "underdetermined": False, "band_used": 0.0}
    else:
        bc = boundary_curvature(eps, L, W, w, alpha, band, mask)
        H = bc["H"]
    # drop = 1/2 g^T H^-1 g  with g = 2 f W d   =>   drop = d^T B d,
    # B = 2 f^2 W H^-1 W  (symmetrized against round-off)
    try:
        Hi = np.linalg.pinv(H, rcond=1e-10)
    except np.linalg.LinAlgError:
        Hi = np.linalg.pinv(H + 1e-12 * np.eye(H.shape[0]))
    B = 2.0 * f * f * (W @ Hi @ W)
    B = 0.5 * (B + B.T)
    d_meas = td["d"]
    # Only the reachable part of the tail displacement can be compensated.
    d = d_meas if P_B is None else np.asarray(P_B, float) @ d_meas
    raw = float(d @ B @ d)
    bias = float(np.trace(B @ td["cov"]) / max(td["n_eff"], 1.0)) if debias else 0.0
    drop = max(raw - bias, 0.0)
    R_quad = min(drop / max(cv, 1e-300), 1.0)
    out = {"R_pred_quadratic": float(R_quad),
           "R_pred_raw": float(min(raw / max(cv, 1e-300), 1.0)),
           "d": d, "d_measured": d_meas, "d_norm2_W": _wnorm2(d, W),
           "d_norm2_W_unprojected": _wnorm2(d_meas, W),
           "reach_frac": (1.0 if P_B is None else
                          float(_wnorm2(d, W) / max(_wnorm2(d_meas, W), 1e-300))),
           "bias": float(bias / max(cv, 1e-300)),
           "m_c_over_alpha": float(f), "n_eff": td["n_eff"], "cvar": cv,
           "cov": td["cov"], "B": B, "H": H,
           "rho_tau": float(bc["rho_tau"]), "n_band": int(bc["n_band"]),
           "boundary_frac": float(bc["boundary_frac"]),
           "underdetermined": bool(bc.get("underdetermined", False)),
           # the displacement CVaR is predicted to make: -H^-1 g
           "step": (-2.0 * f * (Hi @ (W @ d)) if P_B is None else
                    np.asarray(P_B, float) @ (-2.0 * f * (Hi @ (W @ d))))}
    # exact line search along that direction, on the same draws
    if line_search:
        r = refine_along_step(eps, L, W, out["step"], w, alpha, mask)
        out["line_search"] = {k2: v for k2, v in r.items() if k2 != "cvar0"}
        gain = r.get("gain_cv", r["gain"])
        out["R_pred"] = float(max(min(gain, 1.0), 0.0))
    else:
        out["line_search"] = None
        out["R_pred"] = float(R_quad)
    return out


def refine_along_step(eps, L, W, step, w=None, alpha=0.05, mask=None,
                      t_grid=None, cross_fit=True, seed=0):
    """Exact one-dimensional search along the predicted direction, using ONLY
    the runs already in hand.

    The quadratic prediction is a second-order expansion, so it over-predicts
    once the gain saturates at the pre-compensation limit (selftest S3 shows
    77% over-prediction at kappa = 0.5).  But the error vector is measured per
    run, so what the loss WOULD have been under a different coherent offset is
    computable without re-running anything:

        L_k(t) = (eps_k + t s)^T W (eps_k + t s)   for responsive branches
                 L_k                               for inert ones

    and the tail set is re-selected at each t, which makes the boundary effect
    exact rather than second-order.  One scan over data already collected --
    on hardware, a post-processing step on the same tomography record.
    """
    eps = np.atleast_2d(np.asarray(eps, float))
    L = np.asarray(L, float).ravel()
    W = np.asarray(W, float)
    s = np.asarray(step, float)
    ns = math.sqrt(max(_wnorm2(s, W), 0.0))
    if ns <= 0 or not np.all(np.isfinite(s)):
        return {"t_best": 0.0, "gain": 0.0, "cvar0": cvar_from_weights(
            L, tail_weights(L, w, alpha), alpha), "n_grid": 0}
    resp = np.ones(L.size, bool) if mask is None else np.asarray(mask, bool)
    cv0 = cvar_from_weights(L, tail_weights(L, w, alpha), alpha)
    # Seven decades, two-sided.  A rare branch with a large fixed error needs a
    # displacement orders of magnitude beyond the bulk's, and a grid that
    # cannot reach it silently reports "no gain" -- which is the failure mode
    # this whole stage is trying to distinguish from a real null.
    _g = np.geomspace(1e-3, 1e4, 22)
    coarse = (np.concatenate([[0.0], _g, -_g])
              if t_grid is None else np.asarray(t_grid, float))

    def cvar_at(t, idx=None):
        sub = slice(None) if idx is None else idx
        Ls = L[sub]; rs = resp[sub]; ws = None if w is None else w[sub]
        e = eps[sub][rs] + t * s[None]
        Lt = Ls.copy()
        Lt[rs] = np.einsum("ki,ij,kj->k", e, W, e)
        return cvar_from_weights(Lt, tail_weights(Lt, ws, alpha), alpha)

    def search(idx=None):
        v = np.array([cvar_at(t, idx) for t in coarse])
        j = int(np.argmin(v))
        # local refine: bracket the coarse winner among its sorted neighbours
        so = np.sort(coarse)
        k = int(np.searchsorted(so, coarse[j]))
        lo = so[max(k - 1, 0)]; hi = so[min(k + 1, so.size - 1)]
        fine = np.linspace(lo, hi, 11)
        vf = np.array([cvar_at(t, idx) for t in fine])
        jf = int(np.argmin(vf))
        return ((float(coarse[j]), float(v[j])) if v[j] <= vf[jf]
                else (float(fine[jf]), float(vf[jf])))

    best_t, best = search(None)

    out = {"t_best": best_t, "gain": float(1.0 - best / max(cv0, 1e-300)),
           "cvar0": float(cv0), "cvar_best": best, "n_grid": int(coarse.size),
           "at_grid_edge": bool(abs(best_t) >= 0.999 * float(np.max(np.abs(coarse))))}
    if not cross_fit:
        return out
    # Choosing t on the same draws that score it is a selection bias.  Pick t on
    # one half and evaluate on the other, both ways, and report the average.
    # Still measurement-only: it is a split of runs already collected.
    rng = np.random.default_rng(seed)
    half = rng.random(L.size) < 0.5
    w_full = (np.full(L.size, 1.0 / L.size) if w is None
              else np.asarray(w, float).ravel())
    gains = []
    for sel in (half, ~half):
        # responsive draws: this fold's half.  inert branches: ALL of them, so
        # the mixture the fold sees is the real one.
        keep = (resp & sel) | (~resp)
        if int((resp & sel).sum()) < 64:
            continue
        wf = w_full[keep].copy()
        rr = resp[keep]
        if rr.sum() and wf[rr].sum() > 0:                 # restore the mixture
            wf[rr] *= (w_full[resp].sum() / wf[rr].sum())
        tot = wf.sum()
        if tot <= 0:
            continue
        wf = wf / tot
        other = (resp & ~sel) | (~resp)
        wo = w_full[other].copy()
        ro = resp[other]
        if ro.sum() and wo[ro].sum() > 0:
            wo[ro] *= (w_full[resp].sum() / wo[ro].sum())
        so = wo.sum()
        if so <= 0:
            continue
        wo = wo / so

        def cv_sub(t, idx, wgt):
            Ls = L[idx]; rs = resp[idx]
            e = eps[idx][rs] + t * s[None]
            Lt = Ls.copy()
            Lt[rs] = np.einsum("ki,ij,kj->k", e, W, e)
            return cvar_from_weights(Lt, tail_weights(Lt, wgt, alpha), alpha)

        v = np.array([cv_sub(t, keep, wf) for t in coarse])
        tstar = float(coarse[int(np.argmin(v))])
        c0, c1 = cv_sub(0.0, other, wo), cv_sub(tstar, other, wo)
        gains.append(1.0 - c1 / max(c0, 1e-300))
    if gains:
        out["gain_cv"] = float(np.mean(gains))
        out["gain_cv_folds"] = [float(x) for x in gains]
    return out


def resolution_floor(B, cov, n_eff, cvar, k=FLOOR_K):
    """R_min: the scatter of R_pred under the null d = 0.

    Under d = 0, n_eff * d_hat^T B d_hat ~ sum_i lambda_i chi^2_1 with lambda_i
    the eigenvalues of B Sigma, so the debiased estimator has standard
    deviation sqrt(2 sum lambda_i^2)/n_eff.  Dividing by CVaR puts the floor in
    the same units as R_tail.  Nothing is chosen: the boundary is where two
    measured curves cross.
    """
    M = np.asarray(B, float) @ np.asarray(cov, float)
    lam = np.linalg.eigvals(M).real
    sd = math.sqrt(2.0 * float(np.sum(lam ** 2))) / max(n_eff, 1.0)
    return float(k * sd / max(cvar, 1e-300))


def leverage(L, w=None, alpha=0.05, bulk=None):
    """Lambda_alpha: the share of the alpha-tail LOSS carried by the SAME
    process as the bulk.

    This is the Stage 6 correction to the Stage 5 reading.  Relaxation is not
    special: what suppresses the gain is that the tail is dominated by a RARE
    SUB-POPULATION, whatever its nature, because the coherent offset is shared
    with every other draw.  Compensating a rare branch means paying an offset
    the 99.8% of ordinary draws cannot afford, so the optimizer declines.
    A jump branch cannot be compensated at all; a large fluctuator excursion
    can be, and is still declined for the same reason.

    `bulk` marks the draws that come from the bulk process (default: all).
    Lambda = 1 means the tail is just the extreme draws of the ordinary noise,
    which is the only regime in which a tail objective moves the pulse.
    """
    L = np.asarray(L, float).ravel()
    t = tail_weights(L, w, alpha)
    tot = float(np.dot(t, L))
    if tot <= 0:
        return 1.0
    if bulk is None:
        return 1.0
    return float(np.dot(t * np.asarray(bulk, bool), L) / tot)


def _cos_W(a, b, W):
    """Plain cosine in the W metric, no sign convention."""
    W = np.asarray(W, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = math.sqrt(max(_wnorm2(a, W), 0.0)), math.sqrt(max(_wnorm2(b, W), 0.0))
    return float("nan") if na <= 0 or nb <= 0 else float((a @ W @ b) / (na * nb))


def direction_cosine(de0, d, W):
    """H2's statistic.  CVaR wants to move the offset by -d, so a correct
    prediction has cos ~ +1.  Computed in the W metric, which is the one the
    loss actually uses."""
    W = np.asarray(W, float)
    a, b = -np.asarray(de0, float), np.asarray(d, float)
    na, nb = math.sqrt(max(_wnorm2(a, W), 0.0)), math.sqrt(max(_wnorm2(b, W), 0.0))
    if na <= 0 or nb <= 0:
        return float("nan")
    return float((a @ W @ b) / (na * nb))


def cos_null_pvalue(cos, dim):
    """P(|cos| >= c) for a uniformly random direction in R^dim.  Makes the
    n = 1 claim of H2 quantitative rather than rhetorical."""
    from scipy import stats
    if not np.isfinite(cos) or dim < 2:
        return float("nan")
    # cos^2 ~ Beta(1/2, (dim-1)/2) for a uniform direction
    return float(stats.beta.sf(min(cos ** 2, 1.0), 0.5, (dim - 1) / 2.0))


def kappa_critical(R_min, C_alpha):
    """H6: where Cka^2 crosses the floor."""
    if C_alpha <= 0:
        return float("nan")
    return float(math.sqrt(max(R_min, 0.0) / C_alpha))


# ===================================================================== #
# 2.  SYNTHETIC REDUCED MODEL -- used by the selftests and by H1         #
# ===================================================================== #

def reduced_draws(e0, A, Q, n_draws, seed=0):
    """eps = e0 + A z + Q(z) for whitened z, returning (eps, z)."""
    A = np.atleast_2d(np.asarray(A, float))
    nz = A.shape[1]
    z = np.random.default_rng(seed).standard_normal((n_draws, nz))
    eps = np.asarray(e0, float)[None] + z @ A.T
    if Q is not None:
        Q = np.atleast_2d(np.asarray(Q, float))
        cols, k = [], 0
        for i in range(nz):
            for j in range(i, nz):
                cols.append(z[:, i] * z[:, j]); k += 1
        eps = eps + np.column_stack(cols) @ Q.T
    return eps, z


def brute_force_gain(e0, A, Q, W, alpha, n_draws=120_000, seed=0):
    """Re-optimize the offset for mean and for CVaR by direct minimization.
    The ground truth the closed form is checked against in S3/S4."""
    from scipy.optimize import minimize
    eps0, _ = reduced_draws(np.zeros(np.atleast_2d(A).shape[0]), A, Q, n_draws, seed)
    W = np.asarray(W, float)

    def loss(off):
        e = eps0 + off[None]
        return np.einsum("ki,ij,kj->k", e, W, e)

    def f_mean(off):
        return float(loss(off).mean())

    def f_cvar(off):
        L = loss(off)
        t = tail_weights(L, None, alpha)
        return cvar_from_weights(L, t, alpha)

    o0 = np.asarray(e0, float)
    m = minimize(f_mean, o0, method="Nelder-Mead",
                 options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 20000})
    c = minimize(f_cvar, m.x, method="Nelder-Mead",
                 options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 20000})
    cv_m, cv_c = f_cvar(m.x), f_cvar(c.x)
    return {"e0_mean": m.x, "e0_cvar": c.x, "cvar_mean": cv_m, "cvar_cvar": cv_c,
            "R": float(1.0 - cv_c / cv_m), "de0": c.x - m.x}


# ===================================================================== #
# 3.  SELFTESTS                                                          #
# ===================================================================== #

def selftest(verbose=True):
    ok = True
    rep = {}

    def say(tag, good, **kw):
        nonlocal ok
        ok &= bool(good)
        rep[tag] = {"pass": bool(good), **{k: float(v) if np.isscalar(v) else v
                                           for k, v in kw.items()}}
        if verbose:
            print(f"  [{'ok' if good else 'FAIL'}] {tag}: " +
                  "  ".join(f"{k}={v:.4g}" if np.isscalar(v) else f"{k}={v}"
                            for k, v in kw.items()))

    # ---- S1  tail weights and atomic CVaR are exact, including the split
    rng = np.random.default_rng(0)
    L = rng.random(37); w = rng.random(37); w /= w.sum()
    for a in (0.5, 0.2, 0.05, 0.013):
        t = tail_weights(L, w, a)
        o = np.argsort(-L)
        cum, acc, ref = 0.0, 0.0, 0.0
        for i in o:                                  # reference by hand
            take = min(w[i], a - cum)
            if take <= 0:
                break
            acc += take * L[i]; cum += take
            if cum >= a - 1e-15:
                break
        ref = acc / a
        say(f"S1 atomic CVaR alpha={a}", abs(cvar_from_weights(L, t, a) - ref) < 1e-12,
            err=abs(cvar_from_weights(L, t, a) - ref), mass_err=abs(t.sum() - a))

    # ---- S2  Theorem 1 as a measurement: linear-Gaussian => d -> 0
    A = np.array([[1.0, 0.0], [0.3, 0.9], [0.0, 0.4]])
    W = np.eye(3)
    scale = []
    for K in (20_000, 80_000, 320_000):
        eps, _ = reduced_draws(np.zeros(3), A, None, K, seed=3)
        L = np.einsum("ki,ij,kj->k", eps, W, eps)
        g = predicted_gain(eps, L, W, alpha=0.05)
        fl = resolution_floor(g["B"], g["cov"], g["n_eff"], g["cvar"])
        scale.append((K, g["R_pred"], g["R_pred_raw"], fl))
    # debiased estimate must sit at/below the floor at every K, and the RAW
    # (undebiased) one must fall like 1/K -- that is the bias being removed.
    say("S2 linear-Gaussian d->0 (debiased under floor)",
        all(r <= 1.5 * f for _, r, _, f in scale),
        R_20k=scale[0][1], R_320k=scale[2][1], floor_320k=scale[2][3])
    say("S2b raw bias falls like 1/K",
        scale[0][2] / max(scale[2][2], 1e-300) > 8.0,
        raw_20k=scale[0][2], raw_320k=scale[2][2],
        ratio=scale[0][2] / max(scale[2][2], 1e-300))

    # ---- S3  closed form == brute-force re-optimization, nonlinear case.
    # The quadratic form is a second-order expansion and must over-predict once
    # the gain saturates; the line search on the same draws removes that.  The
    # test therefore asks for tight agreement in the ONSET regime (which is
    # where the boundary lives) and conservatism, never under-prediction, above.
    for kap in (0.10, 0.25, 0.50):
        Q = kap * np.array([[1.0, 0.2, 0.0], [0.0, 0.7, 0.1], [0.3, 0.0, 0.5]])
        bf = brute_force_gain(np.zeros(3), A, Q, W, 0.05, n_draws=120_000, seed=5)
        eps, _ = reduced_draws(bf["e0_mean"], A, Q, 120_000, seed=5)
        L = np.einsum("ki,ij,kj->k", eps, W, eps)
        g = predicted_gain(eps, L, W, alpha=0.05)
        gn = predicted_gain(eps, L, W, alpha=0.05, naive=True)
        rel = abs(g["R_pred"] - bf["R"]) / max(bf["R"], 1e-12)
        relq = abs(g["R_pred_quadratic"] - bf["R"]) / max(bf["R"], 1e-12)
        good = rel < 0.30 if kap <= 0.15 else (
            rel < 0.30 or g["R_pred_quadratic"] >= bf["R"])
        say(f"S3 closed form vs brute force kappa={kap}", good,
            R_pred=g["R_pred"], R_quadratic=g["R_pred_quadratic"], R_true=bf["R"],
            rel_linesearch=rel, rel_quadratic=relq,
            R_naive=gn["R_pred_quadratic"], bfrac=g["boundary_frac"])

    # ---- S4  the DIRECTION is right (H2's statistic, on ground truth)
    Q = 0.3 * np.array([[1.0, 0.2, 0.0], [0.0, 0.7, 0.1], [0.3, 0.0, 0.5]])
    bf = brute_force_gain(np.zeros(3), A, Q, W, 0.05, n_draws=120_000, seed=7)
    eps, _ = reduced_draws(bf["e0_mean"], A, Q, 120_000, seed=7)
    L = np.einsum("ki,ij,kj->k", eps, W, eps)
    g = predicted_gain(eps, L, W, alpha=0.05)
    cs = direction_cosine(bf["de0"], -g["step"], W)
    cs_d = direction_cosine(bf["de0"], g["d"], W)
    say("S4 predicted displacement direction", cs > 0.95, cos=cs,
        cos_from_d_alone=cs_d, p_null=cos_null_pvalue(cs, 3))

    # ---- S5  small-kappa limit reproduces the kappa^2 law
    rs = []
    for kap in (0.03, 0.05, 0.08):          # onset only: above it the gain saturates
        Q = kap * np.array([[1.0, 0.2, 0.0], [0.0, 0.7, 0.1], [0.3, 0.0, 0.5]])
        bf = brute_force_gain(np.zeros(3), A, Q, W, 0.05, n_draws=120_000, seed=11)
        eps, _ = reduced_draws(bf["e0_mean"], A, Q, 120_000, seed=11)
        L = np.einsum("ki,ij,kj->k", eps, W, eps)
        rs.append(predicted_gain(eps, L, W, alpha=0.05)["R_pred"] / kap ** 2)
    say("S5 R_pred/kappa^2 approaches a constant (onset regime)",
        max(rs) / max(min(rs), 1e-300) < 1.6, C_lo=min(rs), C_hi=max(rs),
        spread=max(rs) / max(min(rs), 1e-300))

    # ---- S6  leverage: a control-inert heavy atom suppresses R_pred
    Q = 0.3 * np.array([[1.0, 0.2, 0.0], [0.0, 0.7, 0.1], [0.3, 0.0, 0.5]])
    bf = brute_force_gain(np.zeros(3), A, Q, W, 0.05, n_draws=120_000, seed=13)
    eps, _ = reduced_draws(bf["e0_mean"], A, Q, 120_000, seed=13)
    Lc = np.einsum("ki,ij,kj->k", eps, W, eps)
    g_coh = predicted_gain(eps, Lc, W, alpha=0.05)
    # append a rare catastrophic, control-inert branch (a T1 jump)
    p_j, ell = 2.0e-3, 1200.0 * float(np.mean(Lc))
    K = Lc.size
    epsj = np.vstack([eps, np.zeros((1, 3))])
    Lj = np.concatenate([Lc, [ell]])
    wj = np.concatenate([np.full(K, (1.0 - p_j) / K), [p_j]])
    msk = np.concatenate([np.ones(K, bool), [False]])
    g_j = predicted_gain(epsj, Lj, W, w=wj, alpha=0.05, mask=msk)
    lam = leverage(Lj, wj, 0.05, msk)   # msk = the bulk draws
    say("S6 inert heavy tail suppresses R_pred",
        g_j["R_pred"] < 0.2 * g_coh["R_pred"] and lam < 0.2,
        R_coherent=g_coh["R_pred"], R_with_jump=g_j["R_pred"], Lambda=lam,
        suppression=g_j["R_pred"] / max(g_coh["R_pred"], 1e-300))
    # ...and a heavy tail that the control CAN shape is declined for the same
    # reason once it is rare enough.  Sweeping a single fluctuator's amplitude
    # traces the crossover: at small excursion the rare branch is effectively
    # part of the bulk and the full coherent gain survives; at large excursion
    # the offset it wants is one the bulk cannot pay for, and the gain
    # collapses even though nothing about it is un-shapeable.  Verified against
    # a brute-force offset optimization: 0.037 true vs 0.034 predicted.
    nr, curve = 4000, []
    for mult in (1.0, 10.0, 200.0, 1200.0):
        sc = math.sqrt(mult * ell / max(float(np.mean(Lc)), 1e-300) / 1200.0)
        rg = np.random.default_rng(21)
        zb = np.zeros((nr, A.shape[1]))
        zb[:, 0] = rg.choice([-1.0, 1.0], size=nr) * sc
        cl = [zb[:, i] * zb[:, j] for i in range(A.shape[1]) for j in range(i, A.shape[1])]
        bg = bf["e0_mean"][None] + zb @ A.T + np.column_stack(cl) @ Q.T
        er = np.vstack([eps, bg])
        Lr = np.concatenate([Lc, np.einsum("ki,ij,kj->k", bg, W, bg)])
        wr = np.concatenate([np.full(K, (1 - p_j) / K), np.full(nr, p_j / nr)])
        bulk_m = np.concatenate([np.ones(K, bool), np.zeros(nr, bool)])
        gr = predicted_gain(er, Lr, W, w=wr, alpha=0.05)
        curve.append((mult, leverage(Lr, wr, 0.05, bulk_m), gr["R_pred"]))
    say("S6b leverage crossover: rare sub-population, shapeable or not",
        curve[0][2] > 0.7 * g_coh["R_pred"] and curve[-1][2] < 0.25 * g_coh["R_pred"]
        and curve[0][1] > 0.9 and curve[-1][1] < 0.1,
        R_coherent=g_coh["R_pred"], R_small_excursion=curve[0][2],
        R_large_excursion=curve[-1][2], Lambda_small=curve[0][1],
        Lambda_large=curve[-1][1])
    say("S6c Lambda tracks the gain across the crossover",
        all(curve[i][2] >= 0.9 * curve[i + 1][2] for i in range(len(curve) - 1)),
        Lambdas=[round(c[1], 3) for c in curve], R=[round(c[2], 4) for c in curve])

    # ---- S7  the floor really is the null scatter
    draws = []
    for s in range(24):
        eps, _ = reduced_draws(np.zeros(3), A, None, 20_000, seed=100 + s)
        L = np.einsum("ki,ij,kj->k", eps, W, eps)
        g = predicted_gain(eps, L, W, alpha=0.05)
        draws.append(g["R_pred_raw"] - g["bias"])
    eps, _ = reduced_draws(np.zeros(3), A, None, 20_000, seed=999)
    L = np.einsum("ki,ij,kj->k", eps, W, eps)
    g = predicted_gain(eps, L, W, alpha=0.05)
    fl = resolution_floor(g["B"], g["cov"], g["n_eff"], g["cvar"])
    emp = float(np.std(draws))
    say("S7 floor matches empirical null scatter", 0.4 < fl / max(emp, 1e-300) < 2.5,
        floor=fl, empirical_sd=emp, ratio=fl / max(emp, 1e-300))

    # ---- S8  rho/r_alpha agree with the closed forms used elsewhere
    say("S8 rho_alpha(1)=r_alpha, rho(2)=1+ln(1/a)",
        abs(rho_eff(1, 0.05) - r_alpha(0.05)) < 1e-9 and
        abs(rho_eff(2, 0.05) - (1 + math.log(1 / 0.05))) < 1e-9,
        rho1=rho_eff(1, 0.05), rho2=rho_eff(2, 0.05))

    # ---- S9  the retro mixture arm: Lambda is the suppression FACTOR
    # mixture_predictor is what makes H5's T1 arm a measurement instead of a
    # tautology, so it gets checked here rather than discovered in a six-hour
    # run.  Two claims: (a) Lambda lands on the analytic bulk share of the
    # alpha-tail, alpha*rho_alpha/(beta + alpha*rho_alpha), and (b) R_pred on
    # the mixture is Lambda times R_pred on the bulk alone -- i.e. leverage is
    # not merely correlated with the gain, it multiplies it.
    Qm = 0.25 * np.array([[1.0, 0.2, 0.0], [0.0, 0.7, 0.1], [0.3, 0.0, 0.5]])
    em, _ = reduced_draws(np.zeros(3), A, Qm, 60_000, seed=5)
    Lm = np.einsum("ki,ij,kj->k", em, W, em)
    stm = {"eps": em, "L": Lm, "W": W, "e0": em.mean(0), "C_alpha": {}, "n_coord": 3}
    base_R = predict_from_control(stm, (0.05,))["0.05"]["R_pred"]
    # rho is MEASURED on the bulk, not taken as r_alpha: the synthetic bulk has
    # an effective dimension strictly between 1 and 2, so r_alpha(0.05) = 5.582
    # would be the wrong reference by ~9% and the check would be testing
    # m_eff = 1, not the mixture algebra.
    rho_b = (cvar_from_weights(Lm, tail_weights(Lm, None, 0.05), 0.05)
             / float(Lm.mean()))
    lam_err, fac_err, tab = [], [], []
    for bet in (0.3, 1.0, 3.0, 20.0):
        pm, _inf = mixture_predictor(stm, (0.05,), p=2.0e-3, beta=bet)
        lm_ = pm["0.05"]["Lambda"]; rm_ = pm["0.05"]["R_pred"]
        an = 0.05 * rho_b / (bet + 0.05 * rho_b)
        lam_err.append(abs(lm_ / an - 1.0))
        fac_err.append(abs(rm_ / max(lm_ * base_R, 1e-300) - 1.0))
        tab.append((bet, round(lm_, 4), round(an, 4), round(rm_, 5)))
    say("S9 mixture leverage = analytic bulk share, and multiplies R_pred",
        max(lam_err) < 0.03 and max(fac_err) < 0.10,
        R_bulk_only=base_R, rho_bulk=rho_b, worst_Lambda_error=max(lam_err),
        worst_factor_error=max(fac_err),
        beta_Lambda_analytic_Rpred=tab)

    print(("\nselftest PASSED" if ok else "\nselftest FAILED"))
    return ok, rep


# ===================================================================== #
# 4.  SIMULATOR GLUE                                                     #
# ===================================================================== #

def _sim():
    import inspect as _inspect
    import fast_viper as fv
    import stage0_v2 as v2
    import fade_realistic as fr
    if "rho" not in _inspect.signature(v2.Ensemble.__init__).parameters:
        raise SystemExit("stage6.py needs the Stage-1 versions of stage0_v2.py / fast_viper.py.")
    return fv, v2, fr


def _ms(P, *args, **kw):
    """Problem.multistart returns (best_dict, runs); we want the coefficients."""
    best, _ = P.multistart(*args, **kw)
    return best["coef"]


def c_alpha_map(st, alphas):
    """C_alpha per alpha.  The kappa^2 coefficient is alpha-dependent -- that
    is the whole content of Table I -- so evaluating it once at 0.05 and
    reusing it makes kappa_c(alpha) wrong at every other level.  (Bug found in
    review; it silently flattened H6's alpha dependence.)"""
    import stage3 as s3
    out = {}
    for a in alphas:
        try:
            out[f"{a:g}"] = float(s3.C_closed(st["A"][:3], st["Q"][:3], s3.W_EPS, a))
        except Exception:
            out[f"{a:g}"] = float("nan")
    return out


def coherent_state(fv, v2, uI, uQ, ens, n_levels):
    """Everything the predictor needs, from ONE control: error coordinates,
    losses, the metric, the fitted offset, and the two-coordinate geometry."""
    import stage3 as s3
    import stage4 as s4
    from fade_realistic import cfg
    U = s4.colored_propagators(fv, uI, uQ, ens.deltas, ens.gammas)
    x = s3.error_coords(U, n_levels)
    W = s3.metric_W(n_levels)
    L = v2.losses(uI, uQ, ens)
    dt = cfg.Tg / cfg.Nt
    try:
        z1, z2 = s4.noise_coordinates(ens.deltas, dt)
        zs = [z1, z2]
    except Exception:
        z = np.asarray(ens.deltas, float)
        z = (z - z.mean()) / max(float(z.std()), 1e-300)
        zs = [z]
    e0, A, Q, r2 = s3.fit_expansion(x, zs)
    return {"eps": x, "L": np.asarray(L, float).ravel(), "W": W,
            "e0": e0, "A": A, "Q": Q, "fit_R2_min": float(np.min(r2)),
            "kappa": float(s3.kappa_of(A[:3], Q[:3])),
            "C_closed": float(s3.C_closed(A[:3], Q[:3], s3.W_EPS, 0.05)),
            "n_coord": int(x.shape[1])}


def reachable_projector(fv, v2, P, coef, levels, W, idx=None, h=None, tol=1e-6):
    """P_B: the orthogonal projector (in the W metric) onto the subspace of
    error-coordinate space the PULSE can actually reach.

        B = d eps / d c        (finite differences in the coefficient vector)
        P_B = projector onto range(B)

    Without this the predictor assumes every measured error direction is
    controllable.  For a two-level gate the error vector is 3-dimensional and
    a 34-coefficient pulse spans all of it, so P_B comes out as the identity --
    which is itself a check, not an assumption.  It stops being the identity
    for leakage-aware coordinates, where the control cannot reach every
    leakage amplitude independently, and that is where it matters.

    Evaluated on a subsample `idx` (the reachable subspace is a property of
    the control, not of the ensemble, so a few hundred draws suffice and the
    full test ensemble would be wasted here).
    """
    import stage3 as s3
    import stage4 as s4
    ens = P.test
    # deltas is (K,) for quasi-static noise and (K, Nt) for a trajectory; in
    # both cases the FIRST axis is the draw, so index rows and leave the rest
    # alone.  (atleast_2d turns (K,) into (1, K), which silently transposed the
    # ensemble -- the kind of bug that shows up as an unrelated broadcast error
    # three frames away.)
    dl = np.asarray(ens.deltas, float)
    gm = np.atleast_1d(np.asarray(ens.gammas, float))
    K = gm.size
    if idx is None:
        idx = np.arange(K) if K <= 400 else np.linspace(0, K - 1, 400).astype(int)
    d_sub, g_sub = dl[idx], gm[idx]

    def eps_of(c):
        ui, uq = P.filtered(np.asarray(c, float))
        U = s4.colored_propagators(fv, ui, uq, d_sub, g_sub)
        return s3.error_coords(U, levels).mean(0)

    c0 = np.asarray(coef, float).ravel()
    sc = float(np.max(np.abs(c0))) or 1.0
    hh = (1e-4 * sc) if h is None else float(h)
    e0 = eps_of(c0)
    B = np.empty((e0.size, c0.size))
    for i in range(c0.size):
        cp = c0.copy(); cp[i] += hh
        B[:, i] = (eps_of(cp) - e0) / hh
    Wh = np.real(np.linalg.cholesky(np.asarray(W, float)
                                    + 1e-14 * np.eye(W.shape[0])))
    Bw = np.linalg.solve(Wh, B) if False else Wh.T @ B      # whiten: W = Wh Wh^T
    u, sv, _ = np.linalg.svd(Bw, full_matrices=False)
    rank = int(np.sum(sv > tol * max(sv[0], 1e-300)))
    Uk = u[:, :rank]
    Pw = Uk @ Uk.T                                          # projector, whitened
    Wh_inv = np.linalg.inv(Wh.T)
    PB = Wh_inv @ Pw @ Wh.T                                 # back to eps space
    return {"P_B": PB, "B": B, "rank": rank, "n_coord": int(e0.size),
            "singular_values": [float(x) for x in sv[:8]],
            "full_rank": bool(rank >= e0.size)}


def predict_from_control(st, alphas=ALPHA_GRID, w=None, mask=None, L=None,
                         eps=None, bulk=None, P_B=None):
    """R_pred, the floor and the leverage at every alpha, from one control."""
    eps = st["eps"] if eps is None else eps
    L = st["L"] if L is None else L
    W = st["W"]
    out = {}
    for a in alphas:
        g = predicted_gain(eps, L, W, w=w, alpha=a, mask=mask, P_B=P_B)
        gn = predicted_gain(eps, L, W, w=w, alpha=a, mask=mask, naive=True,
                            line_search=False, P_B=P_B)
        fl = resolution_floor(g["B"], g["cov"], g["n_eff"], g["cvar"])
        out[f"{a:g}"] = {
            "R_pred": g["R_pred"],
            "R_pred_quadratic": g["R_pred_quadratic"],
            "R_pred_simple": gn["R_pred_quadratic"],   # ||d||^2_W/CVaR, H = 2W
            "R_pred_raw": g["R_pred_raw"],
            "reach_frac": g["reach_frac"], "boundary_frac": g["boundary_frac"],
            "R_min": fl, "resolved": bool(g["R_pred"] > fl),
            "d": [float(v) for v in g["d"]],
            "d_measured": [float(v) for v in g["d_measured"]],
            "step": [float(v) for v in g["step"]],
            "d_norm2_W": g["d_norm2_W"], "bias": g["bias"],
            "cvar": g["cvar"], "n_eff": g["n_eff"], "n_band": g["n_band"],
            "rho_tau": g["rho_tau"], "m_c_over_alpha": g["m_c_over_alpha"],
            "underdetermined": g["underdetermined"],
            "t_best": (g["line_search"] or {}).get("t_best"),
            "at_grid_edge": (g["line_search"] or {}).get("at_grid_edge"),
            "Lambda": leverage(L, w, a, bulk),
            "C_alpha": st.get("C_alpha", {}).get(f"{a:g}"),
            "kappa_c": kappa_critical(fl, st.get("C_alpha", {}).get(f"{a:g}") or 0.0)}
    return out


def mixture_predictor(st, alphas, p, beta, P_B=None):
    """Splice a recorded rare, control-inert jump branch onto the MEASURED
    dephasing ensemble and re-run the predictor on the mixture.

    This is what makes the T1 arm of H5 a T1 arm at all.  `retro` re-derives
    the mean optimum with stage0_v2's plain Gaussian dephasing ensemble, which
    has no relaxation channel in it, so the leverage there came out Lambda = 1
    by construction -- the summary was reporting "100% of the tail is bulk" for
    a simulator in which the bulk is all there is.  That is not a measurement of
    T1 leverage; it is the absence of one.

    Stage 5 did not sample jumps either: it used the exact no-jump / one-jump
    decomposition, so what it recorded per cell is (p, l_j, E[L_deph]) and
    hence beta = p l_j / E[L_deph] -- the analytic mixture, not a draw set.
    The faithful way to put it back is therefore analytic too: append ONE atom
    of weight p carrying loss l_j, with its error coordinates pinned at e0 so
    it contributes nothing to the displacement.  That encodes the Stage 5
    finding directly (the per-control spread of l_j was 0.88%, of n_int 1.37%,
    against 44.9% for L_deph -- the jump channel is very nearly uncontrollable)
    and it is the same construction selftest S6 validates against brute force.

    l_j is rescaled to the retro cell's own E[L_deph] so that the MIXTURE RATIO
    beta is preserved rather than the absolute jump loss; beta is the invariant
    Proposition 4 is written in, and the re-derived mean optimum is not
    numerically identical to Stage 5's.
    """
    eps = np.asarray(st["eps"], float)
    L = np.asarray(st["L"], float).ravel()
    K = L.size
    p = float(p)
    L_bulk = float(L.mean())
    ell = float(beta) * L_bulk / max(p, 1e-300)
    e0 = np.asarray(st["e0"], float).ravel()
    eps_m = np.vstack([eps, e0[None, :]])
    L_m = np.concatenate([L, [ell]])
    w_m = np.concatenate([np.full(K, (1.0 - p) / K), [p]])
    bulk = np.concatenate([np.ones(K, bool), [False]])
    pred = predict_from_control(st, alphas, w=w_m, mask=bulk, L=L_m, eps=eps_m,
                                bulk=bulk, P_B=P_B)
    return pred, {"p": p, "beta": float(beta), "l_j_effective": ell,
                  "L_deph_mean": L_bulk, "n_bulk": int(K)}


# ===================================================================== #
# 5.  RANDOM TELEGRAPH NOISE -- the positive control for leverage         #
# ===================================================================== #

def rtn_trajectories(K, Nt, dt, rate_ns, amp, seed=0):
    """A two-state fluctuator: the detuning jumps between -amp and +amp at a
    Poisson rate.  Returns (K, Nt) in the same shape and units as
    stage4.ou_draws, so it drops straight into the colored-noise path.

    Rare and catastrophic like relaxation, but the resulting error is a
    COHERENT phase, so the pulse can in principle reshape it.  Sweeping `amp`
    traces the leverage crossover.
    """
    rng = np.random.default_rng(seed)
    s = rng.choice([-1.0, 1.0], size=K)
    out = np.empty((K, Nt))
    p = 1.0 - math.exp(-abs(rate_ns) * dt)
    for k in range(Nt):
        s = np.where(rng.random(K) < p, -s, s)
        out[:, k] = s
    return float(amp) * out


class RTNEnsemble:
    """Drop-in for stage0_v2.Ensemble whose `deltas` is a telegraph trajectory
    per draw, plus an ordinary Gaussian background.  Installed exactly the way
    stage4.ColoredEnsemble is, so v2.losses, the optimizer and
    stage4.colored_propagators all see the fluctuator.

    The previous revision generated the telegraph process and then never fed it
    anywhere -- the arm was not an RTN experiment at all.  Caught in review.
    """
    rate_ns = 1e-3          # module state, set per point by `rtn`
    amp = 0.0               # fluctuator half-amplitude, rad/ns
    p_active = 1.0          # fraction of draws that carry a fluctuator

    def __init__(self, n, sigma_d, sigma_a, seed, name, rho=0.0):
        from fade_realistic import cfg
        dt = cfg.Tg / cfg.Nt
        rng = np.random.default_rng(seed + 991)
        bg = np.repeat(rng.normal(0.0, float(sigma_d), n)[:, None], cfg.Nt, axis=1)
        tel = rtn_trajectories(n, cfg.Nt, dt, RTNEnsemble.rate_ns,
                               RTNEnsemble.amp, seed + 17)
        on = (rng.random(n) < RTNEnsemble.p_active).astype(float)[:, None]
        self.deltas = bg + on * tel
        self.active = on.ravel() > 0.5
        self.gammas = np.random.default_rng(seed + 1).normal(0.0, float(sigma_a), n)
        self.n, self.seed, self.name, self.rho = n, seed, name, float(rho)
        self.phi = self.deltas.sum(1) * dt
        self.amp, self.rate_ns = RTNEnsemble.amp, RTNEnsemble.rate_ns

    def describe(self):
        return {"name": self.name, "n": self.n, "seed": self.seed,
                "amp": self.amp, "rate_ns": self.rate_ns,
                "p_active": float(self.active.mean()),
                "sigma_phi_sq_realized": float(self.phi.var())}


def _install_rtn(fv, v2, amp, rate_ns, p_active):
    """Route every loss and gradient through the fluctuator simulator."""
    import stage4 as s4
    RTNEnsemble.amp = float(amp)
    RTNEnsemble.rate_ns = float(rate_ns)
    RTNEnsemble.p_active = float(p_active)
    prev = (v2.gather_ordered, v2.Ensemble)
    v2.gather_ordered = lambda uI, uQ, d, g: s4.colored_gather(fv, uI, uQ, d, g)
    v2.Ensemble = RTNEnsemble
    fv._prev.clear()

    def restore():
        v2.gather_ordered, v2.Ensemble = prev
        fv._prev.clear()
    return restore


# ===================================================================== #
# 6.  DRIVER                                                             #
# ===================================================================== #

@dataclass
class Point:
    stem: str
    Tg: float
    phase_var: float
    seed: int
    levels: int


TAIL_TARGET = 8          # training draws required in the alpha-tail


def auto_k_train(alphas, base=128, target=TAIL_TARGET):
    """One training ensemble per point, sized for the SMALLEST alpha.

    The CVaR objective only sees the worst alpha fraction of the TRAINING
    ensemble, so alpha*K_train is the number of draws its gradient is actually
    built from.  At the stage0_v2 default K_train = 128 that is 6.4 draws at
    alpha = 0.05, 3.2 at 0.025 and 1.28 at 0.01 -- the last two are noise, and
    they show up in the log as CVaR runs "converging" in 6-10 iterations.

    It has to be ONE ensemble shared by the mean, CVaR and reverse runs at a
    point, not a per-alpha size: giving the CVaR arm more data than its
    comparator would break the budget parity the whole fairness protocol rests
    on.  So the cost of the smallest alpha is paid by every run at that point,
    which is why dropping alpha = 0.01 saves more than half the wall time.
    """
    return int(max(base, math.ceil(target / max(min(alphas), 1e-12))))


def _cell(v2, Tg, pv, seed, levels, sigma_a=0.0, n_test=None, k_train=None):
    base = v2.V2Config()
    base.n_levels = int(levels)
    cc = v2.V2Config(**asdict(base))
    if k_train:
        cc.K_train = int(k_train)
    if n_test:
        # The predictor is a conditional average over the worst alpha fraction,
        # so the TEST ensemble sets its resolution.  The default n_test = 1000
        # leaves 10 draws in the tail at alpha = 0.01, which cannot support a
        # conditional mean, let alone the conditional covariance the curvature
        # needs.  Working rule: alpha_min * n_test >= ~400.
        cc.n_test = int(n_test)
    cc.seed_train += seed; cc.seed_val += seed; cc.seed_test += seed
    cc.sigma_a = float(sigma_a)
    v2.set_gate(Tg, cc)
    cc.sigma_d = math.sqrt(pv) / Tg
    return cc


def _one_point(fv, v2, Tg, pv, seed, levels, alphas, stem, outdir,
               ground_truth=True, log=print, n_test=None, no_projector=False,
               k_train=None, mix_p=None, mix_beta=None):
    """The mean control and its predictor (always), plus the CVaR ground truth
    with the full fairness protocol (optional, and it is what costs time)."""
    import stage3 as s3
    t0 = time.time()
    cc = _cell(v2, Tg, pv, seed, levels, n_test=n_test, k_train=k_train)
    P = v2.Problem(Tg, cc)

    c_mean = _ms(P, "mean", seed=seed, label=f"{stem}:mean")
    uI, uQ = P.filtered(c_mean)
    st = coherent_state(fv, v2, uI, uQ, P.test, levels)
    st["C_alpha"] = c_alpha_map(st, alphas)
    pb = (None if no_projector else
          reachable_projector(fv, v2, P, c_mean, levels, st["W"]))
    pred = predict_from_control(st, alphas, P_B=(pb or {}).get("P_B"))
    bad = [k for k, v in pred.items() if v.get("underdetermined")]
    if bad:
        log(f"  WARNING {stem}: curvature underdetermined at alpha={bad}; "
            f"K={st['eps'].shape[0]} test draws is too few. Raise the test "
            f"ensemble (alpha*K >= ~400 per level is the working rule).")

    rec = {"stem": stem, "Tg": float(Tg), "phase_var": float(pv), "seed": int(seed),
           "levels": int(levels), "alphas": list(alphas), "n_test": int(cc.n_test),
           "K_train": int(cc.K_train),
           "tail_draws_min_alpha": float(min(alphas) * cc.K_train),
           "kappa": st["kappa"], "C_alpha": st["C_alpha"],
           "fit_R2_min": st["fit_R2_min"], "n_coord": st["n_coord"],
           "projector": {k: v for k, v in (pb or {}).items() if k not in ("P_B", "B")},
           "mean_loss": float(st["L"].mean()),
           "e0_mean": [float(v) for v in st["e0"]],
           "coef_mean": [float(v) for v in np.asarray(c_mean).ravel()],
           "predictor": pred, "ground_truth": {}}

    # The rare-channel arm: only present where a previous stage recorded a
    # mixture (p, beta) for this cell.  Without it Lambda is 1 by construction.
    if mix_p and mix_beta and float(mix_p) > 0.0 and float(mix_beta) > 0.0:
        pm, minfo = mixture_predictor(st, alphas, mix_p, mix_beta,
                                      P_B=(pb or {}).get("P_B"))
        rec["predictor_T1"] = pm
        rec["mixture"] = minfo
        log(f"  {stem}: mixture beta={float(mix_beta):.3g} p={float(mix_p):.3g} "
            f"-> Lambda(0.05)="
            f"{pm.get('0.05', {}).get('Lambda')}")

    # H1 needs the inner-optimization predictor on the same fit
    try:
        rec["R_model_q"] = {f"{a:g}": float(s3.r_model(st["e0"], st["A"][:3], st["Q"][:3],
                                                       s3.W_EPS, a)) for a in alphas}
    except Exception as ex:
        rec["R_model_q"] = {"error": repr(ex)}

    if ground_truth:
        controls = {"mean": c_mean}
        for a in alphas:
            controls[f"cvar_{a:g}"] = _ms(P, "cvar", alpha=a, seed=seed + 101,
                                          warm=c_mean, label=f"{stem}:cvar{a:g}")
        for a in alphas:                       # reverse warm start: fairness
            controls[f"rev_{a:g}"] = _ms(P, "mean", seed=seed + 202,
                                         warm=controls[f"cvar_{a:g}"],
                                         label=f"{stem}:rev{a:g}")
        ev = {}
        for k, c in controls.items():
            ui, uq = P.filtered(c)
            Lk = np.asarray(v2.losses(ui, uq, P.test), float).ravel()
            ev[k] = {"mean_loss": float(Lk.mean()),
                     "cvar": {f"{a:g}": cvar_from_weights(Lk, tail_weights(Lk, None, a), a)
                              for a in alphas}}
        cand = ["mean"] + [f"rev_{a:g}" for a in alphas]
        for a in alphas:
            key = f"{a:g}"
            best = min(cand, key=lambda c_: ev[c_]["cvar"][key])
            Rraw = 1.0 - ev[f"cvar_{key}"]["cvar"][key] / ev["mean"]["cvar"][key]
            Rb = 1.0 - ev[f"cvar_{key}"]["cvar"][key] / ev[best]["cvar"][key]
            # H2: the measured offset displacement, mean -> cvar
            ui, uq = P.filtered(controls[f"cvar_{key}"])
            stc = coherent_state(fv, v2, ui, uq, P.test, levels)
            de0 = np.asarray(stc["e0"], float) - np.asarray(st["e0"], float)
            cs = direction_cosine(de0, -np.asarray(pred[key]["step"], float), st["W"])
            rec["ground_truth"][key] = {
                "R_raw": float(Rraw), "R": float(Rb), "comparator": best,
                "cos": float(cs), "cos_p_null": cos_null_pvalue(cs, st["n_coord"]),
                "de0_norm": float(np.linalg.norm(de0)),
                "coef_cvar": [float(v) for v in np.asarray(controls[f"cvar_{key}"]).ravel()]}
        rec["evals"] = ev
        rec["coef_rev"] = {f"{a:g}": [float(v) for v in np.asarray(controls[f"rev_{a:g}"]).ravel()]
                           for a in alphas}
    rec["seconds"] = time.time() - t0
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / f"{stem}.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def probe(a):
    """Measure one point, then extrapolate the whole grid.  This exists
    because the Stage 5 estimate was 3x low: cost scales as Nt^~1.4, not
    linearly, and the only honest way to quote it is to measure it."""
    fv, v2, fr = _sim()
    d = Path(a.outdir) / "probe"
    kaps = list(a.kappa)
    pv = _pv_for_kappa(kaps[len(kaps) // 2], a.Tg)
    t0 = time.time()
    rec = _one_point(fv, v2, a.Tg, pv, a.seeds[0], a.levels, tuple(a.alpha),
                     f"probe_Tg{a.Tg:g}".replace(".", "p"), d, ground_truth=True,
                     n_test=a.n_test, no_projector=a.no_projector,
                     k_train=a.k_train)
    dt = time.time() - t0
    n_main = len(kaps) * len(a.seeds)
    bad = [k for k, v in rec["predictor"].items() if v.get("underdetermined")]
    thin = {k: int(v["n_eff"]) for k, v in rec["predictor"].items()}
    est = {"probe_seconds": dt, "n_points_main": n_main,
           "n_test": rec.get("n_test"), "tail_n_eff_by_alpha": thin,
           "underdetermined_alphas": bad,
           "resolution_ok": not bad and min(thin.values()) >= 100,
           "est_hours_main": n_main * dt / 3600.0,
           "est_hours_with_rtn": 2 * n_main * dt / 3600.0,
           "measured_kappa": rec["kappa"], "requested_kappa": kaps[len(kaps) // 2],
           "note": ("one point = 1 mean + %d cvar + %d reverse multistarts; "
                    "scale linearly in the number of points, not in Tg"
                    % (len(a.alpha), len(a.alpha)))}
    (Path(a.outdir) / "stage6_probe.json").write_text(json.dumps(est, indent=1, default=float))
    print(json.dumps(est, indent=1, default=float))
    return est


def _pv_for_kappa(kappa, Tg):
    """kappa grows roughly linearly in sigma_phi on a driven qubit, and the
    Stage 2/3 data put kappa ~ 0.37 sigma_phi.  Only a starting guess: the
    achieved kappa is measured and recorded per point, and the analysis uses
    the MEASURED value, never this one."""
    return float((kappa / 0.37) ** 2)


def run(a):
    fv, v2, fr = _sim()
    d = Path(a.outdir) / f"stage6_{a.levels}level"
    d.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(a.kappa, a.seeds))
    for i, (kap, seed) in enumerate(grid):
        pv = _pv_for_kappa(kap, a.Tg)
        stem = ("s6_Tg%g_k%g_s%d" % (a.Tg, kap, seed)).replace(".", "p")
        if (d / f"{stem}.json").exists() and not a.force:
            v2.log(f"  {stem} already done, skipping"); continue
        v2.log(f"=== Stage 6 point {i+1}/{len(grid)}: Tg={a.Tg} ns, "
               f"kappa~{kap}, sigma_phi^2={pv:.4g}, seed {seed} ===")
        r = _one_point(fv, v2, a.Tg, pv, seed, a.levels, tuple(a.alpha), stem, d,
                       ground_truth=not a.predictor_only, log=v2.log,
                       n_test=a.n_test, no_projector=a.no_projector,
                     k_train=a.k_train)
        g = r["ground_truth"].get("0.05", {})
        p5 = r["predictor"]["0.05"]
        v2.log(f"  -> kappa={r['kappa']:.4f}  R_pred={p5['R_pred']:+.4f} "
               f"(floor {p5['R_min']:.4f})  R={g.get('R', float('nan')):+.4f}  "
               f"cos={g.get('cos', float('nan')):+.3f}  [{r['seconds']:.0f}s]")


def rtn(a):
    """6B: a single two-state fluctuator, amplitude swept.

    The crossover, not a 2x2.  At small excursion the fluctuator is
    effectively part of the bulk and the full coherent gain survives; at large
    excursion the offset the rare branch wants is one the bulk cannot pay for
    and the gain collapses -- even though, unlike relaxation, nothing about it
    is un-shapeable.  That is why this is the right positive control: it
    interpolates between the two regimes on one axis.
    """
    fv, v2, fr = _sim()
    from fade_realistic import cfg
    d = Path(a.outdir) / "stage6_rtn"
    d.mkdir(parents=True, exist_ok=True)
    # RARITY is swept, not held fixed.  The first sweep varied only the
    # amplitude at p = 0.02, which moves the conflict cosine but confounds it
    # with kappa; the obvious repair -- drop p to 0.002 to satisfy the p/alpha
    # expansion -- would have destroyed the effect, because the rare branch's
    # influence on the mixture displacement scales with p/alpha.  At p/alpha =
    # 0.04 the cosine never leaves 1.0 and the sweep measures nothing.
    #
    # The conflict-angle claim does not need the expansion: it needs the
    # two-population predictor checked against ground truth, which is valid at
    # any p/alpha.  So p becomes the swept axis and the existing amplitude
    # sweep at p = 0.02 becomes the second axis of a 2-D grid.
    plist = list(a.rtn_p if isinstance(a.rtn_p, (list, tuple)) else [a.rtn_p])
    for i, (amp, pj, seed) in enumerate(itertools.product(a.rtn_amp, plist, a.seeds)):
        stem = ("s6rtn_Tg%g_a%g_p%g_r%g_s%d"
                % (a.Tg, amp, pj, a.rtn_rate, seed)).replace(".", "p")
        if (d / f"{stem}.json").exists() and not a.force:
            v2.log(f"  {stem} already done, skipping"); continue
        v2.log(f"=== Stage 6B RTN {i+1}: amp={amp}/Tg, rate={a.rtn_rate}/ns, "
               f"p_active={pj}, seed {seed} ===")
        t0 = time.time()
        restore = _install_rtn(fv, v2, amp / a.Tg, a.rtn_rate, pj)
        try:
            cc = _cell(v2, a.Tg, a.rtn_pv, seed, a.levels, n_test=a.n_test,
                       k_train=a.k_train)
            P = v2.Problem(a.Tg, cc)
            c_mean = _ms(P, "mean", seed=seed, label=f"{stem}:mean")
            uI, uQ = P.filtered(c_mean)
            st = coherent_state(fv, v2, uI, uQ, P.test, a.levels)
            st["C_alpha"] = c_alpha_map(st, tuple(a.alpha))
            pb = (reachable_projector(fv, v2, P, c_mean, a.levels, st["W"])
                  if not a.no_projector else None)
            bulk = ~np.asarray(P.test.active, bool)      # draws with NO fluctuator
            pred = predict_from_control(st, tuple(a.alpha), bulk=bulk,
                                        P_B=(pb or {}).get("P_B"))
            # The counterfactual: THE SAME CONTROL, evaluated on the bulk draws
            # alone.  Without this the amplitude sweep proves nothing, because
            # amplitude moves two things at once -- it lowers Lambda (the rare
            # branch takes over the tail) and it raises kappa (bigger phase
            # excursions, so more curvature for the bulk to exploit).  R_pred
            # then rises with amplitude for a reason that has nothing to do with
            # leverage, and a rank correlation between Lambda and R_pred across
            # the sweep is exactly -1 whatever the mechanism, because both are
            # monotone in amplitude.  What is scale-free and unconfounded is the
            # RATIO of the two predictors at one control and one kappa, which is
            # the quantity the theory says equals Lambda -- and which the T1 arm
            # already reproduces to three decimals.
            pred_bulk = predict_from_control(
                st, tuple(a.alpha), eps=np.asarray(st["eps"], float)[bulk],
                L=np.asarray(st["L"], float).ravel()[bulk],
                P_B=(pb or {}).get("P_B"))
            suppression = {}
            for al in a.alpha:
                k = f"{al:g}"
                den = pred_bulk[k]["R_pred"]
                # The CONFLICT ANGLE.  Lambda turned out not to be a universal
                # suppression factor: it is the suppression factor for a rare
                # channel that is control-INERT and therefore has no preferred
                # offset of its own, which is why the T1 arm reproduces
                # R_mixture/R_bulk = Lambda to 0.65% while this arm misses it by
                # up to 100x.  What actually governs the outcome is whether the
                # rare branch wants the SAME offset as the bulk.  A fluctuator
                # along the coordinate the bulk already loads wants more of the
                # same, so it costs nothing (cos ~ 1, ratio ~ 1) and at large
                # excursion it takes the tail over and RAISES the gain.
                # NOT direction_cosine: that helper negates its first argument,
                # because H2 compares a measured offset CHANGE against a
                # predicted step and CVaR moves the offset by -d.  Here both
                # arguments are steps, so using it sign-flips every value --
                # which it did, reporting conflict_cos = -0.999 at p = 0.002
                # where the fluctuator is 0.2% of draws and the two steps are
                # necessarily near-parallel.
                cs = _cos_W(np.asarray(pred_bulk[k]["step"], float),
                            np.asarray(pred[k]["step"], float), st["W"])
                suppression[k] = {
                    "R_pred_mixture": pred[k]["R_pred"],
                    "R_pred_bulk_only": den,
                    "ratio": (float(pred[k]["R_pred"] / den) if den > 1e-300 else None),
                    "Lambda": pred[k]["Lambda"],
                    "conflict_cos": float(cs),
                    "p_over_alpha": float(np.mean(P.test.active) / al),
                    "d_norm_bulk": float(np.linalg.norm(pred_bulk[k]["d"])),
                    "d_norm_mixture": float(np.linalg.norm(pred[k]["d"]))}
            # ground truth on the same installed noise
            gt = {}
            for al in a.alpha:
                c_c = _ms(P, "cvar", alpha=al, seed=seed + 101, warm=c_mean,
                          label=f"{stem}:cvar{al:g}")
                c_r = _ms(P, "mean", seed=seed + 202, warm=c_c,
                          label=f"{stem}:rev{al:g}")
                cv = {}
                for k, c in (("mean", c_mean), ("cvar", c_c), ("rev", c_r)):
                    ui, uq = P.filtered(c)
                    Lk = np.asarray(v2.losses(ui, uq, P.test), float).ravel()
                    cv[k] = cvar_from_weights(Lk, tail_weights(Lk, None, al), al)
                best = min(("mean", "rev"), key=lambda k: cv[k])
                gt[f"{al:g}"] = {"R_raw": float(1.0 - cv["cvar"] / cv["mean"]),
                                 "R": float(1.0 - cv["cvar"] / cv[best]),
                                 "comparator": best,
                                 "coef_cvar": [float(x) for x in np.asarray(c_c).ravel()]}
            rec = {"stem": stem, "Tg": float(a.Tg), "phase_var": float(a.rtn_pv),
                   "levels": int(a.levels), "seed": int(seed), "n_test": int(cc.n_test),
                   # needed by `polish`: without it the pooled runs would rebuild
                   # the training ensemble at the default size rather than the
                   # one these controls were actually optimized against
                   "K_train": int(cc.K_train), "alphas": [float(x) for x in a.alpha],
                   "predictor_bulk_only": pred_bulk, "suppression": suppression,
                   "rtn_amp_over_Tg": float(amp), "rtn_p_requested": float(pj),
                   "rtn_rate_ns": float(a.rtn_rate),
                   "p_active": float(np.mean(P.test.active)),
                   "kappa": st["kappa"], "C_alpha": st["C_alpha"],
                   "fit_R2_min": st["fit_R2_min"], "n_coord": st["n_coord"],
                   "mean_loss": float(st["L"].mean()),
                   "sigma_phi_sq_realized": float(np.var(P.test.phi)),
                   "coef_mean": [float(x) for x in np.asarray(c_mean).ravel()],
                   "projector": {k: v for k, v in (pb or {}).items() if k not in ("P_B", "B")},
                   "predictor": pred, "ground_truth": gt,
                   "seconds": time.time() - t0}
        finally:
            restore()
        (d / f"{stem}.json").write_text(json.dumps(rec, indent=1, default=float))
        p5 = pred[f"{a.alpha[min(2, len(a.alpha)-1)]:g}"]
        v2.log(f"  -> Lambda={p5['Lambda']:.3f}  R_pred={p5['R_pred']:+.4f} "
               f"(floor {p5['R_min']:.4f})  [{rec['seconds']:.0f}s]")


def _dedupe(coefs, rtol=1e-6):
    """Distinct controls only.  The nine controls saved per point collapse to
    three or four in most cells -- the reverse runs frequently land on the same
    optimum -- and polishing a duplicate costs a full optimization for nothing.
    """
    out = []
    for c in coefs:
        if c is None:
            continue
        c = np.asarray(c, float).ravel()
        if not any(np.linalg.norm(c - o) <= rtol * max(1.0, np.linalg.norm(o))
                   for o in out):
            out.append(c)
    return out


def _polish_point(fv, v2, j, alphas, log=print):
    """Score both objectives from a SHARED pool of starts.

    Why this exists.  In the main run the mean arm gets one multistart
    (n_restarts = 4: the Hann seed plus three perturbations) and the CVaR arm
    gets one warm-started from it; the four reverse runs then re-optimize the
    MEAN objective from each CVaR solution.  Scoring R against the best of
    {mean, rev_*} was meant as a fairness guard, but in 10 of the 16 cells at
    least one reverse run found a strictly BETTER MEAN OPTIMUM than the mean
    multistart did -- by a factor of four in one cell.  Where that happens the
    comparator is not a fairness correction: the mean arm has effectively had
    five restarts, four of them warm-started with inside information about
    where the good basins are, and the CVaR arm has had one.  R then measures
    whether the mean multistart converged, not whether tail-risk optimization
    helps.  (Note it is NOT selection-on-outcome: choosing the comparator by
    lowest mean loss instead of lowest CVaR gives numerically identical R at
    every point.  The asymmetry is in the restart budget.)

    The repair is symmetry.  Pool every control any arm found at this point,
    deduplicate, and run EVERY objective -- mean and CVaR at each alpha -- from
    EVERY start in that pool.  Each arm then reports its best, selected on the
    VALIDATION ensemble by that arm's own risk measure, exactly as
    Problem.multistart does; the test ensemble is touched only for the final
    evaluation, so no arm selects on the quantity being compared.

    Cost: len(pool) x (1 + len(alphas)) single optimizations, all warm, which
    is typically 20 against the 36 the original point cost.
    """
    import stage0_v2 as _v2mod  # for risk(); same module object as v2
    stem = j["stem"]
    restore = None
    if j.get("rtn_amp_over_Tg") is not None:
        # the fluctuator has to be installed before the ensembles are built, or
        # the pooled runs would optimize a different system from the one the
        # saved controls came from
        restore = _install_rtn(fv, v2, j["rtn_amp_over_Tg"] / j["Tg"],
                               j["rtn_rate_ns"],
                               j.get("rtn_p_requested", j.get("p_active", 0.02)))
    kt = j.get("K_train") or auto_k_train(alphas)
    if not j.get("K_train"):
        log(f"  NOTE {stem}: no K_train recorded; using auto_k_train = {kt}. "
            f"If this point predates that field, confirm it matches the run.")
    cc = _cell(v2, j["Tg"], j["phase_var"], j["seed"], j["levels"],
               n_test=j.get("n_test"), k_train=kt)
    P = v2.Problem(j["Tg"], cc)

    pool = [j.get("coef_mean")]
    for a in list(j.get("alphas", alphas)) + list(alphas):
        k = f"{a:g}"
        pool.append((j.get("ground_truth", {}).get(k) or {}).get("coef_cvar"))
        pool.append((j.get("coef_rev", {}) or {}).get(k))
    pool = _dedupe(pool)
    log(f"=== polish {stem}: {len(pool)} distinct starts x {1 + len(alphas)} arms")

    def loss_test(c):
        ui, uq = P.filtered(np.asarray(c, float))
        return np.asarray(v2.losses(ui, uq, P.test), float).ravel()

    def arm(mode, al):
        best = None
        for i, s in enumerate(pool):
            r = P.optimize(np.asarray(s, float), mode, al, 0.5,
                           tag=f"{stem}/{mode}{al if al else ''}/start{i}")
            ui, uq = P.filtered(r["coef"])
            Lv = np.asarray(v2.losses(ui, uq, P.val), float).ravel()
            vr = float(v2.risk(Lv, mode, al, 0.5))
            if best is None or vr < best[0]:
                best = (vr, np.asarray(r["coef"], float).ravel(), i)
        return {"val_risk": best[0], "coef": best[1], "start_index": best[2]}

    arms = {"mean": arm("mean", 0.05)}
    for a in alphas:
        arms[f"cvar_{a:g}"] = arm("cvar", float(a))

    ev = {}
    for k, v in arms.items():
        L = loss_test(v["coef"])
        ev[k] = {"mean_loss": float(L.mean()),
                 "cvar": {f"{a:g}": cvar_from_weights(L, tail_weights(L, None, a), a)
                          for a in alphas},
                 "val_risk": v["val_risk"], "start_index": v["start_index"]}

    rec = {"stem": stem, "Tg": j["Tg"], "phase_var": j["phase_var"],
           "seed": j["seed"], "levels": j["levels"], "kappa": j["kappa"],
           "alphas": [float(a) for a in alphas],
           "n_pool": len(pool), "n_test": int(cc.n_test), "K_train": int(cc.K_train),
           "evals_polished": ev,
           "coef": {k: [float(x) for x in v["coef"]] for k, v in arms.items()},
           "R_pool": {}, "predictor": j.get("predictor", {})}

    for a in alphas:
        k = f"{a:g}"
        rec["R_pool"][k] = {
            "R": float(1.0 - ev[f"cvar_{k}"]["cvar"][k] / ev["mean"]["cvar"][k]),
            "cvar_mean_arm": ev["mean"]["cvar"][k],
            "cvar_cvar_arm": ev[f"cvar_{k}"]["cvar"][k]}

    # Did the shared pool improve on the original run?  This is the diagnostic
    # that says whether the main run's mean baseline was converged.
    o = j.get("evals", {})
    if o:
        rec["baseline_check"] = {
            "mean_loss_original": float(o["mean"]["mean_loss"]),
            "mean_loss_polished": ev["mean"]["mean_loss"],
            "mean_loss_ratio": float(ev["mean"]["mean_loss"] / o["mean"]["mean_loss"]),
            "best_original_mean_loss_ratio":
                float(min(x["mean_loss"] for x in o.values()) / o["mean"]["mean_loss"]),
            "R_original_raw": {f"{a:g}": float(
                1.0 - o[f"cvar_{a:g}"]["cvar"][f"{a:g}"] / o["mean"]["cvar"][f"{a:g}"])
                for a in alphas},
            "R_original_best": {f"{a:g}": (j.get("ground_truth", {}).get(f"{a:g}") or {}).get("R")
                                for a in alphas}}
    if restore is not None:
        rec["rtn_amp_over_Tg"] = j["rtn_amp_over_Tg"]
        rec["rtn_p_requested"] = j.get("rtn_p_requested")
        restore()
    return rec


def polish(a):
    """Re-score points with a shared basin pool.  Reuses the saved
    coefficients, so every optimization is warm-started.

    `--src` selects which arm to repair.  The fluctuator arm needs this as much
    as the main grid did: its gains were obtained from a single multistart with
    a two-element comparator set, which is the state the main grid was in when
    its slope read -1.15.
    """
    fv, v2, fr = _sim()
    sub = getattr(a, "src", None) or f"stage6_{a.levels}level"
    src = Path(a.outdir) / sub
    out = Path(a.outdir) / ("polish_" + Path(sub).name)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(list(src.glob("s6_*.json")) + list(src.glob("s6rtn_*.json")))
    if not files:
        print(f"no point files under {src}"); return
    for f in files:
        j = json.loads(f.read_text())
        g = out / f"polish_{j['stem']}.json"
        if g.exists() and not a.force:
            continue
        t0 = time.time()
        rec = _polish_point(fv, v2, j, tuple(a.alpha), log=v2.log)
        rec["seconds"] = time.time() - t0
        g.write_text(json.dumps(rec, indent=1, default=float))
        bc = rec.get("baseline_check", {})
        v2.log(f"  -> {j['stem']}: pool={rec['n_pool']} "
               f"mean_loss {bc.get('mean_loss_ratio', float('nan')):.3f}x original, "
               f"R_pool(0.05)={rec['R_pool'].get('0.05', {}).get('R', float('nan')):+.4f} "
               f"(was raw {bc.get('R_original_raw', {}).get('0.05', float('nan')):+.4f}, "
               f"best {bc.get('R_original_best', {}).get('0.05', float('nan'))}) "
               f"{rec['seconds']:.0f}s")
    print(f"polish: {len(files)} points under {out}")


def _noise_spec(row):
    """What noise model the source stage actually ran this cell under."""
    if "tau_c" in row or row.get("quasi_static") is not None:
        tau = row.get("tau_c")
        return {"kind": "ou", "tau_c": (None if tau is None else float(tau)),
                "quasi_static": bool(row.get("quasi_static", tau is None))}
    return {"kind": "quasi_static_gaussian"}


def _install_source_noise(fv, v2, row):
    """Re-derive a previous stage's cell UNDER THAT STAGE'S NOISE MODEL.

    The first revision read only (Tg, phase_var, seed, levels) from the source
    summary and then ran stage0_v2's plain quasi-static Gaussian ensemble.  For
    Stage 5 that silently removed the relaxation channel (fixed separately, by
    splicing the recorded mixture back in analytically); for Stage 4 it removed
    the OU correlation time, so all 36 cells were re-derived as quasi-static.
    The damage was visible: the re-derived kappa came out 1.395x the recorded
    value and the predictor over-shot the recorded R by a factor of 30.  A
    retrospective run against the wrong noise model is not a weak test of the
    predictor, it is a test of a different system.

    Returns a restore() callable; call it before moving to the next cell.
    """
    spec = _noise_spec(row)
    if spec["kind"] != "ou":
        return lambda: None
    import stage4 as s4
    prev = (v2.gather_ordered, v2.Ensemble, s4.ColoredEnsemble.tau_c)
    s4.ColoredEnsemble.tau_c = (s4.QUASI if spec["quasi_static"] or spec["tau_c"] is None
                                else float(spec["tau_c"]))
    v2.gather_ordered = lambda uI, uQ, d, g: s4.colored_gather(fv, uI, uQ, d, g)
    v2.Ensemble = s4.ColoredEnsemble
    fv._prev.clear()

    def restore():
        v2.gather_ordered, v2.Ensemble, s4.ColoredEnsemble.tau_c = prev
        fv._prev.clear()
    return restore


def offsettest(a):
    """Decompose the calibration constant, with NO optimization at all.

    The predictor solves for an offset in ERROR-COORDINATE space and reports
    the gain that offset would buy.  The measured gain comes from re-optimizing
    the control.  Between the two sit two distinct losses, and the constant
    s = 0.588 is their product:

      (i) DELIVERABILITY.  The control may be unable to produce the requested
          offset.  Solving dc = B^+ (-t d) through the Jacobian already
          computed for the reachability projector and applying it gives
          R_offset: the gain from moving along the predicted direction by the
          predicted amount, linearly, without re-optimizing anything.

      (ii) RE-OPTIMIZATION.  Whatever the optimizer adds or loses on top.

    R_offset / R_pred measures (i); R_measured / R_offset measures (ii).  If
    (i) is the whole story the offset is simply not deliverable and the
    predictor should be rescaled; if (ii) is, the optimizer is leaving gain on
    the table and the predictor is right about what is available.  This costs a
    Jacobian and a handful of loss evaluations per point -- no multistarts --
    so it is the cheapest experiment in Stage 6.
    """
    fv, v2, fr = _sim()
    src = Path(a.outdir) / f"stage6_{a.levels}level"
    out = Path(a.outdir) / "offsettest"
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("s6_*.json"))
    if not files:
        print(f"no point files under {src}"); return
    rows = []
    for f in files:
        j = json.loads(f.read_text())
        g = out / f"off_{j['stem']}.json"
        if g.exists() and not a.force:
            rows.append(json.loads(g.read_text())); continue
        t0 = time.time()
        cc = _cell(v2, j["Tg"], j["phase_var"], j["seed"], j["levels"],
                   n_test=j.get("n_test"), k_train=j.get("K_train"))
        P = v2.Problem(j["Tg"], cc)
        c0 = np.asarray(j["coef_mean"], float).ravel()
        # W is a fixed function of the level count, so there is no reason to
        # rebuild it with a full propagator pass over 50 000 draws.
        import stage3 as _s3
        pb = reachable_projector(fv, v2, P, c0, j["levels"], _s3.metric_W(j["levels"]))
        B = np.asarray(pb["B"], float)              # d eps / d c
        Bp = np.linalg.pinv(B)

        def losses_of(c):
            ui, uq = P.filtered(np.asarray(c, float))
            return np.asarray(v2.losses(ui, uq, P.test), float).ravel()

        def cvars_of(c, alphas):
            """One propagator pass over the test ensemble, every alpha from it.
            The first version called a separate pass per alpha, which is a 4x
            waste: the loss vector does not depend on alpha, only the tail
            selection does."""
            L = losses_of(c)
            return {al: cvar_from_weights(L, tail_weights(L, None, al), al)
                    for al in alphas}

        cv0_all = cvars_of(c0, list(a.alpha))      # baseline, shared by all alpha

        rec = {"stem": j["stem"], "Tg": j["Tg"], "phase_var": j["phase_var"],
               "seed": j["seed"], "levels": j["levels"], "kappa": j["kappa"],
               "projector": {k: v for k, v in pb.items() if k not in ("P_B", "B")},
               "offset": {}}
        for al in a.alpha:
            k = f"{al:g}"
            if k not in j.get("predictor", {}):
                continue
            # THE TARGET IS t_best * step, NOT step.  refine_along_step scores
            # eps + t*s and reports the optimal t; R_pred is the gain at that t,
            # not at t = 1.  Applying the raw step compares a gain measured at
            # t_best against an offset applied at 1, which is wrong wherever the
            # line search did not land on 1 -- and on the main grid t_best is
            # 0.372, 0.451, 0.529 and 0.802 at four of the sixteen points.
            t_pred = j["predictor"][k].get("t_best")
            t_pred = 1.0 if t_pred is None else float(t_pred)
            step = np.asarray(j["predictor"][k]["step"], float)
            target = t_pred * step                 # the eps-offset R_pred refers to
            dc = Bp @ target
            achieved = B @ dc                      # what the control can actually do
            frac = (float(np.linalg.norm(achieved) / np.linalg.norm(target))
                    if np.linalg.norm(target) > 0 else float("nan"))
            cv0 = cv0_all[al]
            # A short scan along the same direction, in case the linear step
            # over- or under-shoots once the geometry co-moves.  t = 1 is the
            # predicted target itself, so it is part of the scan rather than a
            # separate evaluation.
            scan, cv1 = {}, None
            best_t, best_cv = 0.0, cv0
            for t in (0.5, 0.75, 1.0, 1.25, 1.5):
                cvt = cvars_of(c0 + t * dc, [al])[al]
                scan[f"{t:g}"] = float(cvt)
                if abs(t - 1.0) < 1e-12:
                    cv1 = cvt
                if cvt < best_cv:
                    best_t, best_cv = t, cvt
            rec["offset"][k] = {
                "R_pred": j["predictor"][k]["R_pred"], "t_pred": t_pred,
                "R_offset_t1": float(1.0 - cv1 / cv0),
                "R_offset_best": float(1.0 - best_cv / cv0), "t_best": best_t,
                # This will read ~1.000 on our data and that is not a result:
                # predicted_gain already projects the step onto range(B) before
                # storing it, so the target is reachable by construction. The
                # number that measures the geometric loss is R_offset/R_pred --
                # the gain a LINEAR move actually buys, against the gain the
                # quadratic model says it should -- and the t-scan is what
                # exposes the nonlinearity of eps(c) that the Jacobian misses.
                "deliverable_fraction": frac,
                "cvar_mean_control": float(cv0), "scan": scan,
                "R_measured_main": (j.get("ground_truth", {}).get(k) or {}).get("R")}
        rec["seconds"] = time.time() - t0
        g.write_text(json.dumps(rec, indent=1, default=float))
        rows.append(rec)
        o5 = rec["offset"].get("0.05", {})
        v2.log(f"  {j['stem']}: deliverable {o5.get('deliverable_fraction', float('nan')):.3f}"
               f"  R_pred {o5.get('R_pred', float('nan')):.4f}"
               f"  R_offset {o5.get('R_offset_best', float('nan')):+.4f}"
               f"  (t={o5.get('t_best')})  {rec['seconds']:.0f}s")
    print(f"offsettest: {len(rows)} points under {out}")


def ktrain(a):
    """Is the calibration constant finite-training-tail starvation?

    The CVaR objective's gradient is built only from the worst alpha fraction
    of its TRAINING ensemble, so alpha*K_train is the number of draws it
    actually sees.  If the constant s = 0.588 were statistical, it would climb
    toward 1 as K_train grows.  The indirect evidence says it is not -- s is
    flat across an 8x range in training-tail size (64 draws at alpha = 0.2, 8
    at alpha = 0.025: 0.558 vs 0.582) -- but flatness in alpha changes the tail
    size and the risk measure together, so it is suggestive rather than
    decisive.  Here alpha is fixed and only K_train moves.

    Both arms are resized together at every K_train, so budget parity holds at
    each point of the sweep and the comparison is not between a well-fed CVaR
    run and a starved comparator.
    """
    fv, v2, fr = _sim()
    d = Path(a.outdir) / "ktrain"
    d.mkdir(parents=True, exist_ok=True)
    al = float(a.alpha[0])
    for kap, kt in itertools.product(a.kappa, a.k_train_grid):
        pv = _pv_for_kappa(kap, a.Tg)
        stem = ("s6kt_Tg%g_k%g_K%d_s%d" % (a.Tg, kap, kt, a.seeds[0])).replace(".", "p")
        if (d / f"{stem}.json").exists() and not a.force:
            v2.log(f"  {stem} already done, skipping"); continue
        v2.log(f"=== K_train sweep: kappa~{kap}, K_train={kt} "
               f"(alpha*K_train = {al * kt:.0f} training-tail draws) ===")
        rec = _one_point(fv, v2, a.Tg, pv, a.seeds[0], a.levels, (al,), stem, d,
                         ground_truth=True, log=v2.log, n_test=a.n_test,
                         no_projector=a.no_projector, k_train=int(kt))
        rec["K_train_requested"] = int(kt)
        rec["alpha_K_train"] = float(al * kt)
        (d / f"{stem}.json").write_text(json.dumps(rec, indent=1, default=float))
    print(f"ktrain: grid under {d}")


def retro(a):
    """Re-derive the mean optimum at a previous stage's cells and evaluate the
    predictor there.  Costs one mean multistart per cell -- about an eighth of
    the original -- because stage4.py and stage5.py did not save coefficients.
    """
    fv, v2, fr = _sim()
    src = Path(a.stage)
    rows = []
    summ = next(iter(sorted(src.glob("*_summary.json"))), None)
    if summ is None:
        print(f"no summary under {src}"); return
    S = json.loads(summ.read_text())
    out = Path(a.outdir) / ("retro_" + src.name)
    out.mkdir(parents=True, exist_ok=True)
    seen = set()
    # Stale files from before levels entered the stem: they are ambiguous, so
    # say so rather than mixing them into the summary.
    old_style = [q for q in out.glob("retro_*.json") if "_lv" not in q.stem]
    if old_style:
        v2.log(f"  NOTE {len(old_style)} pre-existing files under {out} have no "
               f"_lv suffix and may be either level; delete them before "
               f"trusting the summary.")
    for r in S.get("rows", []):
        Tg = float(r.get("Tg", a.Tg)); pv = float(r.get("phase_var", 0.04))
        seed = int(r.get("seed", 0)); lv = int(r.get("levels", 2))
        # Levels MUST be in the filename.  stage5.py names a cell by
        # (Tg, T1, phase_var, seed) and runs 2-level and 4-level grids as
        # separate invocations, so both rows carry the SAME stem.  Without this
        # the two collide on one output file and whichever row comes last in
        # the summary silently overwrites the other -- which is exactly what
        # happened: re-running `retro --force` flipped
        # retro_s5_Tg12p7_T110_pv0p04_s0 from the 2-level cell (n_coord = 3,
        # beta = 1.924) to the 4-level one (n_coord = 11, beta = 0.099), and
        # the summary's median beta moved from 1.545 to 0.892 with nothing in
        # the pipeline having changed.  Two correct cells, one filename.
        stem = "retro_" + str(r.get("stem", f"{Tg}_{pv}_{seed}")) + f"_lv{lv}"
        if stem in seen:
            v2.log(f"  WARNING duplicate cell {stem} in {summ.name}; skipping "
                   f"the second copy rather than overwriting the first")
            continue
        seen.add(stem)
        f = out / f"{stem}.json"
        if f.exists() and not a.force:
            rows.append(json.loads(f.read_text())); continue
        restore = _install_source_noise(fv, v2, r)
        v2.log(f"=== retro {stem} ==="
               + (f"  [tau_c={r.get('tau_c')}]" if "tau_c" in r or
                  r.get("quasi_static") is not None else ""))
        rec = _one_point(fv, v2, Tg, pv, seed, lv, tuple(a.alpha), stem, out,
                         ground_truth=False, n_test=a.n_test,
                         no_projector=a.no_projector, k_train=a.k_train,
                         mix_p=r.get("p"), mix_beta=r.get("beta"))
        rec["recorded_R"] = r.get("R")
        rec["recorded_kappa"] = r.get("kappa")
        rec["recorded_beta"] = r.get("beta")
        rec["recorded_p"] = r.get("p")
        rec["recorded_l_j"] = r.get("l_j")
        rec["source_noise"] = _noise_spec(r)
        restore()
        f.write_text(json.dumps(rec, indent=1, default=float))
        rows.append(rec)
    print(f"retro: {len(rows)} cells under {out}")


# ===================================================================== #
# 7.  SUMMARY -- scores H1..H6                                           #
# ===================================================================== #

def _sp(x, y):
    from scipy import stats
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3:
        return [None, None]
    r = stats.spearmanr(x, y)
    return [float(r.statistic), float(r.pvalue)]


# The mean multistart is treated as converged at a cell when no other control
# found there beats it on ITS OWN objective by more than this.  Below it, the
# cell's mean baseline -- and therefore every R scored against it -- is invalid.
BASIN_TOL = 0.95
# |R_raw| below this and the CVaR control did not move the CVaR, so the
# displacement direction it implies is noise, not a disagreement.
NULL_R = 0.005


def _through_origin(x, y):
    """Through-origin slope, skipping missing entries."""
    xs = [float(a) for a, b in zip(x, y) if a is not None and b is not None]
    ys = [float(b) for a, b in zip(x, y) if a is not None and b is not None]
    if len(xs) < 2:
        return None
    xs, ys = np.asarray(xs), np.asarray(ys)
    return float(np.sum(xs * ys) / np.sum(xs ** 2)) if np.sum(xs ** 2) > 0 else None


def _seed_blocked(rows, rp, R, mask):
    """Slope and rank correlation within each seed, and pooled after centring
    each seed.  Stage 1b needed exactly this (rho 0.712 -> 0.962): a per-seed
    offset destroys a pooled correlation without touching the within-seed one.
    """
    from scipy import stats
    out = {"by_seed": {}}
    seeds = sorted({r["seed"] for r in rows})
    cx, cy = [], []
    for sd in seeds:
        m = mask & np.array([r["seed"] == sd for r in rows])
        if m.sum() < 3:
            continue
        x, y = rp[m], R[m]
        out["by_seed"][str(sd)] = {
            "n": int(m.sum()),
            "slope": float(np.sum(x * y) / np.sum(x ** 2)) if np.sum(x ** 2) > 0 else None,
            "spearman": _sp(x, y)}
        cx.append(x - x.mean()); cy.append(y - y.mean())
    if len(cx) >= 2:
        X, Y = np.concatenate(cx), np.concatenate(cy)
        r_ = stats.spearmanr(X, Y)
        out["pooled_seed_centred"] = [float(r_.statistic), float(r_.pvalue)]
        out["n_pooled"] = int(X.size)
    return out


def _replicate_sd(rows, key):
    cells = {}
    for r in rows:
        cells.setdefault((round(r["Tg"], 6), round(r["phase_var"], 8),
                          r["levels"]), []).append(r[key])
    dif = [abs(v[0] - v[1]) for v in cells.values() if len(v) == 2]
    return float(np.mean(dif) / math.sqrt(2.0)) if dif else float("nan")


def summary(a):
    from scipy import stats
    base = Path(a.outdir)
    files = [p for p in base.rglob("s6_*.json")] + [p for p in base.rglob("retro_*.json")]
    recs = []
    for p in sorted(files):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        if "predictor" in j:
            recs.append(j)
    if not recs:
        print(f"no Stage 6 point files under {base}"); return
    alphas = [f"{x:g}" for x in a.alpha]

    # Polished points, keyed by stem.  When `polish` has been run, R comes from
    # the shared basin pool (symmetric restart budget) instead of the main
    # run's best-of-{mean, rev_*} rule -- see _polish_point for why that rule
    # is not a fairness guard when the mean multistart has not converged.
    pol = {}
    for p in sorted(base.rglob("polish_s6_*.json")):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        pol[j.get("stem")] = j
    for r in recs:
        ev = r.get("evals")
        r["_basin_ok"] = (bool(min(x["mean_loss"] for x in ev.values())
                               >= BASIN_TOL * ev["mean"]["mean_loss"]) if ev else None)
        r["_basin_ratio"] = (float(min(x["mean_loss"] for x in ev.values())
                                   / ev["mean"]["mean_loss"]) if ev else None)
        r["_pol"] = pol.get(r.get("stem"))
    n_gt = sum(1 for r in recs if r.get("evals"))
    n_ok = sum(1 for r in recs if r.get("_basin_ok"))
    # Post-repair convergence.  The number above is computed from the MAIN
    # run's evals and therefore keeps reporting the pre-polish state forever;
    # what matters after `polish` is whether the pooled mean arm was itself
    # beaten, which is what baseline_check records.
    pol_ratio = [v.get("baseline_check", {}).get("mean_loss_ratio")
                 for v in pol.values()]
    pol_ratio = [x for x in pol_ratio if x is not None]
    out = {"n_points": len(recs), "alphas": alphas,
           "R_source": ("shared basin pool (polish)" if pol else
                        "main run, best of {mean, rev_*}"),
           "n_polished": len(pol),
           "baseline_convergence": {
               "n_with_ground_truth": n_gt, "n_converged": n_ok,
               "tolerance": BASIN_TOL,
               "worst_ratio": (min([r["_basin_ratio"] for r in recs
                                    if r.get("_basin_ratio") is not None], default=None)),
               "note": ("PRE-REPAIR: fraction of cells in which the main run's "
                        "mean multistart was NOT beaten by any other control on "
                        "its own objective. A low number is what motivated the "
                        "shared basin pool; it does not describe the pooled "
                        "results, which are summarized below.")},
           "baseline_after_polish": ({
               "n": len(pol_ratio),
               "median_mean_loss_ratio": float(np.median(pol_ratio)),
               "worst_mean_loss_ratio": float(np.min(pol_ratio)),
               "n_improved_over_main": int(sum(1 for x in pol_ratio if x < 0.999)),
               "note": ("mean loss of the POOLED mean arm divided by the main "
                        "run's; below 1 means the pool found a better mean "
                        "optimum than the original multistart did")}
               if pol_ratio else None),
           "by_alpha": {}}

    # ---- H1: closed form vs inner optimization
    h1q, h1l = [], []
    for r in recs:
        if not isinstance(r.get("R_model_q"), dict) or "error" in r.get("R_model_q", {}):
            continue
        if r.get("fit_R2_min", 0) < 0.94:
            continue
        for k in alphas:
            rm = r["R_model_q"].get(k)
            if not rm or rm <= 1e-6:
                continue
            # Only where the predictor is resolved.  The boundary sweep adds
            # points whose gain is at or below the resolution floor; a RELATIVE
            # difference between two closed forms that both tend to zero there
            # is dominated by the floor, not by any disagreement of substance.
            if not (r["predictor"].get(k, {}) or {}).get("resolved"):
                continue
            q = r["predictor"].get(k, {}).get("R_pred_quadratic")
            l = r["predictor"].get(k, {}).get("R_pred")
            if q:
                h1q.append(abs(q / rm - 1.0))
            if l:
                h1l.append(abs(l / rm - 1.0))
    out["H1_closed_form"] = {
        "n": len(h1l),
        # The line-search statistic is the one validated against brute force
        # (S3: 0.6 / 3.3 / 4.4% at kappa = 0.1 / 0.25 / 0.5); the quadratic one
        # is known to over-predict by 16-77% there.  Both are reported because
        # H1 compares two INDEPENDENT closed forms, neither of which is ground
        # truth, so the number is a calibration, not a verdict.
        "median_rel_diff": float(np.median(h1l)) if h1l else None,
        "median_rel_diff_quadratic": float(np.median(h1q)) if h1q else None,
        "H1_pass": bool(h1l and np.median(h1l) < 0.15)}

    for k in alphas:
        # Two populations.  H2/H3/H4 compare against a MEASURED gain and so need
        # a point that was run with ground truth; H6 asks only where R_pred
        # crosses its own floor and is therefore scored on every point that has
        # a predictor, including the predictor-only boundary sweep.  Conflating
        # them is what kept 24 boundary points out of the scoring entirely.
        # Main-grid family only: the kappa sweep and the predictor-only boundary
        # extension, both written as s6_*.  The retrospective cells (retro_*)
        # and the fluctuator arm (s6rtn_*) are different gate times and
        # different noise models, and pooling them into a boundary location
        # would be comparing systems, not kappa values.
        rows_pred = [r for r in recs
                     if str(r.get("stem", "")).startswith("s6_")
                     and k in r.get("predictor", {})]
        rows = [r for r in rows_pred if k in r.get("ground_truth", {})]
        if not rows_pred:
            continue
        kp_all = np.array([r["kappa"] for r in rows_pred])
        rp_all = np.array([r["predictor"][k]["R_pred"] for r in rows_pred])
        fl_all = np.array([r["predictor"][k]["R_min"] for r in rows_pred])
        res_all = rp_all > fl_all
        o_all = np.argsort(kp_all)
        kc_obs_all = next((float(kp_all[i]) for i in o_all if res_all[i]), None)
        kc_hi = next((float(kp_all[i]) for i in reversed(o_all) if not res_all[i]), None)
        boundary = {
            "n_points": int(len(rows_pred)), "n_resolved": int(res_all.sum()),
            "kappa_range": [float(kp_all.min()), float(kp_all.max())],
            "kappa_smallest_resolved": kc_obs_all,
            "kappa_largest_unresolved": kc_hi,
            "median_floor": float(np.median(fl_all)),
            "grid": [{"kappa": float(kp_all[i]), "seed": rows_pred[i]["seed"],
                      "R_pred": float(rp_all[i]), "R_min": float(fl_all[i]),
                      "resolved": bool(res_all[i])} for i in o_all]}
        if not rows:
            out["by_alpha"][k] = {"n": 0, "n_predictor_only": int(len(rows_pred)),
                                  "H6_boundary_location": boundary}
            continue
        rp = np.array([r["predictor"][k]["R_pred"] for r in rows])
        fl = np.array([r["predictor"][k]["R_min"] for r in rows])

        def _R_of(r):
            q = (r.get("_pol") or {}).get("R_pool", {}).get(k)
            if q is not None:
                return float(q["R"])
            return float(r["ground_truth"][k]["R"])

        R = np.array([_R_of(r) for r in rows])
        R_raw = np.array([float(r["ground_truth"][k]["R_raw"]) for r in rows])
        basin = np.array([bool(r.get("_basin_ok")) for r in rows])
        cs = np.array([r["ground_truth"][k]["cos"] for r in rows], float)
        kp = np.array([r["kappa"] for r in rows])
        flat = [{"Tg": r["Tg"], "phase_var": r["phase_var"], "levels": r["levels"],
                 "R": _R_of(r)} for r in rows]
        sd_rep = _replicate_sd(flat, "R")
        sd_tot = float(np.std(R, ddof=1)) if R.size > 1 else float("nan")
        ceil = (math.sqrt(max(0.0, 1.0 - (sd_rep / sd_tot) ** 2))
                if np.isfinite(sd_rep) and sd_tot > 0 else float("nan"))

        res = rp > fl
        # H2 direction.  The claim is that the CVaR control sits DISPLACED from
        # the mean control in the predicted direction, so the test is restricted
        # to points where that premise holds.  Two gates, both per-alpha:
        #   same basin -- the CVaR run must not have found a BETTER mean optimum
        #     (mean loss below the mean control's).  When it has, it did not
        #     trade mean for tail, it jumped, and a local displacement model
        #     does not describe the move.
        #   non-null -- the CVaR control must actually have moved the CVaR.  At
        #     |R_raw| ~ 0 there is no displacement to have a direction.
        # This is a per-alpha gate on the CVaR arm itself, NOT the cell-level
        # baseline-convergence gate used for R: whether some reverse run found
        # a better basin elsewhere says nothing about this displacement.
        ml_ratio = np.array([r["evals"][f"cvar_{k}"]["mean_loss"]
                             / r["evals"]["mean"]["mean_loss"]
                             if r.get("evals") else 1.0 for r in rows])
        nonnull = np.abs(R_raw) > NULL_R
        good = res & (ml_ratio >= BASIN_TOL) & nonnull & np.isfinite(cs)
        loose = res & np.isfinite(cs)          # the old, ungated criterion
        # Absence of evidence is not evidence against: cos = 0.53 in three
        # dimensions has null p = 0.47, so it is uninformative, not
        # contradictory.  What would refute H2 is SIGNIFICANT anti-alignment.
        n_anti = int(sum(1 for i in range(len(rows))
                         if good[i] and cs[i] < 0.0 and
                         cos_null_pvalue(cs[i], rows[i]["n_coord"]) < 0.05))
        # H3 level, through-origin slope on the resolved points
        slope = (float(np.sum(rp[res] * R[res]) / np.sum(rp[res] ** 2))
                 if res.sum() >= 2 and np.sum(rp[res] ** 2) > 0 else None)
        # H4 boundary: below the floor, |R| small and the sign random
        below = ~res
        nb = int(below.sum())
        npos = int(np.sum(R[below] > 0))
        sign_p = (float(stats.binomtest(npos, nb, 0.5).pvalue) if nb >= 3 else None)
        # H6 boundary location
        C = float(np.median([(r.get("C_alpha") or {}).get(k) or np.nan
                              for r in rows_pred]))
        kc_pred = kappa_critical(boundary["median_floor"], C)
        kc_obs = boundary["kappa_smallest_resolved"]

        out["by_alpha"][k] = {
            "n": len(rows), "n_resolved": int(res.sum()),
            "H2_direction": {
                "n": int(good.sum()),
                "median_cos": float(np.median(cs[good])) if good.any() else None,
                "min_cos": float(np.min(cs[good])) if good.any() else None,
                # A single 3-dimensional agreement is WEAK: cos is uniform on
                # [-1,1] there, so cos > 0.9 has null probability 0.05 one-sided.
                # Power comes from combining points (Fisher) and from the
                # 11-coordinate leakage-aware arm, not from one point.
                "per_point_p_null": [float(cos_null_pvalue(c, rows[i]["n_coord"]))
                                     for i, c in enumerate(cs) if good[i]],
                "fisher_p": (float(stats.combine_pvalues(
                    [max(cos_null_pvalue(c, rows[i]["n_coord"]), 1e-12)
                     for i, c in enumerate(cs) if good[i]], method="fisher")[1])
                    if int(good.sum()) >= 2 else None),
                # Reported so the gating is auditable: the same statistics over
                # every resolved point, gated or not, and what the gate removed.
                "n_ungated": int(loose.sum()),
                "median_cos_ungated": float(np.median(cs[loose])) if loose.any() else None,
                "min_cos_ungated": float(np.min(cs[loose])) if loose.any() else None,
                "excluded": [{"stem": rows[i]["stem"], "cos": float(cs[i]),
                              "cvar_arm_mean_loss_ratio": float(ml_ratio[i]),
                              "R_raw": float(R_raw[i])}
                             for i in range(len(rows)) if loose[i] and not good[i]],
                "n_significantly_anti_aligned": n_anti,
                "H2_pass": bool(int(good.sum()) >= 3 and n_anti == 0 and
                                float(np.median(cs[good])) > COS_MIN)},
            "H3_level": {
                "slope": slope, "n": int(res.sum()),
                "spearman": _sp(rp[res], R[res]) if res.sum() >= 3 else [None, None],
                "replicate_sd_R": sd_rep, "sd_R": sd_tot,
                # sd_rep is scatter in the DEPENDENT variable, so it caps the
                # attainable correlation; it does not attenuate the slope.
                # Compare spearman against this ceiling, not the slope.
                "attenuation_ceiling": ceil,
                "spearman_over_ceiling": (
                    None if not np.isfinite(ceil) or ceil <= 0 or res.sum() < 3
                    else float(abs(_sp(rp[res], R[res])[0] or 0.0) / ceil)),
                # Seed-blocked, and the raw arm.  Both exist because R against
                # the main run's best-of-{mean, rev_*} comparator is dominated
                # by whether the mean multistart converged in that cell.
                "seed_blocked": _seed_blocked(rows, rp, R, res),
                "slope_raw": (float(np.sum(rp * R_raw) / np.sum(rp ** 2))
                              if np.sum(rp ** 2) > 0 else None),
                "spearman_raw": _sp(rp, R_raw),
                "slope_same_basin": (float(np.sum(rp[res & basin] * R[res & basin])
                                           / np.sum(rp[res & basin] ** 2))
                                     if (res & basin).sum() >= 2 and
                                     np.sum(rp[res & basin] ** 2) > 0 else None),
                "spearman_same_basin": (_sp(rp[res & basin], R[res & basin])
                                        if (res & basin).sum() >= 3 else [None, None]),
                "n_same_basin": int((res & basin).sum()),
                "H3_pass": bool(slope is not None and SLOPE_BAND[0] <= slope <= SLOPE_BAND[1]),
                # H3 as stated fails; what holds instead is a constant slope with
                # a near-perfect ordering.  Both are reported so neither claim
                # can be mistaken for the other.
                "H3_calibrated_pass": bool(
                    slope is not None and CALIBRATION_BAND[0] <= slope <= CALIBRATION_BAND[1]
                    and (_sp(rp[res], R[res])[0] or 0.0) > 0.8)},
            "H4_boundary": {
                # pointwise: |R_i| < 2 R_min,i at every point below its OWN floor.
                # (The previous revision compared against 2 * median(floor),
                # which is not the estimator H4 states.  Caught in review.)
                "n_below_floor": nb, "n_positive": npos, "sign_binomial_p": sign_p,
                "n_exceeding_own_floor": int(np.sum(np.abs(R[below]) >= 2 * fl[below]))
                if nb else 0,
                "worst_ratio_R_over_floor": float(np.max(np.abs(R[below]) / fl[below]))
                if nb else None,
                "median_floor": float(np.median(fl)),
                # H4 needs a MEASURED gain below the floor, so predictor-only
                # boundary points cannot score it however far down they reach.
                "n_predictor_only_below_floor":
                    int(np.sum(~res_all)) - int(np.sum(~res_all & np.array(
                        [k in r.get("ground_truth", {}) for r in rows_pred]))),
                "untested": bool(nb == 0),
                "untested_reason": (None if nb else
                                    "no cell with ground truth falls below its own "
                                    "floor; run `main` at a few kappa inside the "
                                    "bracket reported under H6"),
                "H4_pass": bool(nb >= 3 and sign_p is not None and sign_p > 0.05 and
                                np.all(np.abs(R[below]) < 2 * fl[below]))},
            "H6_boundary_location": {
                **boundary,
                "C_alpha_median": C, "kappa_c_predicted": kc_pred,
                "kappa_c_observed": kc_obs,
                "kappa_min_on_grid": float(np.min(kp_all)),
                "bracket": ([boundary["kappa_largest_unresolved"], kc_obs]
                            if boundary["kappa_largest_unresolved"] and kc_obs
                            else None),
                # If the SMALLEST kappa on the grid is already resolved, the
                # boundary is below the grid and kappa_c_observed is a censoring
                # bound, not a measurement.  Scoring that as a failure is wrong;
                # it is untested.  Extend the grid downward (predictor-only is
                # seconds per point) before reading anything into H6 or H4.
                "censored": bool(kc_obs is not None and
                                 abs(kc_obs - float(np.min(kp_all))) < 1e-12),
                "H6_pass": bool(kc_obs is not None and kc_pred > 0 and
                                not abs(kc_obs - float(np.min(kp_all))) < 1e-12 and
                                0.5 <= kc_obs / kc_pred <= 2.0)},
        }

    # ---- H5: the leverage 2x2, if both arms are present
    t1 = [json.loads(p.read_text()) for p in base.rglob("retro_s5_*.json")]
    rt = [json.loads(p.read_text()) for p in base.rglob("s6rtn_*.json")]
    if t1 or rt:
        def lam(rs, key="predictor"):
            v = [r[key]["0.05"]["Lambda"] for r in rs
                 if "0.05" in (r.get(key) or {})]
            return float(np.median(v)) if v else None

        def rp(rs, key="predictor"):
            v = [r[key]["0.05"]["R_pred"] for r in rs
                 if "0.05" in (r.get(key) or {})]
            return float(np.median(v)) if v else None

        # The T1 arm MUST come from predictor_T1 -- the plain `predictor` block
        # of a retro cell is the dephasing-only ensemble, where Lambda = 1 by
        # construction and the number says nothing about relaxation.
        n_t1_mix = sum(1 for r in t1 if "predictor_T1" in r)
        lam_t1 = lam(t1, "predictor_T1")
        lam_rtn = lam(rt)
        # The RTN arm is an AMPLITUDE SWEEP: Lambda is supposed to fall from ~1
        # to ~0 across it, so a median over the sweep is not a statistic of
        # anything.  Score the crossover -- the value at the smallest and
        # largest excursion, and whether R_pred falls monotonically with it.
        def _amp(r):
            v = r.get("rtn_amp_over_Tg")
            return float(v) if v is not None else float("nan")
        rts = sorted([r for r in rt if "0.05" in (r.get("predictor") or {})],
                     key=lambda r: (_amp(r), r.get("rtn_p_requested") or 0.0))
        cross = None
        if len(rts) >= 2:
            lamv = [float(r["predictor"]["0.05"]["Lambda"]) for r in rts]
            rpv = [float(r["predictor"]["0.05"]["R_pred"]) for r in rts]
            sup = [((r.get("suppression") or {}).get("0.05") or {}).get("ratio")
                   for r in rts]
            kpv = [float(r.get("kappa", float("nan"))) for r in rts]
            have = [i for i in range(len(rts)) if sup[i] is not None]
            cross = {
                "amps": [_amp(r) for r in rts],
                "kappa": kpv,
                "Lambda": lamv,
                "R_pred": rpv,
                "suppression_ratio": sup,
                # R_pred vs Lambda across the sweep is CONFOUNDED: amplitude
                # drives kappa up and Lambda down, so this correlation is -1 by
                # construction and is reported only to make that visible.
                "spearman_Lambda_Rpred_CONFOUNDED": _sp(lamv, rpv),
                "spearman_kappa_amp": _sp(kpv, [_amp(r) for r in rts]),
                # The real test: at one control and one kappa, does removing the
                # rare branch recover the gain by exactly the factor Lambda?
                "suppression_vs_Lambda_rel_error": (
                    [abs(sup[i] / max(lamv[i], 1e-300) - 1.0) for i in have]
                    if have else None),
                "conflict_cos": [((r.get("suppression") or {}).get("0.05") or {}).get("conflict_cos")
                                 for r in rts],
                "p_over_alpha": [((r.get("suppression") or {}).get("0.05") or {}).get("p_over_alpha")
                                 for r in rts],
                "fit_R2_min": [r.get("fit_R2_min") for r in rts],
                # The identity Lambda = R_mixture/R_bulk is only testable where
                # the p/alpha expansion applies AND the reduced model still fits.
                # Points failing either are reported, not scored.
                "valid_for_identity": [bool((((r.get("suppression") or {}).get("0.05") or {})
                                             .get("p_over_alpha") or 1.0) < 0.1
                                            and (r.get("fit_R2_min") or 0) > 0.99)
                                       for r in rts],
                "Lambda_small_amp": lamv[0], "Lambda_large_amp": lamv[-1],
                "suppression_small_amp": sup[0], "suppression_large_amp": sup[-1],
                # The claim that survives both arms: the two-population
                # predictor tracks the MEASURED gain.  Lambda alone does not --
                # it is the inert limit.  This is scored against ground truth,
                # at whatever p/alpha, so it needs no expansion to be valid.
                "R_pred_vs_truth": [
                    {"amp": _amp(r), "p": r.get("rtn_p_requested"),
                     "p_over_alpha": ((r.get("suppression") or {}).get("0.05") or {}).get("p_over_alpha"),
                     "conflict_cos": ((r.get("suppression") or {}).get("0.05") or {}).get("conflict_cos"),
                     "Lambda": float(r["predictor"]["0.05"]["Lambda"]),
                     "R_pred": float(r["predictor"]["0.05"]["R_pred"]),
                     "R_measured": ((r.get("ground_truth") or {}).get("0.05") or {}).get("R"),
                     "fit_R2_min": r.get("fit_R2_min")}
                    for r in rts],
                "slope_pred_vs_truth": _through_origin(
                    [float(r["predictor"]["0.05"]["R_pred"]) for r in rts],
                    [((r.get("ground_truth") or {}).get("0.05") or {}).get("R") for r in rts])}
        out["H5_leverage"] = {
            "n_T1_cells": len(t1), "n_T1_with_mixture": n_t1_mix,
            "n_RTN": len(rt), "rtn_crossover": cross,
            "median_Lambda_T1": lam_t1,
            "median_Lambda_RTN": lam_rtn,
            "median_Lambda_T1_dephasing_only": lam(t1),
            "median_R_pred_T1_mixture": rp(t1, "predictor_T1"),
            "median_R_pred_T1_dephasing_only": rp(t1),
            "median_beta_T1": (float(np.median([r["mixture"]["beta"] for r in t1
                                                if "mixture" in r]))
                               if n_t1_mix else None),
            # T1: a rare, control-INERT channel suppresses the gain.
            # RTN: a rare channel the control CAN shape does too, once it is
            # rare enough -- so the pass condition is the CROSSOVER, not a
            # median.  Lambda near 1 at the smallest excursion, small at the
            # largest, and R_pred falling with it.
            "H5_pass": bool(
                n_t1_mix and (lam_t1 or 1) < 0.1 and cross is not None
                and cross["Lambda_small_amp"] > 0.5
                and cross["Lambda_large_amp"] < 0.2
                and cross["suppression_vs_Lambda_rel_error"] is not None
                and max(cross["suppression_vs_Lambda_rel_error"]) < 0.25)}
        if t1 and not n_t1_mix:
            out["H5_leverage"]["note"] = (
                "retro cells carry no recorded (p, beta): the T1 arm was run "
                "against a dephasing-only simulator and Lambda = 1 is an "
                "artefact of that, not a measurement. Re-run `retro` with a "
                "stage5 summary that records p and beta.")

    p = base / "stage6_summary.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {p}")
    return out


# ===================================================================== #
# 8.  CLI                                                                #
# ===================================================================== #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(q):
        q.add_argument("--outdir", default="results_stage6")
        q.add_argument("--Tg", type=float, default=20.0)
        q.add_argument("--levels", type=int, default=2)
        q.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
        q.add_argument("--alpha", type=float, nargs="+", default=list(ALPHA_GRID))
        q.add_argument("--force", action="store_true")
        q.add_argument("--n-test", type=int, default=50000,
                       help="test-ensemble size; the predictor's resolution. "
                            "Rule: min(alpha) * n_test >= 400. Enforced.")
        q.add_argument("--k-train", type=int, default=None,
                       help="training-ensemble size, shared by the mean, CVaR "
                            "and reverse runs at a point. Default: "
                            "max(128, ceil(8/min(alpha))). Enforced.")
        q.add_argument("--allow-thin", action="store_true",
                       help="run anyway when min(alpha)*n_test < 400")
        q.add_argument("--no-projector", action="store_true",
                       help="skip the reachable-subspace Jacobian (34 extra "
                            "small propagator evaluations per point)")
        return q

    sub.add_parser("selftest")
    q = common(sub.add_parser("probe"))
    q.add_argument("--kappa", type=float, nargs="+",
                   default=[0.03, 0.05, 0.08, 0.12, 0.18, 0.26, 0.36, 0.50])
    q = common(sub.add_parser("run"))
    q.add_argument("--kappa", type=float, nargs="+",
                   default=[0.03, 0.05, 0.08, 0.12, 0.18, 0.26, 0.36, 0.50])
    q.add_argument("--predictor-only", action="store_true",
                   help="skip the CVaR ground truth; seconds per point instead of minutes")
    q = common(sub.add_parser("polish"))
    q.add_argument("--src", default=None,
                   help="subdirectory of --outdir holding the points to repair "
                        "(default: stage6_<levels>level; use stage6_rtn for the "
                        "fluctuator arm)")
    common(sub.add_parser("offsettest"))
    q = common(sub.add_parser("ktrain"))
    q.add_argument("--kappa", type=float, nargs="+", default=[0.12, 0.26, 0.5],
                   help="a few kappa values spanning the resolved range")
    q.add_argument("--k-train-grid", type=float, nargs="+",
                   default=[128, 320, 1280, 5120],
                   help="training-ensemble sizes; alpha*K_train is the number "
                        "of draws the CVaR gradient is actually built from")
    q = common(sub.add_parser("rtn"))
    q.add_argument("--rtn-amp", type=float, nargs="+",
                   default=[0.2, 0.5, 1.0, 2.0, 4.0],
                   help="fluctuator half-amplitude in units of 1/Tg; this is "
                        "the crossover axis")
    q.add_argument("--rtn-rate", type=float, default=1e-3, help="flips per ns")
    q.add_argument("--rtn-p", type=float, nargs="+", default=[0.02],
                   help="fraction of draws carrying a fluctuator (its rarity)")
    q.add_argument("--rtn-pv", type=float, default=0.04)
    q = common(sub.add_parser("retro"))
    q.add_argument("--stage", required=True)
    common(sub.add_parser("summary"))

    a = ap.parse_args()
    if a.cmd == "ktrain":
        # The guard below exists to stop an accidentally starved training tail.
        # Here a starved tail IS the experiment, so report the grid and skip it.
        print("[stage6] K_train sweep at alpha=%g: " % float(a.alpha[0])
              + ", ".join(f"K={int(k)} ({float(a.alpha[0])*k:.1f} tail draws)"
                          for k in a.k_train_grid))
    elif a.cmd not in ("selftest",) and getattr(a, "alpha", None):
        if a.k_train is None:
            a.k_train = auto_k_train(a.alpha)
        tail_tr = min(a.alpha) * a.k_train
        if tail_tr < TAIL_TARGET and not getattr(a, "allow_thin", False):
            raise SystemExit(
                f"min(alpha)*K_train = {tail_tr:.1f} < {TAIL_TARGET}: the CVaR "
                f"objective would be built from {tail_tr:.1f} training draws at "
                f"alpha={min(a.alpha):g}. Raise --k-train to "
                f"{auto_k_train(a.alpha)}, drop the smallest alpha, or pass "
                f"--allow-thin.")
        print(f"[stage6] K_train={a.k_train} "
              f"({tail_tr:.1f} training draws in the alpha={min(a.alpha):g} tail), "
              f"n_test={a.n_test} "
              f"({min(a.alpha)*a.n_test:.0f} test draws in that tail)")
        need = 400
        got = min(a.alpha) * a.n_test
        if got < need and not getattr(a, "allow_thin", False):
            raise SystemExit(
                f"min(alpha)*n_test = {got:.0f} < {need}: the tail-conditional "
                f"mean and its covariance cannot be estimated at alpha="
                f"{min(a.alpha):g} with n_test={a.n_test}. Raise --n-test to "
                f"{int(math.ceil(need / min(a.alpha)))}, drop the smallest "
                f"alpha, or pass --allow-thin to override.")
    if a.cmd == "selftest":
        ok, _ = selftest(); sys.exit(0 if ok else 1)
    {"probe": probe, "run": run, "polish": polish, "offsettest": offsettest,
     "ktrain": ktrain, "rtn": rtn, "retro": retro, "summary": summary}[a.cmd](a)


if __name__ == "__main__":
    main()