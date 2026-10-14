"""
stage3.py -- VIPER Stage 3.  Two experiments, both of them limitations the
paper currently states in its own Limitations section.

=============================================================================
PART A -- THE CONTROLLED EXPERIMENT AT FIXED kappa   (steps: calibrate, run)
=============================================================================
Stage 1 showed that rho, Tg and the noise strength predict R_tail only
through kappa = |Q|/|A| (partial correlations ~ 0 given kappa).  That test is
OBSERVATIONAL: the knobs were never varied independently of kappa, so the
paper can only say "within these data".  Stage 3 makes it a designed
experiment.  For every cell of a (rho, Tg) grid we SOLVE for the noise scale
s that puts the mean control's kappa on a target value kappa*, then run the
standard mean-vs-CVaR comparison there.  Rows of the design therefore differ
in kappa and nothing else; columns differ in the knobs at (almost) constant
kappa.

  design:  kappa* in {0.20, 0.35, 0.55}   x   rho in {0, 0.5, 0.9}
                                          x   Tg  in {8, 12.7, 20} ns
           27 points, 2 levels, base noise (sigma_d, sigma_a) = (2.5 MHz,
           0.127) scaled by one number s per cell.

  calibration:  kappa(s) is measured on the MEAN control that the standard
           multistart returns at that s (kappa is a property of the mean
           control, so it is only known after an optimization).  Secant
           iteration in log s, started from the empirical kappa ~ s^0.5,
           stops at |log kappa/kappa*| < log(1.05), <= 6 optimizations/cell.

PRE-REGISTERED PREDICTIONS (fixed before the run; `summary` scores them)
  A1  Between rows: median R_tail increases with kappa*  (0.20 < 0.35 < 0.55).
  A2  Within a row: no significant dependence of R_tail on rho or on Tg
      (Kruskal-Wallis, Holm-corrected, alpha = 0.05).  This is the controlled
      version of the Stage-1 independence claim.
  A3  Within a row, the residual scatter is GEOMETRY: R_tail correlates with
      the zero-parameter reduced-model prediction R_model (Spearman > 0).
  A4  R_tail <= R_model at every point (upper-envelope property, prospective).
  A5* Reported as an EFFECT SIZE, not only as a p-value: with three cells per
      knob value a non-significant Kruskal-Wallis is weak evidence of nothing,
      so `summary` also gives the largest between-group gap of the residual
      and a one-sided 95% bootstrap upper bound on it, as a factor.  The claim
      to make is "at fixed kappa, varying rho over {0, 0.5, 0.9} moves
      R/(C kappa^2) by at most a factor F", with F from the data.
  A5  THE SHARP ONE.  The mechanism predicts a VALUE per cell, not only an
      ordering: the reduced model R_model (exact for any kappa) and, inside
      the domain where it was verified (kappa < 0.35), its analytic limit
      C kappa^2 with C in closed form from the fitted A and Q (see C_closed).
      The residual log R_tail - log R_model should not depend on rho or Tg.
      Passing A2 while failing A5 would mean kappa orders the results but
      does not explain them.
  Primary statistic: closed-system R_tail(0.05) -- the T1/T2 floor dilutes the
  Lindblad ratio by a different factor at each cell (the cells have different
  loss magnitudes by construction), which would confound A2 and A5.  Both are
  stored, and the dilution model of the paper relates them.

=============================================================================
PART B -- LEAKAGE-AWARE kappa                  (steps: diagnose-lk, summary)
=============================================================================
kappa is read off the qubit-block error rotation vector, so it is blind to
leakage; the paper notes this is probably why the transmon diverges at
somewhat smaller kappa (0.1-0.24) than the qubit (0.13).  Part B extends the
error coordinate to the leakage amplitudes and re-diagnoses EVERY saved
control of every stage (no new optimizations).

  coordinates   x = (eps ; Re l ; Im l),  l_{kj} = <k|U|j>, k >= 2, j = 0,1,
                in the same global-phase gauge as eps.
  metric        L ~ x^T W x with W = diag(W_eps, W_leak),
                W_eps = diag(3,3,2)/16     (four-state average, as in the paper)
                W_leak = (1+lambda)/4 * [[I(x)Gr, -I(x)Gi], [I(x)Gi, I(x)Gr]],
                G = conj(S), S = sum_psi |psi><psi| = 2 I + (X+Y)/2.
                Derivation: 1-F_psi = (eps^T W_eps eps)_psi + <psi|l^dag l|psi>
                and P_leak carries the extra lambda, so the leakage block gets
                (1 + lambda).  `selftest` checks it against the exact
                four-state loss of random unitaries (no simulator needed).
  variables     kappa      = |Q_eps| / |A_eps|                 (the paper's)
                kappa_W    = |W^1/2 Q_eps| / |W^1/2 A_eps|     (metric-weighted)
                kappa_lk   = |W^1/2 Q_x| / |W^1/2 A_x|         (+ leakage)
                R_model_lk = reduced model in the full coordinates.

PRE-REGISTERED PREDICTIONS
  B1  kappa_lk separates divergent from non-divergent points at least as well
      as kappa, pooled over 2- and 4-level points (AUC_lk >= AUC).
  B2  The 4-level and 2-level decision thresholds move TOGETHER in kappa_lk:
      |log(thr_4/thr_2)| smaller for kappa_lk than for kappa.
  B3  R_model_lk is an upper envelope for the transmon too (R_tail <= R_model
      at >= 90% of 4-level points), which the qubit-only model is not required
      to be.

=============================================================================
PART C -- GENERALITY                                    (steps: gates, summary)
=============================================================================
The mechanism was established for one target (X_pi/2), one anharmonicity
(-200 MHz) and one pulse basis (17 Hann-windowed sine modes per quadrature).
Part C varies all three.  A note on what is not varied: with an x/y drive and
no z control, a Hadamard equals X_pi/2 (or Y_pi/2) up to virtual-Z frames,
which are free, so "another gate" in this model means another ROTATION ANGLE
-- which is the physically relevant axis anyway, since the angle sets the Rabi
rate Omega and therefore delta/Omega, the ratio that generates the
second-order error in the first place.

  grid   theta in {pi/4, pi/2, pi} x anharmonicity in {-150, -200, -300} MHz
         x basis in {sine (17 modes), sine (6 modes), windowed piecewise
         constant} x noise scale in {1, 2}, at Tg = 12.7 ns.
  note   the rotation-angle points are CLOSED-SYSTEM ONLY: the Lindblad
         worker has the X_pi/2 target compiled into fade_realistic, so
         `validate` is not run on them and their JSON says so.  The
         anharmonicity and basis points keep the standard target and are
         validated normally.

PRE-REGISTERED PREDICTIONS
  C1  kappa still orders the divergence across all of Part C.
  C2  The kappa^2 law holds with its closed-form coefficient: the ratio
      R_tail / (C kappa^2) stays within a factor of about two for kappa < 0.35,
      as it does for the X_pi/2 family.
  C3  The residual log R_tail - log(C kappa^2) does not depend on the target
      angle, the anharmonicity or the basis (Kruskal-Wallis).  If it does, the
      mechanism is gate-specific and the paper must say so.

=============================================================================
USAGE  (from the iris folder, with the Stage-1 versions of stage0_v2.py,
fast_viper.py and stage1.py already there)
=============================================================================
  python stage3.py selftest                       # pure maths, seconds
  bash run_stage3.sh 2>&1 | tee stage3_log.txt    # the overnight batch
or step by step:
  python stage3.py calibrate --kappa-star 0.20 0.35 0.55 \
         --rho 0 0.5 0.9 --Tg 8 12.7 20          # ~2 h, resumable
  python stage3.py run                            # ~2.5 h, resumable
  python stage3.py gates --theta pi/4 pi/2 pi --anharm -150 -200 -300 \
         --basis sine:17 sine:6 pc --noise-scale 1 2      # Part C, ~3 h
  python fast_viper.py validate --outdir results_stage3
  python stage1.py diagnose results_stage3
  python stage3.py diagnose-lk results_stage3 results_fast results_2lvl_long \
         results_strong results_strong_2lvl results_det_only results_amp_only \
         results_collapse_T8 results_collapse_T20 results_seed1 results_seed2 \
         results_stage1 results_stage2
  python stage3.py summary

WHAT TO SEND BACK (all small)
  results_stage3/stage3_summary.json, stage3_calib.json, validate.json,
  stage1_diagnose.json, stage3_lk.json, stage3_lk.csv, stage3_log.txt

NOT executed on Claude's side (per your preference): `selftest` covers the
maths on synthetic data; the simulator paths are syntax-checked only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# constants shared with the rest of the project
# ---------------------------------------------------------------------------
PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)
U_TARGET = np.array([[1.0, -1j], [-1j, 1.0]]) / np.sqrt(2)       # X(pi/2)
W_EPS = np.diag([3.0, 3.0, 2.0]) / 16.0                          # four-state loss metric
LEAK_LAMBDA = 7.0                                                # cfg.leak_lambda
KAPPA_TOL = 1.05                                                 # calibration: within 5%
S_MIN, S_MAX = 0.01, 3.0                                         # noise-scale bracket
MEAN_LOSS_FLOOR = 1e-8                                           # refuse cells below this
KAPPA2_DOMAIN = 0.35                                             # where R = C kappa^2 was verified


# ===========================================================================
# 1. PURE MATHS  (no simulator; everything here is covered by `selftest`)
# ===========================================================================

def cvar(L, alpha):
    L = np.asarray(L, float)
    n = len(L)
    m = max(int(round(alpha * n)), 1)
    return float(np.partition(L, n - m)[n - m:].mean())


def whiten(d, g, sd, sa, rho):
    """Standardised, independent noise coordinates; dominant direction first."""
    xs = ([d / sd] if sd > 0 else []) + ([g / sa] if sa > 0 else [])
    if len(xs) == 2:
        x1, x2 = xs
        zp = (x1 + x2) / np.sqrt(2 * (1 + rho))
        zm = (x1 - x2) / np.sqrt(2 * (1 - rho))
        xs = [zp, zm] if rho >= 0 else [zm, zp]
    return xs


def monomials(zs):
    """[1, z_i, z_i z_j (i<=j)] design matrix columns for the quadratic fit."""
    n = len(zs)
    cols = [np.ones_like(zs[0])] + list(zs)
    cols += [zs[i] * zs[j] for i in range(n) for j in range(i, n)]
    return np.column_stack(cols), n


def fit_expansion(x, zs):
    """x: (K, ncoord) error coordinates, zs: list of n whitened noise arrays.
    Least-squares fit x = e0 + A z + Q(z).  Returns e0 (ncoord,), A (ncoord,n),
    Q (ncoord, n(n+1)/2), R^2 per coordinate (ncoord,)."""
    X, n = monomials(zs)
    B, *_ = np.linalg.lstsq(X, np.asarray(x, float), rcond=None)      # (1+n+nq, ncoord)
    pred = X @ B
    ss = ((x - x.mean(0)) ** 2).sum(0)
    r2 = 1.0 - ((x - pred) ** 2).sum(0) / np.where(ss > 0, ss, 1.0)
    return B[0], B[1:1 + n].T, B[1 + n:].T, r2


def kappa_of(A, Q, W=None):
    """|Q| / |A| in the plain Frobenius norm (W=None) or in the loss metric."""
    A, Q = np.asarray(A, float), np.asarray(Q, float)
    if W is not None:
        Wh = _sqrtm_psd(W)
        A, Q = Wh @ A, Wh @ Q
    na = np.linalg.norm(A)
    return float(np.linalg.norm(Q) / na) if na > 0 else float("inf")


def C_closed(A, Q, W, alpha=0.05, N=400_000, band=0.05, seed=0):
    """Coefficient of the kappa^2 law:  R_alpha = C kappa^2 + O(kappa^3).

    Reduced model eps = e0 + a(A z + kappa Q(z)) with |A| = |Q| = 1.  At
    kappa = 0 the noise part is symmetric, the alpha-tail is two-sided, and a
    coherent offset gains on one side exactly what it loses on the other, so
    the O(kappa) terms cancel and the gain is second order.  Writing
    e0 = kappa f, CVaR = CVaR(L0) + kappa^2 J(f) + O(kappa^3) with J quadratic:

        J(f) = E[(Q+f)^T W (Q+f) 1{L0>tau}]/alpha
               + (rho(tau)/2 alpha) Var[2 (Az)^T W (Q+f) | L0 = tau],

    Hessian H = 2W + (4 rho/alpha) Cov[W A z | L0 = tau], so both optima are
    explicit and C = [J(f_mean) - J(f*)] / CVaR(L0), f_mean = -E Q.
    For one dominant direction with Q parallel to A this reduces to
    C = (c*-1)^2, c* = 1 + z_a^2 (1 - 1/r_a) -- see C_scalar().
    """
    A, Q, W = np.asarray(A, float), np.asarray(Q, float), np.asarray(W, float)
    na, nq = np.linalg.norm(A), np.linalg.norm(Q)
    if na <= 0 or nq <= 0:
        return 0.0
    Ah, Qh = A / na, Q / nq
    n = Ah.shape[1]
    z = np.random.default_rng(seed).standard_normal((n, N))
    mon = [z[i] * z[j] for i in range(n) for j in range(i, n)]
    Az = np.column_stack(list(z)) @ Ah.T
    Qz = np.column_stack(mon) @ Qh.T
    L0 = np.einsum("ki,ij,kj->k", Az, W, Az)
    tau = np.quantile(L0, 1 - alpha)
    tail = L0 > tau
    lo, hi = np.quantile(L0, 1 - alpha - band / 2), np.quantile(L0, 1 - alpha + band / 2)
    bnd = (L0 >= lo) & (L0 <= hi)
    if hi <= lo or bnd.sum() < 50:
        return 0.0
    dens = band / (hi - lo)
    u = (Az @ W)[bnd]
    h0 = 2 * np.einsum("ki,ki->k", Az[bnd], Qz[bnd] @ W)
    Su = np.atleast_2d(np.cov(u.T, bias=True))
    c0 = ((h0 - h0.mean())[:, None] * (u - u.mean(0))).mean(0)
    H = 2 * W + (4 * dens / alpha) * Su
    b = 2 * W @ Qz[tail].mean(0) + (2 * dens / alpha) * c0
    d = -Qz.mean(0) + np.linalg.solve(H, b)
    return float(0.5 * float(d @ H @ d) / cvar(L0, alpha))


def C_scalar(alpha):
    """One dominant direction, Q parallel to A: closed form."""
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    r = 1 + 2 * z * norm.pdf(z) / alpha          # tail-amplification ratio (Prop. 1)
    c_star = 1 + z ** 2 * (1 - 1 / r)
    return {"z": float(z), "r": float(r), "c_star": float(c_star), "C": float((c_star - 1) ** 2)}


def _sqrtm_psd(W):
    w, V = np.linalg.eigh(np.asarray(W, float))
    return V @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ V.T


def leak_metric(n_levels, lam=LEAK_LAMBDA):
    """Real quadratic form for the leakage amplitudes, ordered
    [Re l_{k j}] then [Im l_{k j}] with k = 2..N-1 outer, j = 0,1 inner."""
    nk = max(n_levels - 2, 0)
    if nk == 0:
        return np.zeros((0, 0))
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    S = 2 * np.eye(2) + (X + Y) / 2.0            # sum_psi |psi><psi| over the four inputs
    G = np.conj(S)                               # <psi|l^dag l|psi> summed  ->  l^dag G l
    Gr, Gi = np.real(G), np.imag(G)
    I = np.eye(nk)
    top = np.hstack([np.kron(I, Gr), -np.kron(I, Gi)])
    bot = np.hstack([np.kron(I, Gi), np.kron(I, Gr)])
    return (1.0 + lam) / 4.0 * np.vstack([top, bot])


def metric_W(n_levels, lam=LEAK_LAMBDA):
    """Block-diagonal loss metric on x = (eps ; Re l ; Im l)."""
    Wl = leak_metric(n_levels, lam)
    n = 3 + Wl.shape[0]
    W = np.zeros((n, n))
    W[:3, :3] = W_EPS
    if Wl.size:
        W[3:, 3:] = Wl
    return W


def error_coords(U, n_levels, Ut=U_TARGET):
    """(K,N,N) closed-system propagators -> (K, 3 + 4(N-2)) error coordinates.

    eps: rotation vector of the polar (nearest-unitary) factor of the qubit
    block of U_target^dag U, exactly as in stage1.rotation_vectors.
    l:   leakage amplitudes <k|U|j> (k >= 2, j = 0,1) in the SAME global-phase
    gauge, so that the pair (eps, l) transforms consistently draw to draw."""
    U = np.asarray(U, complex)
    K, N = U.shape[0], U.shape[1]
    Ut_full = np.eye(N, dtype=complex)
    Ut_full[:2, :2] = Ut
    E = np.conj(Ut_full.T)[None] @ U
    Eq = E[:, :2, :2]
    u, _, vh = np.linalg.svd(Eq)
    Wu = u @ vh                                            # nearest unitary
    ph = np.sqrt(np.linalg.det(Wu))                        # global phase of the qubit block
    Wn = Wu / ph[:, None, None]
    c = np.real(np.trace(Wn, axis1=1, axis2=2)) / 2
    v = -np.imag(np.einsum("kab,iba->ki", Wn, PAULI)) / 2
    sign = np.where(c < 0, -1.0, 1.0)                      # pick the rotation angle <= pi
    c, v = c * sign, v * sign[:, None]
    nv = np.linalg.norm(v, axis=1)
    th = 2 * np.arctan2(nv, c)
    scale = np.where(nv > 1e-15, th / np.where(nv > 1e-15, nv, 1.0), 2.0)
    eps = v * scale[:, None]
    if N <= 2:
        return eps
    gauge = (sign / ph)[:, None, None]                     # |gauge| = 1: a global phase
    l = (E[:, 2:, :2] * gauge).reshape(K, -1)              # (K, 2(N-2)), k outer, j inner
    return np.concatenate([eps, np.real(l), np.imag(l)], axis=1)


def exact_four_state_loss(U, lam=LEAK_LAMBDA, Ut=U_TARGET):
    """The model's loss for a unitary error, pure inputs, no thermal
    background: L = 1 - Fbar + lambda * Pleak.  Used by `selftest` to check
    the metric W; the production numbers always come from the simulator."""
    U = np.asarray(U, complex)
    K, N = U.shape[0], U.shape[1]
    Ut_full = np.eye(N, dtype=complex)
    Ut_full[:2, :2] = Ut
    E = np.conj(Ut_full.T)[None] @ U
    kets = [np.array([1, 0], complex), np.array([0, 1], complex),
            np.array([1, 1], complex) / np.sqrt(2), np.array([1, 1j], complex) / np.sqrt(2)]
    F = np.zeros(K)
    P = np.zeros(K)
    for psi2 in kets:
        psi = np.zeros(N, complex)
        psi[:2] = psi2
        out = E @ psi                                       # E|psi>
        F += np.abs(out[:, :2] @ np.conj(psi[:2])) ** 2
        P += np.sum(np.abs(out[:, 2:]) ** 2, axis=1)
    return 1.0 - F / 4.0 + lam * P / 4.0


def r_model(e0, A, Q, W, alpha=0.05, offset_dims=None, N_mc=60_000, seed=1):
    """Reduced model (Prop. 4): loss = (e0 + A z + Q(z))^T W (...), each
    objective picks only the offset e0.  offset_dims = None: the full
    coordinate space; = 3: only the qubit block (leakage offsets frozen at
    their fitted value).  Returns 1 - CVaR_min / CVaR(mean-optimal offset)."""
    from scipy.optimize import minimize
    A, Q, W = np.asarray(A, float), np.asarray(Q, float), np.asarray(W, float)
    n = A.shape[1]
    z = list(np.random.default_rng(seed).standard_normal((n, N_mc)))
    mon = z + [z[i] * z[j] for i in range(n) for j in range(i, n)]
    V = np.column_stack(mon) @ np.concatenate([A, Q], axis=1).T        # (N_mc, ncoord)
    nd = V.shape[1] if offset_dims is None else int(offset_dims)
    fixed = np.zeros(V.shape[1])
    if nd < V.shape[1]:
        fixed[nd:] = np.asarray(e0, float)[nd:]                        # frozen coordinates

    def loss(off):
        e = fixed.copy()
        e[:nd] = off
        X = V + e
        return np.einsum("ki,ij,kj->k", X, W, X)

    e_mean = -V.mean(0)[:nd]
    base = cvar(loss(e_mean), alpha)
    res = minimize(lambda o: cvar(loss(o), alpha), e_mean, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 6000, "maxfev": 12000})
    return float(1.0 - min(res.fun, base) / base) if base > 0 else 0.0


def secant_log(f, x0, target, lo, hi, slope=0.5, tol=math.log(KAPPA_TOL), max_eval=6, log=print):
    """Solve f(x) = target for x in [lo, hi], f > 0 and increasing, working in
    log-log.  f is expensive (one optimization per call), so: one evaluation,
    one power-law jump using the assumed local slope d log f / d log x, then
    secant steps on the accumulated (log x, log f) samples.  Returns
    (x_best, f_best, trace)."""
    lx0, lt = math.log(x0), math.log(target)
    pts = []

    def ev(lx):
        lx = min(max(lx, math.log(lo)), math.log(hi))
        val = f(math.exp(lx))
        pts.append((lx, math.log(max(val, 1e-300))))
        log(f"    calib: s={math.exp(lx):.4g} -> kappa={val:.4g} (target {target:g})")
        return pts[-1]

    lx, lf = ev(lx0)
    for _ in range(max_eval - 1):
        if abs(lf - lt) < tol:
            break
        if len(pts) == 1:
            lx_new = lx + (lt - lf) / slope
        else:
            (xa, fa), (xb, fb) = pts[-2], pts[-1]
            m = (fb - fa) / (xb - xa) if abs(xb - xa) > 1e-12 else slope
            m = m if abs(m) > 0.05 else math.copysign(0.05, m if m else 1.0)
            lx_new = xb + (lt - fb) / m
            lx_new = min(max(lx_new, min(xa, xb) - 1.5), max(xa, xb) + 1.5)   # damping
        lx, lf = ev(lx_new)
    best = min(pts, key=lambda p: abs(p[1] - lt))
    return math.exp(best[0]), math.exp(best[1]), [(math.exp(a), math.exp(b)) for a, b in pts]


# ===========================================================================
# 2. SELFTEST  (synthetic; no simulator, no optimizer)
# ===========================================================================

def selftest(outdir="results_stage3"):
    from scipy.linalg import expm
    rng = np.random.default_rng(0)
    res, ok = {}, True

    # (a) the metric W reproduces the exact four-state loss to second order,
    #     and the residual scales like the cube of the error size.
    for N in (2, 4):
        W = metric_W(N)
        errs = []
        for eta in (3e-3, 1e-3, 3e-4):
            H = rng.standard_normal((40, N, N)) + 1j * rng.standard_normal((40, N, N))
            H = eta * (H + np.conj(np.swapaxes(H, 1, 2))) / 2
            U = np.array([expm(-1j * h) for h in H])
            Uf = np.einsum("ab,kbc->kac", np.eye(N, dtype=complex), U)
            Ut_full = np.eye(N, dtype=complex); Ut_full[:2, :2] = U_TARGET
            Utot = np.einsum("ab,kbc->kac", Ut_full, Uf)          # U = U_target * error
            x = error_coords(Utot, N)
            quad = np.einsum("ki,ij,kj->k", x, W, x)
            exact = exact_four_state_loss(Utot)
            errs.append(float(np.max(np.abs(quad - exact)) / max(np.mean(exact), 1e-300)))
        res[f"W_rel_err_N{N}"] = errs
        ok &= errs[-1] < 2e-2 and errs[-1] < errs[0]              # small, and shrinking
    res["W_leak_eigmin"] = float(np.linalg.eigvalsh(metric_W(4))[0])
    ok &= res["W_leak_eigmin"] > 0

    # (b) the expansion fit recovers a known (e0, A, Q) and hence kappa
    n, nc = 2, 3
    e0 = rng.standard_normal(nc) * 1e-3
    A = rng.standard_normal((nc, n)) * 1e-2
    Q = rng.standard_normal((nc, 3)) * 3e-3
    z = rng.standard_normal((n, 4000))
    mon = list(z) + [z[i] * z[j] for i in range(n) for j in range(i, n)]
    x = (e0[None] + np.column_stack(mon) @ np.concatenate([A, Q], axis=1).T)
    e0f, Af, Qf, r2 = fit_expansion(x, list(z))
    res["fit_max_err"] = float(max(np.abs(e0f - e0).max(), np.abs(Af - A).max(), np.abs(Qf - Q).max()))
    res["fit_kappa"] = [kappa_of(Af, Qf), float(np.linalg.norm(Q) / np.linalg.norm(A))]
    ok &= res["fit_max_err"] < 1e-9 and abs(res["fit_kappa"][0] - res["fit_kappa"][1]) < 1e-9
    ok &= bool(np.all(r2 > 0.999999))

    # (c) reduced model: zero at kappa = 0 (Prop. 4 iii), positive for Q != 0
    W3 = metric_W(2)
    res["r_model_Q0"] = r_model(np.zeros(3), A, np.zeros((3, 3)), W3, N_mc=20000)
    res["r_model_Q"] = r_model(e0, A, Q, W3, N_mc=20000)
    ok &= abs(res["r_model_Q0"]) < 5e-3 and res["r_model_Q"] > 0.02

    # (d) one-dimensional pre-compensation limit: offset -> 2.76 |q| at 0.05
    q = np.array([0.0, 0.0, 1.0]) * 1e-3
    A1 = np.zeros((3, 1)); A1[0, 0] = 1e-6                        # linear error ~ cancelled
    res["r_model_1d"] = r_model(np.zeros(3), A1, q[:, None], W3, N_mc=40000)
    ok &= res["r_model_1d"] > 0.3                                 # the toy trade is large

    # (e) calibration solver on a synthetic kappa(s) = c s^0.6
    calls = [0]

    def fake(s):
        calls[0] += 1
        return 0.9 * s ** 0.6

    s_hit, k_hit, trace = secant_log(fake, 1.0, 0.25, S_MIN, S_MAX, log=lambda *_: None)
    res["calib_evals"], res["calib_kappa"] = calls[0], k_hit
    ok &= abs(math.log(k_hit / 0.25)) < math.log(KAPPA_TOL) and calls[0] <= 6

    # (f) leakage coordinates are gauge invariant (global phase) and the loss is too
    N = 4
    H = rng.standard_normal((5, N, N)) + 1j * rng.standard_normal((5, N, N))
    H = 1e-3 * (H + np.conj(np.swapaxes(H, 1, 2))) / 2
    Ut_full = np.eye(N, dtype=complex); Ut_full[:2, :2] = U_TARGET
    U = np.einsum("ab,kbc->kac", Ut_full, np.array([expm(-1j * h) for h in H]))
    ph = np.exp(1j * rng.uniform(0, 6.283, 5))[:, None, None]
    res["gauge_max_diff"] = float(np.max(np.abs(error_coords(U, N) - error_coords(U * ph, N))))
    ok &= res["gauge_max_diff"] < 1e-9

    # (g) the kappa^2 law: closed form vs a direct scalar simulation
    cs = C_scalar(0.05)
    res["C_scalar_0.05"], res["c_star_0.05"] = cs["C"], cs["c_star"]
    zz = rng.standard_normal(400_000)
    v = zz + 0.05 * zz ** 2
    f = lambda c: cvar((v - 0.05 * c) ** 2, 0.05)
    from scipy.optimize import minimize_scalar
    rr = minimize_scalar(f, bounds=(0.5, 8.0), method="bounded", options={"xatol": 1e-7})
    res["C_scalar_MC"] = float((1 - rr.fun / f(1.0)) / 0.05 ** 2)
    res["c_star_MC"] = float(rr.x)
    ok &= abs(res["C_scalar_MC"] / cs["C"] - 1) < 0.25 and abs(res["c_star_MC"] / cs["c_star"] - 1) < 0.1
    # aligned three-dimensional geometry must reproduce the scalar coefficient
    dvec = np.array([1.0, 0.0, 0.0])
    res["C_closed_aligned"] = C_closed(dvec.reshape(3, 1), dvec.reshape(3, 1), W_EPS, 0.05, N=200_000)
    ok &= abs(res["C_closed_aligned"] / cs["C"] - 1) < 0.2

    res["PASS"] = bool(ok)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "stage3_selftest.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    print(f"STAGE 3 SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


# ===========================================================================
# 3. SIMULATOR HELPERS  (imports are local: `selftest` must not need them)
# ===========================================================================

def _sim():
    """Import the project modules and check they are the Stage-1 versions."""
    import inspect as _inspect
    import fast_viper as fv                    # patches stage0_v2 with the fast simulator
    import stage0_v2 as v2
    import fade_realistic as fr
    if "rho" not in _inspect.signature(v2.Ensemble.__init__).parameters:
        raise SystemExit("stage3.py needs the Stage-1 versions of stage0_v2.py / fast_viper.py.")
    return fv, v2, fr


def full_propagators(fv, uI, uQ, deltas, gammas):
    """Closed-system total propagator per draw, ALL N levels (K,N,N).
    Same Hamiltonian and time stepping as fast_viper.fast_gather."""
    from fade_realistic import cfg
    M = fv.get_model()
    deltas, gammas = np.asarray(deltas, float), np.asarray(gammas, float)
    dt, g1 = cfg.Tg / cfg.Nt, 1.0 + gammas
    H = (M["H0"][None, None] + deltas[:, None, None, None] * M["n"][None, None]
         + g1[:, None, None, None] * (uI[None, :, None, None] * M["Hx"]
                                      + uQ[None, :, None, None] * M["Hy"]))
    e, V = np.linalg.eigh(H)
    U = (V * np.exp(-1j * e * dt)[..., None, :]) @ np.conj(np.swapaxes(V, -1, -2))
    Ut = np.broadcast_to(np.eye(M["N"], dtype=complex), (len(deltas), M["N"], M["N"])).copy()
    for k in range(U.shape[1]):
        Ut = U[:, k] @ Ut
    return Ut


def control_geometry(fv, uI, uQ, ens, sd, sa, rho, n_levels, with_model=True, alpha=0.05,
                     Ut=U_TARGET):
    """Everything Parts A, B and C need about one control, on one ensemble."""
    zs = whiten(ens.deltas, ens.gammas, sd, sa, rho)
    U = full_propagators(fv, uI, uQ, ens.deltas, ens.gammas)
    x = error_coords(U, n_levels, Ut)
    e0, A, Q, r2 = fit_expansion(x, zs)
    W = metric_W(n_levels)
    out = {"kappa": kappa_of(A[:3], Q[:3]),
           "kappa_W": kappa_of(A[:3], Q[:3], W_EPS),
           "kappa_lk": kappa_of(A, Q, W),
           "fit_R2_min": float(np.min(r2)), "n_noise": len(zs),
           "coherent_norm": float(np.linalg.norm(e0[:3])),
           "leak_share_of_A": float(np.linalg.norm(A[3:]) ** 2 /
                                    max(np.linalg.norm(A) ** 2, 1e-300)) if len(A) > 3 else 0.0}
    if len(A) > 3:
        out["leak_share_of_Q"] = float(np.linalg.norm(Q[3:]) ** 2 / max(np.linalg.norm(Q) ** 2, 1e-300))
    out["C_closed"] = C_closed(A[:3], Q[:3], W_EPS, alpha)                 # kappa^2 law, qubit block
    out["R_kappa2"] = out["C_closed"] * out["kappa"] ** 2                  # its prediction
    if len(A) > 3:
        out["C_closed_lk"] = C_closed(A, Q, W, alpha)
        out["R_kappa2_lk"] = out["C_closed_lk"] * out["kappa_lk"] ** 2
    if with_model:
        out["R_model_q"] = r_model(e0, A[:3], Q[:3], W_EPS, alpha)          # qubit-only (paper's)
        out["R_model_lk"] = r_model(e0, A, Q, W, alpha, offset_dims=None)   # full coordinates
    if len(zs) == 1:                       # one-dimensional pre-compensation offset, in |q| units
        qv = Q[:3, 0]
        nq = float(qv @ qv)
        out["offset_c"] = float(-e0[:3] @ qv / nq) if nq > 0 else None
    return out


# ===========================================================================
# 4. PART A -- CALIBRATE the noise scale so that kappa hits a target
# ===========================================================================

def _cell_key(ks, rho, Tg):
    return f"k{ks:g}_rho{rho:+g}_Tg{Tg:g}"


def calibrate(a):
    fv, v2, fr = _sim()
    base = v2.V2Config()
    base.n_levels = a.levels
    base.n_restarts = a.restarts
    base.maxiter = a.maxiter
    sd0 = float(2 * np.pi * a.base_sigma_d_mhz * 1e-3)
    sa0 = float(a.base_sigma_a)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cf = out / "stage3_calib.json"
    calib = json.loads(cf.read_text()) if cf.exists() else {"cells": {}, "design": {}}
    calib["design"] = {"kappa_star": list(a.kappa_star), "rho": list(a.rho), "Tg": list(a.Tg),
                       "levels": a.levels, "base_sigma_d_MHz": a.base_sigma_d_mhz,
                       "base_sigma_a": sa0, "restarts": a.restarts, "maxiter": a.maxiter,
                       "kappa_tol": KAPPA_TOL, "s_bracket": [S_MIN, S_MAX]}
    grid = list(itertools.product(a.kappa_star, a.rho, a.Tg))
    for i, (ks, rho, Tg) in enumerate(grid):
        key = _cell_key(ks, rho, Tg)
        if key in calib["cells"] and calib["cells"][key].get("done"):
            v2.log(f"  {key} already calibrated (s={calib['cells'][key]['s']:.4g}), skipping")
            continue
        # warm guess: any solved cell with the same rho, rescaled by kappa ~ s^slope
        guess, slope = 1.0, a.slope
        same = [c for k, c in calib["cells"].items()
                if c.get("done") and abs(c["rho"] - rho) < 1e-12 and c["kappa"] > 0]
        if same:
            ref = min(same, key=lambda c: abs(math.log(max(c["Tg"], 1e-9) / Tg)))
            guess = float(np.clip(ref["s"] * (ks / ref["kappa"]) ** (1.0 / slope), S_MIN, S_MAX))
        v2.log(f"=== calibrate {i+1}/{len(grid)}: kappa*={ks:g}, rho={rho:+g}, Tg={Tg:g} ns "
               f"(start s={guess:.4g}) ===")
        t0 = time.monotonic()
        seen = {}

        def kappa_at(s):
            s = float(s)
            if s in seen:
                return seen[s]
            cc = v2.V2Config(**asdict(base))
            cc.sigma_d, cc.sigma_a, cc.rho_da = s * sd0, s * sa0, float(rho)
            P = v2.Problem(Tg, cc)
            best, _ = P.multistart("mean", alpha=0.05, lam=0.5,
                                   seed=7000 + 13 * i + len(seen), label=f"{key}/s{s:.3g}")
            uI, uQ = P.filtered(best["coef"])
            geo = control_geometry(fv, uI, uQ, P.test, cc.sigma_d, cc.sigma_a, rho,
                                   a.levels, with_model=False)
            geo["mean_loss"] = float(v2.losses(uI, uQ, P.test).mean())
            geo["coef"] = best["coef"].tolist()
            seen[s] = geo
            return geo

        s_best, k_best, trace = secant_log(lambda s: kappa_at(s)["kappa"], guess, ks,
                                           S_MIN, S_MAX, slope=slope, log=v2.log)
        g = seen.get(float(s_best)) or seen[min(seen, key=lambda s: abs(math.log(s / s_best)))]
        cell = {"coef": g["coef"], "kappa_star": float(ks), "rho": float(rho),
                "Tg": float(Tg), "s": float(s_best),
                "kappa": float(k_best), "kappa_W": g["kappa_W"], "kappa_lk": g["kappa_lk"],
                "sigma_d_MHz": float(s_best * a.base_sigma_d_mhz), "sigma_a": float(s_best * sa0),
                "mean_loss": g["mean_loss"], "fit_R2_min": g["fit_R2_min"],
                "rel_err": float(abs(math.log(k_best / ks))), "trace": trace,
                "n_optimizations": len(seen), "seconds": time.monotonic() - t0,
                "converged": bool(abs(math.log(k_best / ks)) < math.log(KAPPA_TOL)),
                "above_loss_floor": bool(g["mean_loss"] > MEAN_LOSS_FLOOR), "done": True}
        calib["cells"][key] = cell
        v2.write_json(cf, calib)
        v2.log(f"  -> {key}: s={s_best:.4g} (sigma_d={cell['sigma_d_MHz']:.3g} MHz, "
               f"sigma_a={cell['sigma_a']:.4g}), kappa={k_best:.3f} "
               f"({'ok' if cell['converged'] else 'NOT CONVERGED'}), "
               f"E[L]={cell['mean_loss']:.2e}, {cell['n_optimizations']} solves, "
               f"{cell['seconds']:.0f}s")
    done = [c for c in calib["cells"].values() if c.get("done")]
    v2.log(f"CALIBRATE done: {len(done)} cells, "
           f"{sum(c['converged'] for c in done)} within {100*(KAPPA_TOL-1):.0f}%, "
           f"{sum(c['n_optimizations'] for c in done)} mean optimizations")


# ===========================================================================
# 5. PART A -- RUN the standard comparison at the calibrated settings
# ===========================================================================

def run(a):
    fv, v2, fr = _sim()
    out = Path(a.outdir)
    cf = out / "stage3_calib.json"
    if not cf.exists():
        raise SystemExit("run `python stage3.py calibrate ...` first")
    calib = json.loads(cf.read_text())
    des = calib["design"]
    base = v2.V2Config()
    base.n_levels = int(des["levels"])
    base.n_restarts = a.restarts or int(des["restarts"])
    base.maxiter = a.maxiter or int(des["maxiter"])
    d = out / f"stage3_{base.n_levels}level"
    d.mkdir(parents=True, exist_ok=True)
    cells = [c for c in calib["cells"].values() if c.get("done")]
    cells.sort(key=lambda c: (c["kappa_star"], c["rho"], c["Tg"]))
    cells = [(c, sd) for c in cells for sd in a.seeds]          # replication seeds
    for i, (c, seed) in enumerate(cells):
        stem = f"fk_k{c['kappa_star']:g}_rho{c['rho']:+g}_Tg{c['Tg']:g}_s{seed}"
        pf = d / f"{stem}.json"
        if pf.exists():
            v2.log(f"  {stem} already done, skipping")
            continue
        if not c["above_loss_floor"]:
            v2.log(f"  {stem} SKIPPED: mean loss {c['mean_loss']:.1e} below the floor "
                   f"{MEAN_LOSS_FLOOR:.0e} (optimizer precision)")
            continue
        cc = v2.V2Config(**asdict(base))
        cc.sigma_d = float(2 * np.pi * c["sigma_d_MHz"] * 1e-3)
        cc.sigma_a = float(c["sigma_a"])
        cc.rho_da = float(c["rho"])
        if seed:
            cc.seed_train += 1000 * seed
            cc.seed_val += 1000 * seed
            cc.seed_test += 1000 * seed
        v2.log(f"=== Stage 3 point {i+1}/{len(cells)}: kappa*={c['kappa_star']:g} "
               f"(achieved {c['kappa']:.3f}), rho={c['rho']:+g}, Tg={c['Tg']:g} ns, "
               f"sigma_d={c['sigma_d_MHz']:.3g} MHz, sigma_a={c['sigma_a']:.4g}, "
               f"seed={seed} ===")
        t0 = time.monotonic()
        P = v2.Problem(c["Tg"], cc)
        if a.warm_from_calib and c.get("coef") is not None and len(c["coef"]) == len(P.c_seed):
            # Give the MEAN side the calibrated control as one of its restarts, so the
            # run reproduces the kappa the calibration solved for instead of drifting to
            # another basin.  fast_viper's continuation hook replaces one random mean
            # restart with it, so restart counts stay equal; the CVaR side is warm-started
            # from the selected mean control as always, and the mean side keeping the
            # better start is the conservative direction (Sec. fairness).
            uI_c, uQ_c = P.param.pulse(np.asarray(c["coef"], float))
            fv._prev["pulse"] = (float(c["Tg"]), uI_c.copy(), uQ_c.copy())
        else:
            fv._prev.clear()
        pt, controls, test_L = v2.compare_mean_vs_cvar(P, cc.alpha_sweep,
                                                       30000 + 100 * i + 10000 * seed, stem)
        pt.update(stage="3-fixed-kappa", kappa_star=c["kappa_star"], kappa_calib=c["kappa"],
                  noise_scale=c["s"], sigma_d_MHz=c["sigma_d_MHz"],
                  sigma_phi_sq=float((cc.sigma_d * c["Tg"]) ** 2),
                  amp_phase_sq=float((cc.sigma_a * np.pi / 2) ** 2))
        v2.write_json(pf, pt)
        v2.save_arrays(d / f"{stem}.npz", P, controls, test_L)
        # geometry of every control at this point (kappa variants + reduced model)
        geo = {}
        for k, coef in controls.items():
            uI, uQ = P.filtered(coef)
            geo[k] = control_geometry(fv, uI, uQ, P.test, cc.sigma_d, cc.sigma_a, cc.rho_da,
                                      base.n_levels, with_model=(k == "mean"))
            geo[k]["mean_loss"] = float(v2.losses(uI, uQ, P.test).mean())
            geo[k]["cvar05"] = cvar(v2.losses(uI, uQ, P.test), 0.05)
        v2.write_json(d / f"{stem}_geo.json",
                      {"stem": stem, "cell": {**c, "seed": int(seed)}, "geometry": geo,
                       "R_tail_fast": {str(x["alpha"]): x["R_tail"] for x in pt["cells"]},
                       "seconds": time.monotonic() - t0})
        for x in pt["cells"]:
            v2.log(f"  -> alpha={x['alpha']}: R_tail(fast)={x['R_tail']:+.3f} "
                   f"[{x['R_lo']:+.3f},{x['R_hi']:+.3f}]")
        v2.log(f"  -> kappa(mean control, re-measured)={geo['mean']['kappa']:.3f}, "
               f"R_model_q={geo['mean'].get('R_model_q', float('nan')):.3f}, "
               f"{time.monotonic()-t0:.0f}s")


# ===========================================================================
# 6. PART C -- GENERALITY: other target rotations, anharmonicities, bases
# ===========================================================================
#
# The mechanism was found for one gate (X_pi/2), one anharmonicity
# (-200 MHz) and one pulse basis (17 Hann-windowed sine modes per
# quadrature).  Part C asks whether kappa and the kappa^2 law survive when
# those change.  Note what is NOT a separate test here: with an x/y drive and
# no z control, a Hadamard is X_pi/2 (or Y_pi/2) up to virtual-Z frame
# changes, which are free and do not change the control problem.  The
# physically distinct axis is the ROTATION ANGLE, which changes the Rabi rate
# and hence delta/Omega, the very ratio that generates the second-order error.
#
# Targets are installed by rebuilding the closed-system model, so Part C runs
# in the fast simulator.  The Lindblad worker has the X_pi/2 target compiled
# into fade_realistic, so `validate` is NOT run on the rotation-angle points
# (their JSON carries closed_system_only = true).  Anharmonicity and basis
# points keep the standard target and can be validated normally.

PAULI_X = np.array([[0, 1], [1, 0]], complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], complex)


def target_unitary(theta, axis="x"):
    n = PAULI_X if axis == "x" else PAULI_Y
    return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * n


def install_target(fv, theta, axis="x"):
    """Rebuild fast_viper's closed-system model for a different target."""
    from fade_realistic import cfg, TWO_PI
    cache = {}

    def build():
        N = cfg.N_LEVELS
        a = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
        ad = a.conj().T
        n_op = ad @ a
        alpha = cfg.alpha_MHz * 1e-3 * TWO_PI
        H0 = (0.5 * alpha * (ad @ ad @ a @ a)).astype(complex)
        Hx, Hy = (a + ad) / 2.0, 1j * (ad - a) / 2.0
        P = np.diag([0, 0] + [1] * (N - 2)).astype(complex)
        x = (TWO_PI * cfg.qubit_freq_GHz * 1e9 * 6.62607015e-34 / (2 * np.pi)) / \
            (1.380649e-23 * cfg.temperature_mK * 1e-3)
        n_th = 1.0 / (np.exp(x) - 1.0)
        r = n_th / (1.0 + n_th) if n_th > 0 else 0.0
        pops = np.array([(1 - r) * r ** k for k in range(N)]); pops /= pops.sum()
        rho_th = np.diag(pops).astype(complex)
        leak_bg = float(np.real(np.trace(P @ rho_th)))
        Uemb = np.eye(N, dtype=complex)
        Uemb[0:2, 0:2] = target_unitary(theta, axis)
        kets = [np.array([1, 0] + [0] * (N - 2), complex),
                np.array([0, 1] + [0] * (N - 2), complex),
                np.array([1, 1] + [0] * (N - 2), complex) / np.sqrt(2),
                np.array([1, 1j] + [0] * (N - 2), complex) / np.sqrt(2)]
        rhos, Os = [], []
        for psi in kets:
            rin = (1.0 - leak_bg) * np.outer(psi, psi.conj()) + P @ rho_th @ P
            tgt = Uemb @ rin @ Uemb.conj().T
            rhos.append(rin)
            Os.append(tgt.conj().T - cfg.leak_lambda * P)
        return {"N": N, "H0": H0, "n": n_op, "Hx": Hx, "Hy": Hy,
                "rhos": np.array(rhos), "Os": np.array(Os)}

    def get_model():
        key = (cfg.N_LEVELS, cfg.alpha_MHz, cfg.leak_lambda, cfg.temperature_mK,
               cfg.qubit_freq_GHz, float(theta), axis)
        if key not in cache:
            cache[key] = build()
        return cache[key]

    fv._model_cache.clear()
    fv.get_model = get_model


def install_nominal(fv, v2):
    """P.nominal without fade_realistic's compiled X_pi/2 target."""
    from fade_realistic import cfg

    def nominal(self, coef):
        uI, uQ = self.filtered(coef)
        J = float(fv.fast_gather(uI, uQ, np.zeros(1), np.zeros(1))[0][0])
        U = full_propagators(fv, uI, uQ, np.zeros(1), np.zeros(1))[0]
        leak = float(np.sum(np.abs(U[2:, :2]) ** 2) / 2) if U.shape[0] > 2 else 0.0
        return {"F_nom_4ket": J + cfg.leak_lambda * leak, "leak_nom": leak, "L_nom": 1.0 - J}

    v2.Problem.nominal = nominal


class MatrixParam:
    """Same interface as stage0_v2.PulseParam with an arbitrary basis matrix."""

    def __init__(self, BI, BQ, A):
        self.BI, self.BQ, self.A, self.M = BI, BQ, A, BI.shape[1]

    def pulse(self, c):
        A, M = self.A, self.M
        return A * np.tanh((self.BI @ c[:M]) / A), A * np.tanh((self.BQ @ c[M:]) / A)

    def grad_c(self, c, gI_raw, gQ_raw):
        uI, uQ = self.pulse(c)
        sI, sQ = 1.0 - (uI / self.A) ** 2, 1.0 - (uQ / self.A) ** 2
        return np.concatenate([self.BI.T @ (gI_raw * sI), self.BQ.T @ (gQ_raw * sQ)])

    def coeffs(self, uI, uQ):
        A = self.A
        zI = A * np.arctanh(np.clip(uI / A, -0.999999, 0.999999))
        zQ = A * np.arctanh(np.clip(uQ / A, -0.999999, 0.999999))
        return np.concatenate([np.linalg.lstsq(self.BI, zI, rcond=None)[0],
                               np.linalg.lstsq(self.BQ, zQ, rcond=None)[0]])


def make_basis(kind, n_modes, A):
    """sine:M  the paper's Hann-windowed sine modes, M per quadrature.
       pc:M    windowed piecewise constant on M equal segments (M = 0 or >= Nt
               gives one parameter per sample).  The segment count is settable
               so that a basis comparison can hold the PARAMETER COUNT fixed:
               pc:17 has as many knobs as sine:17, and any difference is then
               the shape of the basis, not the size of the search space."""
    from fade_realistic import cfg, TWO_PI
    Nt, Tg = cfg.Nt, cfg.Tg
    t = np.linspace(0.0, Tg, Nt)
    hann = 0.5 * (1.0 - np.cos(TWO_PI * np.linspace(0, 1, Nt)))
    if kind == "pc":
        M = Nt if (not n_modes or n_modes >= Nt) else int(n_modes)
        seg = np.minimum((np.arange(Nt) * M) // Nt, M - 1)      # sample -> segment
        B = np.zeros((Nt, M))
        B[np.arange(Nt), seg] = hann
        return MatrixParam(B, B.copy(), A)
    BI = np.stack([np.sin((2 * n - 1) * np.pi * t / Tg) * hann for n in range(1, n_modes + 1)], axis=1)
    BQ = np.stack([np.sin(2 * n * np.pi * t / Tg) * hann for n in range(1, n_modes + 1)], axis=1)
    return MatrixParam(BI, BQ, A)


def setup_point(v2, fv, fr, Tg, cc, theta, axis, basis, n_modes):
    """Problem with the rotation angle, amplitude budget, seed and basis of a
    Part-C cell.  The amplitude budget scales with the rotation angle, so the
    budget is as tight for X_pi as it is for X_pi/2."""
    from fade_realistic import cfg
    P = v2.Problem(Tg, cc)
    scale = theta / (np.pi / 2)                        # Hann seed has area pi/2
    amp = float(cc.kappa * 2.0 * theta / Tg)           # 2.5 x the Hann peak 2 theta/Tg
    cfg.max_amp = amp
    if axis == "y":
        P.seed_uI, P.seed_uQ = P.seed_uQ * scale, P.seed_uI * scale
    else:
        P.seed_uI, P.seed_uQ = P.seed_uI * scale, P.seed_uQ * scale
    if basis != "sine" or n_modes != P.grid["n_modes"]:
        P.param = make_basis(basis, n_modes, amp)
    else:
        P.param.A = amp
    P.c_seed = P.param.coeffs(P.seed_uI, P.seed_uQ)
    P.grid = {**P.grid, "max_amp": amp, "theta": float(theta), "axis": axis,
              "basis": basis, "n_basis": int(P.param.M)}
    return P


def gates(a):
    """Part C grid: (theta, axis, anharmonicity, basis, n_modes, noise scale)."""
    fv, v2, fr = _sim()
    from fade_realistic import cfg, TWO_PI
    base = v2.V2Config()
    base.n_levels = a.levels
    base.n_restarts = a.restarts
    base.maxiter = a.maxiter
    out = Path(a.outdir)
    d = out / f"stage3c_{a.levels}level"
    d.mkdir(parents=True, exist_ok=True)
    alpha_default = cfg.alpha_MHz
    cells = []
    for th_name, th in a.theta:
        for anh in a.anharm:
            for basis, nm in a.basis:
                for s in a.noise_scale:
                    cells.append((th_name, th, anh, basis, nm, s))
    for i, (th_name, th, anh, basis, nm, s) in enumerate(cells):
        safe = str(th_name).replace("/", "").replace(".", "p")      # "pi/4" -> "pi4"
        stem = (f"c_{safe}_anh{abs(anh):g}_{basis}{nm}_s{s:g}").replace(".", "p")
        pf = d / f"{stem}.json"
        if pf.exists():
            v2.log(f"  {stem} already done, skipping")
            continue
        cc = v2.V2Config(**asdict(base))
        cc.sigma_d = float(TWO_PI * a.sigma_d_mhz * 1e-3 * s)
        cc.sigma_a = float(a.sigma_a * s)
        cc.rho_da = float(a.rho)
        cfg.alpha_MHz = float(anh)
        fv._prev.clear()          # no continuation start from a cell with another target
        install_target(fv, th, a.axis)
        install_nominal(fv, v2)
        v2.log(f"=== Stage 3C point {i+1}/{len(cells)}: theta={th_name} ({th:.4f} rad), "
               f"anharm={anh} MHz, basis={basis}{nm}, noise x{s:g}, levels={a.levels} ===")
        t0 = time.monotonic()
        P = setup_point(v2, fv, fr, a.Tg, cc, th, a.axis, basis, nm)
        pt, controls, test_L = v2.compare_mean_vs_cvar(P, cc.alpha_sweep, 40000 + 100 * i, stem)
        Ut = target_unitary(th, a.axis)
        pt.update(stage="3C-generality", theta=float(th), theta_name=th_name, axis=a.axis,
                  anharm_MHz=float(anh), basis=basis, n_basis=int(P.param.M),
                  noise_scale=float(s), sigma_d_MHz=float(a.sigma_d_mhz * s),
                  sigma_a=float(cc.sigma_a), rho_da=float(cc.rho_da),
                  sigma_phi_sq=float((cc.sigma_d * a.Tg) ** 2),
                  closed_system_only=bool(abs(th - np.pi / 2) > 1e-9 or a.axis != "x"
                                          or abs(anh - alpha_default) > 1e-9))
        if pt["closed_system_only"]:
            # Hide the point from `fast_viper.py validate` and `stage1.py diagnose`:
            # both skip a point whose JSON has no top-level "sigma_d".  Two kinds of
            # point need this.  A non-standard TARGET: the Lindblad worker has
            # X_pi/2 compiled into fade_realistic.  A non-standard ANHARMONICITY:
            # `validate` runs in its own process and rebuilds the config from
            # V2Config, so it would re-score the point at the default -200 MHz.
            pt["sigma_d_rad_per_ns"] = pt.pop("sigma_d", None)
            pt["sigma_a_value"] = pt.pop("sigma_a", None)
            pt["validate_skipped_reason"] = (
                "non-standard target or anharmonicity: the Lindblad path would score it "
                "with the default X_pi/2 gate at -200 MHz, so only the closed-system "
                "numbers in this file and in *_geo.json are meaningful")
        v2.write_json(pf, pt)
        v2.save_arrays(d / f"{stem}.npz", P, controls, test_L)
        geo = {}
        for k, coef in controls.items():
            uI, uQ = P.filtered(coef)
            geo[k] = control_geometry(fv, uI, uQ, P.test, cc.sigma_d, cc.sigma_a, cc.rho_da,
                                      a.levels, with_model=(k == "mean"), Ut=Ut)
            L = v2.losses(uI, uQ, P.test)
            geo[k]["mean_loss"], geo[k]["cvar05"] = float(L.mean()), cvar(L, 0.05)
        v2.write_json(d / f"{stem}_geo.json",
                      {"stem": stem, "cell": {"theta": float(th), "theta_name": th_name,
                                              "axis": a.axis, "anharm_MHz": float(anh),
                                              "basis": basis, "n_basis": int(P.param.M),
                                              "noise_scale": float(s), "Tg": float(a.Tg),
                                              "levels": int(a.levels),
                                              "sigma_d_MHz": float(a.sigma_d_mhz * s),
                                              "sigma_a": float(cc.sigma_a), "rho": float(cc.rho_da)},
                       "geometry": geo,
                       "R_tail_fast": {str(x["alpha"]): x["R_tail"] for x in pt["cells"]},
                       "seconds": time.monotonic() - t0})
        m = geo["mean"]
        for x in pt["cells"]:
            v2.log(f"  -> alpha={x['alpha']}: R_tail(fast)={x['R_tail']:+.3f} "
                   f"[{x['R_lo']:+.3f},{x['R_hi']:+.3f}]")
        in_dom = m["kappa"] < KAPPA2_DOMAIN
        v2.log(f"  -> kappa={m['kappa']:.3f}, C={m['C_closed']:.2f}, "
               + (f"C kappa^2={m['R_kappa2']:.3f}, " if in_dom else
                  f"C kappa^2={min(m['R_kappa2'], 1.0):.3f} (kappa > {KAPPA2_DOMAIN}: "
                  f"beyond the expansion, the law only bounds from above), ")
               + f"R_model={m.get('R_model_q', float('nan')):.3f}, "
                 f"E[L]={m['mean_loss']:.2e}, {time.monotonic()-t0:.0f}s")
        if m["mean_loss"] < 10 * MEAN_LOSS_FLOOR:
            v2.log(f"  !! mean loss {m['mean_loss']:.1e} is near the optimizer's precision "
                   f"floor and far below the T1/T2 floor (~3e-4): the closed-system ratio "
                   f"is still meaningful, but on hardware this divergence would be diluted "
                   f"away.  Consider a larger --noise-scale for this cell.")
    cfg.alpha_MHz = alpha_default


# ===========================================================================
# 6. PART B -- LEAKAGE-AWARE DIAGNOSIS of every saved control
# ===========================================================================

def diagnose_lk(folders, outdir, alpha=0.05):
    fv, v2, fr = _sim()
    import stage1 as s1
    from fade_realistic import cfg, TWO_PI
    rows = []
    for folder in folders:
        if not Path(folder).is_dir():
            v2.log(f"  (no folder {folder}, skipping)")
            continue
        vf = Path(folder) / "validate.json"
        vmap = {}
        if vf.exists():
            for r in json.loads(vf.read_text())["rows"]:
                vmap[(r["file"], r["alpha"])] = r
        for sub, jf, pt, npz in s1._points(folder):
            c = v2.V2Config()
            c.n_levels = int(pt["grid"].get("n_levels", 4))
            v2.set_gate(pt["grid"]["Tg"], c)
            sd, sa, rho = pt["sigma_d"], pt["sigma_a"], pt.get("rho_da", 0.0)
            te = next(e for e in pt["ensembles"] if e["name"] == "test")
            ens = v2.Ensemble(te["n"], sd, sa, te["seed"], "test", rho)
            arr = np.load(npz)
            bwf = fr.BandwidthFilter(cfg.Nt, cfg.dt, cfg.awg_bandwidth_GHz, cfg.awg_filter_order)
            keys = (["mean"] + [f"cvar_{x['alpha']}" for x in pt["cells"]]
                    + sorted(f[6:] for f in arr.files if f.startswith("uIraw_rev_")))
            for k in keys:
                if f"uIraw_{k}" not in arr.files:
                    continue
                uI, uQ = bwf.apply(arr[f"uIraw_{k}"]), bwf.apply(arr[f"uQraw_{k}"])
                geo = control_geometry(fv, uI, uQ, ens, sd, sa, rho, c.n_levels,
                                       with_model=(k == "mean"), alpha=alpha)
                L = v2.losses(uI, uQ, ens)
                row = {"folder": Path(folder).name, "file": f"{sub.name}/{jf.name}",
                       "stem": jf.stem, "control": k, "Tg": pt["grid"]["Tg"],
                       "n_levels": c.n_levels, "sigma_d_MHz": sd / TWO_PI * 1e3,
                       "sigma_a": sa, "rho_da": rho, "sigma_phi_sq": (sd * pt["grid"]["Tg"]) ** 2,
                       "mean_L": float(L.mean()), "cvar05": cvar(L, 0.05),
                       "L_nominal": float(v2.losses(uI, uQ, v2.Ensemble(1, 0.0, 0.0, 0, "nom"))[0]),
                       **geo}
                vr = vmap.get((row["file"], alpha))
                if k == "mean" and vr is not None:
                    row.update(R_tail_lindblad=vr["R_tail_vs_best_mean_lindblad"],
                               R_lo=vr["R_lo_vs_best_mean"], R_hi=vr["R_hi_vs_best_mean"],
                               R_tail_fast=vr["R_tail_fast"],
                               comparator=vr.get("best_mean_source", "mean"),
                               significant=bool(vr["R_lo_vs_best_mean"] > 0))
                rows.append(row)
                v2.log(f"{row['folder']}/{row['file']:32s} {k:9s} L={c.n_levels} "
                       f"kappa={geo['kappa']:.3f} kappa_W={geo['kappa_W']:.3f} "
                       f"kappa_lk={geo['kappa_lk']:.3f}"
                       + (f" R_model_q={geo['R_model_q']:.3f} R_model_lk={geo['R_model_lk']:.3f}"
                          if k == "mean" else "")
                       + (f" | R_tail={row['R_tail_lindblad']:+.3f}"
                          if row.get("R_tail_lindblad") is not None else ""))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    v2.write_json(out / "stage3_lk.json", {"rows": rows, "alpha": alpha})
    v2.write_csv(out / "stage3_lk.csv", rows)
    v2.log(f"DIAGNOSE-LK done: {len(rows)} controls -> {out/'stage3_lk.json'}")


# ===========================================================================
# 7. SUMMARY -- score the pre-registered predictions
# ===========================================================================

def _auc(score, label):
    """Rank AUC of `score` for the binary `label` (1 = divergent)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = np.argsort(np.argsort(score)) + 1.0
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _best_threshold(score, label):
    """Youden-J threshold (geometric mid-point between the bracketing values)."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    if label.all() or not label.any():
        return float("nan"), float("nan")
    order = np.argsort(score)
    s, y = score[order], label[order]
    best, thr = -np.inf, float("nan")
    for i in range(len(s) - 1):
        if s[i] == s[i + 1]:
            continue
        pred = np.arange(len(s)) > i                       # above the cut -> divergent
        tpr = (pred & y).sum() / max(y.sum(), 1)
        fpr = (pred & ~y).sum() / max((~y).sum(), 1)
        if tpr - fpr > best:
            best, thr = tpr - fpr, math.sqrt(max(s[i], 1e-12) * max(s[i + 1], 1e-12))
    return float(thr), float(best)


def _gap_bound(groups, n_boot=4000, seed=5):
    """Effect-size companion to the Kruskal-Wallis tests.

    A p-value above 0.05 with three points per group is weak evidence of
    nothing; what the design can honestly report is a BOUND on how much the
    knob moves the statistic.  For log-ratios grouped by a knob, this returns
    the largest between-group gap of medians and a one-sided 95% bootstrap
    upper bound on it, in log units and as a multiplicative factor."""
    gl = {k: np.asarray(v, float) for k, v in groups.items() if len(v)}
    if len(gl) < 2:
        return None
    med = {k: float(np.median(v)) for k, v in gl.items()}
    gap = max(med.values()) - min(med.values())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        m = [np.median(rng.choice(v, size=len(v), replace=True)) for v in gl.values()]
        boot[b] = max(m) - min(m)
    up = float(np.quantile(boot, 0.95))
    return {"max_gap_log": float(gap), "upper95_log": up,
            "factor": float(np.exp(gap)), "factor_upper95": float(np.exp(up)),
            "n_per_group": {k: int(len(v)) for k, v in gl.items()},
            "medians": med}


def _holm(pvals):
    idx = np.argsort(pvals)
    m, adj = len(pvals), np.empty(len(pvals))
    run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(run, 1.0)
    return adj


def summary(outdir, alpha=0.05):
    from scipy.stats import spearmanr, kruskal
    out = Path(outdir)
    S = {"alpha": alpha}

    # ---------------- Part A ----------------
    calib = json.loads((out / "stage3_calib.json").read_text()) if (out / "stage3_calib.json").exists() else None
    vmap = {}
    vf = out / "validate.json"
    if vf.exists():
        for r in json.loads(vf.read_text())["rows"]:
            vmap[(r["file"], r["alpha"])] = r
    rowsA = []
    for gf in sorted(out.glob("stage3_*level/*_geo.json")):      # Part A only; Part C is stage3c_*
        g = json.loads(gf.read_text())
        c, m = g["cell"], g["geometry"]["mean"]
        if "kappa_star" not in c:
            continue
        f = f"{gf.parent.name}/{g['stem']}.json"
        vr = vmap.get((f, alpha), {})
        rowsA.append({"file": f, "kappa_star": c["kappa_star"], "rho": c["rho"], "Tg": c["Tg"],
                      "seed": c.get("seed", 0),
                      "s": c["s"], "sigma_d_MHz": c["sigma_d_MHz"], "sigma_a": c["sigma_a"],
                      "kappa_calib": c["kappa"], "kappa_run": m["kappa"], "kappa_lk": m["kappa_lk"],
                      "C_closed": m.get("C_closed"), "R_kappa2": m.get("R_kappa2"),
                      "R_model_q": m.get("R_model_q"), "R_model_lk": m.get("R_model_lk"),
                      "mean_loss": m["mean_loss"],
                      "R_fast": g["R_tail_fast"].get(str(alpha)),
                      "R_lind": vr.get("R_tail_vs_best_mean_lindblad"),
                      "R_lo": vr.get("R_lo_vs_best_mean"), "R_hi": vr.get("R_hi_vs_best_mean"),
                      "comparator": vr.get("best_mean_source")})
    S["partA_rows"] = rowsA
    if rowsA:
        ks = sorted({r["kappa_star"] for r in rowsA})
        A = {"kappa_star_levels": ks, "n_points": len(rowsA), "rows": {}}
        pvals, ptags = [], []
        for k in ks:
            sub = [r for r in rowsA if r["kappa_star"] == k]
            Rf = np.array([r["R_fast"] for r in sub], float)
            kap = np.array([r["kappa_run"] for r in sub], float)
            ent = {"n": len(sub), "R_fast_median": float(np.median(Rf)),
                   "R_fast_min": float(Rf.min()), "R_fast_max": float(Rf.max()),
                   "kappa_achieved_min": float(kap.min()), "kappa_achieved_max": float(kap.max()),
                   "kappa_spread_pct": float(100 * (kap.max() / kap.min() - 1.0)),
                   "R_lind_median": float(np.median([r["R_lind"] for r in sub
                                                     if r["R_lind"] is not None] or [np.nan]))}
            for knob in ("rho", "Tg"):
                groups = {}
                for r in sub:
                    groups.setdefault(r[knob], []).append(r["R_fast"])
                gl = [np.array(v, float) for v in groups.values() if len(v) > 0]
                if len(gl) > 1 and min(len(g) for g in gl) >= 2:
                    st, p = kruskal(*gl)
                    ent[f"kruskal_{knob}"] = {"H": float(st), "p": float(p),
                                              "groups": {str(kk): list(map(float, v))
                                                         for kk, v in groups.items()}}
                    pvals.append(float(p)); ptags.append(f"kappa*={k:g}/{knob}")
            # The mechanism's prediction for a cell is the reduced model, which is exact
            # for any kappa; the closed-form C kappa^2 is its small-kappa limit and is
            # only used where it was verified (kappa < KAPPA2_DOMAIN).
            for tag, pred in (("model", lambda r: r.get("R_model_q")),
                              ("kappa2", lambda r: r.get("R_kappa2")
                               if r["kappa_run"] < KAPPA2_DOMAIN else None)):
                rat = [float(np.log(max(r["R_fast"], 1e-6) / max(pred(r), 1e-9)))
                       for r in sub if pred(r)]
                if rat:
                    ent[f"log_ratio_R_over_{tag}"] = {
                        "n": len(rat), "median": float(np.median(rat)),
                        "min": float(np.min(rat)), "max": float(np.max(rat))}
                for knob in ("rho", "Tg"):          # the sharp test: is the residual flat?
                    groups = {}
                    for r in sub:
                        if pred(r):
                            groups.setdefault(r[knob], []).append(
                                float(np.log(max(r["R_fast"], 1e-6) / max(pred(r), 1e-9))))
                    gl = [np.array(v, float) for v in groups.values()]
                    if len(gl) > 1 and min(len(g) for g in gl) >= 2:
                        st, pv = kruskal(*gl)
                        ent[f"kruskal_residual_{tag}_{knob}"] = {"H": float(st), "p": float(pv)}
                        ent[f"gap_residual_{tag}_{knob}"] = _gap_bound(groups)
                        if tag == "model":          # one family of tests enters the correction
                            pvals.append(float(pv))
                            ptags.append(f"kappa*={k:g}/residual~{knob}")
            mod = [(r["R_model_q"], r["R_fast"]) for r in sub if r["R_model_q"] is not None]
            if len(mod) >= 4:
                rs = spearmanr([m[0] for m in mod], [m[1] for m in mod])
                ent["spearman_R_model_vs_R_fast"] = {"rho": float(rs.statistic), "p": float(rs.pvalue)}
            A["rows"][f"{k:g}"] = ent
        if pvals:
            adj = _holm(np.array(pvals))
            A["A2_knob_tests"] = [{"test": t, "p": p, "p_holm": float(q)}
                                  for t, p, q in zip(ptags, pvals, adj)]
            A["A2_pass"] = bool(np.all(adj > 0.05))
        med = [A["rows"][f"{k:g}"]["R_fast_median"] for k in ks]
        A["A1_medians"] = med
        A["A1_pass"] = bool(all(med[i] < med[i + 1] for i in range(len(med) - 1)))
        rs = spearmanr([r["kappa_run"] for r in rowsA], [r["R_fast"] for r in rowsA])
        A["A1_spearman_kappa_vs_R"] = {"rho": float(rs.statistic), "p": float(rs.pvalue)}
        good = [r for r in rowsA if r["R_model_q"] is not None]
        A["A4_n_within_envelope"] = int(sum(r["R_fast"] <= r["R_model_q"] + 1e-12 for r in good))
        A["A4_n"] = len(good)
        A["A4_pass"] = bool(A["A4_n_within_envelope"] == A["A4_n"])
        rs = spearmanr([r["R_model_q"] for r in good], [r["R_fast"] for r in good])
        A["A3_spearman_pooled"] = {"rho": float(rs.statistic), "p": float(rs.pvalue)}
        A["A3_pass"] = bool(rs.statistic > 0 and rs.pvalue < 0.05)
        S["partA"] = A
    if calib:
        S["partA_calibration"] = {
            "cells": len(calib["cells"]),
            "converged": int(sum(c.get("converged", False) for c in calib["cells"].values())),
            "total_mean_optimizations": int(sum(c.get("n_optimizations", 0)
                                                for c in calib["cells"].values())),
            "worst_rel_err": float(max([c.get("rel_err", 0.0) for c in calib["cells"].values()] or [0])),
            "design": calib.get("design", {})}

    # ---------------- Part C ----------------
    rowsC = []
    for gf in sorted(out.glob("stage3c_*/**/*_geo.json")):      # ** also finds points
        # written before the angle name was sanitised, which landed one level deeper
        g = json.loads(gf.read_text())
        c, m = g["cell"], g["geometry"]["mean"]
        f = f"{gf.parent.name}/{g['stem']}.json"
        vr = vmap.get((f, alpha), {})
        rowsC.append({"file": f, **c, "kappa": m["kappa"], "kappa_lk": m["kappa_lk"],
                      "C_closed": m.get("C_closed"), "R_kappa2": m.get("R_kappa2"),
                      "R_model_q": m.get("R_model_q"), "R_model_lk": m.get("R_model_lk"),
                      "mean_loss": m["mean_loss"],
                      "R_fast": g["R_tail_fast"].get(str(alpha)),
                      "R_lind": vr.get("R_tail_vs_best_mean_lindblad")})
    if rowsC:
        S["partC_rows"] = rowsC
        C = {"n_points": len(rowsC)}
        good = [r for r in rowsC if r.get("R_kappa2") and r["kappa"] < KAPPA2_DOMAIN
                and r["R_fast"] is not None]
        if good:
            rat = np.array([max(r["R_fast"], 1e-6) / r["R_kappa2"] for r in good])
            C["kappa2_law"] = {"n": len(good), "ratio_median": float(np.median(rat)),
                               "ratio_q25": float(np.percentile(rat, 25)),
                               "ratio_q75": float(np.percentile(rat, 75))}
        mods = [r for r in rowsC if r.get("R_model_q") is not None and r["R_fast"] is not None]
        if mods:
            C["n_within_envelope"] = int(sum(r["R_fast"] <= r["R_model_q"] + 1e-12 for r in mods))
            C["n_model"] = len(mods)
            rs = spearmanr([r["R_model_q"] for r in mods], [r["R_fast"] for r in mods])
            C["spearman_model_vs_R"] = {"rho": float(rs.statistic), "p": float(rs.pvalue)}
        for key in ("theta_name", "anharm_MHz", "basis"):
            groups = {}
            for r in rowsC:
                if r.get("R_model_q") and r["R_fast"] is not None:
                    groups.setdefault(str(r[key]), []).append(
                        float(np.log(max(r["R_fast"], 1e-6) / max(r["R_model_q"], 1e-9))))
            gl = [np.array(v, float) for v in groups.values()]
            if len(gl) > 1 and min(len(g) for g in gl) >= 2:
                st, pv = kruskal(*gl)
                C[f"kruskal_residual_model_{key}"] = {"H": float(st), "p": float(pv),
                                                "groups": {k2: float(np.median(v)) for k2, v in groups.items()}}
                C[f"gap_residual_model_{key}"] = _gap_bound(groups)
        C["kappa_by_cell"] = {r["file"]: {"theta": r["theta_name"], "anharm": r["anharm_MHz"],
                                          "basis": r["basis"], "noise_scale": r["noise_scale"],
                                          "kappa": r["kappa"], "R_fast": r["R_fast"]} for r in rowsC}
        S["partC"] = C

    # ---------------- Part B ----------------
    lf = out / "stage3_lk.json"
    if lf.exists():
        rows = [r for r in json.loads(lf.read_text())["rows"]
                if r["control"] == "mean" and r.get("significant") is not None]
        B = {"n_points": len(rows),
             "n_significant": int(sum(r["significant"] for r in rows)),
             "n_2level": int(sum(r["n_levels"] == 2 for r in rows)),
             "n_4level": int(sum(r["n_levels"] == 4 for r in rows))}
        lab = np.array([r["significant"] for r in rows], bool)
        for var in ("kappa", "kappa_W", "kappa_lk"):
            sc = np.array([r[var] for r in rows], float)
            e = {"auc_all": _auc(sc, lab)}
            thr_all, j_all = _best_threshold(sc, lab)
            e.update(threshold_all=thr_all, youden_all=j_all)
            for lv in (2, 4):
                sel = np.array([r["n_levels"] == lv for r in rows], bool)
                if sel.sum() > 3 and 0 < lab[sel].sum() < sel.sum():
                    t, j = _best_threshold(sc[sel], lab[sel])
                    e[f"auc_{lv}level"] = _auc(sc[sel], lab[sel])
                    e[f"threshold_{lv}level"] = t
                    e[f"youden_{lv}level"] = j
            if e.get("threshold_2level") and e.get("threshold_4level"):
                e["log_threshold_gap"] = float(abs(math.log(e["threshold_4level"] / e["threshold_2level"])))
            rs = spearmanr(sc, [r.get("R_tail_lindblad", np.nan) for r in rows],
                           nan_policy="omit")
            e["spearman_vs_R_tail"] = {"rho": float(rs.statistic), "p": float(rs.pvalue)}
            B[var] = e
        B["B1_pass"] = bool(B["kappa_lk"]["auc_all"] >= B["kappa"]["auc_all"])
        if "log_threshold_gap" in B["kappa_lk"] and "log_threshold_gap" in B["kappa"]:
            B["B2_pass"] = bool(B["kappa_lk"]["log_threshold_gap"] < B["kappa"]["log_threshold_gap"])
        four = [r for r in rows if r["n_levels"] == 4 and r.get("R_tail_lindblad") is not None
                and r.get("R_model_lk") is not None]
        if four:
            within = sum(r["R_tail_lindblad"] <= r["R_model_lk"] + 1e-12 for r in four)
            B["B3_n_within_envelope"], B["B3_n"] = int(within), len(four)
            B["B3_frac"] = float(within / len(four))
            B["B3_pass"] = bool(within / len(four) >= 0.9)
            withinq = sum(r["R_tail_lindblad"] <= r["R_model_q"] + 1e-12 for r in four
                          if r.get("R_model_q") is not None)
            B["B3_frac_qubit_only_model"] = float(withinq / len(four))
        S["partB"] = B

    out.mkdir(parents=True, exist_ok=True)
    (out / "stage3_summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps({k: v for k, v in S.items() if k not in ("partA_rows", "partC_rows")}, indent=1, default=float))
    print(f"SUMMARY -> {out/'stage3_summary.json'}")
    return S


# ===========================================================================
# 8. CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0])
    sp = ap.add_subparsers(dest="step", required=True)

    st = sp.add_parser("selftest")
    st.add_argument("--outdir", default="results_stage3")

    cal = sp.add_parser("calibrate", help="solve for the noise scale that hits each kappa*")
    cal.add_argument("--kappa-star", type=float, nargs="+", default=[0.20, 0.35, 0.55])
    cal.add_argument("--rho", type=float, nargs="+", default=[0.0, 0.5, 0.9])
    cal.add_argument("--Tg", type=float, nargs="+", default=[8.0, 12.7, 20.0])
    cal.add_argument("--levels", type=int, choices=[2, 4], default=2)
    cal.add_argument("--base-sigma-d-mhz", type=float, default=2.5)
    cal.add_argument("--base-sigma-a", type=float, default=0.127)
    cal.add_argument("--slope", type=float, default=0.5, help="assumed d log kappa / d log s")
    cal.add_argument("--restarts", type=int, default=4)
    cal.add_argument("--maxiter", type=int, default=500)
    cal.add_argument("--outdir", default="results_stage3")

    rn = sp.add_parser("run", help="standard mean-vs-CVaR comparison at the calibrated cells")
    rn.add_argument("--no-warm-from-calib", dest="warm_from_calib", action="store_false",
                    help="do not give the mean side the calibrated control as a restart")
    rn.add_argument("--seeds", type=int, nargs="+", default=[0, 1],
                    help="noise seeds per cell; more seeds tighten the effect-size bound "
                         "of A5 (3 cells per knob value per seed)")
    rn.add_argument("--restarts", type=int, default=None)
    rn.add_argument("--maxiter", type=int, default=None)
    rn.add_argument("--outdir", default="results_stage3")

    dl = sp.add_parser("diagnose-lk", help="leakage-aware kappa for every saved control")
    dl.add_argument("folders", nargs="+")
    dl.add_argument("--outdir", default="results_stage3")
    dl.add_argument("--alpha", type=float, default=0.05)

    gt = sp.add_parser("gates", help="Part C: other rotations, anharmonicities, bases")
    gt.add_argument("--theta", nargs="+", default=["pi/4", "pi/2", "pi"],
                    help="rotation angles, e.g. pi/4 pi/2 pi")
    gt.add_argument("--axis", choices=["x", "y"], default="x")
    gt.add_argument("--anharm", type=float, nargs="+", default=[-200.0],
                    help="anharmonicity in MHz (negative)")
    gt.add_argument("--basis", nargs="+", default=["sine:0"],
                    help="basis specs kind:n_modes, e.g. sine:0 (default modes) sine:6 pc:0")
    gt.add_argument("--noise-scale", type=float, nargs="+", default=[1.0, 2.0])
    gt.add_argument("--sigma-d-mhz", type=float, default=2.5)
    gt.add_argument("--sigma-a", type=float, default=0.0)
    gt.add_argument("--rho", type=float, default=0.0)
    gt.add_argument("--Tg", type=float, default=12.7)
    gt.add_argument("--levels", type=int, choices=[2, 4], default=2)
    gt.add_argument("--restarts", type=int, default=4)
    gt.add_argument("--maxiter", type=int, default=500)
    gt.add_argument("--outdir", default="results_stage3")

    sm = sp.add_parser("summary")
    sm.add_argument("--outdir", default="results_stage3")
    sm.add_argument("--alpha", type=float, default=0.05)

    a = ap.parse_args()
    try:
        if a.step == "selftest":
            raise SystemExit(0 if selftest(a.outdir) else 1)
        if a.step == "calibrate":
            calibrate(a)
        elif a.step == "run":
            run(a)
        elif a.step == "gates":
            names = {"pi/4": np.pi / 4, "pi/2": np.pi / 2, "pi": np.pi, "3pi/4": 3 * np.pi / 4}
            a.theta = [(t, names[t] if t in names else float(t)) for t in a.theta]
            a.basis = [(b.split(":")[0], int(b.split(":")[1]) if ":" in b else 0) for b in a.basis]
            a.basis = [(k, n if n else 17) for k, n in a.basis]
            gates(a)
        elif a.step == "diagnose-lk":
            diagnose_lk(a.folders, a.outdir, a.alpha)
        elif a.step == "summary":
            summary(a.outdir, a.alpha)
    finally:
        if a.step in ("calibrate", "run", "gates", "diagnose-lk"):
            try:
                import fade_realistic as fr
                fr._shutdown_pool()
            except Exception:
                pass


if __name__ == "__main__":
    main()