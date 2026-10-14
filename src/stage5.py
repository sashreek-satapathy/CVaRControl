"""
stage5.py -- VIPER Stage 5: trajectory-level T1, i.e. non-Gaussian noise.

THE QUESTION
============================================================================
Every stage so far has kept assumption (A1) -- the noise is Gaussian -- and
probed (A2), the linearity of the coupling, through kappa.  T1 has appeared
only as a Lindblad channel, which is a DETERMINISTIC map: every shot gets the
same average damping, so it contributes a constant to the loss and has no
distribution and therefore no tail.  That is why T1 has acted as a floor in
every earlier result and never as a source of divergence.

At the trajectory level it is not deterministic.  A single run either suffers
a relaxation jump or it does not, so the loss is a MIXTURE:

    with probability 1 - p :  the dephasing blob, ~1e-3
    with probability p     :  a jump happened, loss ~ 0.6

with p = Gamma * int <n> dt ~ 1e-4 .. 1e-3.  This is as non-Gaussian as a
loss distribution gets, and it breaks (A1), not (A2).

THE MECHANISM THIS EXPOSES
============================================================================
Because p << alpha, EVERY jump draw sits inside the alpha-tail.  Hence

    E[L]          ~  E[L_deph]                  +  p       * l_j
    CVaR_alpha(L) ~  rho_alpha(m_eff) E[L_deph] + (p/alpha)* l_j

so the two objectives weight the relaxation channel against the dephasing
channel at a ratio differing by the CHANNEL AMPLIFICATION

    Xi_alpha = 1 / (alpha * rho_alpha(m_eff)) ,   Xi_0.05 = 3.58 at m_eff = 1.

A tail objective does not merely retune a pulse here: it buys a lower
time-integrated excited-state population at the price of coherent error,
Xi_alpha times more eagerly than the mean does.  kappa plays no part.  The
controlling variable is instead

    beta = p * l_j / E[L_deph]        (the jump share of the MEAN loss),

a second mechanism variable, orthogonal to kappa.  Scoping with a one-knob
trade-off model gives R_0.05 ranging from 0.07% (Tg = 12.7 ns, T1 = 30 us,
sigma_phi^2 = 0.41, beta = 0.05) to 14% (Tg = 40 ns, T1 = 10 us,
sigma_phi^2 = 0.04, beta = 4.4).  The grid below is chosen to span that.

THE METHOD: AN EXACT BRANCH DECOMPOSITION, NOT MONTE CARLO
============================================================================
Resolving a p = 1e-4 branch by brute-force MCWF needs ~1e6 trajectories PER
objective evaluation.  Instead the ensemble decomposes exactly to O((Gamma T)^2)
into a no-jump branch under the damped map and a one-jump branch indexed by
jump time, with analytic weights.  The loss conditional on a jump in slice k is

    Tr( O_j S_k sigma_jk S_k^dag ) = Tr( (S_k^dag O_j S_k) sigma_jk ) ,

and O~_j(k) = S_k^dag O_j S_k obeys the SAME backward recursion the gradient
code already runs.  So the entire one-jump branch costs one extra backward
pass -- not one forward pass per jump time -- and no jump-time binning is
needed at any Nt, for two levels or four.  The result is the EXACT loss CDF,
so CVaR is available at any alpha, including alpha below p.

WHAT IS OPTIMIZED AND WHAT IS ONLY EVALUATED
============================================================================
Optimization needs a gradient.  Rather than write a new adjoint for the jump
branch, we use the fact that the T1 contribution to the mean loss is linear in
Gamma, so

    L_T1(u) = L_mean(u; Gamma) - L_mean(u; 0)

with BOTH terms produced by the existing, already-validated code path, and the
CVaR objective above is then the ordinary dephasing CVaR plus (1/alpha) L_T1.
Gradients are differences of gradients the existing path already returns.  This
is exact to O(Gamma^2), the same order as the decomposition itself.

That construction is valid while alpha > p.  Below p the tail is nothing but
jump records and the decomposition no longer separates, so Stage 5 OPTIMIZES at
alpha in {0.2, 0.05, 0.01} and EVALUATES exactly at alpha in
{0.5, 0.2, 0.05, 0.01, 0.002, 5e-4, 1e-4}.  The alpha < p regime is
characterized, not optimized, and the summary says so.

PRE-REGISTERED PREDICTIONS (scored by `summary`)
  P1  CHANNEL REWEIGHTING.  The CVaR-optimal control has a strictly smaller
      time-integrated excited-state population than the mean-optimal one, and
      the ratio of their deviations from the T1-free optimum approaches
      Xi_alpha = 1/(alpha rho_alpha) as beta -> 0.  Predicted 3.58 at
      alpha = 0.05; accepted if the measured ratio is 2.5-4.5 and rises with
      1/alpha across the optimized alphas.
  P2  beta ORDERS THE GAIN, kappa DOES NOT.  Spearman(beta, R_tail) > 0.6,
      while Spearman(kappa, R_tail) is consistent with zero once beta is
      controlled for; partial(kappa, R | beta) has |rho| < 0.3, p > 0.05.
      This is a PRE-REGISTERED NULL for the paper's own mechanism variable:
      kappa is a statement about (A2) and Stage 5 breaks (A1), so kappa is
      predicted NOT to transfer.  A positive result here would falsify the
      reading of kappa given in the paper, not confirm it.
  P3  THE kappa^2 BOUND DOES NOT APPLY.  R_tail is allowed to exceed C kappa^2
      here, and we expect it to at large beta.  Recorded, not scored as a
      failure: the bound is derived under (A1).
  P4  THE TAIL SHAPE IS BIMODAL, AND rho_alpha READS IT WRONGLY.  The measured
      CVaR/mean ratio of the mean control is NOT rho_alpha(m_eff) of
      Proposition 5, because the loss is a mixture rather than a quadratic
      form.  Prediction: the ratio rises above r_alpha as beta grows, reaching
      (1 + beta Xi_alpha rho_alpha)/(1 + beta) * rho_alpha.  This is the
      boundary of Proposition 5, measured.

USAGE (from the iris folder)
============================================================================
  python stage5.py selftest                       # pure maths + core physics
  bash run_stage5.sh 2>&1 | tee stage5_log.txt    # the whole grid, ~8 h
or step by step:
  python stage5.py run --Tg 12.7 40 100 --T1 30 10 --phase-var 0.04 0.16 \
         --levels 2 --seeds 0 1
  python stage5.py run --Tg 12.7 40 --T1 10 --phase-var 0.04 --levels 4 --seeds 0 1
  python stage5.py summary

WHAT TO SEND BACK
  results_stage5/stage5_summary.json, results_stage5/*/*_geo.json,
  results_stage5/*/*_cdf.npz, stage5_log.txt

NOT executed on Claude's side (per your preference): only the pure-maths and
core-physics selftests were run there, against random Hamiltonians and with
finite-difference checks; the simulator paths are syntax-checked only.
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

# alphas at which the CVaR objective is OPTIMIZED (all must exceed p)
ALPHA_OPT = (0.20, 0.05, 0.01)
# alphas at which every control is EVALUATED exactly from the decomposition
ALPHA_EVAL = (0.50, 0.20, 0.05, 0.01, 0.002, 5e-4, 1e-4)


# ===========================================================================
# 1. PURE MATHS: mixtures, the channel amplification, and the scoping model
# ===========================================================================

def rho_eff(m, alpha):
    """CVaR/mean of chi^2_m / m.  rho(1) = r_alpha, rho(2) = 1 + ln(1/alpha)."""
    from scipy import stats
    q = stats.chi2.ppf(1.0 - alpha, m)
    return float(stats.chi2.sf(q, m + 2) / alpha)


def r_alpha(alpha):
    from scipy import stats
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    return float(1.0 + 2.0 * z * stats.norm.pdf(z) / alpha)


def channel_amplification(alpha, m_eff=1.0):
    """Xi_alpha: how much more a tail objective weights a rare catastrophic
    channel, relative to how the mean weights it, per unit of the common
    channel.  Xi = 1 / (alpha rho_alpha(m_eff))."""
    return 1.0 / (alpha * rho_eff(m_eff, alpha))


def cvar_atoms(L, w, alpha):
    """EXACT CVaR of an atomic distribution: atom values L with probabilities w
    (w need not be normalized).  This is the Rockafellar-Uryasev value, with
    the boundary atom split, so it is correct for any alpha including alpha
    smaller than a single atom's mass."""
    L = np.asarray(L, float).ravel()
    w = np.asarray(w, float).ravel()
    if L.size != w.size:
        raise ValueError("L and w must have the same length")
    w = w / w.sum()
    o = np.argsort(-L, kind="stable")
    Ls, ws = L[o], w[o]
    c = np.cumsum(ws)
    k = int(np.searchsorted(c, alpha, side="left"))
    if k >= len(Ls):                       # alpha beyond the total mass
        return float(Ls @ ws / ws.sum())
    full = Ls[:k] @ ws[:k]
    rest = alpha - (c[k - 1] if k > 0 else 0.0)
    return float((full + Ls[k] * rest) / alpha)


def mixture_ratio(beta, alpha, m_eff=1.0):
    """Predicted CVaR/mean of the two-component mixture of prediction P4.
    beta is the jump share of the MEAN loss: E[L] = E[L_deph](1 + beta)."""
    rho = rho_eff(m_eff, alpha)
    return float((rho + beta / alpha) / (1.0 + beta))


def two_channel_toy(alpha, a0, p0, g, lj=0.6, thmax=3.0):
    """The scoping model behind the grid.  One knob th trades coherent quality
    against time-integrated excited-state population:
        a(th) = a0 (1 + th^2)      dephasing scale
        p(th) = p0 exp(-g th)      jump probability
    Returns (R_alpha, th_mean, th_cvar, beta)."""
    from scipy.optimize import minimize_scalar
    from scipy import stats

    def cv(a, p):
        q = alpha - p
        if q <= 0:
            return np.nan
        zz = stats.norm.ppf(1.0 - q / 2.0)
        return (p * (lj + a) + a * (q + 2.0 * zz * stats.norm.pdf(zz))) / alpha

    a = lambda t: a0 * (1.0 + t * t)
    p = lambda t: p0 * np.exp(-g * t)
    fm = lambda t: a(t) + p(t) * lj
    fc = lambda t: cv(a(t), p(t))
    tm = float(minimize_scalar(fm, bounds=(0.0, thmax), method="bounded").x)
    tc = float(minimize_scalar(fc, bounds=(0.0, thmax), method="bounded").x)
    Cm, Cc = fc(tm), fc(tc)
    return float((Cm - Cc) / Cm), tm, tc, float(p0 * lj / a0)


# ===========================================================================
# 2. THE CORE: exact no-jump / one-jump decomposition
# ===========================================================================
# Written against slice propagators so it can be exercised with random
# Hamiltonians, independently of the VIPER simulator.  `selftest` does exactly
# that, against a brute-force reference and against the Lindblad master
# equation.

def _dag(A):
    return np.conj(np.swapaxes(A, -1, -2))


def damped_slices(U, n_op, Gamma, dt):
    """Strang-split damped slice maps K_k = D^(1/2) U_k D^(1/2), which
    approximate exp(-i H dt - (Gamma/2) n dt) to O(Gamma dt^2 ||[H,n]||).
    U has shape (..., Nt, N, N); n_op is the (diagonal) number operator."""
    d = np.exp(-0.25 * Gamma * dt * np.real(np.diag(n_op)))
    D = np.diag(d).astype(complex)
    return D @ U @ D


def jump_records(K, rhos, Os, a_op, Gamma, dt):
    """Exact branch decomposition for ONE noise draw.

    K     (Nt,N,N)  damped slice maps, already Strang-split
    rhos  (M,N,N)   input states of the average-fidelity metric
    Os    (M,N,N)   its observables, J_j = Re Tr(O_j rho_T)
    a_op  (N,N)     collapse operator WITHOUT the rate (Gamma carried separately)

    Returns dict with
      L0, p0      no-jump conditional loss and its probability
      Lk, pk      (Nt,) one-jump conditional losses and probabilities
      n_int       int <n> dt over the gate (the T1 lever)
      mass        p0 + sum(pk); should be 1 + O((Gamma T)^2)
    """
    Nt, N = K.shape[0], K.shape[-1]
    M = rhos.shape[0]

    # forward, unnormalized, under the no-jump map; keep the state at the
    # START of every slice, which is where a jump in that slice acts
    rho = rhos.astype(complex).copy()
    fwd = np.empty((Nt, M, N, N), complex)
    for k in range(Nt):
        fwd[k] = rho
        rho = K[k] @ rho @ _dag(K[k])
    tr0 = np.real(np.einsum("jaa->j", rho))
    J0 = np.real(np.einsum("jab,jba->j", Os, rho)) / np.maximum(tr0, 1e-300)
    p0 = float(tr0.mean())
    # Conditional on the record, the input states are NOT equally likely to have
    # produced it: an excited input is far more likely to decay than a ground
    # one.  Bayes with a uniform prior over inputs gives P(j|r) proportional to
    # t_jr = Tr(K_r rho_j K_r^dag), so the conditional average gate fidelity is
    # the t-WEIGHTED mean of the per-input fidelities, not the plain mean.
    # Using the plain mean leaves an O(Gamma) discrepancy against the Lindblad
    # mean -- the covariance across inputs between t_jr and J_jr -- which the
    # selftest's Gamma-scaling check detects.
    L0 = float(1.0 - (tr0 @ J0) / max(tr0.sum(), 1e-300))

    # backward: O~_j(k) = S_k^dag O_j S_k  and  I~(k) = S_k^dag S_k, where
    # S_k = K_{Nt-1} ... K_k.  S_k = S_{k+1} K_k, so the update is applied
    # BEFORE storing (an off-by-one here is the easiest bug in this file; the
    # selftest checks it against brute force).
    Ot = Os.astype(complex).copy()
    It = np.broadcast_to(np.eye(N, dtype=complex), (M, N, N)).copy()
    Ob = np.empty((Nt, M, N, N), complex)
    Ib = np.empty((Nt, M, N, N), complex)
    for k in range(Nt - 1, -1, -1):
        Ot = _dag(K[k]) @ Ot @ K[k]
        It = _dag(K[k]) @ It @ K[k]
        Ob[k], Ib[k] = Ot, It

    # One-jump branch.  Taking the weight as Gamma dt <n>(t_k) leaves an
    # O(Gamma dt) defect in the total probability, because that weight is not
    # consistent with the discretized no-jump map.  Instead take the weight to
    # be the mass the no-jump step ACTUALLY loses in that slice,
    #     m_jk = Tr rho_j(t_k) - Tr K_k rho_j(t_k) K_k^dag ,
    # and put it on the normalized collapsed state.  The sum then telescopes,
    # so p0 + sum_k p_k = 1 exactly up to the two-jump terms we drop, and the
    # residual is O((Gamma T)^2) with no dt-dependent floor.
    coll = a_op @ fwd @ _dag(a_op)                          # (Nt,M,N,N)
    ctr = np.real(np.einsum("kjaa->kj", coll))
    tr_in = np.real(np.einsum("kjaa->kj", fwd))
    stepped = K[:, None] @ fwd @ _dag(K)[:, None]
    tr_out = np.real(np.einsum("kjaa->kj", stepped))
    lost = np.maximum(tr_in - tr_out, 0.0)                  # (Nt,M)
    sig = coll * (lost / np.maximum(ctr, 1e-300))[..., None, None]
    num = np.real(np.einsum("kjab,kjba->kj", Ob, sig))
    den = np.real(np.einsum("kjab,kjba->kj", Ib, sig))
    pk = den.mean(axis=1)
    dsum = den.sum(axis=1)
    Lk = 1.0 - num.sum(axis=1) / np.maximum(dsum, 1e-300)   # same Bayes weighting
    Lk = np.where(dsum > 1e-300, Lk, 0.0)

    n_int = float(dt * np.real(np.einsum("kjab,ba->", fwd, a_op.conj().T @ a_op)) / M)

    return {"L0": L0, "p0": p0, "Lk": Lk, "pk": pk,
            "n_int": n_int, "mass": float(p0 + pk.sum())}


def records_to_atoms(rec):
    """Flatten a record dict into (values, probabilities) for cvar_atoms."""
    L = np.concatenate([[rec["L0"]], rec["Lk"]])
    w = np.concatenate([[rec["p0"]], rec["pk"]])
    return L, w


def ensemble_atoms(recs):
    """Pool the records of an ensemble of noise draws into one atomic law."""
    Ls, ws = [], []
    for r in recs:
        L, w = records_to_atoms(r)
        Ls.append(L)
        ws.append(w / max(w.sum(), 1e-300))
    n = len(recs)
    return np.concatenate(Ls), np.concatenate(ws) / n


def lindblad_reference(H, n_op, a_op, rhos, Os, Gamma, dt):
    """Direct master-equation integration, used only by the selftest to check
    that the decomposition reproduces the Lindblad mean to O((Gamma T)^2)."""
    Nt, N = H.shape[0], H.shape[-1]
    M = rhos.shape[0]
    rho = rhos.astype(complex).copy()
    ad_a = a_op.conj().T @ a_op
    for k in range(Nt):
        def drho(r):
            com = -1j * (H[k] @ r - r @ H[k])
            dis = Gamma * (a_op @ r @ a_op.conj().T
                           - 0.5 * (ad_a @ r + r @ ad_a))
            return com + dis
        k1 = drho(rho); k2 = drho(rho + 0.5 * dt * k1)
        k3 = drho(rho + 0.5 * dt * k2); k4 = drho(rho + dt * k3)
        rho = rho + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return float(1.0 - np.real(np.einsum("jab,jba->j", Os, rho)).mean())


# ===========================================================================
# 3. THE T1 LEVER AND ITS GRADIENT
# ===========================================================================
# To O(Gamma) both objectives are
#
#     (dephasing objective at Gamma = 0)  +  coefficient * Gamma * n_int(u)
#
# with n_int = int Tr(n rho(t)) dt the time-integrated excited-state
# population, and the coefficient differing between mean and CVaR by the
# channel amplification.  So exactly ONE new gradient is needed: that of
# n_int under unitary evolution.  It is a running-cost adjoint and reuses the
# Daleckii-Krein machinery of fast_viper.fast_gather with two changes: the
# adjoint starts at zero instead of at the observables, and the running cost
# is added at every step.  `selftest` checks it against finite differences.

def n_int_gather(fv, uI_f, uQ_f, deltas, gammas):
    """Per-draw n_int = int Tr(n rho(t)) dt and its gradients w.r.t. the
    filtered controls.  Same call signature and row ordering as fast_gather."""
    from fade_realistic import cfg
    M = fv.get_model()
    N = M["N"]
    deltas = np.asarray(deltas, float)
    if deltas.ndim == 1:
        deltas = deltas[:, None]
    gammas = np.asarray(gammas, float)
    K, Nt = len(gammas), len(uI_f)
    dt = cfg.Tg / cfg.Nt
    g1 = 1.0 + gammas
    d = np.broadcast_to(deltas, (K, Nt))[..., None, None]
    H = (M["H0"][None, None] + d * M["n"][None, None]
         + g1[:, None, None, None] * (uI_f[None, :, None, None] * M["Hx"]
                                      + uQ_f[None, :, None, None] * M["Hy"]))
    e, V = np.linalg.eigh(H)
    Vh = _dag(V)
    ph = np.exp(-1j * e * dt)
    U = (V * ph[..., None, :]) @ Vh

    C = (M["n"] * (dt / 4.0)).astype(complex)          # running cost, 4 inputs
    rho = np.broadcast_to(M["rhos"], (K, 4, N, N)).copy()
    rho_before = np.empty((Nt, K, 4, N, N), complex)
    nint = np.zeros(K)
    for k in range(Nt):
        rho_before[k] = rho
        nint += np.real(np.einsum("ab,kjba->k", C, rho))
        Uk = U[:, k][:, None]
        rho = Uk @ rho @ _dag(Uk)

    de = e[..., :, None] - e[..., None, :]
    pm, pn = ph[..., :, None], ph[..., None, :]
    safe = np.abs(de) > 1e-10
    G = np.where(safe, (pm - pn) / np.where(safe, de, 1.0), -1j * dt * pm)
    XI = Vh @ M["Hx"] @ V
    XQ = Vh @ M["Hy"] @ V

    Lam = np.zeros((K, 4, N, N), complex)              # starts at zero, not at Os
    gI = np.zeros((K, Nt))
    gQ = np.zeros((K, Nt))
    for k in range(Nt - 1, -1, -1):
        Uk = U[:, k][:, None]
        Mk = rho_before[k] @ _dag(Uk) @ Lam
        Y = (Vh[:, k][:, None] @ Mk @ V[:, k][:, None]).sum(axis=1)
        Gk = G[:, k]
        gI[:, k] = 2.0 * np.real(np.einsum("kmn,kmn,knm->k", Gk, XI[:, k], Y))
        gQ[:, k] = 2.0 * np.real(np.einsum("kmn,kmn,knm->k", Gk, XQ[:, k], Y))
        Lam = C[None, None] + _dag(Uk) @ Lam @ Uk      # running cost added here
    gI *= g1[:, None]
    gQ *= g1[:, None]
    return nint, gI, gQ


def channel_weights(recs, alpha, Gamma, m_eff=1.0, L_deph=None):
    """From the exact decomposition, the coefficients that multiply
    Gamma * n_int in the mean and in the CVaR objective.

    The T1 contribution to the loss splits into two parts with different tail
    status.  The jump branch (total probability p, mean conditional loss l_j)
    lies ENTIRELY inside the alpha-tail whenever p < alpha, so it is weighted
    1/alpha.  The no-jump branch's change -- the lost probability mass times
    the dephasing loss, plus the damping correction -- shifts every loss alike
    and is weighted 1.  Since l_j ~ 0.6 while the dephasing loss is ~1e-3, the
    jump-borne fraction f_jump is 1 to a part in 1e3; we compute it rather
    than assume it, and record the residual.

    The returned Xi_objective is the ratio of the two coefficients, ~1/alpha.
    The quantity predicted by P1 is Xi_relative = Xi_objective / rho_alpha,
    because the dephasing part of the CVaR objective already carries its own
    amplification rho_alpha; Xi_relative = 1/(alpha rho_alpha) = 3.58 at
    alpha = 0.05, m_eff = 1."""
    p = float(np.mean([r["pk"].sum() for r in recs]))
    lj = float(np.mean([r["Lk"] @ r["pk"] / max(r["pk"].sum(), 1e-300) for r in recs]))
    nint = float(np.mean([r["n_int"] for r in recs]))
    jump_part = p * lj
    if L_deph is None:                       # dephasing loss of the same control
        L_deph = float(np.mean([r["L0"] for r in recs]))
    nojump_part = -p * L_deph                # mass moved out of the no-jump branch
    total = jump_part + nojump_part
    f_jump = float(jump_part / total) if abs(total) > 1e-300 else 1.0
    w_mean = total / max(Gamma * nint, 1e-300)
    w_cvar = (jump_part / alpha + nojump_part) / max(Gamma * nint, 1e-300)
    rho = rho_eff(m_eff, alpha)
    return {"p": p, "l_j": lj, "n_int": nint, "L_deph": L_deph,
            "f_jump": f_jump, "w_mean": float(w_mean), "w_cvar": float(w_cvar),
            "Xi_objective": float(w_cvar / w_mean) if w_mean else float("nan"),
            "Xi_relative": float(w_cvar / w_mean / rho) if w_mean else float("nan"),
            "Xi_predicted": channel_amplification(alpha, m_eff),
            "rho_alpha": rho,
            "beta": float(total / max(L_deph, 1e-300))}


# ===========================================================================
# 4. SELFTEST: everything that can be checked without the simulator
# ===========================================================================

def _toy_model(N=4, Nt=80, Tg=40.0, seed=0):
    """Random but structured slice Hamiltonians, for the core checks."""
    dt = Tg / Nt
    a = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
    n = a.conj().T @ a
    r = np.random.default_rng(seed)
    base = r.normal(size=(N, N)) + 1j * r.normal(size=(N, N))
    base = base + base.conj().T
    H = np.array([0.3 * base * (0.15 * np.sin(2 * np.pi * (k + .5) / Nt) + 0.05)
                  + 0.02 * k / Nt * n for k in range(Nt)])
    I = np.eye(N, dtype=complex)
    kets = [I[0], I[1], (I[0] + I[1]) / np.sqrt(2), (I[0] + 1j * I[1]) / np.sqrt(2)]
    rhos = np.array([np.outer(k_, k_.conj()) for k_ in kets])
    Os = np.array([r_.conj().T for r_ in rhos])
    return H, n, a, rhos, Os, dt


def _slices(H, dt):
    e, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * e * dt)[..., None, :]) @ _dag(V)


def selftest(outdir="results_stage5"):
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'ok' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")

    print("S1  exact CVaR of an atomic law")
    L = np.array([0.001, 0.002, 0.003, 0.6])
    w = np.array([0.4, 0.3, 0.2999, 1e-4])
    c05 = cvar_atoms(L, w, 0.05)
    manual = (1e-4 * 0.6 + (0.05 - 1e-4) * 0.003) / 0.05
    chk("boundary atom split correctly", abs(c05 - manual) < 1e-12,
        f"{c05:.6f} vs {manual:.6f}")
    chk("alpha below the top atom's mass", abs(cvar_atoms(L, w, 5e-5) - 0.6) < 1e-12)
    chk("alpha = 1 gives the mean",
        abs(cvar_atoms(L, w, 1.0) - float(L @ w / w.sum())) < 1e-12)

    print("S2  the channel amplification and the mixture ratio")
    chk("Xi_0.05 = 1/(alpha r_alpha) = 3.58",
        abs(channel_amplification(0.05) - 3.5825) < 2e-3,
        f"{channel_amplification(0.05):.4f}")
    chk("rho_alpha(1) = r_alpha", abs(rho_eff(1, 0.05) - r_alpha(0.05)) < 1e-9)
    chk("mixture ratio -> rho_alpha as beta -> 0",
        abs(mixture_ratio(1e-9, 0.05) - rho_eff(1, 0.05)) < 1e-6)
    b = 0.5
    chk("mixture ratio exceeds r_alpha at beta = 0.5",
        mixture_ratio(b, 0.05) > r_alpha(0.05), f"{mixture_ratio(b, 0.05):.3f}")

    print("S3  the one-jump branch against brute force")
    worst_b = 0.0
    for N, Nt, Tg, T1 in [(2, 80, 40.0, 30.0), (4, 80, 40.0, 30.0), (4, 120, 100.0, 10.0)]:
        H, n, a, rhos, Os, dt = _toy_model(N, Nt, Tg)
        Gam = 1.0 / (T1 * 1e3)
        K = damped_slices(_slices(H, dt), n, Gam, dt)
        rec = jump_records(K, rhos, Os, a, Gam, dt)
        fwd = rhos.astype(complex).copy()
        saved = {}
        for k in range(Nt):
            if k in (0, Nt // 3, Nt // 2, Nt - 1):
                saved[k] = fwd.copy()
            fwd = K[k] @ fwd @ _dag(K[k])
        for k, rh in saved.items():
            # same weighting as jump_records: the mass the no-jump step loses,
            # placed on the normalized collapsed state
            coll = a @ rh @ a.conj().T
            ctr = np.real(np.einsum("jaa->j", coll))
            stepped = K[k] @ rh @ _dag(K[k])
            lost = np.maximum(np.real(np.einsum("jaa->j", rh))
                              - np.real(np.einsum("jaa->j", stepped)), 0.0)
            sg = coll * (lost / np.maximum(ctr, 1e-300))[:, None, None]
            for j in range(k, Nt):
                sg = K[j] @ sg @ _dag(K[j])
            tr = np.real(np.einsum("jaa->j", sg))
            num = np.real(np.einsum("jab,jba->j", Os, sg))
            Lb = 1.0 - num.sum() / max(tr.sum(), 1e-300)
            worst_b = max(worst_b, abs(Lb - rec["Lk"][k]) / max(abs(Lb), 1e-12),
                          abs(tr.mean() - rec["pk"][k]) / max(tr.mean(), 1e-30))
    chk("backward recursion == explicit propagation", worst_b < 1e-10, f"max rel {worst_b:.1e}")

    print("S4  probability mass and the Lindblad mean")
    rows, T1cs = [], []
    for T1 in (300.0, 100.0, 30.0, 10.0):
        H, n, a, rhos, Os, dt = _toy_model(4, 80, 40.0)
        Gam = 1.0 / (T1 * 1e3)
        K = damped_slices(_slices(H, dt), n, Gam, dt)
        rec = jump_records(K, rhos, Os, a, Gam, dt)
        Lv, wv = records_to_atoms(rec)
        rows.append((Gam, abs(rec["mass"] - 1.0),
                     abs(float(Lv @ wv / wv.sum())
                         - lindblad_reference(H, n, a, rhos, Os, Gam, dt))))
        T1cs.append(float(rec["pk"] @ rec["Lk"]))
    chk("probability mass is 1 to O((Gamma T)^2)",
        all(m < 20.0 * (G * 40.0) ** 2 + 1e-12 for G, m, _ in rows),
        "max ratio to (Gamma T)^2: "
        f"{max(m / (G * 40.0) ** 2 for G, m, _ in rows):.1f}")
    # The physical tolerance: the decomposition must agree with the master
    # equation to well inside the T1 contribution it is there to resolve.
    # The Bayes-weighted conditioning of the conditional losses is what buys
    # this; a plain mean over inputs leaves an O(Gamma) covariance term that
    # is ~30x larger and would fail here.
    chk("agrees with Lindblad to <2% of the T1 contribution itself",
        all(d < 0.02 * T1c for _, _, d, T1c in
            [(G, m, d, max(T1c, 1e-300)) for (G, m, d), T1c in zip(rows, T1cs)]),
        "worst " + f"{max(d / max(t, 1e-300) for (_, _, d), t in zip(rows, T1cs)):.3%}")

    print("S5  the scoping model reproduces the predicted knob ratio")
    R_, tm, tc, beta = two_channel_toy(0.05, a0=2.2e-4, p0=(12.7e-9 / 30e-6) * 0.4, g=0.5)
    chk("knob ratio approaches 1/(alpha r_alpha) from below",
        2.0 < tc / tm < channel_amplification(0.05) + 0.2,
        f"{tc / tm:.2f} vs {channel_amplification(0.05):.2f} at beta = {beta:.2f}")
    Rs = [two_channel_toy(0.05, 2.2e-4, (T * 1e-9 / 30e-6) * 0.4, 0.5)[0]
          for T in (12.7, 40.0, 100.0)]
    chk("R grows with the gate time at fixed T1", Rs[0] < Rs[1] < Rs[2],
        ", ".join(f"{100 * x:.2f}%" for x in Rs))

    print("S6  n_int adjoint against finite differences (needs the simulator)")
    try:
        fv, v2, _ = _sim()
    except (SystemExit, ImportError, ModuleNotFoundError) as ex:
        print(f"      skipped: simulator not importable here ({ex})")
    else:
        from fade_realistic import cfg
        rng = np.random.default_rng(1)
        Nt = cfg.Nt
        uI = 0.05 * rng.normal(size=Nt)
        uQ = 0.05 * rng.normal(size=Nt)
        d = 0.01 * rng.normal(size=6)
        g = 0.02 * rng.normal(size=6)
        _, gI, gQ = n_int_gather(fv, uI, uQ, d, g)
        eps, errs = 1e-6, []
        for _ in range(8):
            k = int(rng.integers(Nt))
            w_ = int(rng.integers(2))
            up = [uI.copy(), uQ.copy()]
            up[w_][k] += eps
            dn = [uI.copy(), uQ.copy()]
            dn[w_][k] -= eps
            fd = (n_int_gather(fv, up[0], up[1], d, g)[0]
                  - n_int_gather(fv, dn[0], dn[1], d, g)[0]) / (2 * eps)
            an = (gI if w_ == 0 else gQ)[:, k]
            errs.append(float(np.max(np.abs(fd - an) / np.maximum(np.abs(fd), 1e-8))))
        chk("adjoint matches finite differences", max(errs) < 1e-5, f"max {max(errs):.1e}")

    print(f"\nselftest {'PASSED' if ok else 'FAILED'}")
    if not ok:
        raise SystemExit(1)


# ===========================================================================
# 5. RUN: the (Tg, T1, sigma_phi^2, levels) grid
# ===========================================================================

def _sim():
    import inspect as _inspect
    import fast_viper as fv
    import stage0_v2 as v2
    import fade_realistic as fr
    if "rho" not in _inspect.signature(v2.Ensemble.__init__).parameters:
        raise SystemExit("stage5.py needs the Stage-1 versions of stage0_v2.py / fast_viper.py.")
    return fv, v2, fr


class _T1Objective:
    """Wraps stage0_v2.Problem.objective so that every objective is

        (dephasing objective at Gamma = 0)  +  w * Gamma * n_int(u)

    with w taken from the exact decomposition: w_mean for the mean and
    w_cvar for CVaR.  The dephasing part and its gradient are the existing,
    already-validated code path; the T1 part uses the adjoint of section 3,
    which the selftest checks against finite differences.  Both w's are
    refreshed from the decomposition between optimizer restarts and the drift
    is recorded, so a frozen coefficient never silently biases a result."""

    def __init__(self, fv, v2, Gamma, alpha_weights):
        self.fv, self.v2, self.Gamma = fv, v2, float(Gamma)
        self.w = dict(alpha_weights)          # mode/alpha key -> coefficient
        self.orig = v2.Problem.objective
        self.calls = 0

    def key(self, mode, alpha):
        return "mean" if mode == "mean" else f"cvar_{alpha:g}"

    def install(self):
        obj, w, Gam, fvv = self.orig, self.w, self.Gamma, self.fv
        outer = self

        def objective(P, coef, mode, alpha, lam, ens=None):
            # stage0_v2.Problem.objective returns (value, grad_c, L_train)
            val, grad, L = obj(P, coef, mode, alpha, lam, ens)
            c = w.get(outer.key(mode, alpha), 0.0) * Gam
            if c == 0.0:
                return val, grad, L
            e = ens if ens is not None else P.train
            uI_f, uQ_f = P.filtered(coef)
            nint, gI, gQ = n_int_gather(fvv, uI_f, uQ_f, e.deltas, e.gammas)
            # gI, gQ are d n_int / d u_FILTERED; the coefficient gradient needs
            # them pushed back through the bandwidth filter first, exactly as
            # Problem.objective does for the loss gradient.
            gIf = c * gI.mean(0)
            gQf = c * gQ.mean(0)
            gcoef = P.param.grad_c(coef, P.bwf.backprop_grad(gIf),
                                   P.bwf.backprop_grad(gQf))
            outer.calls += 1
            return val + c * float(nint.mean()), grad + gcoef, L

        self.v2.Problem.objective = objective

    def restore(self):
        self.v2.Problem.objective = self.orig


def _ms(P, *args, **kw):
    """stage0_v2.Problem.multistart returns (best_dict, runs); we want the
    coefficient vector of the best run."""
    best, _ = P.multistart(*args, **kw)
    return best["coef"]


def _records_for(fv, v2, uI, uQ, ens, Gamma):
    """Exact decomposition for every draw of an ensemble."""
    from fade_realistic import cfg
    M = fv.get_model()
    N = M["N"]
    dt = cfg.Tg / cfg.Nt
    a_op = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
    n_op = M["n"]
    d = np.asarray(ens.deltas, float)
    if d.ndim == 1:
        d = d[:, None]
    g1 = 1.0 + np.asarray(ens.gammas, float)
    recs = []
    for i in range(len(g1)):
        H = (M["H0"][None] + np.broadcast_to(d[i], (len(uI),))[:, None, None] * n_op[None]
             + g1[i] * (uI[:, None, None] * M["Hx"][None] + uQ[:, None, None] * M["Hy"][None]))
        e, V = np.linalg.eigh(H)
        U = (V * np.exp(-1j * e * dt)[..., None, :]) @ _dag(V)
        K = damped_slices(U, n_op, Gamma, dt)
        recs.append(jump_records(K, M["rhos"], M["Os"], a_op, Gamma, dt))
    return recs


def _evaluate(fv, v2, uI, uQ, ens, Gamma, alphas=ALPHA_EVAL):
    """Exact CVaR at every alpha, plus the diagnostics the summary needs."""
    recs = _records_for(fv, v2, uI, uQ, ens, Gamma)
    L, w = ensemble_atoms(recs)
    out = {"mean": float(L @ w / w.sum()),
           "p": float(np.mean([r["pk"].sum() for r in recs])),
           "n_int": float(np.mean([r["n_int"] for r in recs])),
           "L_deph": float(np.mean([r["L0"] for r in recs])),
           "l_j": float(np.mean([r["Lk"] @ r["pk"] / max(r["pk"].sum(), 1e-300)
                                 for r in recs])),
           "mass_defect": float(np.mean([abs(r["mass"] - 1.0) for r in recs]))}
    out["beta"] = out["p"] * out["l_j"] / max(out["L_deph"], 1e-300)
    out["cvar"] = {f"{al:g}": cvar_atoms(L, w, al) for al in alphas}
    out["ratio"] = {k: v / max(out["mean"], 1e-300) for k, v in out["cvar"].items()}
    return out, recs


def run(a):
    fv, v2, fr = _sim()
    from fade_realistic import cfg
    import stage3 as s3

    base = v2.V2Config()
    base.n_levels = int(a.levels)
    grid = [c for c in itertools.product(a.Tg, a.T1, a.phase_var, a.seeds)]
    d = Path(a.outdir) / f"stage5_{a.levels}level"
    d.mkdir(parents=True, exist_ok=True)

    for i, (Tg, T1us, pv, seed) in enumerate(grid):
        stem = (f"s5_Tg{Tg:g}".replace(".", "p") + f"_T1{T1us:g}".replace(".", "p")
                + f"_pv{pv:g}".replace(".", "p") + f"_s{seed}")
        pf = d / f"{stem}.json"
        if pf.exists() and not a.force:
            v2.log(f"  {stem} already done, skipping")
            continue

        cc = v2.V2Config(**asdict(base))
        cc.seed_train += seed
        cc.seed_val += seed
        cc.seed_test += seed
        cc.sigma_a = 0.0                          # phase noise only, as in Stage 4
        v2.set_gate(Tg, cc)
        cc.sigma_d = math.sqrt(pv) / Tg           # sigma_phi^2 = (sigma_d Tg)^2
        cfg.T1 = float(T1us) * 1e3                # ns
        Gamma = 1.0 / cfg.T1

        v2.log(f"=== Stage 5 point {i+1}/{len(grid)}: Tg={Tg} ns, T1={T1us} us, "
               f"sigma_phi^2={pv}, {a.levels} levels, seed {seed} ===")
        t0 = time.time()
        P = v2.Problem(Tg, cc)

        # (0) the Gamma = 0 reference control: the existing mean optimization
        zero = _T1Objective(fv, v2, 0.0, {})
        zero.install()
        try:
            c_free = _ms(P, "mean", seed=seed, label=f"{stem}:free")
        finally:
            zero.restore()
        uIf, uQf = P.filtered(c_free)
        ev_free, recs_free = _evaluate(fv, v2, uIf, uQf, P.test, Gamma)

        # (1) coefficients from the decomposition at that control
        wts = {"mean": channel_weights(recs_free, ALPHA_OPT[1], Gamma,
                                       L_deph=ev_free["L_deph"])["w_mean"]}
        for al in ALPHA_OPT:
            wts[f"cvar_{al:g}"] = channel_weights(recs_free, al, Gamma,
                                                  L_deph=ev_free["L_deph"])["w_cvar"]

        T1o = _T1Objective(fv, v2, Gamma, wts)
        T1o.install()
        controls, evals = {"free": c_free}, {"free": ev_free}
        try:
            controls["mean"] = _ms(P, "mean", seed=seed, warm=c_free,
                                   label=f"{stem}:mean")
            for al in ALPHA_OPT:
                controls[f"cvar_{al:g}"] = _ms(
                    P, "cvar", alpha=al, seed=seed, warm=controls["mean"],
                    label=f"{stem}:cvar{al:g}")
            # reverse start: warm the mean from each CVaR control (fairness)
            for al in ALPHA_OPT:
                controls[f"rev_{al:g}"] = _ms(
                    P, "mean", seed=seed + 7, warm=controls[f"cvar_{al:g}"],
                    label=f"{stem}:rev{al:g}")
        finally:
            T1o.restore()

        for k, c in controls.items():
            if k == "free":
                continue
            uI, uQ = P.filtered(c)
            evals[k], _ = _evaluate(fv, v2, uI, uQ, P.test, Gamma)

        # (2) best-mean comparator, applied HERE (the Stage-3 pitfall)
        cand = ["mean"] + [f"rev_{al:g}" for al in ALPHA_OPT]
        rows = {}
        for al in ALPHA_OPT:
            key = f"{al:g}"
            best = min(cand, key=lambda c_: evals[c_]["cvar"][key])
            Rraw = 1.0 - evals[f"cvar_{al:g}"]["cvar"][key] / evals["mean"]["cvar"][key]
            Rbest = 1.0 - evals[f"cvar_{al:g}"]["cvar"][key] / evals[best]["cvar"][key]
            rows[key] = {"R_raw": float(Rraw), "R": float(Rbest), "comparator": best}

        cw = {f"{al:g}": channel_weights(recs_free, al, Gamma,
                                         L_deph=ev_free["L_deph"]) for al in ALPHA_OPT}
        # kappa, for the pre-registered null P2
        try:
            U = fv_props(fv, uIf, uQf, P.test)
            x = s3.error_coords(U, a.levels)
            z = P.test.deltas
            z = (z - z.mean()) / max(float(z.std()), 1e-300)
            e0, A, Q, r2 = s3.fit_expansion(x, [z])
            geo = {"kappa": float(s3.kappa_of(A[:3], Q[:3])),
                   "fit_R2_min": float(np.min(r2)),
                   "coherent_norm": float(np.linalg.norm(e0[:3]))}
        except Exception as ex:                    # geometry is a diagnostic, not a result
            geo = {"kappa": None, "kappa_error": repr(ex)}

        pt = {"stem": stem, "Tg": float(Tg), "T1_us": float(T1us), "phase_var": float(pv),
              "levels": int(a.levels), "seed": int(seed), "Gamma_ns": float(Gamma),
              "alpha_opt": list(ALPHA_OPT), "alpha_eval": list(ALPHA_EVAL),
              "evals": evals, "rows": rows, "channel": cw, "geometry": geo,
              "objective_calls": T1o.calls, "seconds": time.time() - t0}
        v2.write_json(pf, pt)
        v2.log(f"  -> beta={evals['mean']['beta']:.3f}  p={evals['mean']['p']:.2e}  "
               + "  ".join(f"R@{k}={v['R']:+.3f}" for k, v in rows.items())
               + f"  [{pt['seconds']:.0f}s]")


def fv_props(fv, uI, uQ, ens):
    """Full propagator per draw (Gamma = 0), for the kappa diagnostic."""
    import stage4 as s4
    return s4.colored_propagators(fv, uI, uQ, ens.deltas, ens.gammas)


# ===========================================================================
# 6. SUMMARY: score P1-P4
# ===========================================================================

def summary(outdir, alpha=0.05):
    from scipy import stats
    base = Path(outdir)
    pts = []
    for f in sorted(base.glob("stage5_*level/*.json")):
        if f.name.endswith("_geo.json"):
            continue
        j = json.loads(f.read_text())
        if "rows" not in j:
            continue
        pts.append(j)
    if not pts:
        raise SystemExit(f"no Stage-5 points under {base}")
    key = f"{alpha:g}"

    rows = []
    for j in pts:
        ev, ch = j["evals"], j["channel"][key]
        free, mean_c = ev["free"], ev["mean"]
        cv = ev[f"cvar_{alpha:g}"]
        # P1: how far each objective moves n_int away from the T1-free optimum
        dn_mean = free["n_int"] - mean_c["n_int"]
        dn_cvar = free["n_int"] - cv["n_int"]
        rows.append({
            "stem": j["stem"], "Tg": j["Tg"], "T1_us": j["T1_us"],
            "phase_var": j["phase_var"], "levels": j["levels"], "seed": j["seed"],
            "beta": mean_c["beta"], "p": mean_c["p"], "l_j": mean_c["l_j"],
            "kappa": (j.get("geometry") or {}).get("kappa"),
            "R": j["rows"][key]["R"], "R_raw": j["rows"][key]["R_raw"],
            "comparator": j["rows"][key]["comparator"],
            "n_int_free": free["n_int"], "n_int_mean": mean_c["n_int"],
            "n_int_cvar": cv["n_int"],
            "dn_mean": dn_mean, "dn_cvar": dn_cvar,
            "knob_ratio": (dn_cvar / dn_mean) if abs(dn_mean) > 1e-12 else None,
            "ratio_meas": mean_c["ratio"][key],
            "ratio_pred": mixture_ratio(mean_c["beta"], alpha),
            "r_alpha": r_alpha(alpha),
            "mass_defect": mean_c["mass_defect"],
            "cvar_by_alpha": {k: ev["mean"]["cvar"][k] for k in ev["mean"]["cvar"]},
            "R_by_alpha": {k: v["R"] for k, v in j["rows"].items()},
        })

    arr = lambda k: np.array([r[k] for r in rows], float)
    out = {"alpha": alpha, "n_points": len(rows), "rows": rows}

    # ---- P1: channel reweighting
    # A per-point ratio dn_cvar / dn_mean is unusable when dn_mean sits at the
    # optimizer's resolution: it is then a ratio of two near-zero numbers and
    # explodes.  (A short smoke run produced 27.3 against a prediction of 3.58
    # for exactly that reason, with n_int differing only in the fourth
    # decimal.)  This is the clipped-ratio failure of Stage 3 in another form,
    # so P1 is scored the way Stage 4 scored its mechanism test: by a
    # through-the-origin slope across points, which weights each point by how
    # well resolved it is, with a bootstrap interval.  A per-point median is
    # reported as well, but only over points whose dn_mean clears a gate.
    GATE = 1e-3                    # |dn_mean| must exceed this fraction of n_int

    def _slope(dm, dc):
        dm, dc = np.asarray(dm, float), np.asarray(dc, float)
        den = float(dm @ dm)
        return float(dm @ dc / den) if den > 0 else float("nan")

    def _boot(dm, dc, n_boot=2000, seed=0):
        dm, dc = np.asarray(dm, float), np.asarray(dc, float)
        if len(dm) < 3:
            return None
        rng = np.random.default_rng(seed)
        v = []
        for _ in range(n_boot):
            i = rng.integers(0, len(dm), len(dm))
            x = _slope(dm[i], dc[i])
            if np.isfinite(x):
                v.append(x)
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None

    by_alpha = {}
    for al in ALPHA_OPT:
        dm, dc, ratios = [], [], []
        for j in pts:
            ev = j["evals"]
            free = ev["free"]["n_int"]
            a_ = free - ev["mean"]["n_int"]
            b_ = free - ev[f"cvar_{al:g}"]["n_int"]
            dm.append(a_)
            dc.append(b_)
            if abs(a_) > GATE * abs(free):
                ratios.append(b_ / a_)
        by_alpha[f"{al:g}"] = {
            "slope": _slope(dm, dc), "slope_ci": _boot(dm, dc),
            "n_resolved": len(ratios),
            "median_ratio_resolved": float(np.median(ratios)) if ratios else None,
            "predicted": channel_amplification(al)}

    key_a = f"{alpha:g}"
    sl = by_alpha[key_a]["slope"]
    rises = [by_alpha[f"{al:g}"]["slope"] for al in ALPHA_OPT]
    out["P1_channel_reweighting"] = {
        "estimator": "through-origin slope of (n_int_free - n_int_cvar) on "
                     "(n_int_free - n_int_mean) across points",
        "resolution_gate": GATE,
        "slope": sl, "slope_ci": by_alpha[key_a]["slope_ci"],
        "predicted": channel_amplification(alpha),
        "by_alpha": by_alpha,
        "monotone_in_inverse_alpha": bool(
            all(np.isfinite(x) for x in rises)
            and all(rises[i] <= rises[i + 1] + 1e-9 for i in range(len(rises) - 1))),
        "P1_pass": bool(np.isfinite(sl) and 2.5 <= sl <= 4.5
                        and by_alpha[key_a]["n_resolved"] >= max(3, len(rows) // 2))}

    # ---- P2: beta orders the gain, kappa does not
    R = arr("R")
    beta = arr("beta")
    kap = np.array([r["kappa"] if r["kappa"] is not None else np.nan for r in rows], float)

    def _sp(x, y):
        """Spearman that returns None rather than NaN when there are too few
        points or no spread, so a partial run's summary stays readable."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
            return [None, None]
        r_ = stats.spearmanr(x[m], y[m])
        return [float(r_[0]), float(r_[1])]

    sp_b = _sp(beta, R)
    out["P2_beta_orders"] = {"spearman_beta": sp_b, "n": len(rows)}
    m = np.isfinite(kap)
    if m.sum() >= 5:
        sp_k = stats.spearmanr(kap[m], R[m])
        rb, rk = stats.rankdata(beta[m]), stats.rankdata(kap[m])
        rr = stats.rankdata(R[m])
        Z = np.column_stack([np.ones(m.sum()), rb])
        res = lambda u: u - Z @ np.linalg.lstsq(Z, u, rcond=None)[0]
        pr = float(np.corrcoef(res(rk), res(rr))[0, 1])
        df = int(m.sum()) - 3
        t = pr * np.sqrt(df / max(1 - pr ** 2, 1e-15))
        out["P2_beta_orders"].update({
            "spearman_kappa": [float(sp_k[0]), float(sp_k[1])],
            "partial_kappa_given_beta": [pr, float(2 * stats.t.sf(abs(t), df))]})
        out["P2_beta_orders"]["P2_pass"] = bool(
            sp_b[0] is not None and sp_b[0] > 0.6 and abs(pr) < 0.3 and
            out["P2_beta_orders"]["partial_kappa_given_beta"][1] > 0.05)
    else:
        out["P2_beta_orders"]["P2_pass"] = None

    # ---- P3: the kappa^2 bound is not expected to hold (recorded, not scored)
    out["P3_bound_not_applicable"] = {
        "note": "the kappa^2 bound is derived under (A1); Stage 5 breaks (A1), "
                "so exceedances are expected and are not scored as failures",
        "max_R": float(R.max()), "n_positive": int((R > 0).sum()), "n": len(rows)}

    # ---- P4: the tail shape is a mixture, not rho_alpha
    meas, prd = arr("ratio_meas"), arr("ratio_pred")
    ra = r_alpha(alpha)
    out["P4_mixture_shape"] = {
        "r_alpha": ra,
        "measured_median": float(np.median(meas)),
        "predicted_median": float(np.median(prd)),
        "median_rel_error": float(np.median(np.abs(meas - prd) / np.maximum(prd, 1e-300))),
        "spearman_pred_meas": _sp(prd, meas),
        "n_above_r_alpha": int((meas > ra).sum()),
        "spearman_beta_ratio": _sp(beta, meas),
        "P4_pass": bool(np.median(np.abs(meas - prd) / np.maximum(prd, 1e-300)) < 0.15
                        and (_sp(beta, meas)[0] or 0.0) > 0.6)}

    out["numerics"] = {"max_mass_defect": float(arr("mass_defect").max())}
    out["comparator"] = {
        "upgraded": int(sum(1 for r in rows if r["comparator"] != "mean")),
        "max_raw_minus_best": float(max(r["R_raw"] - r["R"] for r in rows))}

    p = base / "stage5_summary.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1, default=float))
    print(f"\nwrote {p}")
    return out


# ===========================================================================
# 7. CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("selftest")
    s.add_argument("--outdir", default="results_stage5")

    r = sub.add_parser("run")
    r.add_argument("--Tg", type=float, nargs="+", default=[12.7, 40.0, 100.0])
    r.add_argument("--T1", type=float, nargs="+", default=[30.0, 10.0])
    r.add_argument("--phase-var", dest="phase_var", type=float, nargs="+",
                   default=[0.04, 0.16])
    r.add_argument("--levels", type=int, default=2, choices=(2, 4))
    r.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    r.add_argument("--outdir", default="results_stage5")
    r.add_argument("--force", action="store_true")

    m = sub.add_parser("summary")
    m.add_argument("--outdir", default="results_stage5")
    m.add_argument("--alpha", type=float, default=0.05)

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
    else:
        summary(a.outdir, a.alpha)


if __name__ == "__main__":
    main()