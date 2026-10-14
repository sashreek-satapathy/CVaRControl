"""
stage4.py -- VIPER Stage 4: colored (Ornstein-Uhlenbeck) noise.

THE QUESTION
============================================================================
Every result so far assumes QUASI-STATIC noise: one scalar (delta, gamma) per
shot, frozen for the duration of the gate.  Assumption (A1) of Theorem 1 is
weaker than that -- it asks only that the noise be Gaussian -- and the
corollary on temporal correlation says so explicitly: for any stationary
Gaussian process, the accumulated phase

        phi[u] = int_0^Tg f_u(t) eta(t) dt

is a linear functional of a Gaussian process, hence Gaussian, so Theorem 1
applies verbatim and memory changes only the SIZE of sigma_phi^2, not the
equivalence.  That corollary has never been run.  Stage 4 runs it.

For an Ornstein-Uhlenbeck process with stationary width sigma_ou and
correlation time tau_c, the quasi-static limit is tau_c >> Tg and the white
limit is tau_c << Tg.  The accumulated phase variance over a gate is

    Var(phi) = 2 sigma_ou^2 tau_c [ Tg - tau_c (1 - e^{-Tg/tau_c}) ] ,

so at each tau_c we can solve for the sigma_ou that holds sigma_phi^2 fixed.
The design sweeps tau_c at fixed sigma_phi^2: if the corollary is right, the
correlation time then has NO effect on the mean-vs-CVaR comparison.

PRE-REGISTERED PREDICTIONS (scored by `summary`)
  T1  EQUIVALENCE IS BLIND TO MEMORY.  At fixed sigma_phi^2, R_tail does not
      depend on tau_c (partial rank correlation with log tau_c, |rho| < 0.3
      and p > 0.05), and at sigma_phi^2 = 0.04 it stays within the resolution
      of the study (|R_tail| <~ 0.04) at every tau_c.
  T2  THE SHAPE IS BLIND TO MEMORY.  CVaR_0.05/mean of the mean control stays
      at the chi^2_1 value 5.58 of Proposition 1 across tau_c, to within a few
      percent, wherever the losses are in the quadratic regime.
  T3  THE MECHANISM TRANSFERS.  kappa and the reduced model predict R_tail
      across the whole grid as well as they do for quasi-static noise
      (Spearman > 0.6), and R_tail <= R_model at every point.
  T4  ONE SCALAR IS ENOUGH -- OR NOT.  The theorem's premise is that the loss
      depends on the noise only through the scalar phi.  With colored noise
      the control's own modulation f_u(t) weights the trajectory, so the raw
      accumulated phase need not be the whole story.  We therefore fit the
      error in TWO noise coordinates: z1, the normalized accumulated phase,
      and z2, the normalized first moment of the trajectory about the centre
      of the gate (orthogonalized against z1).  The share of the linear
      response that lies along z2 measures how much the time structure
      matters.  Prediction: that share grows as tau_c falls below Tg, while
      R_tail stays flat -- memory changes the error GEOMETRY without changing
      the mean-CVaR comparison.

WHAT IS AND IS NOT VALIDATED IN LINDBLAD
============================================================================
The Lindblad worker takes one scalar detuning per draw, so it cannot score a
trajectory.  Stage 4 is therefore closed-system only, and its points are
written so that `fast_viper.py validate` and `stage1.py diagnose` skip them.
To avoid the trap Part C of Stage 3 fell into, the best-mean comparator of the
fairness protocol (multistart, continuation or reverse start, chosen on
validation data) is applied HERE, at save time: every point records both
R_tail against the multistart mean control and against the best mean control.
The second is the number to report.

USAGE (from the iris folder)
============================================================================
  python stage4.py selftest                       # pure maths, seconds
  bash run_stage4.sh 2>&1 | tee stage4_log.txt    # the whole grid, ~2 h
or step by step:
  python stage4.py run --tau 1.27 3.81 12.7 38.1 127 --quasi-static \
         --phase-var 0.04 0.16 0.41 --seeds 0 1
  python stage4.py summary

WHAT TO SEND BACK
  results_stage4/stage4_summary.json, results_stage4/*/*_geo.json,
  stage4_log.txt

NOT executed on Claude's side (per your preference): the maths selftest was
run there; the simulator paths are syntax-checked only.
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

QUASI = float("inf")            # tau_c label for the quasi-static limit


# ===========================================================================
# 1. PURE MATHS: the OU process, its accumulated phase, and the calibration
# ===========================================================================

def ou_corr(tau_c, dt, n):
    """Correlation matrix of the sampled stationary OU process, rho^|i-j|."""
    if not np.isfinite(tau_c):
        return np.ones((n, n))
    r = math.exp(-dt / tau_c)
    idx = np.arange(n)
    return r ** np.abs(idx[:, None] - idx[None, :])


def phase_variance_factor(tau_c, dt, n):
    """Var(phi) / sigma_ou^2 for phi = dt * sum_j eta_j, the piecewise-constant
    accumulated phase of the sampled process.  Exact, no Monte Carlo."""
    C = ou_corr(tau_c, dt, n)
    return float(dt ** 2 * C.sum())


def sigma_for_phase(sigma_phi_sq, tau_c, dt, n):
    """Stationary OU width that gives the requested accumulated phase variance."""
    f = phase_variance_factor(tau_c, dt, n)
    return float(math.sqrt(sigma_phi_sq / f)) if f > 0 else 0.0


def ou_draws(n_draws, tau_c, dt, n, sigma_ou, seed):
    """Exact stationary sampling of the OU process (AR(1) in discrete time).
    Returns (n_draws, n).  tau_c = inf gives the quasi-static case: one
    constant per draw, which reproduces the earlier stages bit for bit when
    the same seed is used."""
    rng = np.random.default_rng(seed)
    if not np.isfinite(tau_c):
        return np.repeat(rng.normal(0.0, sigma_ou, n_draws)[:, None], n, axis=1)
    r = math.exp(-dt / tau_c)
    s = math.sqrt(max(1.0 - r * r, 0.0))
    x = np.empty((n_draws, n))
    x[:, 0] = rng.normal(0.0, sigma_ou, n_draws)
    for j in range(1, n):
        x[:, j] = r * x[:, j - 1] + sigma_ou * s * rng.normal(0.0, 1.0, n_draws)
    return x


def noise_coordinates(traj, dt):
    """Two scalars per draw: z1, the accumulated phase, and z2, the first
    moment of the trajectory about the centre of the gate, orthogonalized
    against z1 and standardized.  z1 is the theorem's scalar; the share of the
    linear response along z2 measures how much the time structure matters."""
    n = traj.shape[1]
    t = (np.arange(n) + 0.5) * dt
    w = t - t.mean()
    z1 = traj.sum(1) * dt
    z2 = traj @ (w * dt)
    z1 = z1 - z1.mean()
    z2 = z2 - z2.mean()
    z1 = z1 / max(z1.std(), 1e-300)
    z2 = z2 - np.dot(z2, z1) / max(np.dot(z1, z1), 1e-300) * z1
    z2 = z2 / max(z2.std(), 1e-300)
    return z1, z2


def cvar(L, alpha):
    L = np.asarray(L, float)
    n = len(L)
    m = max(int(round(alpha * n)), 1)
    return float(np.partition(L, n - m)[n - m:].mean())


def selftest(outdir="results_stage4"):
    res, ok = {}, True
    dt, n = 0.249, 51
    # (a) the exact phase-variance factor matches Monte Carlo
    for tau in (1.0, 5.0, 50.0, QUASI):
        f = phase_variance_factor(tau, dt, n)
        x = ou_draws(200_000, tau, dt, n, 1.0, 3)
        mc = float((x.sum(1) * dt).var())
        res[f"phase_var_tau{tau}"] = [f, mc, abs(mc / f - 1)]
        ok &= abs(mc / f - 1) < 0.02
    # (b) the continuous-time formula is recovered for tau >> dt
    tau, Tg = 12.7, dt * n
    cont = 2 * tau * (Tg - tau * (1 - math.exp(-Tg / tau)))
    res["discrete_vs_continuous"] = [phase_variance_factor(tau, dt, n), cont]
    ok &= abs(phase_variance_factor(tau, dt, n) / cont - 1) < 0.05
    # (c) the calibration hits the requested accumulated phase variance
    for tau in (1.0, 12.7, 127.0, QUASI):
        s = sigma_for_phase(0.16, tau, dt, n)
        x = ou_draws(100_000, tau, dt, n, s, 5)
        got = float((x.sum(1) * dt).var())
        res[f"calib_tau{tau}"] = [s, got]
        ok &= abs(got / 0.16 - 1) < 0.03
    # (d) quasi-static draws are constant in time; short tau are not
    x = ou_draws(50, QUASI, dt, n, 1.0, 1)
    res["quasi_static_is_constant"] = float(np.abs(x - x[:, :1]).max())
    ok &= res["quasi_static_is_constant"] == 0.0
    x = ou_draws(5000, 0.5, dt, n, 1.0, 1)
    res["short_tau_autocorr_one_step"] = float(np.corrcoef(x[:, 10], x[:, 11])[0, 1])
    ok &= abs(res["short_tau_autocorr_one_step"] - math.exp(-dt / 0.5)) < 0.05
    # (e) the two noise coordinates are standardized and uncorrelated
    z1, z2 = noise_coordinates(ou_draws(20000, 3.0, dt, n, 1.0, 7), dt)
    res["coords"] = [float(z1.std()), float(z2.std()), float(abs(np.corrcoef(z1, z2)[0, 1]))]
    ok &= abs(z1.std() - 1) < 1e-9 and abs(z2.std() - 1) < 1e-9 and abs(np.corrcoef(z1, z2)[0, 1]) < 1e-9
    # (f) in the quasi-static limit z2 is degenerate (no time structure to see)
    zq1, zq2 = noise_coordinates(ou_draws(2000, QUASI, dt, n, 1.0, 9), dt)
    res["quasi_static_z2_is_noise"] = float(abs(np.corrcoef(zq1, zq2)[0, 1]))
    res["PASS"] = bool(ok)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    (Path(outdir) / "stage4_selftest.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))
    print(f"STAGE 4 SELFTEST {'PASS' if ok else 'FAIL'}")
    return ok


# ===========================================================================
# 2. THE COLORED SIMULATOR
# ===========================================================================
# stage0_v2.gather_ordered and fast_viper.fast_gather take ONE detuning per
# draw.  The only change a trajectory needs is that delta carries a time index
# as well, which broadcasts into the same Hamiltonian construction; the
# Daleckii-Krein gradient with respect to the control is untouched, because
# the noise does not multiply the control.  Everything else -- loss, basis,
# filter, CVaR weights, fairness protocol -- is the existing code path.

def _dag(A):
    return np.conj(np.swapaxes(A, -1, -2))


def colored_gather(fv, uI_f, uQ_f, deltas, gammas):
    """Per-draw (J, dJ/duI_f, dJ/duQ_f) with deltas of shape (K,) or (K, Nt)."""
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
    d = np.broadcast_to(deltas, (K, Nt))[..., None, None]          # (K,Nt,1,1)
    H = (M["H0"][None, None] + d * M["n"][None, None]
         + g1[:, None, None, None] * (uI_f[None, :, None, None] * M["Hx"]
                                      + uQ_f[None, :, None, None] * M["Hy"]))
    e, V = np.linalg.eigh(H)
    Vh = _dag(V)
    ph = np.exp(-1j * e * dt)
    U = (V * ph[..., None, :]) @ Vh
    rho = np.broadcast_to(M["rhos"], (K, 4, N, N)).copy()
    rho_before = np.empty((Nt, K, 4, N, N), complex)
    for k in range(Nt):
        rho_before[k] = rho
        Uk = U[:, k][:, None]
        rho = Uk @ rho @ _dag(Uk)
    J = np.real(np.einsum("jab,kjba->k", M["Os"], rho)) / 4.0
    de = e[..., :, None] - e[..., None, :]
    pm, pn = ph[..., :, None], ph[..., None, :]
    safe = np.abs(de) > 1e-10
    G = np.where(safe, (pm - pn) / np.where(safe, de, 1.0), -1j * dt * pm)
    XI = Vh @ M["Hx"] @ V
    XQ = Vh @ M["Hy"] @ V
    Lam = np.broadcast_to(M["Os"], (K, 4, N, N)).copy()
    gI = np.zeros((K, Nt))
    gQ = np.zeros((K, Nt))
    for k in range(Nt - 1, -1, -1):
        Uk = U[:, k][:, None]
        Mk = rho_before[k] @ _dag(Uk) @ Lam
        Y = (Vh[:, k][:, None] @ Mk @ V[:, k][:, None]).sum(axis=1)
        Gk = G[:, k]
        gI[:, k] = 2.0 * np.real(np.einsum("kmn,kmn,knm->k", Gk, XI[:, k], Y))
        gQ[:, k] = 2.0 * np.real(np.einsum("kmn,kmn,knm->k", Gk, XQ[:, k], Y))
        Lam = _dag(Uk) @ Lam @ Uk
    gI *= g1[:, None] / 4.0
    gQ *= g1[:, None] / 4.0
    return J, gI, gQ


def colored_propagators(fv, uI, uQ, deltas, gammas):
    """Full N-level propagator per draw, trajectory detuning."""
    from fade_realistic import cfg
    M = fv.get_model()
    deltas = np.asarray(deltas, float)
    if deltas.ndim == 1:
        deltas = deltas[:, None]
    gammas = np.asarray(gammas, float)
    K, Nt = len(gammas), len(uI)
    dt, g1 = cfg.Tg / cfg.Nt, 1.0 + gammas
    d = np.broadcast_to(deltas, (K, Nt))[..., None, None]
    H = (M["H0"][None, None] + d * M["n"][None, None]
         + g1[:, None, None, None] * (uI[None, :, None, None] * M["Hx"]
                                      + uQ[None, :, None, None] * M["Hy"]))
    e, V = np.linalg.eigh(H)
    U = (V * np.exp(-1j * e * dt)[..., None, :]) @ _dag(V)
    Ut = np.broadcast_to(np.eye(M["N"], dtype=complex), (K, M["N"], M["N"])).copy()
    for k in range(U.shape[1]):
        Ut = U[:, k] @ Ut
    return Ut


class ColoredEnsemble:
    """Drop-in replacement for stage0_v2.Ensemble whose `deltas` is a
    trajectory per draw.  sigma_d is read as the STATIONARY OU width; the
    correlation time is module state, set per point by `run`."""
    tau_c = QUASI

    def __init__(self, n, sigma_d, sigma_a, seed, name, rho=0.0):
        from fade_realistic import cfg
        dt = cfg.Tg / cfg.Nt
        self.deltas = ou_draws(n, ColoredEnsemble.tau_c, dt, cfg.Nt, float(sigma_d), seed)
        self.gammas = np.random.default_rng(seed + 1).normal(0.0, float(sigma_a), n)
        self.n, self.seed, self.name, self.rho = n, seed, name, float(rho)
        self.tau_c = ColoredEnsemble.tau_c
        self.phi = self.deltas.sum(1) * dt          # accumulated phase per draw

    def describe(self):
        return {"name": self.name, "n": self.n, "seed": self.seed, "rho": self.rho,
                "tau_c": self.tau_c, "sigma_phi_sq_realized": float(self.phi.var())}


# ===========================================================================
# 3. RUN: the tau_c x sigma_phi^2 grid
# ===========================================================================

def _sim():
    import inspect as _inspect
    import fast_viper as fv
    import stage0_v2 as v2
    import fade_realistic as fr
    if "rho" not in _inspect.signature(v2.Ensemble.__init__).parameters:
        raise SystemExit("stage4.py needs the Stage-1 versions of stage0_v2.py / fast_viper.py.")
    return fv, v2, fr


def _install(fv, v2):
    """Route every loss and gradient through the colored simulator."""
    v2.gather_ordered = lambda uI, uQ, d, g: colored_gather(fv, uI, uQ, d, g)
    v2.Ensemble = ColoredEnsemble
    fv._prev.clear()


def _fit_geometry(fv, v2, uI, uQ, ens, n_levels, with_model=True, alpha=0.05):
    """kappa, the closed-form coefficient and the reduced model, in the two
    noise coordinates (accumulated phase and first moment)."""
    import stage3 as s3
    from fade_realistic import cfg
    dt = cfg.Tg / cfg.Nt
    z1, z2 = noise_coordinates(ens.deltas, dt)
    U = colored_propagators(fv, uI, uQ, ens.deltas, ens.gammas)
    x = s3.error_coords(U, n_levels)
    e0, A, Q, r2 = s3.fit_expansion(x, [z1, z2])
    W = s3.metric_W(n_levels)
    a1, a2 = np.linalg.norm(A[:3, 0]), np.linalg.norm(A[:3, 1])
    out = {"kappa": s3.kappa_of(A[:3], Q[:3]),
           "kappa_lk": s3.kappa_of(A, Q, W),
           "C_closed": s3.C_closed(A[:3], Q[:3], s3.W_EPS, alpha),
           "fit_R2_min": float(np.min(r2)),
           "linear_share_z2": float(a2 ** 2 / max(a1 ** 2 + a2 ** 2, 1e-300)),
           "coherent_norm": float(np.linalg.norm(e0[:3]))}
    out["R_kappa2"] = out["C_closed"] * out["kappa"] ** 2
    if with_model:
        out["R_model_q"] = s3.r_model(e0, A[:3], Q[:3], s3.W_EPS, alpha)
    return out


def run(a):
    fv, v2, fr = _sim()
    from fade_realistic import cfg, TWO_PI
    import stage3 as s3
    _install(fv, v2)
    base = v2.V2Config()
    base.n_levels = a.levels
    base.n_restarts = a.restarts
    base.maxiter = a.maxiter
    base.sigma_a = 0.0                     # detuning only: the setting of Theorem 1
    d = Path(a.outdir) / f"stage4_{a.levels}level"
    d.mkdir(parents=True, exist_ok=True)
    taus = list(a.tau) + ([QUASI] if a.quasi_static else [])
    grid = [(t, pv, s) for t in taus for pv in a.phase_var for s in a.seeds]
    for i, (tau, pv, seed) in enumerate(grid):
        tag = "qs" if not np.isfinite(tau) else f"{tau:g}"
        stem = f"ou_tau{tag}_pv{pv:g}_s{seed}".replace(".", "p")
        pf = d / f"{stem}.json"
        if pf.exists():
            v2.log(f"  {stem} already done, skipping")
            continue
        cc = v2.V2Config(**asdict(base))
        if seed:
            cc.seed_train += 1000 * seed
            cc.seed_val += 1000 * seed
            cc.seed_test += 1000 * seed
        # set the gate first, so dt and Nt are known, then calibrate sigma_ou
        v2.set_gate(a.Tg, cc)
        dt = cfg.Tg / cfg.Nt
        ColoredEnsemble.tau_c = tau
        cc.sigma_d = sigma_for_phase(pv, tau, dt, cfg.Nt)
        v2.log(f"=== Stage 4 point {i+1}/{len(grid)}: tau_c={tag} ns "
               f"(tau_c/Tg={tau/a.Tg if np.isfinite(tau) else float('inf'):.3g}), "
               f"sigma_phi^2={pv:g}, sigma_ou={cc.sigma_d/TWO_PI*1e3:.3g} MHz, seed={seed} ===")
        t0 = time.monotonic()
        P = v2.Problem(a.Tg, cc)
        pt, controls, test_L = v2.compare_mean_vs_cvar(P, cc.alpha_sweep, 50000 + 100 * i, stem)
        # ---- the fairness comparator, applied here (no Lindblad step for trajectories)
        cands = []
        for c in pt["cells"]:
            if "reverse_val_mean" in c and f"rev_{c['alpha']}" in test_L:
                cands.append(("mean", c["reverse_val_mean"] - c["reverse_vs_mean_ctrl"]))
                cands.append((f"rev_{c['alpha']}", c["reverse_val_mean"]))
        best = min(cands, key=lambda t: t[1])[0] if cands else "mean"
        R_best = {}
        for c in pt["cells"]:
            al = c["alpha"]
            R_best[str(al)] = float(1 - cvar(test_L[f"cvar_{al}"], al) / cvar(test_L[best], al))
        pt.update(stage="4-colored", tau_c=(None if not np.isfinite(tau) else float(tau)),
                  quasi_static=bool(not np.isfinite(tau)), sigma_phi_sq_target=float(pv),
                  sigma_phi_sq_realized=float(P.test.phi.var()),
                  sigma_ou_MHz=float(cc.sigma_d / TWO_PI * 1e3), seed=int(seed),
                  comparator=best, R_tail_best_mean=R_best,
                  closed_system_only=True,
                  validate_skipped_reason=("trajectory noise: the Lindblad worker takes one "
                                           "scalar detuning per draw, so only the closed-system "
                                           "numbers here are meaningful"))
        pt["sigma_d_ou"] = pt.pop("sigma_d", None)      # hide from validate / diagnose
        pt["sigma_a_value"] = pt.pop("sigma_a", None)
        v2.write_json(pf, pt)
        v2.save_arrays(d / f"{stem}.npz", P, controls, test_L)
        geo = {}
        for k, coef in controls.items():
            uI, uQ = P.filtered(coef)
            geo[k] = _fit_geometry(fv, v2, uI, uQ, P.test, a.levels, with_model=(k == "mean"))
            L = v2.losses(uI, uQ, P.test)
            geo[k]["mean_loss"], geo[k]["cvar05"] = float(L.mean()), cvar(L, 0.05)
            geo[k]["ratio05"] = geo[k]["cvar05"] / max(geo[k]["mean_loss"], 1e-300)
            geo[k]["L_nominal"] = float(v2.losses(uI, uQ, _nominal_ens(cfg))[0])
        v2.write_json(d / f"{stem}_geo.json",
                      {"stem": stem, "cell": {"tau_c": (None if not np.isfinite(tau) else float(tau)),
                                              "quasi_static": bool(not np.isfinite(tau)),
                                              "tau_over_Tg": (None if not np.isfinite(tau) else float(tau / a.Tg)),
                                              "phase_var": float(pv), "seed": int(seed),
                                              "Tg": float(a.Tg), "levels": int(a.levels),
                                              "sigma_ou_MHz": float(cc.sigma_d / TWO_PI * 1e3)},
                       "geometry": geo, "comparator": best,
                       "R_tail_raw": {str(x["alpha"]): x["R_tail"] for x in pt["cells"]},
                       "R_tail_best_mean": R_best, "seconds": time.monotonic() - t0})
        m = geo["mean"]
        v2.log(f"  -> R_tail(0.05) raw={pt['cells'][0]['R_tail']:+.3f}, "
               f"vs best mean ({best})={R_best['0.05']:+.3f} | kappa={m['kappa']:.3f}, "
               f"C={m['C_closed']:.2f}, R_model={m.get('R_model_q', float('nan')):.3f}, "
               f"CVaR/mean={m['ratio05']:.2f}, z2 share of |A|^2={m['linear_share_z2']:.3f}, "
               f"E[L]={m['mean_loss']:.2e}, {time.monotonic()-t0:.0f}s")


def _nominal_ens(cfg):
    """A one-draw noiseless ensemble, for the coherent (noise-free) loss."""
    e = ColoredEnsemble.__new__(ColoredEnsemble)
    e.deltas = np.zeros((1, cfg.Nt))
    e.gammas = np.zeros(1)
    e.n, e.seed, e.name, e.rho, e.tau_c = 1, 0, "nominal", 0.0, ColoredEnsemble.tau_c
    e.phi = np.zeros(1)
    return e


# ===========================================================================
# 4. SUMMARY: score T1-T4
# ===========================================================================

def summary(outdir, alpha=0.05):
    from scipy.stats import spearmanr, rankdata, pearsonr
    out = Path(outdir)
    rows = []
    for gf in sorted(out.glob("stage4_*level/*_geo.json")):
        g = json.loads(gf.read_text())
        c, m = g["cell"], g["geometry"]["mean"]
        rows.append({"stem": g["stem"], **c, "kappa": m["kappa"], "C": m["C_closed"],
                     "R_kappa2": m["R_kappa2"], "R_model": m.get("R_model_q"),
                     "ratio05": m["ratio05"], "mean_loss": m["mean_loss"],
                     "L_nominal": m.get("L_nominal"), "fit_R2_min": m["fit_R2_min"],
                     "z2_share": m["linear_share_z2"], "comparator": g["comparator"],
                     "R_raw": g["R_tail_raw"].get(str(alpha)),
                     "R": g["R_tail_best_mean"].get(str(alpha))})
    S = {"alpha": alpha, "n_points": len(rows), "rows": rows}
    if not rows:
        (out / "stage4_summary.json").write_text(json.dumps(S, indent=1))
        print("no points yet"); return S

    def partial(a, b, c):
        ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
        res = lambda u: u - np.polyval(np.polyfit(rc, u, 1), rc)
        return pearsonr(res(ra), res(rb))

    lt = np.array([np.log(r["tau_over_Tg"]) if r["tau_over_Tg"] else np.log(1e3) for r in rows])
    R = np.array([r["R"] for r in rows], float)
    pv = np.array([r["phase_var"] for r in rows], float)
    # T1: memory should not matter at fixed phase variance
    T1 = {"partial_logtau_given_phasevar": list(map(float, partial(lt, R, pv))),
          "by_phase_var": {}}
    for p in sorted(set(pv)):
        sub = [r for r in rows if r["phase_var"] == p]
        rs = spearmanr([np.log(r["tau_over_Tg"]) if r["tau_over_Tg"] else np.log(1e3) for r in sub],
                       [r["R"] for r in sub])
        T1["by_phase_var"][str(p)] = {
            "n": len(sub), "R_median": float(np.median([r["R"] for r in sub])),
            "R_min": float(np.min([r["R"] for r in sub])), "R_max": float(np.max([r["R"] for r in sub])),
            "spearman_logtau": [float(rs.statistic), float(rs.pvalue)],
            "quasi_static_R": [r["R"] for r in sub if r["quasi_static"]]}
    small = [r for r in rows if r["phase_var"] <= 0.05]
    if small:
        T1["equivalence_region_max_abs_R"] = float(max(abs(r["R"]) for r in small))
        T1["T1_pass"] = bool(abs(T1["partial_logtau_given_phasevar"][0]) < 0.3
                             and T1["partial_logtau_given_phasevar"][1] > 0.05
                             and T1["equivalence_region_max_abs_R"] < 0.05)
    S["T1_memory_independence"] = T1
    # T2: the chi^2_1 shape ratio
    quad = [r for r in rows if r["phase_var"] <= 0.05]
    S["T2_shape"] = {"pred": 5.58,
                     "ratio_by_tau": {str(r["tau_over_Tg"]): round(r["ratio05"], 3) for r in quad},
                     "median": float(np.median([r["ratio05"] for r in quad])) if quad else None,
                     "T2_pass": bool(quad and max(abs(r["ratio05"] / 5.58 - 1) for r in quad) < 0.1)}
    # T3: the mechanism still predicts
    mod = [r for r in rows if r["R_model"] is not None]
    rs_k = spearmanr([r["kappa"] for r in mod], [r["R"] for r in mod])
    rs_m = spearmanr([r["R_model"] for r in mod], [r["R"] for r in mod])
    S["T3_mechanism"] = {"spearman_kappa": [float(rs_k.statistic), float(rs_k.pvalue)],
                         "spearman_model": [float(rs_m.statistic), float(rs_m.pvalue)],
                         "n_above_envelope": int(sum(r["R"] > r["R_model"] for r in mod)),
                         "n": len(mod),
                         "T3_pass": bool(rs_m.statistic > 0.6 and sum(r["R"] > r["R_model"] for r in mod) <= 1)}
    # T4: how much of the linear response is in the time structure
    S["T4_time_structure"] = {
        "z2_share_by_tau": {str(r["tau_over_Tg"]): round(r["z2_share"], 4) for r in rows},
        "spearman_logtau_vs_z2share": list(map(float, spearmanr(
            lt, [r["z2_share"] for r in rows]))),
        "fit_R2_min": float(min(r["fit_R2_min"] for r in rows))}
    # the comparator correction, for the record
    S["comparator"] = {"upgraded": int(sum(r["comparator"] != "mean" for r in rows)),
                       "max_raw_minus_best": float(max(r["R_raw"] - r["R"] for r in rows))}
    (out / "stage4_summary.json").write_text(json.dumps(S, indent=1, default=float))
    print(json.dumps({k: v for k, v in S.items() if k != "rows"}, indent=1, default=float))
    print(f"SUMMARY -> {out/'stage4_summary.json'}")
    return S


# ===========================================================================
# 5. CLI
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("USAGE")[0])
    sp = ap.add_subparsers(dest="step", required=True)
    st = sp.add_parser("selftest"); st.add_argument("--outdir", default="results_stage4")
    rn = sp.add_parser("run", help="the tau_c x sigma_phi^2 grid")
    rn.add_argument("--tau", type=float, nargs="+", default=[1.27, 3.81, 12.7, 38.1, 127.0],
                    help="OU correlation times in ns")
    rn.add_argument("--quasi-static", action="store_true", default=False,
                    help="also run the tau_c -> infinity limit, which reproduces Stage 0")
    rn.add_argument("--phase-var", type=float, nargs="+", default=[0.04, 0.16, 0.41])
    rn.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    rn.add_argument("--Tg", type=float, default=12.7)
    rn.add_argument("--levels", type=int, choices=[2, 4], default=2)
    rn.add_argument("--restarts", type=int, default=4)
    rn.add_argument("--maxiter", type=int, default=500)
    rn.add_argument("--outdir", default="results_stage4")
    sm = sp.add_parser("summary")
    sm.add_argument("--outdir", default="results_stage4")
    sm.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args()
    try:
        if a.step == "selftest":
            raise SystemExit(0 if selftest(a.outdir) else 1)
        if a.step == "run":
            run(a)
        elif a.step == "summary":
            summary(a.outdir, a.alpha)
    finally:
        if a.step == "run":
            try:
                import fade_realistic as fr
                fr._shutdown_pool()
            except Exception:
                pass


if __name__ == "__main__":
    main()