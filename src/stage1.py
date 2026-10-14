"""
stage1.py -- VIPER Stage 1: several Gaussian noise channels acting at once.

THE QUESTION
============================================================================
Plan, Stage 1: "leakage suppressed; detuning + amplitude noise jointly.  Run
the linearity diagnostic first, then R_tail."  The plan's Corollary predicts
R_tail ~ 0 because a jointly Gaussian, linearly coupled noise vector gives a
Gaussian phase.  That holds when every channel drives the SAME error axis.
Detuning (sigma_z) and amplitude error (sigma_x, along the drive) generally do
not: to leading order the per-draw error is a rotation VECTOR  delta*b + gamma*a
with a, b not parallel, so

        L  ~  L0 + x^T M x ,   x = (delta/sd, gamma/sa) jointly Gaussian
           =  L0 + lam1 z1^2 + lam2 z2^2    (z independent, lam = eig(C^1/2 M C^1/2))

a generalised chi-square with TWO weights.  Mean = lam1 + lam2; CVaR_alpha is
Schur-convex in (lam1, lam2) (Marshall-Olkin-Arnold, Ch. 11, applied to (y - tau)_+ for
every tau, then min over tau).  At equal mean, CVaR prefers EQUAL weights:
alpha = 0.05 CVaR is 5.58 for one axis vs 4.00 for two equal axes (-28%).
So two non-parallel channels need not be ordered by stochastic dominance, and
the theory makes three testable predictions:

  P1  one axis (each channel alone, quadratic regime):  R_tail ~ 0
  P2  two independent axes:  R_tail can be > 0; where it is, the CVaR control
      has comparable or slightly larger total error (lam1 + lam2) but a MORE
      ISOTROPIC split, A = (lam1 - lam2)/(lam1 + lam2) smaller than the mean
      control's.  The claim is about GEOMETRY, not about a lower-error pulse.
  P3  rank-one noise geometry restores equivalence.  The standardised noise
      covariance [[1, rho], [rho, 1]] has eigenvalues 1 + rho, 1 - rho, so
      |rho| -> 1 is rank -> 1.  Expect R_tail -> 0 in that LIMIT, not a
      monotone fall with |rho|: at intermediate rho the loss weights can get
      more unequal.  The analysis x-axis is the weight anisotropy A, not rho.

STEPS
============================================================================
  diagnose   Geometry of the error, for every saved control (mean, each CVaR,
             each reverse start); no optimisation, closed-system model:
             (a) ERROR-VECTOR CLOUD.  Per draw, the rotation vector eps in
                 U_target^dag U = exp(-i eps.sigma/2) (qubit block; nearest
                 unitary if there is leakage).  Reports the 3x3 covariance
                 Sigma_eps: eigenvalues e1 >= e2 >= e3, tr, A_eps =
                 (e1 - e2)/(e1 + e2), and e3/tr -- with two linear channels
                 Sigma_eps has rank <= 2, so e3/tr > 0 measures NONLINEAR
                 error (mechanism B), separating it from the two-axis
                 mechanism A.  Coherent share |mean eps|^2 / E|eps|^2 too.
                 The clouds are saved (stage1_error_clouds.npz) for figures.
             (b) LOSS-METRIC WEIGHTS.  The loss is not |eps|^2: the 4-ket
                 average weights rotation axes unequally and leakage adds its
                 own term.  A quadratic fit of the per-draw loss in
                 (delta, gamma) gives the weights CVaR actually sees:
                 w1 >= w2, tr = w1 + w2, A_w = (w1 - w2)/(w1 + w2), fit R^2.
             (c) noise-scaling exponent p: mean excess loss ~ s^p when the
                 noise is scaled by s (p = 2: quadratic regime; p = 4:
                 first-order-robust control, where (b) is not the whole story)
             (d) R_iso: the largest tail reduction mechanism A alone could
                 give at fixed mean (equalise w1, w2), as a fraction of the
                 control's CVaR.  An upper bound for P2.

  channels   New optimisations (fast model; `fast_viper.py validate` re-scores
             them in full Lindblad afterwards).  Grid = product of
             --sigma-a-grid and --rho-grid at one Tg and one sigma_d.

USAGE (from the iris folder, after downloading the updated stage0_v2.py and
fast_viper.py -- they add the rho option; every earlier result is unchanged)
============================================================================
  python stage1.py diagnose results_fast results_2lvl_long results_strong results_strong_2lvl
  python stage1.py channels --levels 2 --Tg 12.7 --sigma-d-mhz 2.5 \
        --sigma-a-grid 0.01 0.02 0.04 0.08 0.127 --rho-grid 0
  python stage1.py channels --levels 2 --Tg 12.7 --sigma-d-mhz 0 --sigma-a-grid 0.127
  python stage1.py channels --levels 2 --Tg 12.7 --sigma-d-mhz 2.5 \
        --sigma-a-grid 0.127 --rho-grid -0.99 -0.9 -0.5 0.5 0.9 0.99
  (rho = 0 at sigma_a = 0.127 is already in the first channels run)
  python fast_viper.py validate --outdir results_stage1
  python stage1.py diagnose results_stage1

NOT executed on Claude's side (per your preference); the fitting / bound
maths is unit-tested on synthetic data, the rest is syntax-checked.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import fast_viper as fv                 # importing patches stage0_v2 with the fast simulator
import stage0_v2 as v2
import fade_realistic as fr
from fade_realistic import cfg, TWO_PI

import inspect as _inspect
if ("rho" not in _inspect.signature(v2.Ensemble.__init__).parameters
        or "rho_da" not in _inspect.getsource(fv)):
    raise SystemExit("stage1.py needs the UPDATED stage0_v2.py and fast_viper.py (they add the "
                     "rho option).  Download both again, replace the old copies, and re-run.")

SCALES = (0.5, 1.0, 2.0)                # noise-scaling test
N_MC = 400_000                          # Monte Carlo draws for the R_iso bound


# ============================================================================
# 1. PURE MATHS (unit-tested; no simulator)
# ============================================================================

def cvar(L, alpha):
    return float(np.asarray(L, float) @ v2.tail_weights(L, alpha))


def fit_quadratic(xs, L):
    """Least-squares fit L ~ c0 + g.x + x^T M x over the ACTIVE channels.
    xs: list of standardised draw arrays (1 or 2).  Returns c0, g, M, R^2."""
    L = np.asarray(L, float)
    cols, n = [np.ones_like(L)], len(xs)
    cols += list(xs)
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    cols += [xs[i] * xs[j] for i, j in pairs]
    X = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(X, L, rcond=None)
    pred = X @ coef
    ss = float(((L - L.mean()) ** 2).sum())
    r2 = 1.0 - float(((L - pred) ** 2).sum()) / ss if ss > 0 else float("nan")
    M = np.zeros((n, n))
    for (i, j), q in zip(pairs, coef[1 + n:]):
        if i == j:
            M[i, i] = q
        else:
            M[i, j] = M[j, i] = q / 2.0
    return float(coef[0]), coef[1:1 + n], M, r2


def weights(M, rho):
    """Eigenvalues (descending) of C^1/2 M C^1/2, C = corr matrix of x."""
    if M.shape == (1, 1):
        return np.array([M[0, 0]])
    C = np.array([[1.0, rho], [rho, 1.0]])
    w, V = np.linalg.eigh(C)
    Ch = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
    return np.sort(np.linalg.eigvalsh(Ch @ M @ Ch))[::-1]


def iso_bound(lam, cvar_total, alpha, seed=0):
    """Tail reduction available by equalising the weights at fixed mean,
    as a fraction of the control's actual CVaR (same denominator as R_tail).
    0 for a single weight / rank-1 form -- the theorem's case."""
    lam = np.clip(np.asarray(lam, float), 0.0, None)
    if len(lam) < 2 or lam.sum() <= 0 or cvar_total <= 0:
        return 0.0
    z2 = np.random.default_rng(seed).standard_normal((N_MC, 2)) ** 2
    q = z2 @ lam
    q_iso = z2 @ np.full(2, lam.sum() / 2)
    return float((cvar(q, alpha) - cvar(q_iso, alpha)) / cvar_total)


PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], complex)
U_TARGET = np.array([[1.0, -1j], [-1j, 1.0]]) / np.sqrt(2)          # X(pi/2), as in the model


def rotation_vectors(Uq, Ut=U_TARGET):
    """(K,2,2) qubit-block propagators -> (K,3) error rotation vectors eps with
    Ut^dag U = exp(-i eps.sigma/2) up to global phase.  Sub-unitary blocks
    (leakage) are replaced by their nearest unitary (polar factor)."""
    E = np.conj(Ut.T)[None] @ np.asarray(Uq, complex)
    u, _, vh = np.linalg.svd(E)
    E = u @ vh
    E = E / np.sqrt(np.linalg.det(E))[:, None, None]
    c = np.real(np.trace(E, axis1=1, axis2=2)) / 2
    v = -np.imag(np.einsum("kab,iba->ki", E, PAULI)) / 2               # sin(th/2) n
    flip = c < 0                                                        # E and -E: pick th < pi
    c, v = np.where(flip, -c, c), np.where(flip[:, None], -v, v)
    nv = np.linalg.norm(v, axis=1)
    th = 2 * np.arctan2(nv, c)
    scale = np.where(nv > 1e-15, th / np.where(nv > 1e-15, nv, 1.0), 2.0)
    return v * scale[:, None]


def geometry(eps):
    """Covariance geometry of an error cloud (K,3)."""
    S2 = np.einsum("ki,kj->ij", eps, eps) / len(eps)                   # second moment
    Sig = np.cov(eps.T, bias=True)
    e = np.sort(np.linalg.eigvalsh(Sig))[::-1]
    tr = float(e.sum())
    return {"eps_e1": float(e[0]), "eps_e2": float(e[1]), "eps_e3": float(e[2]),
            "eps_tr": tr, "eps_A": float((e[0] - e[1]) / (e[0] + e[1])) if e[0] + e[1] > 0 else 0.0,
            "eps_nonlin_share": float(e[2] / tr) if tr > 0 else 0.0,
            "eps_coherent_share": float(np.dot(eps.mean(0), eps.mean(0)) / np.trace(S2))
            if np.trace(S2) > 0 else 0.0}


def _unit_tests():
    rng = np.random.default_rng(1)
    d, g = rng.standard_normal(20000), rng.standard_normal(20000)
    L = 0.3 + 0.01 * d + 2.0 * d ** 2 + 0.5 * d * g + 1.0 * g ** 2
    c0, gl, M, r2 = fit_quadratic([d, g], L)
    assert abs(c0 - 0.3) < 1e-9 and abs(M[0, 1] - 0.25) < 1e-9 and r2 > 0.999999
    assert np.allclose(weights(np.diag([2.0, 1.0]), 0.0), [2.0, 1.0])
    assert np.allclose(weights(np.ones((2, 2)), 1.0), [4.0, 0.0], atol=1e-9)   # rho=1: rank 1
    one = iso_bound([1.0, 0.0], 5.58, 0.05)
    assert 0.25 < one < 0.31, one                    # 5.58 -> 4.00 at alpha = 0.05
    assert iso_bound([1.0, 1.0], 4.0, 0.05) == 0.0 or abs(iso_bound([1.0, 1.0], 4.0, 0.05)) < 1e-12
    # rotation vectors: exp(-i eps.sigma/2) with a random global phase and target
    eps = 0.3 * rng.standard_normal((50, 3))
    th = np.linalg.norm(eps, axis=1)
    n = eps / th[:, None]
    E = (np.cos(th / 2)[:, None, None] * np.eye(2)
         - 1j * np.sin(th / 2)[:, None, None] * np.einsum("ki,iab->kab", n, PAULI))
    U = U_TARGET[None] @ E * np.exp(1j * rng.uniform(0, 6.28, 50))[:, None, None]
    assert np.allclose(rotation_vectors(U), eps, atol=1e-10)
    # two linear channels on non-parallel axes: rank-2 cloud, e3 ~ 0
    d, g = rng.standard_normal(5000), rng.standard_normal(5000)
    G = geometry(np.outer(d, [0.0, 0.3, 1.0]) * 1e-2 + np.outer(g, [1.0, 0.0, 0.0]) * 1e-2)
    assert G["eps_nonlin_share"] < 1e-12 and G["eps_tr"] > 0
    return True


# ============================================================================
# 2. DIAGNOSE saved controls
# ============================================================================

def _points(folder):
    for sub in sorted(p for p in Path(folder).iterdir() if p.is_dir() and p.name != "characterize"):
        for jf in sorted(sub.glob("*.json")):
            if jf.name in ("summary.json", "pareto.json", "validate.json"):
                continue
            pt = json.loads(jf.read_text())
            npz = jf.with_suffix(".npz")
            if not npz.exists():
                npz = sub / "audit_arrays.npz"
            if "cells" in pt and "sigma_d" in pt and npz.exists():
                yield sub, jf, pt, npz


def _loss(uI, uQ, deltas, gammas):
    return 1.0 - fv.fast_gather(uI, uQ, np.asarray(deltas, float), np.asarray(gammas, float))[0]


def qubit_propagators(uI, uQ, deltas, gammas):
    """Closed-system total propagator per draw, qubit block (K,2,2).  Same
    Hamiltonian and time stepping as fast_viper.fast_gather."""
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
    return Ut[:, :2, :2]


def diagnose(folders, outdir):
    assert _unit_tests()
    rows, clouds = [], {}
    for folder in folders:
        for sub, jf, pt, npz in _points(folder):
            c = v2.V2Config()
            c.n_levels = int(pt["grid"].get("n_levels", 4))
            v2.set_gate(pt["grid"]["Tg"], c)
            sd, sa, rho = pt["sigma_d"], pt["sigma_a"], pt.get("rho_da", 0.0)
            te = next(e for e in pt["ensembles"] if e["name"] == "test")
            ens = v2.Ensemble(te["n"], sd, sa, te["seed"], "test", rho)
            xs = ([ens.deltas / sd] if sd > 0 else []) + ([ens.gammas / sa] if sa > 0 else [])
            arr = np.load(npz)
            bwf = fr.BandwidthFilter(cfg.Nt, cfg.dt, cfg.awg_bandwidth_GHz, cfg.awg_filter_order)
            keys = (["mean"] + [f"cvar_{x['alpha']}" for x in pt["cells"]]
                    + sorted(f[6:] for f in arr.files if f.startswith("uIraw_rev_")))
            cells = {x["alpha"]: x for x in pt["cells"]}
            per = {}
            for k in keys:
                if f"uIraw_{k}" not in arr.files:
                    continue
                uI, uQ = bwf.apply(arr[f"uIraw_{k}"]), bwf.apply(arr[f"uQraw_{k}"])
                L = _loss(uI, uQ, ens.deltas, ens.gammas)
                L0 = float(_loss(uI, uQ, [0.0], [0.0])[0])
                ex = [float(_loss(uI, uQ, s * ens.deltas, s * ens.gammas).mean()) - L0 for s in SCALES]
                p = (float(np.polyfit(np.log(SCALES), np.log(np.clip(ex, 1e-300, None)), 1)[0])
                     if min(ex) > 0 else float("nan"))
                c0, gl, M, r2 = fit_quadratic(xs, L)
                lam = weights(M, rho)
                a_ref = 0.05 if not k.startswith("cvar_") else float(k.split("_")[1])
                eps = rotation_vectors(qubit_propagators(uI, uQ, ens.deltas, ens.gammas))
                geo = geometry(eps)
                clouds[f"{Path(folder).name}__{sub.name}__{jf.stem}__{k}"] = eps.astype(np.float32)
                w1, w2 = float(lam[0]), (float(lam[1]) if len(lam) > 1 else 0.0)
                row = {"folder": Path(folder).name, "file": f"{sub.name}/{jf.name}",
                       "Tg": pt["grid"]["Tg"], "n_levels": c.n_levels,
                       "sigma_d_MHz": sd / TWO_PI * 1e3, "sigma_a": sa, "rho_da": rho,
                       "sigma_phi_sq": (sd * pt["grid"]["Tg"]) ** 2,
                       "amp_phase_sq": (sa * np.pi / 2) ** 2,
                       "control": k, "mean_L": float(L.mean()), "L_nominal": L0,
                       "cvar05": cvar(L, 0.05), "cvar20": cvar(L, 0.2),
                       "quad_R2": r2, "scaling_exp": p,
                       "w1": w1, "w2": w2, "w_tr": w1 + w2,
                       "w_A": (w1 - w2) / (w1 + w2) if len(lam) > 1 and w1 + w2 > 0 else 1.0,
                       "w_ratio": w1 / w2 if w2 > 0 else float("inf"),
                       "linear_norm": float(np.linalg.norm(gl)),
                       "R_iso_bound": iso_bound(lam, cvar(L, a_ref), a_ref),
                       "R_tail_fast": cells[a_ref]["R_tail"] if k.startswith("cvar_") else None,
                       **geo}
                per[k] = row
                rows.append(row)
                v2.log(f"{row['folder']}/{row['file']:30s} {k:9s} Tg={row['Tg']:5.1f} L={c.n_levels} "
                       f"sa={sa:.3f} rho={rho:+.2f} | E[L]={row['mean_L']:.2e} "
                       f"CVaR05={row['cvar05']:.2e} | w_tr={w1 + w2:.2e} A_w={row['w_A']:.2f} "
                       f"R2={r2:.3f} p={p:4.2f} | eps: tr={geo['eps_tr']:.2e} A={geo['eps_A']:.2f} "
                       f"nonlin={geo['eps_nonlin_share']:.3f} | R_iso<={row['R_iso_bound']:.3f}"
                       + (f"  R_tail={row['R_tail_fast']:+.3f}" if row["R_tail_fast"] is not None else ""))
            # P2 check, per CVaR control vs the mean control at the same point
            if "mean" in per and len(xs) == 2:
                m = per["mean"]
                for k, r in per.items():
                    if k.startswith("cvar_"):
                        r["dA_w_vs_mean"] = r["w_A"] - m["w_A"]            # < 0: more isotropic
                        r["dtr_w_rel_vs_mean"] = (r["w_tr"] - m["w_tr"]) / m["w_tr"] if m["w_tr"] else None
                        r["dA_eps_vs_mean"] = r["eps_A"] - m["eps_A"]
                        r["more_isotropic_than_mean"] = bool(r["w_A"] < m["w_A"])
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    v2.write_json(out / "stage1_diagnose.json", {"rows": rows})
    v2.write_csv(out / "stage1_diagnose.csv", rows)
    np.savez_compressed(out / "stage1_error_clouds.npz", **clouds)
    v2.log(f"DIAGNOSE done: {len(rows)} controls -> {out/'stage1_diagnose.json'}")


# ============================================================================
# 3. CHANNELS: new optimisations
# ============================================================================

def channels(a):
    base = v2.V2Config()
    base.n_levels = a.levels
    if a.restarts:
        base.n_restarts = a.restarts
    if a.maxiter:
        base.maxiter = a.maxiter
    if a.seed:
        base.seed_train += 1000 * a.seed
        base.seed_val += 1000 * a.seed
        base.seed_test += 1000 * a.seed
    sd = float(TWO_PI * a.sigma_d_mhz * 1e-3)
    d = Path(a.outdir) / f"channels_{a.levels}level"
    d.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(a.sigma_a_grid, a.rho_grid))
    for i, (sa, rho) in enumerate(grid):
        if sd == 0 and rho != 0:
            continue
        stem = f"ch_Tg{a.Tg:g}_sd{a.sigma_d_mhz:g}_sa{sa:g}_rho{rho:+g}_s{a.seed}"
        pf = d / f"{stem}.json"
        if pf.exists():
            v2.log(f"  {stem} already done, skipping")
            continue
        cc = v2.V2Config(**asdict(base))
        cc.sigma_d, cc.sigma_a, cc.rho_da = sd, float(sa), float(rho)
        v2.log(f"=== channels point {i+1}/{len(grid)}: Tg={a.Tg} ns, sigma_d={a.sigma_d_mhz} MHz, "
               f"sigma_a={sa:g}, rho={rho:+g}, levels={a.levels} ===")
        P = v2.Problem(a.Tg, cc)
        pt, controls, test_L = v2.compare_mean_vs_cvar(P, cc.alpha_sweep,
                                                       11000 + 100 * i + 10000 * a.seed, stem)
        pt.update(sigma_d_MHz=float(a.sigma_d_mhz), sigma_phi_sq=float((sd * a.Tg) ** 2),
                  amp_phase_sq=float((sa * np.pi / 2) ** 2), stage="1-channels")
        v2.write_json(pf, pt)
        v2.save_arrays(d / f"{stem}.npz", P, controls, test_L)
        for x in pt["cells"]:
            v2.log(f"  -> alpha={x['alpha']}: R_tail(fast)={x['R_tail']:+.3f} "
                   f"[{x['R_lo']:+.3f},{x['R_hi']:+.3f}]")


# ============================================================================
# 4. CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("STEPS")[0])
    sp = ap.add_subparsers(dest="step", required=True)
    dg = sp.add_parser("diagnose")
    dg.add_argument("folders", nargs="+")
    dg.add_argument("--outdir", default=None,
                    help="default: the first folder given, so runs on different inputs never overwrite each other")
    ch = sp.add_parser("channels")
    ch.add_argument("--levels", type=int, choices=[2, 3, 4], default=2)
    ch.add_argument("--Tg", type=float, default=12.7)
    ch.add_argument("--sigma-d-mhz", type=float, default=2.5)
    ch.add_argument("--sigma-a-grid", type=float, nargs="+", required=True)
    ch.add_argument("--rho-grid", type=float, nargs="+", default=[0.0])
    ch.add_argument("--restarts", type=int)
    ch.add_argument("--maxiter", type=int)
    ch.add_argument("--seed", type=int, default=0)
    ch.add_argument("--outdir", default="results_stage1")
    a = ap.parse_args()
    try:
        if a.step == "diagnose":
            diagnose(a.folders, a.outdir or a.folders[0])
        else:
            channels(a)
    finally:
        fr._shutdown_pool()


if __name__ == "__main__":
    main()