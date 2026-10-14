"""
fast_viper.py -- same experiments as stage0_v2.py, roughly 10-50x cheaper.

WHY IT IS FASTER
============================================================================
stage0_v2 spends almost all of its time in one place: for every noise draw,
every input state and every time step it builds a 16x16 Lindblad
superoperator and exponentiates it, one draw at a time, across a process pool.
For the question VIPER asks (does CVaR pick a different control than the
mean?) that is mostly wasted work:

  * The randomness that CVaR ranks is the quasi-static (delta, gamma) draw.
    T1/T2 act the same way on every draw, so they add a near-constant shift to
    every per-draw loss.  CVaR(L + c) = CVaR(L) + c, so a draw-independent
    shift cannot move the mean-vs-CVaR comparison.  `selftest` MEASURES how
    draw-independent it really is at your parameters; it does not assume it.
  * Without dissipation each draw is an N x N unitary (4x4, or 2x2 for the
    ablation) instead of a 16x16 superoperator, and all K draws and all time
    steps are diagonalised in ONE vectorised numpy call.  The exact gradient
    comes from the eigen-decomposition (Daleckii-Krein), no finite differences.

So: optimise on the fast closed-system model, then `validate` re-scores every
saved control in the FULL Lindblad model (evaluation only, no optimisation --
cheap) and reports R_tail both ways.  Only the Lindblad numbers go in the paper.

WHAT IS REUSED UNCHANGED
============================================================================
Everything else is stage0_v2: loss (1 - F_4ket + lambda*leak), tanh-bounded
pulse, train/val/test split, exact-RU CVaR, warm start, mean continuation,
reverse warm start, paired bootstrap, sweep / sigma / pareto steps, figures,
JSON outputs.  This file only swaps the per-draw simulator and adds one thing:

  * CONTINUATION ALONG THE SWEEP.  One mean restart starts from the previous
    sweep point's mean-optimal pulse (time-rescaled, area preserved).  It
    replaces a random restart, so restart counts stay equal.  This tracks one
    basin smoothly across Tg / sigma -- the non-monotone mean loss at
    15.4 ns in the Lindblad sweep is exactly what it is meant to catch.

USAGE (outputs go to results_fast/ unless --outdir is given)
============================================================================
    python fast_viper.py selftest            # match vs Lindblad, gradient check, timing
    python fast_viper.py characterize
    python fast_viper.py audit --Tg-audit 12.7
    python fast_viper.py sweep --Tg 6.337 7.351 8.529 9.894 11.479 12.7 15.449 20.793 32.467 --restarts 6
    python fast_viper.py sigma --levels 2 --Tg-sigma 12.7 --sigma-grid 0.5 1 2.5 5 8 16
    python fast_viper.py sweep --levels 2 --Tg 6.337 12.7 15.449 --outdir results_fast
    python fast_viper.py validate            # re-score everything in full Lindblad

No FADE_WORKERS needed for the optimisation (single process, numpy);
`validate` still uses the Lindblad pool.

NOT executed on Claude's side (per your preference); syntax-checked only.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import fade_realistic as fr
import stage0 as s0
import stage0_v2 as v2
from fade_realistic import cfg, TWO_PI

_lindblad_gather = v2.gather_ordered          # kept for selftest / validate


# ============================================================================
# 1. CLOSED-SYSTEM MODEL (same Hamiltonian, states, target, leak as the worker)
# ============================================================================

_model_cache = {}


def get_model():
    key = (cfg.N_LEVELS, cfg.alpha_MHz, cfg.leak_lambda, cfg.temperature_mK, cfg.qubit_freq_GHz)
    if key in _model_cache:
        return _model_cache[key]
    N = cfg.N_LEVELS
    a = np.diag(np.sqrt(np.arange(1, N)), 1).astype(complex)
    ad = a.conj().T
    n_op = ad @ a
    alpha = cfg.alpha_MHz * 1e-3 * TWO_PI
    H0 = (0.5 * alpha * (ad @ ad @ a @ a)).astype(complex)
    Hx = (a + ad) / 2.0
    Hy = 1j * (ad - a) / 2.0
    P = np.diag([0, 0] + [1] * (N - 2)).astype(complex)

    x = (TWO_PI * cfg.qubit_freq_GHz * 1e9 * 6.62607015e-34 / (2 * np.pi)) / \
        (1.380649e-23 * cfg.temperature_mK * 1e-3)
    n_th = 1.0 / (np.exp(x) - 1.0)
    r = n_th / (1.0 + n_th) if n_th > 0 else 0.0
    pops = np.array([(1 - r) * r ** k for k in range(N)]); pops /= pops.sum()
    rho_th = np.diag(pops).astype(complex)
    leak_bg = float(np.real(np.trace(P @ rho_th)))
    Uemb = np.eye(N, dtype=complex)
    Uemb[0:2, 0:2] = np.array([[1.0, -1j], [-1j, 1.0]]) / np.sqrt(2)
    kets = [np.array([1, 0] + [0] * (N - 2), complex),
            np.array([0, 1] + [0] * (N - 2), complex),
            np.array([1, 1] + [0] * (N - 2), complex) / np.sqrt(2),
            np.array([1, 1j] + [0] * (N - 2), complex) / np.sqrt(2)]
    rhos, Os = [], []
    for psi in kets:
        rin = (1.0 - leak_bg) * np.outer(psi, psi.conj()) + P @ rho_th @ P
        tgt = Uemb @ rin @ Uemb.conj().T
        rhos.append(rin)
        Os.append(tgt.conj().T - cfg.leak_lambda * P)     # J_j = Re Tr(O_j rho_T)
    m = {"N": N, "H0": H0, "n": n_op, "Hx": Hx, "Hy": Hy,
         "rhos": np.array(rhos), "Os": np.array(Os)}
    _model_cache[key] = m
    return m


def _dag(A):
    return np.conj(np.swapaxes(A, -1, -2))


def fast_gather(uI_f, uQ_f, deltas, gammas):
    """Drop-in replacement for stage0_v2.gather_ordered, without T1/T2.
    Returns per-draw J (K,), dJ/duI_f (K,Nt), dJ/duQ_f (K,Nt), draw i at row i."""
    M = get_model()
    N = M["N"]
    deltas = np.asarray(deltas, float)
    gammas = np.asarray(gammas, float)
    K, Nt = len(deltas), len(uI_f)
    dt = cfg.Tg / cfg.Nt
    g1 = (1.0 + gammas)

    H = (M["H0"][None, None] + deltas[:, None, None, None] * M["n"][None, None]
         + g1[:, None, None, None] * (uI_f[None, :, None, None] * M["Hx"]
                                      + uQ_f[None, :, None, None] * M["Hy"]))
    e, V = np.linalg.eigh(H)                                   # (K,Nt,N), (K,Nt,N,N)
    Vh = _dag(V)
    ph = np.exp(-1j * e * dt)
    U = (V * ph[..., None, :]) @ Vh                            # (K,Nt,N,N)

    # forward: density matrices for the 4 input states
    rho = np.broadcast_to(M["rhos"], (K, 4, N, N)).copy()
    rho_before = np.empty((Nt, K, 4, N, N), complex)
    for k in range(Nt):
        rho_before[k] = rho
        Uk = U[:, k][:, None]
        rho = Uk @ rho @ _dag(Uk)
    J = np.real(np.einsum("jab,kjba->k", M["Os"], rho)) / 4.0

    # Daleckii-Krein divided differences for d exp(-i H dt)
    de = e[..., :, None] - e[..., None, :]
    pm, pn = ph[..., :, None], ph[..., None, :]
    safe = np.abs(de) > 1e-10
    G = np.where(safe, (pm - pn) / np.where(safe, de, 1.0), -1j * dt * pm)
    XI = Vh @ M["Hx"] @ V
    XQ = Vh @ M["Hy"] @ V

    # backward
    Lam = np.broadcast_to(M["Os"], (K, 4, N, N)).copy()
    gI = np.zeros((K, Nt))
    gQ = np.zeros((K, Nt))
    for k in range(Nt - 1, -1, -1):
        Uk = U[:, k][:, None]
        Mk = rho_before[k] @ _dag(Uk) @ Lam                    # rho_k U_k^dag Lam_{k+1}
        Y = (Vh[:, k][:, None] @ Mk @ V[:, k][:, None]).sum(axis=1)   # (K,N,N)
        Gk = G[:, k]
        gI[:, k] = 2.0 * np.real(np.einsum("kmn,kmn,knm->k", Gk, XI[:, k], Y))
        gQ[:, k] = 2.0 * np.real(np.einsum("kmn,kmn,knm->k", Gk, XQ[:, k], Y))
        Lam = _dag(Uk) @ Lam @ Uk
    gI *= g1[:, None] / 4.0
    gQ *= g1[:, None] / 4.0
    return J, gI, gQ


# ============================================================================
# 2. PATCH stage0_v2 (every function there looks these names up at call time)
# ============================================================================

v2.gather_ordered = fast_gather

_prev = {}
_orig_compare = v2.compare_mean_vs_cvar
_orig_starts = v2.Problem.starts


def _compare_with_memory(P, alphas, seed_base, label):
    out, controls, test_L = _orig_compare(P, alphas, seed_base, label)
    out["simulator"] = "closed-system (fast_viper); validate re-scores in Lindblad"
    out["sigma_d"], out["sigma_a"] = float(P.c.sigma_d), float(P.c.sigma_a)
    out["rho_da"] = float(getattr(P.c, "rho_da", 0.0))
    uI, uQ = P.param.pulse(controls["mean"])
    _prev["pulse"] = (float(cfg.Tg), uI.copy(), uQ.copy())
    return out, controls, test_L


def _starts_with_continuation(self, n, seed, warm=None):
    out = _orig_starts(self, n, seed, warm)
    if warm is None and "pulse" in _prev and n >= 2:          # mean multistart only
        Tp, uIp, uQp = _prev["pulse"]
        s_old, s_new = np.linspace(0, 1, len(uIp)), np.linspace(0, 1, cfg.Nt)
        sc, A = Tp / cfg.Tg, cfg.max_amp                      # keep rotation area
        uI = np.clip(np.interp(s_new, s_old, uIp) * sc, -0.95 * A, 0.95 * A)
        uQ = np.clip(np.interp(s_new, s_old, uQp) * sc, -0.95 * A, 0.95 * A)
        out[-1] = ("continued_from_neighbor", self.param.coeffs(uI, uQ))
    return out


v2.compare_mean_vs_cvar = _compare_with_memory
v2.Problem.starts = _starts_with_continuation


# ============================================================================
# 3. SELFTEST: exactness vs Lindblad, dissipation as a shift, gradient, speed
# ============================================================================

def selftest(outdir):
    c = v2.V2Config(outdir=outdir)
    cs = v2.V2Config(**{**asdict(c), "K_train": 8, "n_val": 8, "n_test": 8})
    res = {}
    rng = np.random.default_rng(0)
    P = v2.Problem(12.7, cs)
    cc = P.c_seed + 0.3 * rng.standard_normal(len(P.c_seed)) * np.abs(P.c_seed).max()
    uI_f, uQ_f = P.filtered(cc)
    d, g = P.train.deltas, P.train.gammas

    # (a) identical physics when dissipation is switched off in the Lindblad worker
    Jf, gIf, gQf = fast_gather(uI_f, uQ_f, d, g)
    phys = (cfg.alpha_MHz, cfg.N_LEVELS, cfg.Tg, cfg.Nt, 1e15, 1e15,
            cfg.temperature_mK, cfg.qubit_freq_GHz, cfg.leak_lambda)
    Jl, gIl, gQl = s0._stage0_chunk_worker((uI_f, uQ_f, d, g, phys))
    res["J_maxabs_diff_vs_lindblad_nodiss"] = float(np.max(np.abs(Jf - Jl)))
    res["grad_maxrel_diff_vs_lindblad_nodiss"] = float(
        max(np.max(np.abs(gIf - gIl)), np.max(np.abs(gQf - gQl))) /
        max(np.max(np.abs(gIl)), np.max(np.abs(gQl))))

    # (b) how draw-independent is the T1/T2 contribution?  (the surrogate's premise)
    ens = v2.Ensemble(64, c.sigma_d, c.sigma_a, 999, "diss")
    Jr = _lindblad_gather(uI_f, uQ_f, ens.deltas, ens.gammas)[0]
    Jc = fast_gather(uI_f, uQ_f, ens.deltas, ens.gammas)[0]
    shift = Jc - Jr                                          # loss added by T1/T2
    res["dissipation_shift_mean"] = float(shift.mean())
    res["dissipation_shift_std"] = float(shift.std())
    res["loss_std_across_draws"] = float((1 - Jc).std())
    res["shift_std_over_loss_std"] = float(shift.std() / max((1 - Jc).std(), 1e-15))

    # (c) finite-difference check of the full coefficient-space objective, saturated
    c_sat = 4.0 * cc
    fd = {}
    for mode, a in (("mean", 0.05), ("cvar", 0.2), ("mixed", 0.2)):
        v0, g0, _ = P.objective(c_sat, mode, a, 0.5)
        dv = rng.standard_normal(len(c_sat))
        dv *= 1e-6 * np.linalg.norm(c_sat) / np.linalg.norm(dv)
        vp = P.objective(c_sat + dv, mode, a, 0.5)[0]
        vm = P.objective(c_sat - dv, mode, a, 0.5)[0]
        num, ana = (vp - vm) / 2, float(g0 @ dv)
        fd[mode] = {"fd": num, "analytic": ana, "rel_err": abs(num - ana) / max(abs(num), 1e-15)}
    res["fd"] = fd

    # (d) speed at the real ensemble size
    e128 = v2.Ensemble(128, c.sigma_d, c.sigma_a, 5, "t")
    t0 = time.monotonic()
    for _ in range(5):
        fast_gather(uI_f, uQ_f, e128.deltas, e128.gammas)
    tf = (time.monotonic() - t0) / 5
    t0 = time.monotonic()
    _lindblad_gather(uI_f, uQ_f, e128.deltas, e128.gammas)
    tl = time.monotonic() - t0
    res["seconds_per_eval_fast"], res["seconds_per_eval_lindblad"] = tf, tl
    res["speedup"] = tl / tf

    res["PASS"] = bool(res["J_maxabs_diff_vs_lindblad_nodiss"] < 1e-7
                       and res["grad_maxrel_diff_vs_lindblad_nodiss"] < 1e-6
                       and all(v["rel_err"] < 1e-4 for v in fd.values()))
    res["read_me"] = ("shift_std_over_loss_std << 1 means T1/T2 is a near-constant "
                      "offset and cannot change the mean-vs-CVaR ranking; if it is "
                      "not small, trust only the `validate` numbers")
    out = Path(outdir) / "selftest.json"
    v2.write_json(out, res)
    v2.log(json.dumps(v2._jsonable(res), indent=1))
    v2.log(f"FAST SELFTEST {'PASS' if res['PASS'] else 'FAIL'} -> {out}")
    return res["PASS"]


# ============================================================================
# 4. VALIDATE: re-score every saved control in the full Lindblad model
# ============================================================================

def _lindblad_J(uI, uQ, deltas, gammas):
    """Full-Lindblad per-draw J.  The worker pool was spawned with the default
    4-level transmon baked into each worker's fade_realistic.sys_, so for 2-
    or 3-level points run the SAME worker serially in this process, where
    stage0_v2.set_gate has already rebuilt sys_ for the right level count."""
    if cfg.N_LEVELS == 4:
        return _lindblad_gather(uI, uQ, deltas, gammas)[0]
    phys = (cfg.alpha_MHz, cfg.N_LEVELS, cfg.Tg, cfg.Nt, cfg.T1, cfg.T2s,
            cfg.temperature_mK, cfg.qubit_freq_GHz, cfg.leak_lambda)
    return s0._stage0_chunk_worker((uI, uQ, np.asarray(deltas), np.asarray(gammas), phys))[0]


def validate(outdir):
    base = Path(outdir)
    c = v2.V2Config(outdir=outdir)
    rows = []
    for sub in sorted(p for p in base.iterdir() if p.is_dir() and p.name != "characterize"):
        for jf in sorted(sub.glob("*.json")):
            if jf.name in ("summary.json", "pareto.json", "validate.json"):
                continue
            pt = json.loads(jf.read_text())
            npz = jf.with_suffix(".npz")
            if not npz.exists():
                npz = sub / "audit_arrays.npz"
            if "cells" not in pt or not npz.exists() or "sigma_d" not in pt:
                continue
            arr = np.load(npz)
            c.n_levels = int(pt["grid"].get("n_levels", 4))
            v2.set_gate(pt["grid"]["Tg"], c)
            te = next(e for e in pt["ensembles"] if e["name"] == "test")
            ens = v2.Ensemble(te["n"], pt["sigma_d"], pt["sigma_a"], te["seed"], "test",
                              pt.get("rho_da", 0.0))
            bwf = fr.BandwidthFilter(cfg.Nt, cfg.dt, cfg.awg_bandwidth_GHz, cfg.awg_filter_order)
            Lfull = {}
            for key in arr.files:
                if key.startswith("uIraw_"):
                    k = key[6:]
                    uI, uQ = bwf.apply(arr[f"uIraw_{k}"]), bwf.apply(arr[f"uQraw_{k}"])
                    Lfull[k] = 1.0 - _lindblad_J(uI, uQ, ens.deltas, ens.gammas)
            # Best mean-objective control found ANYWHERE at this point, chosen on the
            # VALIDATION ensemble only: the multistart winner or any reverse-warm-start
            # result (those are mean-objective optimisations too).  Comparing CVaR
            # against this removes the "mean optimizer missed a basin" confound.
            cands = []
            for c2 in pt["cells"]:
                if "reverse_val_mean" in c2 and f"rev_{c2['alpha']}" in Lfull:
                    base_val = c2["reverse_val_mean"] - c2["reverse_vs_mean_ctrl"]
                    cands.append(("mean", base_val))
                    cands.append((f"rev_{c2['alpha']}", c2["reverse_val_mean"]))
            best_key = min(cands, key=lambda t: t[1])[0] if cands else "mean"
            for cell in pt["cells"]:
                a = cell["alpha"]
                bs = v2.paired_bootstrap(Lfull["mean"], Lfull[f"cvar_{a}"], a, 2000, 7)
                bsb = v2.paired_bootstrap(Lfull[best_key], Lfull[f"cvar_{a}"], a, 2000, 7)
                row = {"file": f"{sub.name}/{jf.name}", "Tg": pt["grid"]["Tg"],
                       "n_levels": c.n_levels, "sigma_d_MHz": pt["sigma_d"] / TWO_PI * 1e3,
                       "sigma_a": pt["sigma_a"], "rho_da": pt.get("rho_da", 0.0),
                       "alpha": a, "R_tail_fast": cell["R_tail"],
                       "R_tail_lindblad": bs["R_tail"], "R_lo_lindblad": bs["R_lo"],
                       "R_hi_lindblad": bs["R_hi"], "D_lindblad": bs["D"],
                       "verdict_lindblad": v2.audit_verdict(bs),
                       "mean_loss_mean_ctrl_lindblad": float(Lfull["mean"].mean()),
                       "mean_loss_cvar_ctrl_lindblad": float(Lfull[f"cvar_{a}"].mean()),
                       "best_mean_source": best_key,
                       "R_tail_vs_best_mean_lindblad": bsb["R_tail"],
                       "R_lo_vs_best_mean": bsb["R_lo"], "R_hi_vs_best_mean": bsb["R_hi"],
                       "D_vs_best_mean": bsb["D"], "D_lo_vs_best_mean": bsb["D_lo"],
                       "D_hi_vs_best_mean": bsb["D_hi"],
                       "verdict_vs_best_mean": v2.audit_verdict(bsb)}
                rel = cell.get("reverse_rel_to_mean_ctrl")
                if rel is None and "reverse_vs_mean_ctrl" in cell:      # runs made before the fix
                    base_val = cell["reverse_val_mean"] - cell["reverse_vs_mean_ctrl"]
                    rel = cell["reverse_vs_mean_ctrl"] / base_val
                row["reverse_rel_to_mean_ctrl_fast"] = rel
                row["reverse_class_fast"] = (None if rel is None else
                                             "artifact_mean_missed_basin" if rel < -1e-3 else
                                             "same_basin_returns_to_mean" if rel <= 1e-3 else
                                             "different_basins_mean_prefers_own")
                row["reverse_du_to_mean_fast"] = cell.get("reverse_du_to_mean")
                rows.append(row)
                v2.log(f"{row['file']:28s} a={a:<5} R_tail fast={cell['R_tail']:+.3f}  "
                       f"Lindblad={bs['R_tail']:+.3f} [{bs['R_lo']:+.3f},{bs['R_hi']:+.3f}]  "
                       f"| vs best mean ({best_key}) {bsb['R_tail']:+.3f} "
                       f"[{bsb['R_lo']:+.3f},{bsb['R_hi']:+.3f}]  "
                       f"rev={row['reverse_class_fast']}")
    v2.write_json(base / "validate.json", {"rows": rows})
    v2.write_csv(base / "validate.csv", rows)
    v2.log(f"VALIDATE done -> {base/'validate.json'}")
    return rows


# ============================================================================
# 5. CLI
# ============================================================================

def main():
    argv = sys.argv[1:]
    if "--outdir" not in argv:
        argv += ["--outdir", "results_fast"]
    outdir = argv[argv.index("--outdir") + 1]
    Path(outdir).mkdir(parents=True, exist_ok=True)
    step = argv[0] if argv else ""
    try:
        if step == "selftest":
            sys.exit(0 if selftest(outdir) else 1)
        if step == "validate":
            validate(outdir)
            return
    finally:
        if step in ("selftest", "validate"):
            fr._shutdown_pool()
    v2.main(argv)


if __name__ == "__main__":
    main()