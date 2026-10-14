"""
stage0_leakage_divergence.py -- Stage 0 of the tail-risk quantum control
research program: "Quasi-static leakage divergence (highest priority)"
(see tail_risk_quantum_control_paper.tex, Section 5.4).

WHAT THIS TESTS
============================================================================
Theorem 1 of the paper proves mean-fidelity and CVaR-optimal control
coincide under single-channel, linearly-coupled Gaussian dephasing.
Leakage into non-computational transmon levels is a NONLINEAR function of
the calibration parameters (detuning, amplitude), which is exactly the
mechanism the theorem's Assumption (A2) rules out. Stage 0 tests this
directly and cheaply: reuse the EXISTING, already-validated quasi-static
(delta, gamma) ensemble and Lindblad propagator from fade_realistic.py
completely unmodified, and replace only the training-time REDUCTION over
the K noise draws -- mean (FADE-RIPE) vs. worst-alpha-tail (VIPER) vs. a
mixed objective -- while sweeping the control-amplitude BUDGET (cfg.max_amp)
toward wherever leakage actually turns on. No new physics simulation is
introduced; per Section 3.5 of the paper (ensemble-averaged channel vs.
per-shot distribution), the K per-draw fidelities fade_realistic.py already
computes are exactly the per-shot object CVaR needs.

PREDICTION BEING TESTED (paper Table 1, Stage 0 row):
    R_tail(alpha) ~= 0 far from the operational leakage-onset amplitude,
    rising as it is approached -- where the onset amplitude is an
    EMPIRICALLY DERIVED quantity (see "THE SWEEP VARIABLE" below), not an
    assumption about where in the swept amplitude range it must fall. A
    null result (no rise) is evidence AGAINST the proposed leakage-
    nonlinearity mechanism under this model, not a mathematical
    falsification of it -- see run_stage0_sweep's "INTERPRETATION"
    docstring section for exactly how to read a positive or a null result.

THE SWEEP VARIABLE: max_amp is not itself "distance to the leakage onset"
-- read this before interpreting any plot
============================================================================
Sweeping cfg.max_amp does two things at once: (1) it changes the physical
drive strength available to the pulse, which is what actually controls how
much population leaks into level >=2 through the transmon's anharmonicity;
and (2) because hann_init() and stage0_optimize's coeffs_to_pulse() both
clip to +/-cfg.max_amp, it also changes the FEASIBLE PULSE SPACE and the
initial seed itself. So the honest description of what this sweep measures
is the causal chain

    control-amplitude budget -> leakage onset -> nonlinear loss
    landscape -> mean/CVaR divergence

not a direct measurement of "distance to the leakage boundary" at whatever
max_amp happens to be -- and there is no claim here that nature hands you
a sharp, universal physical boundary in the first place. The paper's m is
this distance, operationalized as distance to an OPERATIONAL leakage-onset
AMPLITUDE: a specific, reproducible threshold-crossing rule applied to a
measured curve (find_leakage_onset), not a unique physical phase boundary.
It is NOT SAFE to assume this onset falls somewhere inside a chosen
0.5x-2.5x sweep -- that would be assuming the answer. Instead,
run_stage0_sweep() first calls characterize_leakage_vs_amplitude() to
measure nominal leakage vs. amplitude directly (cheap: a single
deterministic thermal-state propagation per amplitude, no noise ensemble,
no optimization), then find_leakage_onset() to locate the amplitude at
which leakage crosses an explicit, stated threshold, by interpolation.
Only after that is m := max_amp - onset_amp computed and used as a second,
DERIVED x-axis (alongside the raw max_amp axis, kept for transparency) for
R_tail. leakage_onset_robustness() then re-derives that onset at several
threshold multipliers (3x/5x/10x background by default) from the SAME
curve, at no extra simulation cost, and run_stage0_sweep stores an m_x{mult}
column for each -- so whether "R_tail rises near the onset" is an artifact
of the 5x convention specifically, or a robust qualitative feature, is
checkable directly from the one sweep's output. The threshold, its
multiplier, and the robustness variants are all printed and saved, so
nothing about where the onset sits is silently assumed, and
characterize_leakage_vs_amplitude's cheap fixed-seed curve is plotted
alongside the ACTUAL optimized mean control's leakage at every sweep point
(mean_ctrl_leak_final) so the two can be compared before trusting the
seed-derived onset as representative of what the optimizer itself does.

CVaR ESTIMATOR: fractional-weight Rockafellar-Uryasev tail mean, not a
boolean quantile mask
============================================================================
For finite K, empirical CVaR is evaluated using fractional weighting of
the boundary sample whenever K*alpha is not an integer -- i.e. the
empirical CVaR_alpha of K per-draw samples is the mean of the worst
alpha*K of them, but alpha*K is not generally an integer (e.g. K=128,
alpha=0.01 -> worst 1.28 samples). A boolean mask built from
np.quantile(..., alpha) can only select an integer count and is therefore
NOT the Rockafellar-Uryasev estimator whenever alpha*K has a fractional
part; this matters most at small alpha, exactly where CVaR is most
interesting. _cvar_tail_weights()/_cvar_tail_mean() below implement the
exact finite-sample RU estimator instead: the worst floor(alpha*K) draws
get full weight, the next-worst draw gets the leftover fractional weight,
and the same weighting is applied to the gradient rows so the reported
CVaR value and its training gradient are consistent (an envelope-theorem
subgradient of the exact weighted quantity, not of a rounded proxy of it).
This ONE estimator is used everywhere CVaR is computed in this file --
training-time tail selection, out-of-sample R_tail, and the adversarial
search -- so there is no separate "training CVaR" and "reporting CVaR" to
drift apart.

SIGN / LOSS CONVENTION, AND TRAINING REWARD vs. PHYSICAL EVALUATION LOSS
============================================================================
Two distinct quantities are both informally "J" if you're not careful, and
conflating them is exactly the kind of thing a reviewer would attack, so
they get different names here:

  J_train  (scalar, mode-dependent; what L-BFGS-B actually maximizes)
      := reduction_over_K_draws(F_4ket - leak_lambda * leak) - ff_mu * J_FF
  where reduction is mean (mode="mean"), the fractional RU tail mean
  (mode="cvar"), or a convex combination of both (mode="mixed"), and J_FF
  is the smoothness/filter-function penalty from fade_realistic.py's own
  fade_ripe_objective_and_grad, applied IDENTICALLY across all three modes
  (per the optimizer-fairness protocol, Section 3.9) so it cannot itself
  become a confound between them.

  L_phys  (per-draw, mode-independent; the physical quantity everything
           is actually reported and compared on)
      := -(F_4ket - leak_lambda * leak)          [NO ff_mu term, ever]
  where F_4ket is the 4-ket average-gate-fidelity construction (the same
  one fade_ripe_objective_and_grad uses -- see selftest_matches_fade_ripe).
  The per-draw array all_J returned by _stage0_gather_per_draw is exactly
  -L_phys for each noise draw, and it is what cvar-mode tail selection
  ranks draws by DURING TRAINING. Stage0Ensemble (Section 1 below) computes
  the identical L_phys functional -- same 4-ket AGF, same leak penalty, no
  ff_mu -- on an INDEPENDENT out-of-sample noise ensemble. So the tail
  ranking used during training and the CVaR reported out-of-sample are
  computing the same physical object on different noise draws, differing
  only in which draws are used, never in what is measured.

  (An earlier version of this file used fade_realistic.py's own
  NoiseModel/fidelity_distribution for out-of-sample evaluation. That
  class evaluates a cheaper single-computational-basis-state fidelity
  (see _eval_fidelity_worker's rho0 = |0><0|), not the 4-ket average gate
  fidelity Stage 0 actually trains on -- a second, deeper instance of
  exactly the train/eval mismatch this section is about. Stage0Ensemble
  replaces it for that reason; fade_realistic.py itself is still not
  modified.)

R_tail is a ratio of CVaR_alpha(L_phys) values (mean-optimal vs.
CVaR-optimal control), well-defined and meaningful as long as it is
applied consistently to every compared control on the same evaluation
ensemble -- enforced throughout this file. loss_mean_from_J and
loss_cvar_from_J are the single point where the J -> L_phys sign flip
happens, so a sign error would show up in exactly one place if there is
one.

REQUIRED SETUP
============================================================================
Save (or copy) the uploaded fade_realistic.py into the SAME directory as
this script, named exactly `fade_realistic.py` (not the hashed upload
filename), so that `import fade_realistic` resolves.

WHAT THIS FILE DOES NOT DO
============================================================================
It does not modify fade_realistic.py. It does not implement Stages 1-7,
the full ablation ladder, or the phase-diagram synthesis (Stage 6) --
those are separate, later pieces of the program. It implements, for
Stage 0 specifically: the mean/CVaR/mixed training objective sharing one
gradient path, the optimizer-fairness protocol (Section 3.9), the
distributional-reporting requirement (Section 5.2), the Pareto frontier
(Section 5.3), the adversarial counterexample search (Section 3.8) scoped
to this stage's controls, and the two self-tests the paper's validation
plan requires be green before any divergence claim is trusted.

NOT executed on Claude's side (per your stated preference to run compute
locally). Syntax-checked via py_compile only. Run the self-tests first
(the __main__ block below does this automatically and stops if either
fails) before committing to the full sweep -- see the SMOKE_TEST /
FULL_RUN switch near the bottom.

REVISION NOTE (in response to two rounds of review of this file)
============================================================================
Round 1: (1) training reward vs. physical evaluation loss now have
distinct names and the out-of-sample evaluator (Stage0Ensemble) computes
the SAME 4-ket-AGF-based physical functional training uses, instead of
fade_realistic.py's cheaper single-basis-state fidelity; (2) the sweep no
longer treats cfg.max_amp itself as "distance to the leakage boundary" --
that distance is now measured empirically (characterize_leakage_vs_
amplitude / find_leakage_onset) before any R_tail number is interpreted
against it; (3) CVaR is now the exact fractional-weight Rockafellar-Uryasev
tail mean (_cvar_tail_mean), not a boolean quantile mask, so alpha*K need
not be an integer.

Round 2: (1) [important] restart selection and final reporting no longer
share an ensemble -- stage0_multistart now takes val_model (selection
only, thrown away) separately from the test_model every reported R_tail/
bootstrap/distribution/adversarial-search/Pareto number comes from; (2)
"leakage boundary" is now "operational leakage-onset amplitude"
throughout, with a free robustness check (leakage_onset_robustness) at
3x/5x/10x background re-derived from the one characterization curve; (3)
the leakage-characterization plot now overlays the cheap fixed-seed curve
against the actual optimized mean control's leakage at every sweep point;
(4) this docstring now says explicitly, in one sentence, that finite-K
CVaR uses fractional boundary-sample weighting whenever K*alpha is not an
integer; (5) training/validation/test ensembles are now built ONCE and
reused, unchanged, at every amplitude in the sweep (common random numbers)
instead of getting a fresh seed per sweep point; (6)/(7) apply to the
paper's prose, not this file, but the same spirit holds here: a rising
R_tail is evidence FOR the proposed leakage-nonlinearity mechanism, not
proof that "CVaR wins", and a flat R_tail is evidence against that
mechanism under this model, not a mathematical falsification of it.

No new stages, mechanisms, or sweep dimensions were added in either round
-- per both reviews, Stage 0's job is still just the cleanest possible
answer to whether leakage breaks mean/CVaR equivalence.
"""

import numpy as np
from scipy.optimize import minimize
from concurrent.futures import as_completed

import fade_realistic as fr
from fade_realistic import cfg, sys_, TWO_PI

# ============================================================================
# 1. PER-DRAW GRADIENT GATHER (the only new "physics-adjacent" code here --
#    everything inside is a direct reuse of fade_realistic's own per-draw
#    construction, just kept un-summed so a tail can be selected afterward)
# ============================================================================

def _stage0_chunk_worker(args):
    """Same per-draw physics as fade_realistic._fade_noise_chunk_worker
    (H0 + detuning*n_op, (1+amp_err)*Hx/Hy, exact Lindblad propagation,
    4-ket average gate fidelity with leakage penalty) -- but returns the
    PER-DRAW (J, gI, gQ) arrays instead of a chunk sum, since CVaR needs
    to rank individual draws rather than average over them up front."""
    (uI_f, uQ_f, deltas_chunk, gammas_chunk, phys_params) = args
    (alpha_MHz, N_LEVELS, Tg, Nt, T1, T2s, temperature_mK, qubit_freq_GHz,
     leak_lambda) = phys_params

    N = N_LEVELS
    a = fr._destroy_np(N); adag = a.conj().T; n_op = adag @ a
    alpha = alpha_MHz * 1e-3 * TWO_PI
    H0 = (0.0 * n_op + 0.5 * alpha * (adag @ adag @ a @ a)).astype(complex)
    Hx = (a + adag) / 2.0
    Hy = 1j * (adag - a) / 2.0
    gamma1 = 1.0 / T1
    gamma_phi = max(1.0 / T2s - gamma1 / 2.0, 1e-12)
    c_ops = [np.sqrt(gamma1) * a, np.sqrt(2 * gamma_phi) * n_op]
    P_leak = np.diag([0, 0] + [1] * (N - 2)).astype(complex)
    dt = Tg / Nt

    hbar_omega_over_kT = (TWO_PI * qubit_freq_GHz * 1e9 * 6.62607015e-34
                           / (2 * np.pi)) / (1.380649e-23 * temperature_mK * 1e-3)
    n_th = 1.0 / (np.exp(hbar_omega_over_kT) - 1.0)
    r = n_th / (1.0 + n_th) if n_th > 0 else 0.0
    pops = np.array([(1 - r) * r ** k for k in range(N)]); pops /= pops.sum()
    rho_th = np.diag(pops).astype(complex)
    leak_bg = float(np.real(np.trace(P_leak @ rho_th)))
    Uemb = np.eye(N, dtype=complex)
    Uemb[0:2, 0:2] = np.array([[1.0, -1j], [-1j, 1.0]]) / np.sqrt(2)

    kets = [
        np.array([1, 0] + [0] * (N - 2), dtype=complex),
        np.array([0, 1] + [0] * (N - 2), dtype=complex),
        np.array([1, 1] + [0] * (N - 2), dtype=complex) / np.sqrt(2),
        np.array([1, 1j] + [0] * (N - 2), dtype=complex) / np.sqrt(2),
    ]
    n_kets = len(kets)

    n_chunk = len(deltas_chunk)
    J_arr = np.zeros(n_chunk)
    gI_arr = np.zeros((n_chunk, Nt))
    gQ_arr = np.zeros((n_chunk, Nt))

    for i, (d, g) in enumerate(zip(deltas_chunk, gammas_chunk)):
        H0_d = H0 + d * n_op
        prop_d = fr.LindbladPropagator(H0_d, (1 + g) * Hx, (1 + g) * Hy, c_ops)
        steps = prop_d.step_data(uI_f, uQ_f, dt)
        J_draw, gI_draw, gQ_draw = 0.0, np.zeros(Nt), np.zeros(Nt)
        for psi in kets:
            rho_comp = np.outer(psi, psi.conj())
            rho_in = (1.0 - leak_bg) * rho_comp + P_leak @ rho_th @ P_leak
            rho_target = Uemb @ rho_in @ Uemb.conj().T
            J, gI, gQ, _, _ = fr.lindblad_fidelity_and_grad(
                uI_f, uQ_f, rho_in, rho_target, prop_d, dt, leak_lambda, steps=steps)
            J_draw += J; gI_draw += gI; gQ_draw += gQ
        J_arr[i] = J_draw / n_kets
        gI_arr[i] = gI_draw / n_kets
        gQ_arr[i] = gQ_draw / n_kets

    return J_arr, gI_arr, gQ_arr


def _stage0_gather_per_draw(uI_f, uQ_f, deltas, gammas, parallel=True):
    """Returns (all_J, all_gI, all_gQ) of shape (K,), (K,Nt), (K,Nt): the
    per-draw reward and gradient for every noise draw, unreduced. Draw
    order across the returned arrays is not meaningful (as_completed does
    not preserve submission order) -- this is fine because every reduction
    used downstream (mean, worst-alpha-tail mean) is order-independent."""
    K = len(deltas)
    phys_params = (cfg.alpha_MHz, cfg.N_LEVELS, cfg.Tg, cfg.Nt, cfg.T1, cfg.T2s,
                   cfg.temperature_mK, cfg.qubit_freq_GHz, cfg.leak_lambda)

    if parallel and K >= fr._N_WORKERS:
        n_chunks = min(K, max(fr._N_WORKERS, fr._N_WORKERS * 4))
        d_chunks = np.array_split(deltas, n_chunks)
        g_chunks = np.array_split(gammas, n_chunks)
        pool = fr._get_pool()
        futures = [pool.submit(_stage0_chunk_worker, (uI_f, uQ_f, dc, gc, phys_params))
                   for dc, gc in zip(d_chunks, g_chunks) if len(dc) > 0]
        J_parts, gI_parts, gQ_parts = [], [], []
        for fut in as_completed(futures):
            Ja, gIa, gQa = fut.result()
            J_parts.append(Ja); gI_parts.append(gIa); gQ_parts.append(gQa)
        all_J = np.concatenate(J_parts)
        all_gI = np.vstack(gI_parts)
        all_gQ = np.vstack(gQ_parts)
    else:
        all_J, all_gI, all_gQ = _stage0_chunk_worker((uI_f, uQ_f, deltas, gammas, phys_params))

    return all_J, all_gI, all_gQ


class Stage0Ensemble:
    """A fixed (delta, gamma) noise ensemble for validation or test /
    physical evaluation, built the same way as the training ensemble but
    from a disjoint seed -- run_stage0_sweep constructs one instance for
    validation (restart selection only, then discarded) and a second,
    disjoint instance for test (every reported number; see the module
    docstring's train/validation/test section). Deliberately does NOT
    reuse fade_realistic.NoiseModel:
    that class's fidelity_distribution evaluates a cheaper single-
    computational-basis-state fidelity (see fade_realistic._eval_fidelity_
    worker's rho0 = |0><0|), not the 4-ket average-gate-fidelity
    functional Stage 0 actually trains and reports on -- reusing it here
    would silently reintroduce the exact train/eval mismatch the module
    docstring's "TRAINING REWARD vs. PHYSICAL EVALUATION LOSS" section
    warns about, just one level deeper. This class instead evaluates the
    IDENTICAL physical functional used during training (via
    _stage0_gather_per_draw, gradients simply discarded), just on an
    independent noise draw, and the ff_mu penalty is never added.

    Interface-compatible with fade_realistic.NoiseModel for the methods
    this file actually calls (fidelity_distribution/mean_fidelity/cvar),
    so it drops in wherever a validation or test ensemble is constructed.
    """

    def __init__(self, n_samples, sigma_d, sigma_a, seed):
        rng = np.random.default_rng(seed)
        self.deltas = rng.normal(0.0, sigma_d, n_samples)
        self.gammas = rng.normal(0.0, sigma_a, n_samples)

    def fidelity_distribution(self, uI_f, uQ_f):
        """uI_f, uQ_f: the BANDWIDTH-FILTERED pulse (matching the
        convention used everywhere else in this file, e.g.
        stage0_multistart's uI_opt/uQ_opt). Returns the per-draw physical
        reward array all_J = F_4ket - leak_lambda*leak (no ff_mu term);
        L_phys per draw is -all_J."""
        all_J, _, _ = _stage0_gather_per_draw(uI_f, uQ_f, self.deltas, self.gammas, parallel=True)
        return all_J

    def mean_fidelity(self, uI_f, uQ_f):
        return float(np.mean(self.fidelity_distribution(uI_f, uQ_f)))

    def worst_fidelity(self, uI_f, uQ_f):
        return float(np.min(self.fidelity_distribution(uI_f, uQ_f)))

    def cvar(self, uI_f, uQ_f, alpha=0.05):
        return _cvar_tail_mean(self.fidelity_distribution(uI_f, uQ_f), alpha)


def _cvar_tail_weights(values, alpha):
    """Exact finite-sample Rockafellar-Uryasev tail weights on `values`
    (worst = LOWEST value throughout this file, since we work in reward J
    where low J = bad). Returns (order, n_full, frac): `order` sorts
    `values` ascending (order[0] is the single worst draw); the first
    n_full entries of `order` get full weight 1; the entry at index n_full
    gets fractional weight `frac` in [0, 1); everything after that gets
    weight 0. The weights sum to exactly alpha * len(values) by
    construction (n_full = floor(alpha*n), frac = alpha*n - n_full), which
    is what makes a plain weighted SUM divided by (n_full + frac) the
    correct tail MEAN even when alpha*n is not an integer -- e.g. K=128,
    alpha=0.01 gives n_full=1, frac=0.28, i.e. the single worst draw plus
    28% of the second-worst draw, exactly "the worst 1.28 samples."""
    n = len(values)
    order = np.argsort(values)
    target = alpha * n
    n_full = min(int(np.floor(target)), n)
    frac = 0.0 if n_full == n else (target - n_full)
    return order, n_full, frac


def _weighted_tail_mean(sorted_arr, n_full, frac):
    """sorted_arr must already be sorted worst-first along axis 0 (i.e.
    sorted_arr = arr[order] using the `order` from _cvar_tail_weights).
    Works for 1-D (values) or 2-D (e.g. per-draw gradient rows) arrays."""
    total_weight = n_full + frac
    if total_weight <= 0:
        return sorted_arr[0] * 1.0  # degenerate alpha*n < any weight: single worst draw
    partial = sorted_arr[:n_full].sum(axis=0) if n_full > 0 else 0.0 * sorted_arr[0]
    if frac > 0 and n_full < len(sorted_arr):
        partial = partial + frac * sorted_arr[n_full]
    return partial / total_weight


def _cvar_tail_mean(values, alpha):
    """The canonical CVaR-style tail mean used EVERYWHERE in this file
    (training-time selection, out-of-sample R_tail, adversarial search) --
    see the module docstring's "CVaR ESTIMATOR" section. `values` is a
    reward array (worst = lowest); returns a scalar."""
    values = np.asarray(values, dtype=float)
    order, n_full, frac = _cvar_tail_weights(values, alpha)
    return float(_weighted_tail_mean(values[order], n_full, frac))


# ============================================================================
# 2. TRAINING OBJECTIVE: mean / cvar / mixed, one shared gradient path
# ============================================================================

def stage0_objective_and_grad(uI_raw, uQ_raw, bwf, deltas, gammas,
                               mode="mean", alpha=0.05, lam=0.5, parallel=True):
    """mode in {"mean", "cvar", "mixed"}. Returns (J_reduced, gI_raw, gQ_raw,
    all_J) -- all_J (the unreduced per-draw rewards) is returned for
    diagnostics (adversarial search, distributional plots), not used
    further here.

    cfg.ff_mu is READ, not calibrated, here -- see calibrate_ff_mu() below.
    All three modes must share one calibration so the filter-function
    penalty does not itself become a confound between them (Section 3.9,
    optimizer-fairness protocol).
    """
    dt = cfg.dt
    uI_f = bwf.apply(uI_raw)
    uQ_f = bwf.apply(uQ_raw)

    all_J, all_gI, all_gQ = _stage0_gather_per_draw(uI_f, uQ_f, deltas, gammas, parallel=parallel)

    if mode == "mean":
        J_red = np.mean(all_J)
        gI_red = np.mean(all_gI, axis=0)
        gQ_red = np.mean(all_gQ, axis=0)
    elif mode == "cvar":
        order, n_full, frac = _cvar_tail_weights(all_J, alpha)
        J_red = _weighted_tail_mean(all_J[order], n_full, frac)
        gI_red = _weighted_tail_mean(all_gI[order], n_full, frac)
        gQ_red = _weighted_tail_mean(all_gQ[order], n_full, frac)
    elif mode == "mixed":
        J_mean = np.mean(all_J)
        gI_mean = np.mean(all_gI, axis=0)
        gQ_mean = np.mean(all_gQ, axis=0)
        order, n_full, frac = _cvar_tail_weights(all_J, alpha)
        J_tail = _weighted_tail_mean(all_J[order], n_full, frac)
        gI_tail = _weighted_tail_mean(all_gI[order], n_full, frac)
        gQ_tail = _weighted_tail_mean(all_gQ[order], n_full, frac)
        J_red = (1 - lam) * J_mean + lam * J_tail
        gI_red = (1 - lam) * gI_mean + lam * gI_tail
        gQ_red = (1 - lam) * gQ_mean + lam * gQ_tail
    else:
        raise ValueError(f"unknown mode {mode!r}")

    # Same filter-function penalty term as fade_ripe_objective_and_grad,
    # applied identically regardless of mode (it does not depend on the
    # noise draw, so it cannot itself change which draws are in the tail).
    J_ff, gI_ff, gQ_ff = fr.filter_function_overlap_and_grad(uI_f, uQ_f, dt)
    J_red = J_red - cfg.ff_mu * J_ff
    gI_red = gI_red - cfg.ff_mu * gI_ff
    gQ_red = gQ_red - cfg.ff_mu * gQ_ff

    gI_raw = bwf.backprop_grad(gI_red)
    gQ_raw = bwf.backprop_grad(gQ_red)
    return J_red, gI_raw, gQ_raw, all_J


def calibrate_ff_mu(seed_uI, seed_uQ, bwf, verbose=False):
    """Sets the GLOBAL cfg.ff_mu once, from a fixed seed pulse, so that
    mean/cvar/mixed training at a given sweep point all use the identical
    filter-function penalty weight. Mirrors fade_ripe_optimize's own
    calibration exactly, just factored out so it runs once per sweep point
    rather than once per objective."""
    prop = fr._get_propagator()
    uI_seed_f = bwf.apply(seed_uI)
    uQ_seed_f = bwf.apply(seed_uQ)
    J_fid_seed, _, _, leak_seed, F_seed = fr.average_gate_fidelity_and_grad(
        uI_seed_f, uQ_seed_f, prop, cfg.dt)
    J_ff_seed, _, _ = fr.filter_function_overlap_and_grad(uI_seed_f, uQ_seed_f, cfg.dt)
    cfg.ff_mu = (cfg.ff_target_fraction * J_fid_seed / J_ff_seed) if J_ff_seed > 1e-14 else 0.0
    if verbose:
        print(f"    [calibrate_ff_mu] fidelity={J_fid_seed:.6f} leak={leak_seed:.3e} "
              f"J_ff={J_ff_seed:.3e} -> ff_mu={cfg.ff_mu:.4g}")


# ============================================================================
# 3. SELF-TESTS -- must pass before any sweep result is trusted
#    (paper Section 3.10, "Validation plan")
# ============================================================================

def selftest_matches_fade_ripe(verbose=True):
    """Stage 0's central infrastructure claim is that it reuses the
    existing quasi-static ensemble and Lindblad propagator UNMODIFIED.
    This is checkable, not just assertable: stage0_objective_and_grad in
    mode="mean" must reproduce fade_realistic.fade_ripe_objective_and_grad
    bit-for-bit (up to floating-point summation-order noise) on identical
    inputs. Uses a small throwaway problem (tiny Nt/Tg/K) for speed;
    temporarily mutates cfg.Nt/cfg.Tg and restores them afterward, the
    same pattern fade_realistic.py's own self-test uses for its tiny-Nt
    filter-function check."""
    old_Nt, old_Tg, old_ff_mu = cfg.Nt, cfg.Tg, cfg.ff_mu
    try:
        cfg.Nt = 24
        cfg.Tg = 24.0
        cfg.ff_mu = 0.0  # isolate the noise-ensemble reduction; ff term tested separately upstream

        rng = np.random.default_rng(0)
        uI_raw = 0.5 * cfg.max_amp * rng.standard_normal(cfg.Nt)
        uQ_raw = 0.5 * cfg.max_amp * rng.standard_normal(cfg.Nt)
        deltas = rng.normal(0.0, cfg.fade_sigma, 6)
        gammas = rng.normal(0.0, cfg.fade_amp_sig, 6)

        bwf = fr.BandwidthFilter(cfg.Nt, cfg.dt, cfg.awg_bandwidth_GHz, cfg.awg_filter_order)
        prop = fr._get_propagator()

        J_ref, gI_ref, gQ_ref = fr.fade_ripe_objective_and_grad(
            uI_raw, uQ_raw, bwf, prop, deltas, gammas, parallel=False)
        J_test, gI_test, gQ_test, _ = stage0_objective_and_grad(
            uI_raw, uQ_raw, bwf, deltas, gammas, mode="mean", parallel=False)

        dJ = abs(J_ref - J_test)
        dgI = np.max(np.abs(gI_ref - gI_test))
        dgQ = np.max(np.abs(gQ_ref - gQ_test))
        ok = (dJ < 1e-9) and (dgI < 1e-9) and (dgQ < 1e-9)
        if verbose:
            print(f"[selftest_matches_fade_ripe] dJ={dJ:.3e} "
                  f"max|dgI|={dgI:.3e} max|dgQ|={dgQ:.3e} -> {'PASS' if ok else 'FAIL'}")
        return ok
    finally:
        cfg.Nt, cfg.Tg, cfg.ff_mu = old_Nt, old_Tg, old_ff_mu


def selftest_finite_difference(mode="mean", alpha=0.05, lam=0.5, verbose=True, seed=1):
    """Finite-difference check of stage0_objective_and_grad's gradient
    against central differences of its own returned J, at a random point
    with a random smooth perturbation direction. For mode="cvar" or
    "mixed" the tail reduction uses the exact fractional-weight
    Rockafellar-Uryasev estimator (_cvar_tail_weights/_weighted_tail_mean)
    with n_full and frac fixed once alpha and K are fixed -- but WHICH
    noise draw sits at each rank can still change discretely as the
    control varies (two draws' J values crossing in rank), so the
    reduction is piecewise-smooth with measure-zero kinks at those
    crossings, and the reported gradient is a subgradient there, not a
    classical gradient everywhere -- exactly the situation the paper's
    Rockafellar-Uryasev representation is built for. A finite-difference
    check should therefore agree away from those boundaries; a large
    disagreement on one run is not on its own damning (it may have hit a
    boundary by chance) and should be re-run with a different seed before
    being treated as a bug."""
    old_Nt, old_Tg, old_ff_mu = cfg.Nt, cfg.Tg, cfg.ff_mu
    try:
        cfg.Nt = 24
        cfg.Tg = 24.0
        cfg.ff_mu = 0.0

        rng = np.random.default_rng(seed)
        uI_raw = 0.5 * cfg.max_amp * rng.standard_normal(cfg.Nt)
        uQ_raw = 0.5 * cfg.max_amp * rng.standard_normal(cfg.Nt)
        deltas = rng.normal(0.0, cfg.fade_sigma, 10)
        gammas = rng.normal(0.0, cfg.fade_amp_sig, 10)
        bwf = fr.BandwidthFilter(cfg.Nt, cfg.dt, cfg.awg_bandwidth_GHz, cfg.awg_filter_order)

        J0, gI0, gQ0, _ = stage0_objective_and_grad(
            uI_raw, uQ_raw, bwf, deltas, gammas, mode=mode, alpha=alpha, lam=lam, parallel=False)

        dI = rng.standard_normal(cfg.Nt)
        dQ = rng.standard_normal(cfg.Nt)
        eps = 1e-6
        Jp, _, _, _ = stage0_objective_and_grad(
            uI_raw + eps * dI, uQ_raw + eps * dQ, bwf, deltas, gammas,
            mode=mode, alpha=alpha, lam=lam, parallel=False)
        Jm, _, _, _ = stage0_objective_and_grad(
            uI_raw - eps * dI, uQ_raw - eps * dQ, bwf, deltas, gammas,
            mode=mode, alpha=alpha, lam=lam, parallel=False)

        fd_directional = (Jp - Jm) / (2 * eps)
        analytic_directional = np.dot(gI0, dI) + np.dot(gQ0, dQ)
        rel_err = abs(fd_directional - analytic_directional) / max(abs(fd_directional), 1e-8)
        ok = rel_err < 1e-4
        if verbose:
            print(f"[selftest_finite_difference mode={mode}] "
                  f"fd={fd_directional:.6e} analytic={analytic_directional:.6e} "
                  f"rel_err={rel_err:.3e} -> {'PASS' if ok else 'FAIL (may be a tail boundary; re-run with a different seed)'}")
        return ok
    finally:
        cfg.Nt, cfg.Tg, cfg.ff_mu = old_Nt, old_Tg, old_ff_mu


def selftest_fractional_cvar_matches_definition(verbose=True):
    """Directly validates _cvar_tail_mean against a hand-computed example
    where alpha*K is deliberately NOT an integer, since that is exactly
    the case a boolean quantile mask gets wrong (fix (3) from review).
    K=7, alpha=2/7 ~= 0.2857 -> alpha*K = 2 exactly (integer control case,
    should just be the mean of the worst 2). K=7, alpha=0.3 -> alpha*K=2.1
    -> worst 2 fully + 0.1 of the 3rd-worst, checked against a manual
    weighted average computed independently of _cvar_tail_mean's own code
    path."""
    values = np.array([5.0, 1.0, 9.0, 2.0, 7.0, 3.0, 8.0])  # worst (lowest) first once sorted: 1,2,3,5,7,8,9
    sorted_vals = np.sort(values)

    # integer case: alpha*K = 2 exactly -> mean of the 2 worst (1, 2)
    ok_int = abs(_cvar_tail_mean(values, alpha=2.0 / 7.0) - np.mean(sorted_vals[:2])) < 1e-12

    # fractional case: alpha*K = 0.3*7 = 2.1 -> (1 + 2 + 0.1*3) / 2.1
    expected_frac = (sorted_vals[0] + sorted_vals[1] + 0.1 * sorted_vals[2]) / 2.1
    ok_frac = abs(_cvar_tail_mean(values, alpha=0.3) - expected_frac) < 1e-9

    # sub-single-sample case: alpha*K = 0.1*7 = 0.7 -> pure fractional weight
    # on the single worst sample -> degenerates to that sample itself
    ok_degenerate = abs(_cvar_tail_mean(values, alpha=0.1) - sorted_vals[0]) < 1e-12

    ok = ok_int and ok_frac and ok_degenerate
    if verbose:
        print(f"[selftest_fractional_cvar_matches_definition] integer_case={ok_int} "
              f"fractional_case={ok_frac} (got={_cvar_tail_mean(values, alpha=0.3):.6f}, "
              f"expected={expected_frac:.6f}) degenerate_case={ok_degenerate} "
              f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# 4. OPTIMIZATION: mirrors fade_ripe_optimize exactly, parameterized by mode
# ============================================================================

def stage0_optimize(seed_uI, seed_uQ, deltas, gammas, mode="mean", alpha=0.05,
                     lam=0.5, verbose=True, progress_every=25):
    """progress_every: print a heartbeat line every this many function
    evaluations (elapsed wall-clock time + current objective value), so a
    restart that takes many minutes never sits completely silent -- there
    was previously no way to tell "this is just slow" from "this is stuck"
    without checking CPU usage outside Python. Set to 0/None to disable."""
    Nt, M = cfg.Nt, cfg.fade_nmodes
    tlist = np.linspace(0.0, cfg.Tg, Nt)
    hann = 0.5 * (1.0 - np.cos(TWO_PI * np.linspace(0, 1, Nt)))
    basis_I = np.stack([np.sin((2 * n - 1) * np.pi * tlist / cfg.Tg) * hann
                         for n in range(1, M + 1)], axis=1)
    basis_Q = np.stack([np.sin(2 * n * np.pi * tlist / cfg.Tg) * hann
                         for n in range(1, M + 1)], axis=1)

    def coeffs_to_pulse(c):
        uI = np.clip(basis_I @ c[:M], -cfg.max_amp, cfg.max_amp)
        uQ = np.clip(basis_Q @ c[M:], -cfg.max_amp, cfg.max_amp)
        return uI, uQ

    c0 = np.concatenate([np.linalg.pinv(basis_I) @ seed_uI,
                          np.linalg.pinv(basis_Q) @ seed_uQ])

    bwf = fr.BandwidthFilter(Nt, cfg.dt, cfg.awg_bandwidth_GHz, cfg.awg_filter_order)

    import time
    t_start = time.monotonic()
    eval_count = [0]

    def obj(c):
        uI, uQ = coeffs_to_pulse(c)
        J, gI, gQ, _ = stage0_objective_and_grad(
            uI, uQ, bwf, deltas, gammas, mode=mode, alpha=alpha, lam=lam)
        grad_c = np.concatenate([basis_I.T @ gI, basis_Q.T @ gQ])
        eval_count[0] += 1
        if verbose and progress_every and eval_count[0] % progress_every == 0:
            elapsed = time.monotonic() - t_start
            print(f"      ... eval {eval_count[0]}, {elapsed:.0f}s elapsed, "
                  f"current objective={J:.6f} (still running -- this line proves it, not stuck)")
        return -J, -grad_c

    if verbose:
        print(f"    Stage0[{mode}] alpha={alpha} lam={lam}: Nt={Nt}, modes={M}, "
              f"K={len(deltas)} draws/eval, max_amp={cfg.max_amp:.5f} rad/ns, "
              f"leak_lambda={cfg.leak_lambda}")

    res = minimize(obj, c0, method='L-BFGS-B', jac=True,
                    options={'maxiter': cfg.fade_maxiter, 'ftol': cfg.fade_ftol,
                             'gtol': cfg.fade_gtol})
    if verbose:
        elapsed_total = time.monotonic() - t_start
        print(f"    L-BFGS-B: {res.nit} iters, {res.nfev} evals, {elapsed_total:.0f}s, "
              f"converged={res.success}, message='{res.message}'")

    best_uI_raw, best_uQ_raw = coeffs_to_pulse(res.x)
    best_uI = bwf.apply(best_uI_raw)
    best_uQ = bwf.apply(best_uQ_raw)

    prop = fr._get_propagator()
    J_final, _, _, leak_final, F_final = fr.thermal_gate_fidelity_and_grad(
        best_uI, best_uQ, prop, cfg.dt)
    info = {'J_final': J_final, 'leak_final': leak_final, 'F_final': F_final,
            'nit': res.nit, 'success': res.success, 'mode': mode, 'alpha': alpha, 'lam': lam}
    return best_uI_raw, best_uQ_raw, best_uI, best_uQ, info


def stage0_multistart(seed_uI, seed_uQ, deltas, gammas, mode, alpha=0.05, lam=0.5,
                       n_restarts=None, val_model=None, seed_base=0, verbose=True,
                       progress_every=25):
    """Optimizer-fairness protocol (paper Section 3.9): multiple random
    inits, best-converged result retained under a fixed criterion.

    TRAIN / VALIDATION / TEST SPLIT (review fix (1) -- important): `deltas`/
    `gammas` are the TRAINING ensemble (used for every restart's gradient
    descent). `val_model` is a SEPARATE, disjoint ensemble used ONLY to pick
    which restart wins -- mean training is ranked by validation mean;
    cvar/mixed training by validation CVaR at the SAME alpha it was trained
    on (ranking every mode by validation mean alone would bias the
    comparison toward the mean optimizer by construction). Once a restart
    is selected, val_model's job is done and its numbers are NOT reused for
    anything reported -- the paper promises an independent test ensemble,
    and if the "out-of-sample" set both picks the winner AND supplies the
    reported R_tail/bootstrap/distributions, it is not actually held out.
    The caller (run_stage0_sweep) evaluates the SELECTED control on a
    third, untouched test ensemble for every reported number. See the
    module docstring if one is added there, and run_stage0_sweep's own
    docstring, for the full train -> validation -> test flow."""
    n_restarts = n_restarts or cfg.fade_n_restarts
    pert_rng = np.random.default_rng(seed_base)
    results = []
    for r in range(n_restarts):
        if r == 0:
            init_uI, init_uQ = seed_uI.copy(), seed_uQ.copy()
            tag = "unperturbed Hann seed"
        else:
            pert_scale = 0.3 * cfg.max_amp
            init_uI = np.clip(seed_uI + pert_scale * pert_rng.standard_normal(len(seed_uI)),
                               -cfg.max_amp, cfg.max_amp)
            init_uQ = np.clip(seed_uQ + pert_scale * pert_rng.standard_normal(len(seed_uQ)),
                               -cfg.max_amp, cfg.max_amp)
            tag = f"perturbed start #{r}"
        if verbose:
            print(f"  -- {mode} restart {r + 1}/{n_restarts}: {tag} --")
        uI_raw, uQ_raw, uI_opt, uQ_opt, info = stage0_optimize(
            init_uI, init_uQ, deltas, gammas, mode=mode, alpha=alpha, lam=lam, verbose=verbose,
            progress_every=progress_every)

        if val_model is not None:
            val_mean = val_model.mean_fidelity(uI_opt, uQ_opt)
            val_cvar = val_model.cvar(uI_opt, uQ_opt, alpha=alpha)
        else:
            val_mean, val_cvar = info['J_final'], None

        sel_metric = val_cvar if (mode in ("cvar", "mixed") and val_cvar is not None) else val_mean
        results.append({'sel_metric': sel_metric, 'val_mean': val_mean, 'val_cvar': val_cvar,
                         'uI_raw': uI_raw, 'uQ_raw': uQ_raw,
                         'uI_opt': uI_opt, 'uQ_opt': uQ_opt, 'info': info})

    best = max(results, key=lambda r: r['sel_metric'])
    if verbose:
        spread = max(r['sel_metric'] for r in results) - min(r['sel_metric'] for r in results)
        print(f"  [{mode}] {n_restarts} restarts, selection-metric spread={spread:.4e} "
              f"(validation-only, thrown away after selection), "
              f"kept validation {'CVaR' if mode!='mean' else 'mean'}={best['sel_metric']:.6f}")
    return best


# ============================================================================
# 5. OUT-OF-SAMPLE EVALUATION, R_tail, AND PAIRED BOOTSTRAP
# ============================================================================

def _tail_mean_of_J(J_samples, alpha):
    """Delegates to the canonical fractional-weight RU estimator
    (_cvar_tail_mean) so out-of-sample/reporting CVaR (R_tail, adversarial
    search, Pareto frontier) uses the EXACT SAME estimator as training-time
    tail selection -- see the module docstring's "CVaR ESTIMATOR" section."""
    return _cvar_tail_mean(J_samples, alpha)


def loss_mean_from_J(J_samples):
    return -float(np.mean(J_samples))


def loss_cvar_from_J(J_samples, alpha):
    return -_tail_mean_of_J(J_samples, alpha)


def r_tail_point_estimate(J_mean_ctrl_samples, J_cvar_ctrl_samples, alpha):
    """R_tail(alpha) using loss L := -J (see module docstring)."""
    cvar_mean_ctrl = loss_cvar_from_J(J_mean_ctrl_samples, alpha)
    cvar_cvar_ctrl = loss_cvar_from_J(J_cvar_ctrl_samples, alpha)
    if cvar_mean_ctrl == 0:
        return np.nan
    return (cvar_mean_ctrl - cvar_cvar_ctrl) / cvar_mean_ctrl


def bootstrap_r_tail(model, u_mean, u_cvar, alpha, n_boot=1000, seed=0):
    """Paired bootstrap: u_mean and u_cvar are evaluated against the exact
    same noise draws (model.deltas/model.gammas fixed), so resampling the
    SAME indices for both preserves that pairing and gives a much tighter
    CI than resampling them independently would. `model` must be the
    UNTOUCHED TEST ensemble (review fix (1)) -- never the validation
    ensemble stage0_multistart used to pick these controls in the first
    place, or this number is not actually out-of-sample."""
    uI_m, uQ_m = u_mean
    uI_c, uQ_c = u_cvar
    Jm = model.fidelity_distribution(uI_m, uQ_m)
    Jc = model.fidelity_distribution(uI_c, uQ_c)
    n = len(Jm)
    rng = np.random.default_rng(seed)

    point = r_tail_point_estimate(Jm, Jc, alpha)
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[b] = r_tail_point_estimate(Jm[idx], Jc[idx], alpha)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {'point': point, 'ci_lo': lo, 'ci_hi': hi, 'boot': boot, 'J_mean_ctrl': Jm, 'J_cvar_ctrl': Jc}


# ============================================================================
# 6. ADVERSARIAL SEARCH (paper Section 3.8)
# ============================================================================

def adversarial_search(model, controls, alpha, verbose=True):
    """controls: dict[name] -> (uI, uQ). Scans all pairs for
    mean_loss(u1) < mean_loss(u2) but cvar_loss(u1) > cvar_loss(u2) at the
    given alpha. Flags candidates for human review with bootstrap CIs on
    the CVaR gap -- does not auto-classify significance, per the paper's
    "any confirmed find must be resolved" framing. `model` should be the
    untouched TEST ensemble, same as bootstrap_r_tail (review fix (1))."""
    names = list(controls.keys())
    J_samples = {name: model.fidelity_distribution(*controls[name]) for name in names}
    mean_loss = {name: loss_mean_from_J(J_samples[name]) for name in names}
    cvar_loss = {name: loss_cvar_from_J(J_samples[name], alpha) for name in names}

    flags = []
    for i, n1 in enumerate(names):
        for n2 in names[i + 1:]:
            if mean_loss[n1] < mean_loss[n2] and cvar_loss[n1] > cvar_loss[n2]:
                flags.append((n1, n2))
            elif mean_loss[n2] < mean_loss[n1] and cvar_loss[n2] > cvar_loss[n1]:
                flags.append((n2, n1))

    if verbose:
        if not flags:
            print(f"  [adversarial_search alpha={alpha}] no counterexample among "
                  f"{len(names)} controls.")
        for lo_mean, hi_mean in flags:
            print(f"  [adversarial_search] CANDIDATE COUNTEREXAMPLE: "
                  f"{lo_mean} has lower mean loss ({mean_loss[lo_mean]:.6f}) but "
                  f"HIGHER cvar loss ({cvar_loss[lo_mean]:.6f}) than {hi_mean} "
                  f"(mean={mean_loss[hi_mean]:.6f}, cvar={cvar_loss[hi_mean]:.6f}) "
                  f"-- needs human review, not auto-resolved.")
    return flags, mean_loss, cvar_loss


# ============================================================================
# 7. DISTRIBUTIONAL REPORTING AND PARETO FRONTIER (paper Sections 5.2, 5.3)
# ============================================================================

def save_loss_distribution_plot(model, u_mean, u_cvar, alpha, out_path):
    """`model` should be the untouched TEST ensemble (review fix (1))."""
    import matplotlib.pyplot as plt
    Jm = model.fidelity_distribution(*u_mean)
    Jc = model.fidelity_distribution(*u_cvar)
    Lm, Lc = -Jm, -Jc
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(min(Lm.min(), Lc.min()), max(Lm.max(), Lc.max()), 40)
    ax.hist(Lm, bins=bins, alpha=0.5, label=f"mean-optimal (u_mean)", density=True)
    ax.hist(Lc, bins=bins, alpha=0.5, label=f"CVaR-optimal, alpha={alpha} (u_cvar)", density=True)
    ax.set_xlabel("per-shot loss L = -J")
    ax.set_ylabel("density")
    ax.set_title("Test-ensemble loss distribution: mean- vs. CVaR-optimal control")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    np.savez(str(out_path).replace('.png', '.npz'), L_mean_ctrl=Lm, L_cvar_ctrl=Lc)


def pareto_frontier(seed_uI, seed_uQ, deltas, gammas, val_model, test_model, alpha,
                     lambdas=(0.0, 0.25, 0.5, 0.75, 1.0), n_restarts=None, verbose=True,
                     progress_every=25):
    """lambda=0 recovers the mean objective, lambda=1 the CVaR objective
    (both via mode="mixed" for a single consistent code path). Restart
    selection at each lambda uses val_model only (review fix (1)); the
    reported mean_loss/cvar_loss for each point are then evaluated on the
    untouched test_model, never on val_model."""
    points = []
    for lam in lambdas:
        best = stage0_multistart(seed_uI, seed_uQ, deltas, gammas, mode="mixed",
                                  alpha=alpha, lam=lam, n_restarts=n_restarts,
                                  val_model=val_model, seed_base=int(1000 * lam) + 1,
                                  verbose=verbose, progress_every=progress_every)
        J_samples = test_model.fidelity_distribution(best['uI_opt'], best['uQ_opt'])
        points.append({'lam': lam,
                        'mean_loss': loss_mean_from_J(J_samples),
                        'cvar_loss': loss_cvar_from_J(J_samples, alpha),
                        'uI_opt': best['uI_opt'], 'uQ_opt': best['uQ_opt']})
    return points


def save_pareto_plot(points, alpha, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 5))
    means = [p['mean_loss'] for p in points]
    cvars = [p['cvar_loss'] for p in points]
    ax.plot(means, cvars, 'o-')
    for p in points:
        ax.annotate(f"lam={p['lam']}", (p['mean_loss'], p['cvar_loss']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("E[L] (mean loss)")
    ax.set_ylabel(f"CVaR_{alpha}[L]")
    ax.set_title("Pareto frontier: mixed-objective sweep")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================================
# 7b. OPERATIONAL LEAKAGE-ONSET CHARACTERIZATION (review fix (2)) -- run
#     BEFORE any expensive optimization, so the "operational leakage-onset
#     amplitude" is a measured quantity, not an assumption about where in
#     the sweep it falls. Deliberately NOT called "the leakage boundary":
#     it is a threshold-crossing convention on a characterization curve,
#     not a uniquely-defined physical phase boundary -- see
#     find_leakage_onset's docstring and the robustness check below.
# ============================================================================

def characterize_leakage_vs_amplitude(amp_grid, verbose=True):
    """Cheap, optimization-free characterization of nominal leakage vs.
    control amplitude. For each amp in amp_grid: builds a Hann-envelope
    seed pulse scaled/clipped to that amplitude (fr.hann_init respects
    cfg.max_amp), band-filters it exactly as training would, and evaluates
    its NOMINAL leakage (zero detuning/amplitude error, thermal initial
    population) via fr.thermal_gate_fidelity_and_grad -- a single
    deterministic propagation, no noise ensemble, no optimization. That
    makes this pass cheap enough to run on a much finer amplitude grid
    than the main R_tail sweep can afford.

    This deliberately characterizes the PHYSICAL leakage channel (drive
    populating level >=2 via the transmon's anharmonicity) as a function
    of amplitude BUDGET, decoupled from what a specific optimizer does
    with that budget -- an explicit simplification, not a hidden one: the
    leakage an optimizer actually converges to at a given max_amp can
    differ somewhat from a fixed seed's leakage. run_stage0_sweep already
    records mean_ctrl_leak_final (the REAL optimized mean control's
    leakage) at every sweep point as a cross-check on this cheaper curve;
    compare the two before trusting the derived onset far from where they
    agree.

    Returns {'amps': array, 'leaks': array} sorted ascending by amp.
    """
    old_amp = cfg.max_amp
    amps_out, leaks_out = [], []
    try:
        for amp in amp_grid:
            cfg.max_amp = float(amp)
            uI_seed, uQ_seed = fr.hann_init()
            bwf = fr.BandwidthFilter(cfg.Nt, cfg.dt, cfg.awg_bandwidth_GHz, cfg.awg_filter_order)
            uI_f, uQ_f = bwf.apply(uI_seed), bwf.apply(uQ_seed)
            prop = fr._get_propagator()
            _, _, _, leak, _ = fr.thermal_gate_fidelity_and_grad(uI_f, uQ_f, prop, cfg.dt)
            amps_out.append(float(amp)); leaks_out.append(float(leak))
            if verbose:
                print(f"  [leakage characterization] max_amp={amp:.5f} rad/ns -> nominal leak={leak:.4e}")
    finally:
        cfg.max_amp = old_amp
    order = np.argsort(amps_out)
    return {'amps': np.array(amps_out)[order], 'leaks': np.array(leaks_out)[order]}


def find_leakage_onset(amps, leaks, threshold=None, background_multiplier=5.0, verbose=True):
    """Locates the OPERATIONAL leakage-onset amplitude from a
    characterization curve by linear interpolation to where leakage first
    crosses `threshold`. This is the EMPIRICAL derivation fix (2) asked
    for: the onset is measured from this curve, never assumed to fall
    inside whatever max_amp range the main sweep happens to use.

    Called "operational" deliberately: this is a threshold-crossing
    convention applied to a characterization curve, not a uniquely defined
    physical phase boundary -- nature does not hand you a sharp universal
    leakage boundary, this function hands you a specific, reproducible,
    stated rule for locating where leakage becomes non-negligible GIVEN a
    threshold choice. See leakage_onset_robustness() below for checking
    that a qualitative conclusion (e.g. "R_tail rises near the onset")
    survives changing that choice, rather than being an artifact of it.

    threshold: absolute leakage value defining "onset". If None (default),
    set to background_multiplier * leaks[0] -- i.e. onset is defined as
    where leakage first exceeds `background_multiplier` times the leakage
    at the SMALLEST characterized amplitude (which is dominated by the
    thermal-population floor leak_bg rather than drive-induced leakage).
    background_multiplier=5.0 is an explicit, adjustable convention, not a
    universal constant -- it is returned and printed alongside the onset
    so it is never silently assumed downstream.
    """
    amps = np.asarray(amps, dtype=float)
    leaks = np.asarray(leaks, dtype=float)
    order = np.argsort(amps)
    amps, leaks = amps[order], leaks[order]
    if threshold is None:
        threshold = background_multiplier * leaks[0]

    above = leaks >= threshold
    if not np.any(above):
        if verbose:
            print(f"  [find_leakage_onset] leakage never reaches threshold={threshold:.4e} "
                  f"within the characterized grid (max leak={leaks.max():.4e} at "
                  f"max_amp={amps[-1]:.5f}) -- widen amp_grid to higher amplitudes.")
        return {'onset_amp': None, 'threshold': threshold, 'amps': amps, 'leaks': leaks}

    j = int(np.argmax(above))  # first index where leak >= threshold
    if j == 0:
        onset_amp = float(amps[0])
    else:
        a0, a1, l0, l1 = amps[j - 1], amps[j], leaks[j - 1], leaks[j]
        frac = (threshold - l0) / (l1 - l0) if l1 != l0 else 0.0
        onset_amp = float(a0 + frac * (a1 - a0))
    if verbose:
        print(f"  [find_leakage_onset] threshold={threshold:.4e} "
              f"(background_multiplier={background_multiplier}) -> onset_amp={onset_amp:.5f} rad/ns")
    return {'onset_amp': onset_amp, 'threshold': threshold, 'amps': amps, 'leaks': leaks}


def leakage_onset_robustness(amps, leaks, multipliers=(3.0, 5.0, 10.0), verbose=True):
    """Review fix (2)'s robustness check: re-derives the operational
    leakage-onset amplitude at several background_multiplier choices, all
    from the SAME already-computed characterization curve -- no re-running
    of any physics, let alone the optimization sweep. The point is not to
    pick a "best" multiplier; it is to show whether where R_tail starts
    rising is sensitive to this convention, or whether the qualitative
    picture (onset amplitudes for 3x/5x/10x background all landing in
    roughly the same place) is stable across a reasonable range of it.
    Returns {multiplier: onset_amp_or_None}."""
    out = {}
    for mult in multipliers:
        res = find_leakage_onset(amps, leaks, background_multiplier=mult, verbose=False)
        out[mult] = res['onset_amp']
    if verbose:
        summary = ", ".join(f"{mult}x -> {('%.5f' % a) if a is not None else 'not found'}"
                             for mult, a in out.items())
        print(f"  [leakage_onset_robustness] onset amplitude by background_multiplier: {summary}")
    return out


def _save_leakage_characterization_plot(char, onset, out_path, onset_robustness=None,
                                         optimized_points=None):
    """optimized_points: optional list of (amp, leak_final) from the ACTUAL
    optimized mean control at each sweep point (run_stage0_sweep's own
    mean_ctrl_leak_final column) -- overlaid against the cheap fixed-seed
    characterization curve so the two can be visually compared (review fix
    (3)). If they diverge substantially, don't take the seed-derived onset
    as necessarily the optimizer's own leakage onset."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(char['amps'], char['leaks'], 'o-', label='L_seed(A): fixed Hann-seed leakage', zorder=3)
    if optimized_points:
        pts = sorted(optimized_points, key=lambda p: p[0])
        opt_amps = [p[0] for p in pts]
        opt_leaks = [p[1] for p in pts]
        ax.plot(opt_amps, opt_leaks, 's--', color='tab:green',
                label='L_optimized(A): actual optimized mean-control leakage', zorder=3)
    ax.axhline(onset['threshold'], color='gray', linestyle='--', linewidth=1,
               label=f"primary threshold ({onset['threshold']:.2e})")
    if onset['onset_amp'] is not None:
        ax.axvline(onset['onset_amp'], color='r', linestyle='--', linewidth=1.2,
                    label=f"primary onset={onset['onset_amp']:.4f}")
    if onset_robustness:
        colors = ['tab:orange', 'tab:red', 'tab:purple', 'tab:brown']
        for (mult, a), c in zip(onset_robustness.items(), colors):
            if a is not None:
                ax.axvline(a, color=c, linestyle=':', linewidth=0.8, alpha=0.8,
                            label=f"{mult}x background -> {a:.4f}")
    ax.set_xlabel("cfg.max_amp (rad/ns)")
    ax.set_ylabel("leakage")
    ax.set_yscale('log')
    ax.set_title("Operational leakage-onset characterization\n(threshold convention, not a physical phase boundary)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================================
# 8. TOP-LEVEL DRIVER: the Stage 0 sweep
# ============================================================================

def run_stage0_sweep(max_amp_values, leak_lambda=None,
                      alpha_grid=(0.01, 0.025, 0.05, 0.10, 0.20),
                      reference_alpha=0.05,
                      K_train=None, n_val=None, n_test=None, n_restarts=None, n_boot=1000,
                      run_distributional_at=(0, -1), run_pareto_at=(-1,),
                      leakage_characterization_grid=None,
                      leakage_onset_threshold=None, leakage_onset_background_multiplier=5.0,
                      leakage_onset_multiplier_robustness=(3.0, 5.0, 10.0),
                      train_seed=77, val_seed=10_000, test_seed=20_000,
                      progress_every=25,
                      outdir="stage0_results", verbose=True):
    """max_amp_values: iterable of cfg.max_amp values to sweep, ordered
    from "far from the operational leakage-onset amplitude" (small) to
    "near it" (large) -- though per the module docstring's "THE SWEEP
    VARIABLE" section, that ordering is a design intent, not something
    this file assumes true without checking; the leakage-onset
    characterization below is what actually locates it. It is called
    "operational" throughout, not "the leakage boundary": it is a
    threshold-crossing convention on a measured curve (see
    find_leakage_onset), not a uniquely defined physical phase boundary.

    TRAIN / VALIDATION / TEST (review fix (1), the important one): three
    disjoint noise ensembles are used for three different jobs and never
    swapped. The TRAINING ensemble (K_train draws) is what every restart's
    gradient descent sees. The VALIDATION ensemble (val_model, n_val
    draws) is used ONLY inside stage0_multistart to pick which restart
    wins, then discarded. The TEST ensemble (test_model, n_test draws) is
    untouched by any selection decision and is the ONLY ensemble used for
    every reported number: R_tail, its bootstrap CI, the full loss
    distributions, the adversarial search, and the Pareto frontier. If
    validation and test were the same set, R_tail would be measuring
    "how well did we fit the very data we used to pick a winner", not a
    genuine held-out generalization gap.

    COMMON RANDOM NUMBERS ACROSS THE SWEEP (review fix (5)): all three
    ensembles are built ONCE, from train_seed/val_seed/test_seed, and
    reused UNCHANGED at every amplitude in max_amp_values (previously each
    sweep point got its own seed offset by its index). That means the only
    thing that changes from one sweep point to the next is the amplitude
    budget itself, not which noise realizations happen to be sampled --
    this removes Monte Carlo wiggle between neighboring points that has
    nothing to do with the physics being studied.

    leak_lambda: if given, overrides cfg.leak_lambda for the whole sweep
    (the paper's secondary leakage-margin knob); left at the Config
    default (7.0) if None.
    run_distributional_at / run_pareto_at: indices into max_amp_values at
    which to additionally run the (more expensive) distributional-plot and
    Pareto-frontier analyses; defaults to the first and last sweep points.
    leakage_characterization_grid: amp grid for characterize_leakage_vs_
    amplitude (cheap, no optimization -- fine by default). If None, uses
    25 points spanning [0.5, 1.3] x [min(max_amp_values), max(max_amp_values)]
    so the onset has a good chance of being bracketed even if it sits
    outside the main sweep's own range.
    leakage_onset_threshold / leakage_onset_background_multiplier: passed
    through to find_leakage_onset for the PRIMARY onset used to compute
    the 'm' column (see there for the exact convention).
    leakage_onset_multiplier_robustness: additional background_multiplier
    values (review fix (2)'s robustness check) re-derived from the SAME
    characterization curve at no extra simulation cost, stored as extra
    'm_x{mult}' columns so a plot of R_tail vs. any of them can confirm
    the qualitative divergence-near-onset picture isn't an artifact of
    picking 5x specifically.

    INTERPRETATION (review fix (6)/(7) -- applies to reading this
    function's OWN output, independent of anything in the paper's prose):
    R_tail(alpha) ~= 0 far from the onset is consistent with the paper's
    prediction; a R_tail that rises as m -> 0 is evidence FOR the proposed
    leakage-nonlinearity mechanism, not proof that "CVaR is better" --
    the target of Stage 0 is identifying WHEN risk-sensitive control
    becomes necessary, not declaring a winner between mean and CVaR
    optimization in general. Symmetrically, a flat R_tail across the whole
    sweep is evidence AGAINST the proposed mechanism under this particular
    physical/model setup and grounds for revisiting it -- it does not
    "falsify" the broader mathematical statement that nonlinear
    noise-to-loss maps can break mean/CVaR equivalence, which this one
    experiment does not have the scope to adjudicate either way.
    """
    from pathlib import Path
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    K_train = K_train or cfg.fade_K
    n_val = n_val or cfg.n_noise_eval
    n_test = n_test or cfg.n_noise_eval
    n_restarts = n_restarts or cfg.fade_n_restarts
    if leak_lambda is not None:
        cfg.leak_lambda = leak_lambda

    # reference_alpha (used below to pick which trained CVaR control feeds
    # the adversarial search / distributional plot / Pareto frontier) must
    # actually be one of the alphas being trained at each sweep point, or
    # u_cvar_by_alpha[reference_alpha] raises a bare KeyError deep in the
    # per-point loop -- e.g. calling run_stage0_sweep(..., alpha_grid=(0.2,))
    # with the reference_alpha=0.05 default. Fail clearly and fast instead,
    # picking the closest available alpha rather than crashing mid-sweep.
    if reference_alpha not in alpha_grid:
        closest = min(alpha_grid, key=lambda a: abs(a - reference_alpha))
        if verbose:
            print(f"  [run_stage0_sweep] reference_alpha={reference_alpha} is not in "
                  f"alpha_grid={alpha_grid}; using closest available alpha={closest} for "
                  f"adversarial_search/distributional-plot/Pareto reporting instead.")
        reference_alpha = closest

    # --- leakage-onset characterization, BEFORE any expensive optimization
    # (review fix (2): measure leakage vs. amplitude first, derive the
    # onset from that, only then interpret R_tail against distance to it) ---
    max_amp_values = list(max_amp_values)
    if leakage_characterization_grid is None:
        lo = 0.5 * min(max_amp_values)
        hi = 1.3 * max(max_amp_values)
        leakage_characterization_grid = np.linspace(lo, hi, 25)
    if verbose:
        print(f"\n{'='*70}\nLEAKAGE-ONSET CHARACTERIZATION (cheap, optimization-free, "
              f"{len(leakage_characterization_grid)} points)\n{'='*70}")
    char = characterize_leakage_vs_amplitude(leakage_characterization_grid, verbose=verbose)
    onset = find_leakage_onset(char['amps'], char['leaks'], threshold=leakage_onset_threshold,
                                background_multiplier=leakage_onset_background_multiplier,
                                verbose=verbose)
    onset_amp = onset['onset_amp']
    # Robustness check (review fix (2)): re-derive the onset at several
    # background_multiplier choices from the SAME curve -- free, no rerun.
    onset_robust = (leakage_onset_robustness(char['amps'], char['leaks'],
                                              multipliers=leakage_onset_multiplier_robustness,
                                              verbose=verbose)
                     if leakage_onset_threshold is None else {})
    np.savez(outdir / "leakage_characterization.npz", amps=char['amps'], leaks=char['leaks'],
             threshold=onset['threshold'], onset_amp=np.nan if onset_amp is None else onset_amp,
             robustness_multipliers=np.array(list(onset_robust.keys())),
             robustness_onsets=np.array([np.nan if a is None else a for a in onset_robust.values()]))
    if onset_amp is None and verbose:
        print("  R_tail will still be reported vs. max_amp, but the derived-distance (m) "
              "plot/column will be left blank until the characterization grid is widened.")

    # m_x{mult} field names for the robustness columns, computed once here
    # so both the CSV header and the per-row dict agree.
    m_robust_fields = [f"m_x{mult:g}" for mult in onset_robust.keys()]

    optimized_leak_points = []  # (amp, leak_final) from ACTUAL optimized mean controls,
                                 # filled in during the loop below -- fix (3)'s overlay
    _save_leakage_characterization_plot(char, onset, outdir / "leakage_characterization.png",
                                         onset_robustness=onset_robust,
                                         optimized_points=optimized_leak_points)

    # Checkpointing: every row and plot below is written to disk THE MOMENT
    # it is computed, not buffered until the whole sweep finishes. This
    # sweep is long-running and thermally heavy (dense per-draw Lindblad
    # gradients, multi-core), so on a fanless laptop it is entirely
    # reasonable to Ctrl+C partway through -- whatever is in `outdir` at
    # that point is real, complete, already-written data, not a scratch
    # buffer that gets discarded. Re-running from scratch after a partial
    # kill will overwrite stage0_summary.csv from row 1 (this file does not
    # resume mid-sweep), so save/rename outdir first if you want to keep a
    # partial run's rows.
    import csv
    summary_csv_path = outdir / "stage0_summary.csv"
    summary_fieldnames = (['max_amp', 'm'] + m_robust_fields +
                           ['leak_lambda', 'alpha', 'R_tail', 'ci_lo', 'ci_hi',
                            'mean_ctrl_leak_final', 'mean_ctrl_F_final'])
    with open(summary_csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=summary_fieldnames).writeheader()

    def _append_row_and_flush(row):
        with open(summary_csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=summary_fieldnames).writerow(row)

    summary_rows = []

    # --- ONE fixed training ensemble, ONE fixed validation ensemble, ONE
    # fixed test ensemble, shared by EVERY amplitude in the sweep (review
    # fixes (1) and (5)). Built ONCE here, not per sweep point: the only
    # thing that varies from point to point below is cfg.max_amp. Training
    # ensemble is used for all gradient descent; val_model is used ONLY
    # inside stage0_multistart to pick a winning restart, then never
    # touched again; test_model is untouched by any selection and is the
    # ONLY ensemble behind every reported number.
    train_rng = np.random.default_rng(train_seed)
    deltas = train_rng.normal(0.0, cfg.fade_sigma, K_train)
    gammas = train_rng.normal(0.0, cfg.fade_amp_sig, K_train)
    val_model = Stage0Ensemble(n_val, cfg.sigma_detuning, cfg.sigma_amp, seed=val_seed)
    test_model = Stage0Ensemble(n_test, cfg.sigma_detuning, cfg.sigma_amp, seed=test_seed)
    if verbose:
        print(f"\nCommon-random-numbers ensembles fixed for the WHOLE sweep: "
              f"train_seed={train_seed} (K={K_train}), val_seed={val_seed} (n={n_val}), "
              f"test_seed={test_seed} (n={n_test}).")

    for i, amp in enumerate(max_amp_values):
        cfg.max_amp = float(amp)
        if verbose:
            print(f"\n{'='*70}\nSWEEP POINT {i+1}/{len(max_amp_values)}: "
                  f"max_amp={amp:.5f} rad/ns  (leak_lambda={cfg.leak_lambda})\n{'='*70}")

        m_here = None if onset_amp is None else float(amp - onset_amp)
        m_robust_here = {f"m_x{mult:g}": (None if a is None else float(amp - a))
                          for mult, a in onset_robust.items()}

        seed_uI, seed_uQ = fr.hann_init()
        bwf_for_calib = fr.BandwidthFilter(cfg.Nt, cfg.dt, cfg.awg_bandwidth_GHz, cfg.awg_filter_order)
        calibrate_ff_mu(seed_uI, seed_uQ, bwf_for_calib, verbose=verbose)

        # --- mean-optimal control (FADE-RIPE reduction); restart selection
        # on val_model only --- (review fix (1))
        best_mean = stage0_multistart(seed_uI, seed_uQ, deltas, gammas, mode="mean",
                                       n_restarts=n_restarts, val_model=val_model,
                                       seed_base=1000 * i, verbose=verbose,
                                       progress_every=progress_every)
        u_mean = (best_mean['uI_opt'], best_mean['uQ_opt'])
        optimized_leak_points.append((float(amp), float(best_mean['info']['leak_final'])))

        # --- CVaR-optimal control at each alpha (VIPER reduction); restart
        # selection on val_model only, final numbers from test_model only ---
        u_cvar_by_alpha = {}
        r_tail_by_alpha = {}
        for alpha in alpha_grid:
            best_cvar = stage0_multistart(seed_uI, seed_uQ, deltas, gammas, mode="cvar",
                                           alpha=alpha, n_restarts=n_restarts,
                                           val_model=val_model, seed_base=2000 * i + int(1000 * alpha),
                                           verbose=verbose, progress_every=progress_every)
            u_cvar = (best_cvar['uI_opt'], best_cvar['uQ_opt'])
            u_cvar_by_alpha[alpha] = u_cvar
            boot = bootstrap_r_tail(test_model, u_mean, u_cvar, alpha, n_boot=n_boot, seed=5000 * i)
            r_tail_by_alpha[alpha] = boot
            if verbose:
                print(f"  R_tail(alpha={alpha}) = {boot['point']:.4f} "
                      f"[{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}] (95% bootstrap CI, test ensemble)")
            row = {'max_amp': amp, 'm': m_here, **m_robust_here,
                   'leak_lambda': cfg.leak_lambda, 'alpha': alpha,
                   'R_tail': boot['point'], 'ci_lo': boot['ci_lo'], 'ci_hi': boot['ci_hi'],
                   'mean_ctrl_leak_final': best_mean['info']['leak_final'],
                   'mean_ctrl_F_final': best_mean['info']['F_final']}
            summary_rows.append(row)
            _append_row_and_flush(row)  # written immediately -- safe to Ctrl+C after this point

        # --- adversarial search across all controls obtained at this point
        # (test_model only -- review fix (1)) ---
        controls_here = {'u_mean': u_mean}
        controls_here.update({f'u_cvar[{a}]': u_cvar_by_alpha[a] for a in alpha_grid})
        adversarial_search(test_model, controls_here, reference_alpha, verbose=verbose)

        # --- optional distributional plot (test_model only) ---
        if i in run_distributional_at or (i - len(max_amp_values)) in run_distributional_at:
            save_loss_distribution_plot(
                test_model, u_mean, u_cvar_by_alpha[reference_alpha], reference_alpha,
                outdir / f"loss_distribution_point{i}.png")

        # --- optional Pareto frontier (val_model selects, test_model reports) ---
        if i in run_pareto_at or (i - len(max_amp_values)) in run_pareto_at:
            pts = pareto_frontier(seed_uI, seed_uQ, deltas, gammas, val_model, test_model,
                                   reference_alpha, n_restarts=n_restarts, verbose=verbose,
                                   progress_every=progress_every)
            save_pareto_plot(pts, reference_alpha, outdir / f"pareto_point{i}.png")

        np.savez(outdir / f"controls_point{i}.npz",
                 max_amp=amp, u_mean_uI=u_mean[0], u_mean_uQ=u_mean[1],
                 **{f"u_cvar_{a}_uI": u_cvar_by_alpha[a][0] for a in alpha_grid},
                 **{f"u_cvar_{a}_uQ": u_cvar_by_alpha[a][1] for a in alpha_grid})

        # Regenerate the summary plot after EVERY sweep point (not just at
        # the end) so outdir always has an up-to-date picture of whatever
        # has completed so far -- cheap (matplotlib on <=len(summary_rows)
        # points), and means a kill signal never loses the plot either.
        try:
            _save_summary_plot(summary_rows, alpha_grid, outdir / "R_tail_vs_max_amp.png",
                                x_field='max_amp', x_label="cfg.max_amp (rad/ns)")
        except Exception as e:  # e.g. too few points yet for a sensible plot
            if verbose:
                print(f"  (summary plot vs max_amp not updated yet: {e})")
        if onset_amp is not None:
            try:
                _save_summary_plot(summary_rows, alpha_grid, outdir / "R_tail_vs_m.png",
                                    x_field='m', x_label=f"m = max_amp - onset_amp ({onset_amp:.5f} rad/ns)")
            except Exception as e:
                if verbose:
                    print(f"  (summary plot vs m not updated yet: {e})")
        # Robustness plots (review fix (2)): same R_tail values, just
        # re-indexed against onset estimates from other threshold
        # multipliers -- confirms (or not) that the qualitative picture
        # doesn't depend on having picked leakage_onset_background_multiplier.
        for mult, a in onset_robust.items():
            if a is None:
                continue
            field = f"m_x{mult:g}"
            try:
                _save_summary_plot(summary_rows, alpha_grid, outdir / f"R_tail_vs_{field}.png",
                                    x_field=field, x_label=f"{field} = max_amp - onset_amp({mult:g}x) ({a:.5f} rad/ns)")
            except Exception as e:
                if verbose:
                    print(f"  (summary plot vs {field} not updated yet: {e})")
        # Re-save the leakage-characterization plot with the growing
        # optimized-mean-control overlay (review fix (3)).
        try:
            _save_leakage_characterization_plot(char, onset, outdir / "leakage_characterization.png",
                                                 onset_robustness=onset_robust,
                                                 optimized_points=optimized_leak_points)
        except Exception as e:
            if verbose:
                print(f"  (leakage characterization plot not updated yet: {e})")
        if verbose:
            print(f"  [checkpoint] {len(summary_rows)} row(s) written to {summary_csv_path}")

    if verbose:
        print(f"\nStage 0 sweep complete. Summary written to {summary_csv_path}")
    return summary_rows


def _save_summary_plot(summary_rows, alpha_grid, out_path, x_field='max_amp', x_label=None):
    """x_field: 'max_amp' (raw sweep variable, always available) or 'm'
    (derived distance to the empirically-located leakage onset -- see
    find_leakage_onset; only meaningful once an onset was found)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    xs_all = sorted(set(row[x_field] for row in summary_rows if row.get(x_field) is not None))
    for alpha in alpha_grid:
        xs, ys, los, his = [], [], [], []
        for x in xs_all:
            row = next((r for r in summary_rows if r[x_field] == x and r['alpha'] == alpha), None)
            if row is None:
                continue
            xs.append(x); ys.append(row['R_tail']); los.append(row['ci_lo']); his.append(row['ci_hi'])
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                    marker='o', capsize=3, label=f"alpha={alpha}")
    ax.set_xlabel(x_label or x_field)
    ax.set_ylabel("R_tail")
    ax.axhline(0.0, color='k', linewidth=0.5)
    if x_field == 'm':
        ax.axvline(0.0, color='r', linewidth=0.8, linestyle='--', label="leakage onset (m=0)")
    ax.set_title(f"Stage 0: R_tail vs. {x_field}, by alpha")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ============================================================================
# 9. ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # SMOKE_TEST=True (default): a few minutes, small K/restarts/n_boot,
    # meant to verify the whole pipeline runs end-to-end and produces
    # sane-looking output before committing to the full run below.
    # Set SMOKE_TEST=False (and adjust the sizes) for the actual Stage 0
    # result reported in the paper: K_train=cfg.fade_K (128), n_val=n_test=
    # cfg.n_noise_eval (300 each -- disjoint, review fix (1)),
    # n_restarts=cfg.fade_n_restarts (8), n_boot>=1000, and a finer
    # max_amp grid.
    # ------------------------------------------------------------------
    SMOKE_TEST = True

    print("Running self-tests (paper Section 3.10 validation plan) ...")
    ok1 = selftest_matches_fade_ripe()
    ok2 = selftest_finite_difference(mode="mean")
    ok3 = selftest_finite_difference(mode="cvar", alpha=0.2)  # larger alpha = larger,
                                                               # more finite-difference-
                                                               # friendly tail set
    ok4 = selftest_fractional_cvar_matches_definition()
    if not (ok1 and ok2 and ok3 and ok4):
        print("\nSelf-test FAILURE -- stopping before running the sweep. "
              "Do not trust any Stage 0 divergence result until these pass.")
        raise SystemExit(1)
    print("Self-tests passed.\n")

    if SMOKE_TEST:
        base_amp = cfg.max_amp
        max_amp_grid = np.linspace(0.6 * base_amp, 1.8 * base_amp, 4)
        summary = run_stage0_sweep(
            max_amp_grid,
            alpha_grid=(0.05, 0.20),
            K_train=16, n_val=40, n_test=40, n_restarts=2, n_boot=200,
            outdir="stage0_results_smoketest",
        )
    else:
        base_amp = cfg.max_amp
        max_amp_grid = np.linspace(0.5 * base_amp, 2.5 * base_amp, 9)
        summary = run_stage0_sweep(
            max_amp_grid,
            alpha_grid=(0.01, 0.025, 0.05, 0.10, 0.20),
            K_train=cfg.fade_K, n_val=cfg.n_noise_eval, n_test=cfg.n_noise_eval,
            n_restarts=cfg.fade_n_restarts, n_boot=1000,
            outdir="stage0_results",
        )

    fr._shutdown_pool()