"""
stage7_ibm.py -- the part that actually talks to hardware.

stage7.py stops at `circuit_spec`, which returns a dict that nothing consumes.
This file closes that gap: it builds real circuits, submits them to IBM Quantum
through Qiskit Runtime with fractional gates enabled, pulls the counts back, and
turns them into the epsilon vectors the predictor needs.

STRUCTURE, and why it is split this way
---------------------------------------
Everything is written against a vendor-neutral gate list (the "IR").  Three
things consume it:

  to_qiskit(ir)    -> a QuantumCircuit, for the real run
  simulate_ir(ir)  -> exact 2x2 evolution, no Qiskit needed
  sample_ir(ir)    -> simulate_ir plus binomial sampling and readout error

The middle one exists because the risky part of this file is NOT the Runtime API
calls, which are mechanical and documented.  It is whether the circuit SEMANTICS
match the model in stage7.py -- gate order, the sign convention of the Y-basis
change, whether prep |+> lands on +x, whether amplification composes the way the
analytic model says.  A sign error in the Y basis flips one component of
d_alpha, the predicted step points somewhere else, and the experiment returns a
confident null.  So `selftest` runs the IR through simulate_ir and checks the
reconstructed epsilon against CompositeX.eps to 1e-12, closing the loop without
a QPU and without Qiskit installed.

WHAT IS NOT TESTED HERE
-----------------------
The author could not install Qiskit in the environment where this was written,
so to_qiskit(), the transpiler guard and the Runtime submission path are
UNEXECUTED code.  They are written from the documented API and should be run
first with --dry-run (which uses Aer if present) and then against a single
calibration job, before any science window is spent.  Treat the first real
invocation as a test of this file, not of the physics.

USAGE
    python3 stage7_ibm.py selftest                 # IR semantics, no Qiskit
    python3 stage7_ibm.py dryrun                   # full pipeline, local sampling
    python3 stage7_ibm.py calibrate --backend ibm_torino
    python3 stage7_ibm.py run --backend ibm_torino --arm mean
    python3 stage7_ibm.py analyse --raw raw_*.json
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stage7 import (CompositeX, W_METRIC, cvar, predict, reachable_projector,
                    offset_control, eps_from_counts, bloch_rotation,
                    OPERATING_POINT, PREPS, MEAS, CIRCUITS_PER_DRAW)

PHI = "phi"          # the free parameter name


# ==========================================================================
# 1. The IR
# ==========================================================================
def build_ir(model, c, prep, meas):
    """Gate list for one (prep, measurement-basis) circuit, phi left free.

    Layout, in application order:

        prep   |0> : nothing              |+> : h
        body   reps x [ for i in 1..n :  rz(phi/n) ; rx(theta_i) ; rz(psi_i) ]
        basis  X : h        Y : sdg, h        Z : nothing
        measure

    The phi kick goes FIRST inside each segment, matching CompositeX.unitary
    where the segment is RZ(psi) RX(theta) RZ(phi/n) read right to left.  Get
    this order wrong and the model and the device compute different things while
    both look reasonable.

    No target inverse appears: reps is even and the target is X, so X^reps = I
    and the ideal circuit returns the prep state.  That also removes one gate's
    worth of error from every circuit.
    """
    th, ps = model.split(c)
    ir = []
    if prep == "+":
        ir.append(("h",))
    elif prep == "+i":
        ir += [("h",), ("s",)]
    for _ in range(model.reps):
        for i in range(model.n):
            ir.append(("rz_phi", 1.0 / model.n))     # angle = coeff * phi
            ir.append(("rx", float(th[i])))
            ir.append(("rz", float(ps[i])))
    if model.reps % 2 == 1:
        ir.append(("x",))                            # undo the odd target
    if meas == "X":
        ir.append(("h",))
    elif meas == "Y":
        ir += [("sdg",), ("h",)]
    ir.append(("measure",))
    return ir


_G = {
    "h": np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0),
    "s": np.array([[1, 0], [0, 1j]], dtype=complex),
    "sdg": np.array([[1, 0], [0, -1j]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
}


def _rx(t):
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _rz(p):
    return np.array([[np.exp(-0.5j * p), 0], [0, np.exp(0.5j * p)]], dtype=complex)


def simulate_ir(ir, phi):
    """Exact P(1) for one IR circuit at a given phi.  No Qiskit, no sampling."""
    psi = np.array([1.0, 0.0], dtype=complex)
    for g in ir:
        if g[0] == "measure":
            break
        if g[0] == "rx":
            U = _rx(g[1])
        elif g[0] == "rz":
            U = _rz(g[1])
        elif g[0] == "rz_phi":
            U = _rz(g[1] * phi)
        else:
            U = _G[g[0]]
        psi = U @ psi
    return float(abs(psi[1]) ** 2)


def sample_ir(ir, phi, shots, rng, p_ro=0.0):
    p1 = simulate_ir(ir, phi)
    p1 = p1 * (1.0 - p_ro) + (1.0 - p1) * p_ro
    n1 = int(rng.binomial(shots, min(max(p1, 0.0), 1.0)))
    return {"0": shots - n1, "1": n1}


# ==========================================================================
# 2. counts -> epsilon
# ==========================================================================
def expval_z(counts):
    n0, n1 = counts.get("0", 0), counts.get("1", 0)
    tot = n0 + n1
    return 0.0 if tot == 0 else (n0 - n1) / tot


def eps_from_arm_counts(counts_by_key, p_ro=0.0):
    den = 1.0 - 2.0 * p_ro
    v = {}

    for prep in PREPS:
        vals = []
        for m in MEAS:
            key = (prep, m)

            # Support both tuple keys and saved JSON string keys.
            if key in counts_by_key:
                counts = counts_by_key[key]
            elif f"{prep}|{m}" in counts_by_key:
                counts = counts_by_key[f"{prep}|{m}"]
            else:
                raise KeyError(
                    f"Missing measurement key {key} / {prep}|{m}. "
                    f"Available keys: {list(counts_by_key.keys())}"
                )

            vals.append(expval_z(counts) / den)

        v[prep] = np.array(vals)

    # keep the remainder of your existing function unchanged
    return eps_from_counts(v["0"], v["+"])


# ==========================================================================
# 3. Qiskit construction (UNEXECUTED in the authoring environment)
# ==========================================================================
def to_qiskit(ir, phi_param=None):
    """Build a QuantumCircuit from the IR, with phi as a free Parameter."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    if phi_param is None:
        phi_param = Parameter(PHI)
    qc = QuantumCircuit(1, 1)
    for g in ir:
        if g[0] == "measure":
            qc.measure(0, 0)
        elif g[0] == "rx":
            qc.rx(g[1], 0)
        elif g[0] == "rz":
            qc.rz(g[1], 0)
        elif g[0] == "rz_phi":
            qc.rz(g[1] * phi_param, 0)
        elif g[0] == "h":
            qc.h(0)
        elif g[0] == "s":
            qc.s(0)
        elif g[0] == "sdg":
            qc.sdg(0)
        elif g[0] == "x":
            qc.x(0)
        else:
            raise ValueError(f"unknown IR op {g[0]}")
    return qc, phi_param


def transpile_guarded(qc, backend, n_rx_expected, opt_level=1):
    """Transpile, then REFUSE to continue if the control family did not survive.

    This is calibration check C2 promoted to an assertion.  With
    use_fractional_gates=True the target contains RX(theta) and the preset pass
    manager should leave the body alone.  Without it -- wrong backend flag, a
    device that does not support fractional gates, an optimisation level that
    resynthesises -- every rx(theta_i) is rebuilt out of sx and rz.  The circuit
    still runs, the numbers still come back, and they describe a DIFFERENT
    control family from the one the offline optimisation chose.  That failure is
    silent and would invalidate the whole window, so it is checked rather than
    hoped for.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    isa = pm.run(qc)
    ops = isa.count_ops()
    n_rx = int(ops.get("rx", 0))
    if n_rx != n_rx_expected:
        raise RuntimeError(
            f"fractional gates did not survive transpilation: expected "
            f"{n_rx_expected} rx ops, found {n_rx}. ops={dict(ops)}. "
            f"Load the backend with use_fractional_gates=True, and check the "
            f"device supports them. Do NOT spend the window on this circuit.")
    return isa


def get_backend(name, instance=None):
    """Load a backend with fractional gates enabled."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    try:
        service = (
            QiskitRuntimeService(instance=instance)
            if instance
            else QiskitRuntimeService()
        )   
    except Exception as exc:                            # noqa: BLE001
        # Surface the ROOT cause.  qiskit-ibm-runtime re-raises the IAM failure
        # twice, and the outermost message always says "check that you are using
        # a valid API token" -- which is wrong whenever the real problem is the
        # ACCOUNT rather than the key, and sends you off regenerating tokens
        # that were never the issue.
        root, chain = exc, []
        while root is not None:
            chain.append(f"{type(root).__name__}: {root}")
            root = root.__cause__ or root.__context__
        deepest = chain[-1]
        hint = ("Save it once from a python shell:\n"
                "  from qiskit_ibm_runtime import QiskitRuntimeService\n"
                "  QiskitRuntimeService.save_account(token=..., instance=..., "
                "set_as_default=True, overwrite=True)")
        if "blocked" in deepest.lower():
            hint = ("This is an IBM CLOUD ACCOUNT state, not a bad key -- IAM "
                    "is refusing to issue a token at all, so regenerating the "
                    "API key will not help. Sign in at cloud.ibm.com in a "
                    "browser with the same identity: if that is blocked too, "
                    "the account needs verifying or unblocking with IBM "
                    "support. Only once the web console works is it worth "
                    "making a new API key.")
        raise RuntimeError("could not load a saved IBM account.\n  root cause: "
                           + deepest + "\n  " + hint) from exc
    # fractional gates are opt-in: without the flag RX(theta) and RZZ(theta) do
    # not appear in the target at all
    be = service.backend(name, use_fractional_gates=True)
    tgt = getattr(be, "target", None)
    has_rx = bool(tgt is not None and "rx" in tgt.operation_names)
    print(f"  backend {be.name}: {be.num_qubits} qubits, "
          f"rx in target = {has_rx}")
    if not has_rx:
        raise RuntimeError(
            f"{be.name} has no continuous RX in its target, so the composite "
            f"control family cannot be executed on it. Pick a Heron-class "
            f"device that supports fractional gates.")
    return be


def _counts_list(pub_result, n_sets):
    """Pull per-parameter-set counts out of a SamplerV2 result, defensively.

    BUG THIS REPLACES: `arr.get_counts()` called with NO location argument does
    not return one dict per broadcast set -- it flattens the entire shape and
    merges every set's shots into a single aggregated dict. The old code took
    that merged dict, wrapped it as `[got]` (length 1), and the caller's
    `enumerate(...)` loop then only ever wrote index 0 -- so 7 of every 8
    records in a chunk silently kept EMPTY counts, and record 0 got a wrong,
    shots-times-n_sets-inflated count. The `len(out) != n_sets` guard below
    never caught this because the merged-dict branch bypassed it entirely.
    This is exactly the corruption `recover_cvar.py` had to repair by
    re-fetching every job and calling `arr.get_counts(local_i)` per index --
    the correct API, now used here too.

    The shape of `data.<creg>` depends on the PUB's broadcast shape and has
    changed across Runtime versions, so handle both the shaped and the flat
    case rather than assuming one.
    """
    data = pub_result.data
    creg = next(iter(data.__dict__)) if hasattr(data, "__dict__") else "c"
    arr = getattr(data, creg)
    shape = getattr(arr, "shape", None)
    if shape in (None, ()):
        # No broadcast dimension at all: a single circuit, a single count dict.
        if n_sets != 1:
            raise RuntimeError(
                f"expected {n_sets} count sets, got an unshaped result "
                f"(shape={shape}). This PUB was not broadcast over phi as "
                f"expected -- check the chunk size passed to run_arm_ibm."
            )
        return [arr.get_counts()]
    if tuple(shape) != (n_sets,):
        raise RuntimeError(
            f"expected shape ({n_sets},), got {shape}. The broadcast result "
            f"does not match the chunk size submitted -- do not trust this "
            f"job's counts."
        )
    # One get_counts(i) call per broadcast index. This is the only call that
    # actually returns per-set data; get_counts() with no argument merges them.
    return [arr.get_counts(i) for i in range(n_sets)]


# ==========================================================================
# 4. Running an arm
# ==========================================================================
def arm_controls(op=None):
    """The three frozen controls.  Loaded from disk, never re-optimised here."""
    op = op or OPERATING_POINT
    d = Path(__file__).resolve().parent
    cm = np.load(d / "stage7_c_mean.npy")
    cc = np.load(d / "stage7_c_cvar.npy")
    m1 = CompositeX(op["n"], reps=1)
    mR = CompositeX(op["n"], reps=op["reps"])
    # E2's offset control comes from the MEASURED d_alpha at run time; this is
    # only the offline prediction of it, kept for comparison
    pt = np.random.default_rng(99).normal(0.0, op["sigma"], 30000)
    E, L = m1.eps_batch(cm, pt), m1.loss_batch(cm, pt)
    P_B, _, _ = reachable_projector(m1.jacobian(cm, 0.0))
    pr = predict(E, L, op["alpha"], P_B=P_B)
    c_off, _ = offset_control(m1, cm, pr["step"], 0.0)
    return {"mean": cm, "cvar": cc, "offset_offline": c_off,
            "hard": mR.c_hard()}, pr


def phi_draws(op=None, seed=20260926):
    """The SAME draws for every arm.  Paired comparison: the ensemble draw
    cancels and the error bar sits on the difference, not on two independent
    Monte Carlo samples.  The seed is frozen in the pre-registration."""
    op = op or OPERATING_POINT
    return np.random.default_rng(seed).normal(0.0, op["sigma"], int(op["K"]))


def run_arm_local(model, c, phis, shots, p_ro=0.0, seed=0):
    """Full pipeline with local sampling -- the dry run."""
    rng = np.random.default_rng(seed)
    raw = []
    for k, ph in enumerate(phis):
        cb = {}
        for prep in PREPS:
            for meas in MEAS:
                ir = build_ir(model, c, prep, meas)
                cb[(prep, meas)] = sample_ir(ir, float(ph), shots, rng, p_ro)
        raw.append({"phi": float(ph),
                    "counts": {f"{p}|{m}": v for (p, m), v in cb.items()}})
    return raw


def run_arm_ibm(model, c, phis, shots, backend, max_sets=None, tag=""):
    """Submit one arm to IBM Quantum.

    Six PARAMETERISED circuits per arm, each broadcast over the K phi values,
    rather than 6*K separate circuits.  That is the difference between
    transpiling six circuits and transpiling 1536 of them, and it keeps the job
    count low enough that the 10-minute window is spent on shots rather than on
    per-job overhead.
    """
    from qiskit_ibm_runtime import SamplerV2, Batch
    n_rx_expected = model.n * model.reps
    isa, params = {}, {}
    for prep in PREPS:
        for meas in MEAS:
            qc, p = to_qiskit(build_ir(model, c, prep, meas))
            isa[(prep, meas)] = transpile_guarded(qc, backend, n_rx_expected)
            params[(prep, meas)] = p
    vals = np.asarray(phis, float).reshape(-1, 1)
    chunks = [vals] if not max_sets else [vals[i:i + max_sets]
                                          for i in range(0, len(vals), max_sets)]
    raw = [{"phi": float(p), "counts": {}} for p in phis]
    usage = []
    off = 0
    for ch in chunks:
        pubs = [(isa[(p, m)], ch) for p in PREPS for m in MEAS]
        with Batch(backend=backend):
            sampler = SamplerV2()
            sampler.options.default_shots = int(shots)
            job = sampler.run(pubs)
            print(f"  [{tag}] job {job.job_id()} submitted, "
                  f"{len(pubs)} pubs x {len(ch)} sets x {shots} shots", flush=True)
            res = job.result()
        try:
            usage.append(job.usage())
        except Exception:                              # noqa: BLE001
            usage.append(None)
        for j, (p, m) in enumerate([(p, m) for p in PREPS for m in MEAS]):
            for i, cts in enumerate(_counts_list(res[j], len(ch))):
                raw[off + i]["counts"][f"{p}|{m}"] = {k: int(v) for k, v in cts.items()}
        off += len(ch)
    return raw, usage


def _validate_raw(raw, shots, tag=""):
    """Structural check, run immediately after `run`, not just at `analyse`.

    The old corruption (empty counts / merged counts, see _counts_list above)
    was only discovered by a KeyError deep inside `analyse`, after the window
    was already spent. This catches the same failure mode the moment the job
    finishes, while there is still time left to resubmit.
    """
    expected = {f"{p}|{m}" for p in PREPS for m in MEAS}
    bad = []
    for i, rec in enumerate(raw):
        keys = set(rec["counts"].keys())
        if keys != expected:
            bad.append(f"record {i}: keys={sorted(keys)}")
            continue
        for key, counts in rec["counts"].items():
            total = sum(counts.values())
            if total != shots:
                bad.append(f"record {i} {key}: {total} shots, expected {shots}")
    print(f"  [{tag}] structural check: {len(raw)} records, "
          f"{'PASS' if not bad else f'FAIL ({len(bad)} problems)'}")
    if bad:
        for b in bad[:10]:
            print(f"    {b}")
        if len(bad) > 10:
            print(f"    ... and {len(bad) - 10} more")
        print("  DO NOT trust this raw file for analysis -- re-fetch or "
              "resubmit before spending more window time.")
    return not bad


def analyse_arm(raw, shots, alpha, p_ro=0.0, debias=True):
    """counts -> epsilon -> loss -> CVaR.  The analysis is frozen: the debias
    term and the CVaR convention are fixed in the pre-registration, so this
    function has no tunable knobs by design."""
    E, phis = [], []
    for rec in raw:
        cb = {}
        for key, v in rec["counts"].items():
            p, m = key.split("|")
            cb[(p, m)] = v
        E.append(eps_from_arm_counts(cb, p_ro))
        phis.append(rec["phi"])
    E = np.array(E, float)
    L = np.einsum("ki,ij,kj->k", E, W_METRIC, E)
    if debias:
        # E|eps_hat|^2 = |eps|^2 + shot variance; the bias is equal in both arms
        # but does NOT cancel in a ratio, so remove it before the CVaR
        L = np.maximum(L - (3.0 / shots) / 6.0, 0.0)
    return {"eps": E.tolist(), "phi": phis, "L": L.tolist(),
            "mean_L": float(L.mean()), "cvar": float(cvar(L, alpha))}


def compare(a_mean, a_other, alpha):
    R = 1.0 - a_other["cvar"] / max(a_mean["cvar"], 1e-300)
    ratio = a_other["mean_L"] / max(a_mean["mean_L"], 1e-300)
    return {"R": float(R), "mean_loss_ratio": float(ratio),
            "tail_specific": bool(ratio > 1.0),
            "cvar_mean_arm": a_mean["cvar"], "cvar_other_arm": a_other["cvar"],
            "verdict": ("genuine mean-for-tail trade" if ratio > 1.0 else
                        "mean arm not converged -- gain is baseline slack, "
                        "not a tail trade")}


# ==========================================================================
# 5. Calibration, as circuits rather than prose
# ==========================================================================
def calibration_jobs(op=None):
    """C1-C5 as real IR circuits with numeric acceptance thresholds."""
    op = op or OPERATING_POINT
    n = op["n"]
    jobs = []
    jobs.append({"id": "C1", "what": "RX(pi) is a pi pulse", "shots": 1024,
                 "irs": [[("rx", math.pi), ("measure",)]],
                 "expect": "P(1) >= 0.97", "check": lambda p: p[0] >= 0.97})
    ths = [k * math.pi / 12 for k in range(13)]
    jobs.append({"id": "C2", "what": "fractional RX(theta) follows sin^2(theta/2)",
                 "shots": 512,
                 "irs": [[("rx", t), ("measure",)] for t in ths],
                 "expect": "max |P(1) - sin^2(theta/2)| <= 0.05",
                 "check": lambda p: max(abs(pi - math.sin(t / 2) ** 2)
                                        for pi, t in zip(p, ths)) <= 0.05,
                 "dev": lambda p: max(abs(pi - math.sin(t / 2) ** 2)
                                      for pi, t in zip(p, ths))})
    psis = [k * math.pi / 6 for k in range(13)]
    jobs.append({"id": "C3a", "what": "RZ is a frame change (cos curve)",
                 "shots": 512,
                 "irs": [[("h",), ("rz", ps), ("h",), ("measure",)] for ps in psis],
                 "expect": "P(1) tracks (1-cos psi)/2 to 0.05",
                 "check": lambda p: max(abs(pi - (1 - math.cos(ps)) / 2)
                                        for pi, ps in zip(p, psis)) <= 0.05,
                 "dev": lambda p: max(abs(pi - (1 - math.cos(ps)) / 2)
                                      for pi, ps in zip(p, psis))})
    jobs.append({"id": "C3b", "what": "200 RZs with zero net angle cost nothing",
                 "shots": 1024,
                 "irs": [[("rz", 2 * math.pi / 200)] * 200 + [("measure",)]],
                 "expect": "P(1) <= 0.03",
                 "check": lambda p: p[0] <= 0.03})
    jobs.append({"id": "C4", "what": "six-circuit Bloch reconstruction on a known gate",
                 "shots": 1024, "special": "bloch",
                 "expect": "|eps| <= 4 sigma_shot at phi = 0",
                 "check": None})
    jobs.append({"id": "C5", "what": "throughput", "shots": 256, "special": "usage",
                 "irs": [[("rx", math.pi), ("measure",)]] * 10,
                 "expect": "record job.usage(); pass as --throughput",
                 "check": None})
    return jobs


def run_calibration(backend=None, p_ro=0.0, seed=0, op=None):
    """Run C1-C5 either on hardware or locally, and PASS/FAIL each numerically."""
    op = op or OPERATING_POINT
    rng = np.random.default_rng(seed)
    out = []
    for j in calibration_jobs(op):
        if j.get("special") == "bloch":
            m = CompositeX(op["n"], reps=op["reps"])
            c = np.load(Path(__file__).resolve().parent / "stage7_c_mean.npy")
            cb = {}
            for prep in PREPS:
                for meas in MEAS:
                    ir = build_ir(m, c, prep, meas)
                    if backend is None:
                        cb[(prep, meas)] = sample_ir(ir, 0.0, j["shots"], rng, p_ro)
                    else:
                        raise NotImplementedError("run C4 through `run --arm mean`")
            e = eps_from_arm_counts(cb, p_ro)
            truth = m.eps(c, 0.0)
            tol = 4.0 / math.sqrt(j["shots"])
            ok = bool(np.linalg.norm(e - truth) <= tol)
            out.append({"id": j["id"], "what": j["what"], "ok": ok,
                        "detail": f"|eps_meas - eps_true| = "
                                  f"{np.linalg.norm(e - truth):.4f} vs tol {tol:.4f}"})
            continue
        if j.get("special") == "usage" and backend is None:
            out.append({"id": j["id"], "what": j["what"], "ok": None,
                        "detail": "needs hardware; run with --backend"})
            continue
        ps = []
        for ir in j["irs"]:
            if backend is None:
                cts = sample_ir(ir, 0.0, j["shots"], rng, p_ro)
                ps.append(cts["1"] / j["shots"])
            else:
                raise NotImplementedError("hardware calibration: submit via run_arm_ibm")
        ok = bool(j["check"](ps)) if j["check"] else None
        # report the DEVIATION the check actually tests, not the raw
        # probabilities -- printing P(1) beside a tolerance on |P(1) - model|
        # reads like a failure even when the check passed
        dev = j.get("dev")
        detail = j["expect"] + "; "
        detail += (f"worst deviation {dev(ps):.4f}" if dev else
                   "got " + ", ".join(f"{x:.3f}" for x in ps[:6])
                   + ("..." if len(ps) > 6 else ""))
        out.append({"id": j["id"], "what": j["what"], "ok": ok, "detail": detail})
    return out


# ==========================================================================
# 6. Selftests -- the IR must compute what CompositeX says it computes
# ==========================================================================
def _h1_ir_matches_model():
    """The circuit and the model must agree on epsilon, exactly.

    This is the check that matters.  Everything downstream -- d_alpha, the
    predicted step, the offset control -- is built on the assumption that what
    the device measures is the epsilon of CompositeX.  Gate order, the Y-basis
    sign, prep |+> landing on +x, amplification composing correctly: any one of
    them wrong gives a confident null instead of an error.
    """
    op = OPERATING_POINT
    worst = 0.0
    rng = np.random.default_rng(0)
    for reps in (1, 2, 16):
        m = CompositeX(op["n"], reps=reps)
        for _ in range(6):
            c = m.c0() + 0.3 * rng.standard_normal(m.dim)
            phi = float(rng.normal(0, 0.6))
            cb = {}
            for prep in PREPS:
                for meas in MEAS:
                    ir = build_ir(m, c, prep, meas)
                    p1 = simulate_ir(ir, phi)
                    cb[(prep, meas)] = {"0": 1.0 - p1, "1": p1}
            e_meas = eps_from_arm_counts(cb)
            e_true = m.eps(c, phi)
            worst = max(worst, float(np.linalg.norm(e_meas - e_true)))
    return worst < 1e-10, f"worst |eps_circuit - eps_model| = {worst:.2e}"


def _h2_basis_signs():
    """Each basis change must return the matching Pauli expectation, with sign.

    Tested against bloch_rotation directly, because a Y-basis sign error is the
    single most likely silent bug here and it survives every check that only
    looks at magnitudes.
    """
    m = CompositeX(3, reps=2)
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(8):
        c = m.c0() + 0.4 * rng.standard_normal(6)
        phi = float(rng.normal(0, 0.8))
        R = bloch_rotation(m.error(c, phi))
        for prep, vec in (("0", np.array([0, 0, 1.0])), ("+", np.array([1.0, 0, 0]))):
            want = R @ vec
            got = np.array([1.0 - 2.0 * simulate_ir(build_ir(m, c, prep, mm), phi)
                            for mm in MEAS])
            worst = max(worst, float(np.max(np.abs(got - want))))
    return worst < 1e-10, f"worst |<P>_circuit - (R.v)| = {worst:.2e}"


def _h3_phi_enters_once_per_segment():
    """phi must appear exactly n*reps times, with coefficient 1/n.

    If it appeared once per circuit instead, the model would be a single
    pre-gate frame error rather than a quasi-static detuning, which is different
    physics with the same shape.
    """
    m = CompositeX(3, reps=16)
    ir = build_ir(m, m.c0(), "0", "Z")
    kicks = [g for g in ir if g[0] == "rz_phi"]
    coeff_ok = all(abs(g[1] - 1.0 / m.n) < 1e-15 for g in kicks)
    return (len(kicks) == m.n * m.reps and coeff_ok), \
        f"{len(kicks)} kicks (want {m.n * m.reps}), coeff ok = {coeff_ok}"


def _h4_rx_count_matches_guard():
    """The transpiler guard's expected rx count must match what build_ir emits,
    or the guard fires on every good circuit and never on a bad one."""
    m = CompositeX(3, reps=16)
    bad = []
    for prep in PREPS:
        for meas in MEAS:
            ir = build_ir(m, m.c0(), prep, meas)
            n_rx = sum(1 for g in ir if g[0] == "rx")
            if n_rx != m.n * m.reps:
                bad.append((prep, meas, n_rx))
    return not bad, (f"all {CIRCUITS_PER_DRAW} circuits carry {m.n * m.reps} rx"
                     if not bad else f"mismatch {bad}")


def _h5_hard_pulse_null_through_the_circuit():
    """The E0 null through the CIRCUITS, at reps = 1.

    reps = 1 is not a convenience.  At EVEN reps the hard pulse is a perfect
    spin echo -- U(c_hard, phi)^2 = -I for every phi, exactly -- so epsilon
    vanishes identically and the arm tests nothing: it would return zero whether
    or not the injected ensemble is what we believe.  At odd reps, and reps = 1
    in particular, epsilon = -(phi/n) zhat stays linear, kappa = 0 is a real
    statement, and a measured gain there really does falsify the construction.
    E0 therefore runs UNAMPLIFIED, which it can afford: |eps| reaches 0.8 rad,
    an order of magnitude above the shot floor.
    """
    m = CompositeX(3, reps=1)
    c = m.c_hard()
    phis = np.linspace(-1.2, 1.2, 21)
    E = []
    for phi in phis:
        cb = {}
        for prep in PREPS:
            for meas in MEAS:
                p1 = simulate_ir(build_ir(m, c, prep, meas), float(phi))
                cb[(prep, meas)] = {"0": 1.0 - p1, "1": p1}
        E.append(eps_from_arm_counts(cb))
    E = np.array(E)
    X = np.stack([np.ones_like(phis), phis, phis ** 2], 1)
    coef, *_ = np.linalg.lstsq(X, E, rcond=None)
    q_over_a = np.linalg.norm(coef[2]) / max(np.linalg.norm(coef[1]), 1e-300)
    return q_over_a < 1e-6, f"|Q|/|A| through the circuits at reps=1 = {q_over_a:.2e}"


def _h5b_even_reps_echo():
    """Document the echo as a test so nobody 'fixes' E0 by amplifying it."""
    m = CompositeX(3, reps=2)
    worst = max(float(np.linalg.norm(m.eps(m.c_hard(), ph)))
                for ph in (0.1, 0.5, 1.0, 2.0))
    return worst < 1e-12, (f"hard pulse at reps=2 gives |eps| <= {worst:.1e} for "
                           f"all phi -- an exact echo, hence a vacuous null")


def _h7_amplification_under_the_wrap():
    """Amplified |eps| must stay below pi, where the axis-angle map saturates.

    This is the binding design constraint and it is easy to miss: the
    measurement needs |eps| * reps of order 1 rad to clear the shot floor, and
    the representation saturates at pi, so the useful window is narrow.  Past it
    the far tail -- precisely the draws CVaR is made of -- stops growing with
    phi and the measured gain is biased low.  At the frozen operating point the
    maximum over 40k draws is 3.13 against pi = 3.1416, i.e. the design sits
    just inside the boundary by construction, and this test fails if any change
    pushes it over.
    """
    op = OPERATING_POINT
    m = CompositeX(op["n"], reps=op["reps"])
    c = np.load(Path(__file__).resolve().parent / "stage7_c_mean.npy")
    pt = np.random.default_rng(999).normal(0.0, op["sigma"], 40000)
    mx = float(np.max(np.linalg.norm(m.eps_batch(c, pt), axis=1)))
    return mx < math.pi, f"max |eps| = {mx:.3f} vs pi = {math.pi:.3f}"


def _h6_debias_removes_the_bias():
    """The frozen debias must actually centre the loss at moderate shots."""
    m = CompositeX(3, reps=OPERATING_POINT["reps"])
    c = np.load(Path(__file__).resolve().parent / "stage7_c_mean.npy")
    rng = np.random.default_rng(3)
    shots = 128
    raw = run_arm_local(m, c, np.zeros(400), shots, seed=3)
    a_raw = analyse_arm(raw, shots, 0.2, debias=False)
    a_deb = analyse_arm(raw, shots, 0.2, debias=True)
    truth = float(m.loss_batch(c, np.zeros(1))[0])
    b0 = abs(a_raw["mean_L"] - truth)
    b1 = abs(a_deb["mean_L"] - truth)
    return b1 < b0, f"|bias| {b0:.5f} -> {b1:.5f} (truth {truth:.5f})"


def selftest():
    tests = [("H1 circuit epsilon == model epsilon", _h1_ir_matches_model),
             ("H2 basis-change signs", _h2_basis_signs),
             ("H3 phi enters once per segment", _h3_phi_enters_once_per_segment),
             ("H4 rx count matches the transpiler guard", _h4_rx_count_matches_guard),
             ("H5 hard-pulse null through the circuits", _h5_hard_pulse_null_through_the_circuit),
             ("H5b even reps make the null vacuous", _h5b_even_reps_echo),
             ("H7 amplified |eps| stays under the wrap", _h7_amplification_under_the_wrap),
             ("H6 debias reduces the loss bias", _h6_debias_removes_the_bias)]
    ok = True
    for name, fn in tests:
        try:
            good, msg = fn()
        except Exception as exc:                        # noqa: BLE001
            good, msg = False, f"raised {type(exc).__name__}: {exc}"
        ok &= bool(good)
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {msg}")
    print("\nstage7_ibm selftest " + ("PASSED" if ok else "FAILED"))
    return ok


# ==========================================================================
# 7. CLI
# ==========================================================================
def dryrun(shots=None, p_ro=0.01):
    """The whole protocol with local sampling: proves the pipeline, not the physics."""
    op = OPERATING_POINT
    shots = shots or op["shots"]
    m = CompositeX(op["n"], reps=op["reps"])
    m1 = CompositeX(op["n"], reps=1)
    ctrl, pr_off = arm_controls(op)
    phis = phi_draws(op)
    print(f"dry run: K={len(phis)} shots={shots} reps={op['reps']} "
          f"alpha={op['alpha']} sigma={op['sigma']}\n")
    # Deterministic per-arm sampling seeds.  hash() on a str is salted per
    # process (PYTHONHASHSEED), so `seed=hash(name)` made the dry run
    # irreproducible: three runs of identical code gave R = 0.111, 0.095, 0.139.
    # A frozen protocol cannot have a non-deterministic reference run, and the
    # spread was being mistaken for physics when it was the RNG moving.
    ARM_SEED = {"mean": 101, "cvar": 202, "offset_offline": 303,
                "offset_live": 404, "hard": 505}
    arms = {}
    for name in ("mean", "cvar", "offset_offline"):
        t0 = time.time()
        raw = run_arm_local(m, ctrl[name], phis, shots, p_ro, seed=ARM_SEED[name])
        arms[name] = analyse_arm(raw, shots, op["alpha"], p_ro)
        print(f"  {name:15s} mean_L={arms[name]['mean_L']:.5f} "
              f"CVaR={arms[name]['cvar']:.5f}  [{time.time()-t0:.1f}s]")
    print()
    for name in ("cvar", "offset_offline"):
        cp = compare(arms["mean"], arms[name], op["alpha"])
        print(f"  {name:15s} R = {cp['R']:+.4f}   mean-loss ratio "
              f"{cp['mean_loss_ratio']:.3f}   {cp['verdict']}")
    # E2 as it will actually run: the step from the MEASURED d_alpha.
    #
    # DE-AMPLIFY FIRST.  The device returns the error of the reps-fold sequence,
    # so the measured epsilon is roughly reps times the single-gate one.  The
    # amplification is a measurement amplifier, not a change to the system being
    # optimised: the control we are moving is the SINGLE gate, so the predictor
    # and the Jacobian both belong in single-gate space.
    #
    # Both wrong ways were tried in the dry run and both fail loudly, which is
    # why this is spelled out.  Feeding the amplified d_alpha through the reps=1
    # Jacobian overshoots by a factor of reps: |dc| = 12, R = -11.07.  Keeping
    # everything in amplified space is no better -- the step is physically large,
    # |dc| = 0.74, far outside the linear regime, and R = -3.68.  Dividing the
    # measured epsilon by reps and using the reps=1 Jacobian gives |dc| = 0.006
    # and R = +0.17 against R_pred = +0.11.
    E = np.array(arms["mean"]["eps"]) / float(m.reps)
    L = np.einsum("ki,ij,kj->k", E, W_METRIC, E)
    P_B, _, _ = reachable_projector(m1.jacobian(ctrl["mean"], 0.0))
    pr = predict(E, L, op["alpha"], P_B=P_B)
    c_live, dc_live = offset_control(m1, ctrl["mean"], pr["step"], 0.0)
    print(f"\n  E2 step: |d_alpha| = {np.linalg.norm(pr['d']):.5f}  "
          f"t* = {pr['t_best']:.3f}  |dc| = {np.linalg.norm(dc_live):.5f}")
    raw = run_arm_local(m, c_live, phis, shots, p_ro, seed=ARM_SEED["offset_live"])
    a_live = analyse_arm(raw, shots, op["alpha"], p_ro)
    cp = compare(arms["mean"], a_live, op["alpha"])
    print(f"\n  E2 PRIMARY (step from the measured d_alpha):")
    print(f"    R_pred (from counts) = {pr['R_pred']:+.4f}")
    print(f"    R_offset             = {cp['R']:+.4f}")
    print(f"    delivery             = {cp['R']/max(pr['R_pred'],1e-9):+.2f}")
    print(f"    discriminator        = {cp['verdict']}")
    return arms


def main():
    ap = argparse.ArgumentParser(description="VIPER Stage 7 hardware layer")
    ap.add_argument("cmd", choices=["selftest", "dryrun", "calibrate", "run", "analyse"])
    ap.add_argument("--backend", default=None)
    ap.add_argument("--instance", default=None)
    ap.add_argument("--arm", default="mean",
                    choices=["mean", "cvar", "offset_offline", "hard"])
    ap.add_argument("--shots", type=int, default=None)
    ap.add_argument("--max-sets", type=int, default=None)
    ap.add_argument("--reps", type=int, default=None,
                    help="amplification factor; defaults to the frozen 16. "
                         "E0 (--arm hard) MUST use --reps 1: at even reps the "
                         "hard pulse is an exact echo and the null is vacuous.")
    ap.add_argument("--p-ro", type=float, default=0.01)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    op = OPERATING_POINT

    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)

    if a.cmd == "dryrun":
        dryrun(a.shots, a.p_ro)
        return

    if a.cmd == "calibrate":
        if a.backend:
            print("hardware calibration: submit C1-C3 with `run`, then read "
                  "job.usage() for C5. Local numeric checks below.")
        for r in run_calibration(None, a.p_ro):
            flag = "PASS" if r["ok"] else ("FAIL" if r["ok"] is False else "n/a ")
            print(f"  [{flag}] {r['id']:4s} {r['what']}\n         {r['detail']}")
        return

    if a.cmd == "run":
        if not a.backend:
            sys.exit("run needs --backend (and a configured IBM account)")
        shots = a.shots or op["shots"]
        reps = a.reps if a.reps is not None else op["reps"]
        # E0 is the one arm that must NOT be amplified.  U(c_hard, phi)^2 = -I
        # for every phi, so at any even reps the arm returns zero whether or not
        # the injected ensemble is correct -- a null that cannot fail is not a
        # null.  Refuse rather than warn: this would be invisible in the data.
        if a.arm == "hard" and reps % 2 == 0:
            sys.exit(f"--arm hard with reps={reps} is an exact echo and tests "
                     f"nothing. Use --reps 1 for the E0 null. (Keep the default "
                     f"reps for a throughput probe, but then call it that.)")
        if a.arm != "hard" and reps != op["reps"]:
            print(f"WARNING: reps={reps} departs from the frozen {op['reps']}; "
                  f"this is outside the pre-registration.")
        m = CompositeX(op["n"], reps=reps)
        ctrl, _ = arm_controls(op)
        phis = phi_draws(op)
        backend = get_backend(a.backend, a.instance)
        print(f"backend {backend.name}, arm {a.arm}, reps={reps}, "
              f"K={len(phis)}, shots={shots}")
        raw, usage = run_arm_ibm(m, ctrl[a.arm], phis, shots, backend,
                                 a.max_sets, tag=a.arm)
        out = a.out or f"raw_{a.arm}_{int(time.time())}.json"
        Path(out).write_text(json.dumps(
            {"arm": a.arm, "backend": backend.name, "shots": shots, "reps": reps,
             "operating_point": op, "usage": [str(u) for u in usage],
             "control": np.asarray(ctrl[a.arm]).tolist(), "raw": raw},
            indent=1, default=float))
        print(f"wrote {out}  (raw counts saved: re-analysis never needs a re-run)")
        _validate_raw(raw, shots, tag=a.arm)
        return

    if a.cmd == "analyse":
        if not a.raw:
            sys.exit("analyse needs --raw with one or more files")
        arms = {}
        for f in sorted(Path().glob(a.raw)) or [Path(a.raw)]:
            d = json.loads(Path(f).read_text())
            arms[d["arm"]] = analyse_arm(d["raw"], d["shots"], op["alpha"], a.p_ro)
            print(f"  {d['arm']:15s} mean_L={arms[d['arm']]['mean_L']:.5f} "
                  f"CVaR={arms[d['arm']]['cvar']:.5f}")
        if "mean" in arms:
            for k, v in arms.items():
                if k == "mean":
                    continue
                cp = compare(arms["mean"], v, op["alpha"])
                print(f"\n  {k}: R = {cp['R']:+.4f}  ratio {cp['mean_loss_ratio']:.3f}"
                      f"\n     {cp['verdict']}")
        return


if __name__ == "__main__":
    main()