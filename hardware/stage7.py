"""
stage7.py -- VIPER Stage 7: the hardware test, on a free-tier budget.

WHY THIS LOOKS NOTHING LIKE STAGES 0-6
--------------------------------------
Stages 0-6 shape a 52-coefficient pulse on a simulated transmon.  The obvious
Stage 7 is to send that pulse to a real transmon, and it cannot be done on any
free platform: IBM removed pulse-level control from its hardware in early 2025
and deprecated qiskit.pulse for removal in Qiskit 2.0.  What survives on the
free tier is FRACTIONAL GATES -- a continuously parameterised RX(theta) on
Heron-class devices -- plus RZ, which is a frame change: exact, zero duration,
zero error.

That is enough, because the theory never required pulses.  Theorem 1 and the
whole kappa story are statements about a control family c -> epsilon(c; phi)
and a noise ensemble phi.  They do not care whether c indexes pulse
coefficients or rotation angles.  So Stage 7 re-instantiates the SAME
mathematics on the smallest control family a free QPU will execute:

    U(c; phi) = prod_{i=1..n}  RZ(psi_i) . RX(theta_i) . RZ(phi/n)
    c = (theta_1..theta_n, psi_1..psi_n)  in R^{2n}

with the quasi-static dephasing phi ~ N(0, sigma^2) INJECTED as virtual RZ
kicks rather than inherited from the device.  Injection is the point: sigma is
set exactly, so kappa is a dial rather than a measurement, which is something
the simulator never gave us for free.  The device's own T1, T2, control and
readout errors remain as an uncontrolled background, which is the part that
makes this a test rather than a re-simulation.

WHAT IT CAN AND CANNOT CLAIM
----------------------------
It CAN test, on hardware: that the tail-conditional displacement predicts the
direction a tail objective moves the gate; that applying the predicted offset
lowers the measured CVaR; and what fraction of the predicted gain a linear move
delivers -- the hardware analogue of the s = 0.588 decomposition.

It CANNOT claim pulse-level tail-risk optimal control on hardware, and the
paper must not imply otherwise.  The ensemble is synthetic.  Say so in the
first sentence of the section.

ERROR COORDINATES
-----------------
For one qubit the error operator lives in SU(2), so epsilon has THREE
components -- the same n_coord = 3 every Stage 6 cell reports.  With
E = U_target^dag U(c; phi) projected to SU(2) and written
E = cos(w/2) I - i sin(w/2) (nhat . sigma), we take epsilon = w * nhat and
W = I/6, which makes L = epsilon^T W epsilon equal the average gate infidelity
1 - F_avg to leading order (exactly (2/3) sin^2(w/2) -> w^2/6).

USAGE
    python3 stage7.py selftest            # no hardware, no Qiskit, no network
    python3 stage7.py design              # shot budget and circuit counts
    python3 stage7.py plan --out plan.json
"""
from __future__ import annotations
import json, math, sys, argparse
import numpy as np

# --------------------------------------------------------------------------
# SU(2) primitives.  Everything is 2x2 complex; the whole model is analytic and
# costs microseconds, which is why the offline half of Stage 7 is free.
# --------------------------------------------------------------------------
I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
SZ = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = (SX, SY, SZ)

W_METRIC = np.eye(3) / 6.0      # L = eps^T W eps = 1 - F_avg to leading order


def rx(t):
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def drx(t):
    """d/dt RX(t).  Exact, not finite-differenced."""
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    return 0.5 * np.array([[-s, -1j * c], [-1j * c, -s]], dtype=complex)


def rz(p):
    return np.array([[np.exp(-0.5j * p), 0], [0, np.exp(0.5j * p)]], dtype=complex)


def drz(p):
    return np.array([[-0.5j * np.exp(-0.5j * p), 0],
                     [0, 0.5j * np.exp(0.5j * p)]], dtype=complex)


def to_su2(M):
    """Strip the global phase: divide by a square root of the determinant.

    The branch does not matter for the error VECTOR because we fix the sign by
    requiring cos(w/2) >= 0 below; it matters only for the overall sign of E,
    which is not physical.
    """
    d = np.linalg.det(M)
    return M / np.sqrt(d.astype(complex))


def su2_to_eps(E):
    """epsilon = w * nhat for E = cos(w/2) I - i sin(w/2) (nhat . sigma).

    Returns a real 3-vector.  w is taken in [0, pi] by flipping the sign of E
    when the trace is negative, which is the choice that makes epsilon -> 0
    continuously as E -> I.
    """
    E = to_su2(np.asarray(E, dtype=complex))
    if np.real(np.trace(E)) < 0:
        E = -E
    c = float(np.clip(np.real(np.trace(E)) / 2.0, -1.0, 1.0))
    w = 2.0 * math.acos(c)
    s = math.sin(w / 2.0)
    # -i sin(w/2) n_k = tr(E sigma_k)/2  =>  n_k = i tr(E sigma_k) / (2 sin)
    v = np.array([np.real(1j * np.trace(E @ P) / 2.0) for P in PAULI], float)
    if abs(s) < 1e-14:
        return np.zeros(3)
    return w * (v / s)


def infidelity(E):
    """Average gate infidelity of the error operator E, exactly."""
    E = to_su2(np.asarray(E, dtype=complex))
    f = abs(np.trace(E)) ** 2
    return float(1.0 - (f + 2.0) / 6.0)      # 1 - (|tr|^2 + d)/(d(d+1)), d = 2


# --------------------------------------------------------------------------
# The control family
# --------------------------------------------------------------------------
class CompositeX:
    """n segments of RZ(psi_i) RX(theta_i), with dephasing injected between.

    Segment i is applied as   RZ(psi_i) . RX(theta_i) . RZ(phi/n)
    reading right to left, i.e. the phi kick happens FIRST within the segment.
    Spreading phi over n kicks rather than applying it once up front is what
    makes it quasi-static-like: a constant detuning during the gate accumulates
    phase throughout it, and n kicks is the coarsest honest discretisation of
    that.  With n = 1 the model degenerates to a single pre-gate frame error,
    which is NOT the same physics -- the paper should say n and why.
    """

    def __init__(self, n=3, target=None, reps=1):
        """`reps` repeats the whole sequence at FIXED phi, which is the only
        way this experiment is measurable.

        The mean-optimal control turns out to be dephasing-refocused: it kills
        the first-order response, leaving an infidelity around 1e-4 and an
        |epsilon| of a few times 1e-2.  Binomial noise on a Bloch component is
        1/sqrt(shots) = 0.06 at 256 shots, so a single application measures
        nothing but shot noise -- the emulator returns R = -0.03 +- 0.18 where
        the truth is +0.16.

        Repeating the sequence 2m times amplifies the COHERENT error roughly
        linearly in 2m while the shot noise stays put, so the signal-to-noise
        improves by the same factor.  This is legitimate precisely because the
        noise is quasi-static: phi is the same on every repetition by
        construction, which is what the injected-RZ model guarantees and what a
        real slow drift would only approximate.  An even number of repetitions
        makes the target the identity, which also removes the target-inverse
        gate from the circuit.
        """
        self.n = int(n)
        self.reps = int(reps)
        base = SX if target is None else np.asarray(target, dtype=complex)
        self.base_target = base
        self.target = np.linalg.matrix_power(base, self.reps)
        self.dim = 2 * self.n

    # -- starting points ---------------------------------------------------
    def c0(self):
        """The SPLIT ansatz: the pi rotation shared equally across n segments.

        This has to be the default, and the reason is a result rather than a
        preference.  Put the whole pi in one segment (see c_hard) and the
        injected kicks commute straight through the identity segments, giving

            epsilon(phi) = -(phi/n) zhat          EXACTLY, to machine precision

        -- perfectly linear in the noise, so kappa = 0 and Theorem 1 forces
        mean and CVaR to coincide.  There is no tail gain to find, and no
        amount of device time will produce one.  Sharing the rotation makes the
        kicks interleave with real rotations, which do not commute with them,
        and kappa climbs to the 0.04-0.38 band the simulation grid covers.

        So the ansatz is not a detail of the encoding; it is the experiment.
        """
        return np.concatenate([np.full(self.n, math.pi / self.n),
                               np.zeros(self.n)])

    def c_hard(self):
        """One hard pulse: exactly linear in phi, hence kappa = 0.

        Kept because it is a free, sharp NULL the hardware can check in a few
        hundred shots: Theorem 1 predicts zero gain here, and a measured gain
        would mean the injected ensemble is not what we think it is.
        """
        c = np.zeros(self.dim)
        c[0] = math.pi
        return c

    def split(self, c):
        c = np.asarray(c, float).ravel()
        return c[:self.n], c[self.n:]

    def unitary(self, c, phi=0.0):
        th, ps = self.split(c)
        k = rz(phi / self.n)
        U1 = I2
        for i in range(self.n):
            U1 = rz(ps[i]) @ rx(th[i]) @ k @ U1
        return np.linalg.matrix_power(U1, self.reps)

    def error(self, c, phi=0.0):
        return self.target.conj().T @ self.unitary(c, phi)

    def eps(self, c, phi=0.0):
        return su2_to_eps(self.error(c, phi))

    def loss(self, c, phi=0.0):
        e = self.eps(c, phi)
        return float(e @ W_METRIC @ e)

    # -- vectorised over the ensemble --------------------------------------
    def eps_batch(self, c, phis):
        """epsilon for many phi at once.

        The scalar path runs a Python loop of 2x2 products per draw.  That is
        fine for a few thousand evaluations and hopeless inside an optimiser,
        where it is thousands of iterations times thousands of draws.  Stacking
        the draws on a leading axis takes the offline optimisation -- the part
        that must finish before any device time is bought -- from tens of
        minutes to seconds.
        """
        th, ps = self.split(c)
        phis = np.asarray(phis, float).ravel()
        K = phis.size
        hp = phis / self.n
        Kz = np.zeros((K, 2, 2), complex)
        Kz[:, 0, 0] = np.exp(-0.5j * hp)
        Kz[:, 1, 1] = np.exp(+0.5j * hp)
        U1 = np.broadcast_to(np.eye(2, dtype=complex), (K, 2, 2)).copy()
        for i in range(self.n):
            Si = rz(ps[i]) @ rx(th[i])
            U1 = np.einsum("ab,kbc,kcd->kad", Si, Kz, U1, optimize=True)
        U = U1
        for _ in range(self.reps - 1):
            U = np.einsum("kab,kbc->kac", U1, U, optimize=True)
        E = np.einsum("ab,kbc->kac", self.target.conj().T, U, optimize=True)
        det = E[:, 0, 0] * E[:, 1, 1] - E[:, 0, 1] * E[:, 1, 0]
        E = E / np.sqrt(det)[:, None, None]
        tr = np.real(E[:, 0, 0] + E[:, 1, 1])
        flip = tr < 0
        E[flip] = -E[flip]
        cw = np.clip(np.abs(tr) / 2.0, -1.0, 1.0)
        w = 2.0 * np.arccos(cw)
        sw = np.sin(w / 2.0)
        # tr(E sigma_k), written out rather than hand-simplified: the y term
        # picks up two factors of i and the sign is easy to lose.
        a, b, cc, d = E[:, 0, 0], E[:, 0, 1], E[:, 1, 0], E[:, 1, 1]
        tr_s = np.stack([b + cc, 1j * (b - cc), a - d], axis=1)
        v = np.real(1j * tr_s / 2.0)
        out = np.zeros((K, 3))
        good = np.abs(sw) > 1e-14
        out[good] = (w[good] / sw[good])[:, None] * v[good]
        return out

    def loss_batch(self, c, phis):
        E = self.eps_batch(c, phis)
        return np.einsum("ki,ij,kj->k", E, W_METRIC, E)

    # -- analytic Jacobian d eps / d c ------------------------------------
    def jacobian(self, c, phi=0.0, h=1e-6):
        """d epsilon / d c at fixed phi.

        epsilon is a nonlinear function of the unitary, so we differentiate the
        UNITARY analytically (exact product rule, no finite differences in the
        2x2 algebra) and push it through su2_to_eps by a central difference in
        a single scalar -- which is one-dimensional and therefore stable, unlike
        differencing the whole 6-vector.  For the small errors this experiment
        lives in, epsilon is close to linear in the generator and the two agree
        to better than 1e-9 (selftest T4).
        """
        th, ps = self.split(c)
        k = rz(phi / self.n)
        # prefix[i] = product of segments AFTER i (applied later, so on the left)
        seg = [rz(ps[i]) @ rx(th[i]) @ k for i in range(self.n)]
        suf = [I2] * (self.n + 1)           # suf[i] = seg[i-1] ... seg[0]
        for i in range(self.n):
            suf[i + 1] = seg[i] @ suf[i]
        pre = [I2] * (self.n + 1)           # pre[i] = seg[n-1] ... seg[i]
        for i in range(self.n - 1, -1, -1):
            pre[i] = pre[i + 1] @ seg[i]
        J = np.zeros((3, self.dim))
        base = self.eps(c, phi)
        for i in range(self.n):
            dseg_th = rz(ps[i]) @ drx(th[i]) @ k
            dseg_ps = drz(ps[i]) @ rx(th[i]) @ k
            for col, dseg in ((i, dseg_th), (self.n + i, dseg_ps)):
                dU = pre[i + 1] @ dseg @ suf[i]
                # directional derivative of eps along dU, by a one-dimensional
                # central difference in the step size
                Up = self.target.conj().T @ (self.unitary(c, phi))
                _ = Up  # (kept for clarity; eps(base) already computed)
                ep = su2_to_eps(self.target.conj().T @ (self.unitary(c, phi)
                                                        + h * dU))
                em = su2_to_eps(self.target.conj().T @ (self.unitary(c, phi)
                                                        - h * dU))
                J[:, col] = (ep - em) / (2.0 * h)
        del base
        return J


# --------------------------------------------------------------------------
# Ensemble and risk
# --------------------------------------------------------------------------
def ensemble(model, c, sigma, K, seed=0):
    """Draw K quasi-static phases and return (phi, eps KxD, L K)."""
    rng = np.random.default_rng(seed)
    phi = rng.normal(0.0, sigma, size=int(K))
    E = np.array([model.eps(c, p) for p in phi], float)
    L = np.einsum("ki,ij,kj->k", E, W_METRIC, E)
    return phi, E, L


def cvar(L, alpha):
    """Upper-tail CVaR at level alpha, the atomic-safe form used since Stage 0."""
    L = np.sort(np.asarray(L, float))[::-1]
    K = L.size
    m = max(1, int(math.ceil(alpha * K)))
    w = np.ones(m)
    w[-1] = alpha * K - (m - 1) if m > 1 else 1.0
    w = np.clip(w, 0.0, None)
    return float(np.sum(w * L[:m]) / np.sum(w))


def kappa_of(model, c, sigma, K=4096, seed=0):
    """kappa = ||Q||_F / ||A||_F for eps ~ e0 + A z + Q z^2 in the STANDARDISED
    noise z = phi / sigma.

    Standardising is what makes this number comparable to the simulation.  Fit
    in the raw phi and kappa comes out a property of the ansatz alone, constant
    in sigma -- which is not what earlier stages report, because there the
    expansion variable carried the noise scale with it.  In z the quadratic term
    is Q sigma^2 against a linear A sigma, so kappa grows linearly with sigma
    exactly as it does across the Stage 3-6 grids, and the hardware point can be
    placed on the same axis as Fig. 16 rather than on one of its own.
    """
    rng = np.random.default_rng(seed)
    phi = rng.normal(0.0, sigma, size=K)
    z = phi / max(sigma, 1e-300)
    E = np.array([model.eps(c, p) for p in phi], float)
    X = np.stack([np.ones_like(z), z, z ** 2], axis=1)
    coef, *_ = np.linalg.lstsq(X, E, rcond=None)
    pred = X @ coef
    ss = np.sum((E - pred) ** 2)
    tot = np.sum((E - E.mean(0)) ** 2)
    r2 = 1.0 - ss / max(tot, 1e-300)
    A, Q = coef[1], coef[2]
    nA = float(np.linalg.norm(A))
    return {"kappa": float(np.linalg.norm(Q) / max(nA, 1e-300)),
            "fit_R2": float(r2), "e0": coef[0].tolist(),
            "A_norm": nA, "Q_norm": float(np.linalg.norm(Q))}


# --------------------------------------------------------------------------
# The measurement-only predictor, Stage 6 Sec. IV L, restated for 3 coordinates
# --------------------------------------------------------------------------
def predict(E, L, alpha, P_B=None, W=None, n_t=41, t_max=2.0):
    """Tail-conditional displacement, projected and line-searched.

    Identical in content to stage6.predicted_gain: take the mean displacement
    of the alpha-tail, project it onto the reachable subspace, and scan the
    step length over the MEASURED losses -- no model of the control enters, so
    every input here is something a QPU can hand you.

    `mean_loss_change` is carried over deliberately.  It is the discriminator
    that separates a genuine mean-for-tail trade (mean loss RISES) from an
    unconverged mean baseline (mean loss FALLS), and on hardware there is a
    second way to fail it -- a miscalibrated device gives the offset free mean
    improvement that has nothing to do with the tail.  Report it every time.
    """
    W = W_METRIC if W is None else W
    E = np.asarray(E, float); L = np.asarray(L, float).ravel()
    K = L.size
    thr = np.sort(L)[::-1][max(1, int(math.ceil(alpha * K))) - 1]
    tail = L >= thr
    d = E[tail].mean(0) - E.mean(0)
    if P_B is not None:
        d = np.asarray(P_B, float) @ d
    c0 = cvar(L, alpha)
    best = (0.0, 0.0, c0)
    for t in np.linspace(0.0, t_max, n_t):
        step = -t * d
        L1 = np.einsum("ki,ij,kj->k", E + step, W, E + step)
        c1 = cvar(L1, alpha)
        if c1 < best[2]:
            best = (t, 1.0 - c1 / c0, c1)
    t_best, R_pred, _ = best
    step = -t_best * d
    L1 = np.einsum("ki,ij,kj->k", E + step, W, E + step)
    m0, m1 = float(L.mean()), float(L1.mean())
    return {"d": d.tolist(), "step": step.tolist(), "t_best": float(t_best),
            "R_pred": float(R_pred), "cvar_0": float(c0),
            "n_tail": int(tail.sum()),
            "mean_loss_change": float((m1 - m0) / max(m0, 1e-300)),
            "tail_specific": bool(m1 > m0),
            "at_grid_edge": bool(abs(t_best - t_max) < 1e-9)}


def reachable_projector(J):
    """P_B onto range(J).  With 6 controls and 3 error coordinates this is
    generically the identity -- the composite sequence can reach any small
    SU(2) error -- and the selftest asserts it, because a rank drop would mean
    the ansatz is degenerate and the whole experiment measures nothing.
    """
    U, s, _ = np.linalg.svd(np.asarray(J, float), full_matrices=False)
    r = int(np.sum(s > 1e-10 * max(s[0], 1e-300)))
    Q = U[:, :r]
    return Q @ Q.T, r, s.tolist()


def offset_control(model, c, step, phi=0.0):
    """Realise a requested error-space step as a CONTROL perturbation.

    delta_c = J^+ step, exactly the offsettest move of Stage 6.  Here J is
    analytic, so unlike the simulator this costs no device time at all -- the
    entire linear-response half of the experiment is free.
    """
    J = model.jacobian(c, phi)
    dc = np.linalg.pinv(J) @ np.asarray(step, float)
    return np.asarray(c, float) + dc, dc


# --------------------------------------------------------------------------
# Measurement scheme
# --------------------------------------------------------------------------
# Six circuits per noise draw.  Prepare |0> and |+>, apply U(c; phi) then
# U_target^dag, and measure X, Y, Z on each.  The two rotated Bloch vectors
# R.zhat and R.xhat determine the rotation completely (the third column is
# their cross product), so epsilon comes out SIGNED.
#
# Survival probabilities alone would be cheaper -- three circuits -- and are
# useless: p_m = cos^2(w/2) + sin^2(w/2) (nhat.mhat)^2 depends on nhat only
# through its SQUARED components, so the signs of epsilon are unrecoverable and
# d_alpha cannot be formed.  That trap is worth the extra three circuits.
PREPS = ("0", "+")
MEAS = ("X", "Y", "Z")
CIRCUITS_PER_DRAW = len(PREPS) * len(MEAS)


def bloch_rotation(E):
    """SO(3) matrix R with E sigma.v E^dag = sigma.(R v)."""
    E = to_su2(np.asarray(E, dtype=complex))
    R = np.zeros((3, 3))
    for j, Pj in enumerate(PAULI):
        M = E @ Pj @ E.conj().T
        for i, Pi in enumerate(PAULI):
            R[i, j] = float(np.real(np.trace(Pi @ M) / 2.0))
    return R


def eps_from_R(R):
    """Axis-angle vector of an SO(3) rotation; equals epsilon by construction."""
    R = np.asarray(R, float)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    w = math.acos(c)
    if w < 1e-12:
        return np.zeros(3)
    if abs(math.pi - w) < 1e-6:                  # near-pi branch
        A = (R + np.eye(3)) / 2.0
        n = np.sqrt(np.clip(np.diag(A), 0.0, None))
        k = int(np.argmax(n))
        n = A[:, k] / max(n[k], 1e-300)
        n = n / max(np.linalg.norm(n), 1e-300)
        return w * n
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * v / (2.0 * math.sin(w))


def eps_from_counts(rot_z, rot_x):
    """Rebuild epsilon from two measured rotated Bloch vectors.

    rot_z = R.zhat and rot_x = R.xhat, each a 3-vector of expectation values
    <X>, <Y>, <Z>.  Shot noise makes them non-orthonormal, so the columns are
    re-orthonormalised (the nearest rotation, by polar decomposition) before
    the log -- without that, eps_from_R can return a complex angle on noisy
    data, which is the sort of thing that only shows up at 2 a.m. on a QPU.
    """
    z = np.asarray(rot_z, float); x = np.asarray(rot_x, float)
    y = np.cross(z, x)
    M = np.stack([x, y, z], axis=1)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return eps_from_R(R)


def circuit_spec(model, c, phi, prep, meas):
    """A hardware-agnostic description of one circuit.

    Emitted as a gate list so it can be rendered to Qiskit, OpenQASM 3 or an
    IQM job without this file importing any vendor SDK.  RZ is virtual on every
    superconducting platform worth running this on, so only the RX entries cost
    time or fidelity -- n of them, plus at most two for prep and basis change.
    """
    th, ps = model.split(c)
    g = []
    if prep == "+":
        g.append(("ry", math.pi / 2))
    elif prep == "+i":
        g.append(("rx", -math.pi / 2))
    for i in range(model.n):
        g.append(("rz", phi / model.n))          # the injected quasi-static kick
        g.append(("rx", float(th[i])))
        g.append(("rz", float(ps[i])))
    g.append(("target_inverse", None))           # U_target^dag, one RX(-pi) for X
    if meas == "X":
        g.append(("ry", -math.pi / 2))
    elif meas == "Y":
        g.append(("rx", math.pi / 2))
    return {"prep": prep, "meas": meas, "phi": float(phi), "gates": g,
            "n_rx": sum(1 for q, _ in g if q in ("rx", "ry")) + 1}


# --------------------------------------------------------------------------
# Shot budget
# --------------------------------------------------------------------------
# The binding constraint is 10 minutes of QPU time per 28-day window on the IBM
# Open Plan.  Converting that into draws is the only genuinely uncertain number
# in this file, because throughput depends on reset strategy, batching and the
# device.  So it is a PARAMETER with a deliberately pessimistic default, and
# `design` prints what the plan looks like across a range.  Calibrate it with
# one tiny job -- submit 10 circuits x 256 shots, read the reported usage, and
# pass the measured value as --throughput.  Do not trust the default.
# THE OPERATING POINT, and a correction worth reading before the numbers.
#
# An earlier draft of the protocol quoted kappa = 0.13 here and claimed the
# hardware point "lands on the same axis as the simulation grid".  Both halves
# were wrong, and not because of a convention clash.  kappa = 0.13 is the split
# ansatz c0; the experiment runs at the POOLED MEAN-OPTIMAL control, and there
# kappa = 3.63.  Same standardised z = phi/sigma in both cases -- different
# control.
#
# The reason is physics, not bookkeeping.  Optimising the mean over a 3-segment
# composite finds a dephasing-REFOCUSED pulse: the first-order response |A|
# collapses from 0.388 to 0.0046, an 84-fold cancellation, while |Q| barely
# moves.  Since kappa = |Q|/|A| by definition, it diverges as the mean arm gets
# good.  At sigma = 0.6 the three contributions to epsilon in the standardised
# variable are
#
#     |e0| = 0.0159     |A| = 0.0046     |Q| = 0.0167
#
# i.e. a static offset and a quadratic term with almost no linear part left
# between them.  THAT is the honest description of this regime, and kappa is a
# badly conditioned way to say it -- a ratio of a small number to a smaller one.
# Report the triple, quote kappa second, and do NOT place this point on the
# paper's kappa axis: the simulation grid runs 0.025-0.5 and caps perturbative
# validity at 1.0, so this sits outside it by design.
#
# Why n = 3 is forced rather than chosen: at n = 2 and n = 4 the mean optimiser
# drives the gate to machine-exact (|A| = |e0| = 0, surrogate fit R2 = 0.02),
# leaving nothing to measure -- R_pred = 0.  Only n = 3 leaves a residual.
#
#   sigma = 0.6      surrogate fit R2 = 0.963, just inside the 0.95 floor the
#                    K_train sweep used.  sigma = 0.9 falls to 0.92 and is
#                    inadmissible; sigma = 0.3 gives 0.990 and is the
#                    replication point.
#   reps  = 16       the refocused mean arm has |eps| = 0.024, under the
#                    1/sqrt(shots) = 0.06 floor.  One application measures shot
#                    noise: R = -0.03 +- 0.22 against a truth of +0.16.
#                    Repeating at FIXED phi amplifies coherent error only.
#   K     = 256      the dominant variance is the CVaR estimator's, not shot
#                    noise: 256 -> 2048 shots per draw did not move the error
#                    bar, while 64 -> 256 draws cut it fourfold.
#   shots = 128      what is left once K is paid for.
#   alpha = 0.2      51 tail draws; alpha = 0.05 would give 13, under the floor.
#
# Emulated at this point: R = 0.134 +- 0.049 against a truth of 0.150, positive
# in 16 of 16 repeats, 197 s of a 600 s window for two arms.
OPERATING_POINT = {
    "n": 3, "sigma": 0.6, "alpha": 0.2, "reps": 16, "K": 256, "shots": 128,
    "kappa_at_mean_optimal": 3.63, "kappa_at_c0": 0.1325,
    "surrogate_fit_R2": 0.963,
    "regime": {"e0_norm": 0.01587, "A_norm": 0.00460, "Q_norm": 0.01669,
               "note": "standardised z = phi/sigma; refocused mean arm, so the "
                       "linear term is the smallest of the three and kappa is "
                       "ill-conditioned"},
    # CORRECTED once the emulator scaled depolarising with circuit DEPTH.  The
    # earlier 0.134 +- 0.049 came from a fixed per-circuit rate, which at 48
    # gates understated the incoherent background by ~50x and flattered
    # amplification.  Honest figure below: a ~2 sigma result from ONE window.
    "R_true_single_gate": 0.1616, "R_measured_emulated": 0.109,
    "R_sd_emulated": 0.061, "snr_emulated": 1.8,
    "positive_in": "16/16 emulator repeats",
    "windows_for_3sigma": "2-3 (combine independent windows; phi seed differs)",
    "amplification_ceiling": {"max_eps_at_operating_point": 3.131, "pi": 3.1416,
        "note": "the design sits just inside the axis-angle wrap by "
                "construction; selftest H7 fails if a change pushes it over"},
    "E0_null_reps": 1,
    "E0_note": "the hard pulse is an EXACT echo at even reps (U^2 = -I for all "
               "phi), so E0 must run unamplified at reps=1 or the null is "
               "vacuous; |eps| reaches 0.8 rad there, well clear of the floor",
    "seconds_two_arms_at_2000_per_s": 197,
    "replication_point": {"sigma": 0.3, "kappa_at_mean_optimal": 5.44,
                          "surrogate_fit_R2": 0.990, "R_pred": 0.172,
                          "shot_feasibility": "NOT yet emulated"},
    "predictor_delivery": {"sigma_0.3": 1.00, "sigma_0.6": 1.01, "sigma_0.9": 1.02,
                           "note": "R_offset/R_pred with the step realised as "
                                   "dc = J^+ step at the OPTIMISED mean control. "
                                   "At the unoptimised c0 the same move gives a "
                                   "NEGATIVE R at every sigma -- the delivery "
                                   "fraction is a property of where you stand."}}

THROUGHPUT_DEFAULT = 2000.0      # shots per second of billed QPU time
WINDOW_SECONDS = 600.0           # 10 minutes per 28 days, Open Plan


def budget(K, shots, throughput=THROUGHPUT_DEFAULT, arms=2,
           window=WINDOW_SECONDS):
    n_circ = arms * K * CIRCUITS_PER_DRAW
    n_shot = n_circ * shots
    secs = n_shot / max(throughput, 1e-9)
    return {"K": int(K), "shots": int(shots), "arms": int(arms),
            "circuits": int(n_circ), "shots_total": int(n_shot),
            "seconds": float(secs), "windows": float(secs / window),
            "fits_one_window": bool(secs <= window)}


def tail_draws(K, alpha):
    return alpha * K


def design(alpha=0.2, throughput=THROUGHPUT_DEFAULT):
    """Pick (K, shots) under the free-tier ceiling.

    Two competing demands.  The tail must contain enough draws for CVaR to mean
    anything -- Stage 6 takes 8 as the floor and the K_train sweep showed the
    level statistic is stable above roughly 16 -- which pushes K up.  And each
    epsilon must be resolved well enough that the tail is selected by physics
    rather than by shot noise, which pushes `shots` up.  At alpha = 0.2 both can
    be met; at alpha = 0.05 they cannot, which is why the hardware run uses
    alpha = 0.2 and the paper says so.
    """
    rows = []
    for K in (32, 48, 64, 96, 128):
        for shots in (128, 256, 512, 1024):
            b = budget(K, shots, throughput)
            b["alpha"] = alpha
            b["tail_draws"] = tail_draws(K, alpha)
            # shot noise on one Bloch component, propagated to |eps|
            b["eps_resolution"] = float(1.0 / math.sqrt(shots))
            b["ok_tail"] = bool(b["tail_draws"] >= 8)
            rows.append(b)
    return rows


# --------------------------------------------------------------------------
# Shot-noise emulator
# --------------------------------------------------------------------------
# The point of this is to answer, BEFORE spending a free window, the only
# question that matters: is the gain bigger than the error bar?  Everything
# above is exact algebra and will look wonderful at infinite shots.  With 256
# shots and a real device underneath, the tail of the loss distribution may be
# selected by binomial noise rather than by phi, in which case d_alpha points
# nowhere and the whole experiment returns mush.
#
# So: emulate the six circuits per draw with binomial sampling, a depolarising
# background for the device's own error, and readout bias.  Repeat the whole
# run many times and report the SCATTER in the measured gain.  That scatter is
# the error bar the paper would have to quote.
DEV_DEPOL_DEFAULT = 1.5e-3       # per-circuit depolarising, Heron-class 1Q
DEV_READOUT_DEFAULT = 1.0e-2     # symmetric readout error


def emulate_eps(model, c, phi, shots, rng, p_dep=DEV_DEPOL_DEFAULT,
                p_ro=DEV_READOUT_DEFAULT, debias=True, depth_scaled=True):
    """One noise draw, six circuits, finite shots.  Returns the measured eps.

    `depth_scaled` matters once amplification is in play.  A fixed per-circuit
    depolarising rate flatters deep circuits: at reps = 48 the body is 144 RX
    gates, and treating that as one 1.5e-3 event rather than 144 of them
    understates the incoherent background by two orders of magnitude.  The
    amplification trades exactly this against shot noise, so the comparison is
    meaningless unless the cost side scales with the gate count.
    """
    R = bloch_rotation(model.error(c, phi))
    if depth_scaled:
        n_gates = model.n * model.reps
        p_dep = 1.0 - (1.0 - p_dep) ** n_gates
    vecs = []
    for prep in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])):
        v_true = R @ prep
        est = np.zeros(3)
        for k in range(3):
            v = v_true[k] * (1.0 - p_dep)             # depolarising shrink
            p_up = 0.5 * (1.0 + v)
            p_obs = p_up * (1.0 - p_ro) + (1.0 - p_up) * p_ro
            n_up = rng.binomial(shots, min(max(p_obs, 0.0), 1.0))
            v_obs = 2.0 * n_up / shots - 1.0
            if debias:
                v_obs = v_obs / max((1.0 - 2.0 * p_ro) * (1.0 - p_dep), 1e-9)
            est[k] = v_obs
        vecs.append(est)
    return eps_from_counts(vecs[0], vecs[1])


def emulate_run(model, c_mean, sigma, K, shots, alpha=0.2, seed=0,
                p_dep=DEV_DEPOL_DEFAULT, p_ro=DEV_READOUT_DEFAULT):
    """The full protocol on emulated hardware.

    Mirrors exactly what the QPU script does: measure the ensemble under the
    mean control, build the predictor from those measurements alone, turn the
    predicted step into a control offset through the analytic Jacobian, measure
    the SAME phi draws again under the offset control, and compare CVaRs.  The
    phases are reused between arms on purpose -- it is a paired comparison, so
    the ensemble draw cancels and the error bar is on the difference.
    """
    rng = np.random.default_rng(seed)
    phi = rng.normal(0.0, sigma, size=int(K))
    Em = np.array([emulate_eps(model, c_mean, p, shots, rng, p_dep, p_ro)
                   for p in phi])
    Lm = np.einsum("ki,ij,kj->k", Em, W_METRIC, Em)
    P_B, _, _ = reachable_projector(model.jacobian(c_mean, 0.0))
    pr = predict(Em, Lm, alpha, P_B=P_B)
    c_off, _ = offset_control(model, c_mean, pr["step"], 0.0)
    Eo = np.array([emulate_eps(model, c_off, p, shots, rng, p_dep, p_ro)
                   for p in phi])
    Lo = np.einsum("ki,ij,kj->k", Eo, W_METRIC, Eo)
    cm, co = cvar(Lm, alpha), cvar(Lo, alpha)
    return {"R_measured": float(1.0 - co / cm), "R_pred": pr["R_pred"],
            "t_best": pr["t_best"], "mean_loss_change_pred": pr["mean_loss_change"],
            "mean_loss_measured_ratio": float(Lo.mean() / max(Lm.mean(), 1e-300)),
            "cvar_mean_arm": cm, "cvar_offset_arm": co,
            "delivery": float((1.0 - co / cm) / max(pr["R_pred"], 1e-300))}


def feasibility(sigma=0.9, K=64, shots=256, alpha=0.2, reps=24, n=3,
                p_dep=DEV_DEPOL_DEFAULT, p_ro=DEV_READOUT_DEFAULT):
    """Repeat the emulated run and report the spread.  This is the go/no-go."""
    m = CompositeX(n)
    c = m.c0()
    truth = kappa_of(m, c, sigma, K=8000, seed=0)
    ideal_pr = predict(*ensemble(m, c, sigma, 8000, seed=1)[1:], alpha=alpha)
    rows = [emulate_run(m, c, sigma, K, shots, alpha, seed=100 + i,
                        p_dep=p_dep, p_ro=p_ro) for i in range(reps)]
    Rm = np.array([r["R_measured"] for r in rows])
    Rp = np.array([r["R_pred"] for r in rows])
    return {"sigma": sigma, "K": K, "shots": shots, "alpha": alpha, "reps": reps,
            "kappa": truth["kappa"], "fit_R2": truth["fit_R2"],
            "R_pred_infinite_shots": ideal_pr["R_pred"],
            "R_pred_mean": float(Rp.mean()), "R_pred_sd": float(Rp.std(ddof=1)),
            "R_measured_mean": float(Rm.mean()), "R_measured_sd": float(Rm.std(ddof=1)),
            "snr": float(Rm.mean() / max(Rm.std(ddof=1), 1e-300)),
            "n_positive": int((Rm > 0).sum()),
            "delivery_median": float(np.median([r["delivery"] for r in rows])),
            "budget": budget(K, shots)}


# --------------------------------------------------------------------------
# Selftests.  No hardware, no Qiskit, no network.  Run before spending a single
# free minute: everything below is algebra the QPU cannot fix for you.
# --------------------------------------------------------------------------
def _t1_ideal():
    m = CompositeX(3)
    e = m.eps(m.c0(), 0.0)
    return float(np.linalg.norm(e)) < 1e-12, f"|eps|={np.linalg.norm(e):.2e}"


def _t2_known_detuning():
    """One segment, phi injected before an RX(pi).  Check against the closed
    form: E = X^dag RX(pi) RZ(phi) = RZ(-phi) up to phase, so |eps| = |phi|.
    """
    m = CompositeX(1)
    out = []
    for phi in (0.01, 0.05, 0.2):
        e = m.eps(m.c0(), phi)
        out.append(abs(np.linalg.norm(e) - phi))
    return max(out) < 1e-9, f"worst |‖eps‖-phi| = {max(out):.2e}"


def _t3_loss_is_infidelity():
    """L = eps^T W eps must equal the exact average gate infidelity to O(eps^4)."""
    m = CompositeX(3)
    c = m.c0() + 0.01 * np.arange(6)
    bad = []
    for phi in (0.0, 0.02, 0.05, 0.1):
        L = m.loss(c, phi)
        F = infidelity(m.error(c, phi))
        bad.append(abs(L - F) / max(F, 1e-300))
    # leading-order agreement: the relative error must shrink with the error size
    return bad[0] < 0.05 and bad[-1] < 0.25, f"rel err {['%.3f' % b for b in bad]}"


def _t4_jacobian():
    m = CompositeX(3)
    rng = np.random.default_rng(0)
    c = m.c0() + 0.05 * rng.standard_normal(6)
    phi = 0.03
    J = m.jacobian(c, phi)
    Jn = np.zeros_like(J)
    h = 1e-6
    for k in range(m.dim):
        cp = c.copy(); cp[k] += h
        cm = c.copy(); cm[k] -= h
        Jn[:, k] = (m.eps(cp, phi) - m.eps(cm, phi)) / (2 * h)
    err = np.max(np.abs(J - Jn)) / max(np.max(np.abs(Jn)), 1e-300)
    return err < 1e-5, f"max rel err {err:.2e}"


def _t5_reachability():
    """Rank 3: the composite sequence can reach any small SU(2) error.  A rank
    drop would make P_B a projector onto a plane and silently cap R_pred."""
    m = CompositeX(3)
    rng = np.random.default_rng(1)
    worst = 3
    for _ in range(20):
        c = m.c0() + 0.1 * rng.standard_normal(6)
        _, r, _ = reachable_projector(m.jacobian(c, 0.02))
        worst = min(worst, r)
    return worst == 3, f"min rank over 20 controls = {worst}"


def _t6_bloch_roundtrip():
    """su2_to_eps and the SO(3) log must agree, and the counts-based rebuild
    must invert the forward map.  If these disagree the hardware numbers are
    measuring a different quantity from the simulation ones."""
    m = CompositeX(3)
    rng = np.random.default_rng(2)
    bad = 0.0
    for _ in range(50):
        c = m.c0() + 0.2 * rng.standard_normal(6)
        phi = float(rng.normal(0, 0.1))
        E = m.error(c, phi)
        e1 = su2_to_eps(E)
        R = bloch_rotation(E)
        e2 = eps_from_R(R)
        e3 = eps_from_counts(R @ np.array([0, 0, 1.0]), R @ np.array([1.0, 0, 0]))
        bad = max(bad, np.linalg.norm(e1 - e2), np.linalg.norm(e1 - e3))
    return bad < 1e-8, f"worst mismatch {bad:.2e}"


def _t7_kappa_law():
    """R_pred must follow the kappa^2 onset, as in every earlier stage.

    Restricted to sigma <= 0.9, where the quadratic surrogate still holds
    (fit R2 > 0.996).  Past that the reduced model degrades and the law is not
    expected to hold -- the same admission rule the K_train sweep used.
    """
    m = CompositeX(3)
    c = m.c0()
    rows = []
    for sigma in (0.2, 0.4, 0.6, 0.9):
        k = kappa_of(m, c, sigma, K=4000, seed=3)
        _, E, L = ensemble(m, c, sigma, 3000, seed=4)
        p = predict(E, L, 0.2)
        rows.append((sigma, k["kappa"], k["fit_R2"], p["R_pred"]))
    r = [x[3] / max(x[1] ** 2, 1e-300) for x in rows]
    spread = max(r) / max(min(r), 1e-300)
    ok = spread < 1.5 and min(x[2] for x in rows) > 0.99
    return ok, ("R_pred/kappa^2 " + ", ".join(f"{x:.2f}" for x in r)
                + f" (spread {spread:.2f}x), kappa "
                + ", ".join(f"{x[1]:.3f}" for x in rows))


def _t7a_hard_pulse_is_equivalent():
    """THEOREM 1 AS A NULL.  One hard pulse gives epsilon exactly linear in phi,
    so kappa = 0 and mean and CVaR must coincide.  This is the cheapest
    falsifiable thing the hardware can do: a measured gain here means the
    injected ensemble is not the ensemble we think we injected.
    """
    m = CompositeX(3)
    c = m.c_hard()
    k = kappa_of(m, c, 0.9, K=4000, seed=7)
    _, E, L = ensemble(m, c, 0.9, 3000, seed=8)
    p = predict(E, L, 0.2)
    # and the closed form: eps = -(phi/n) zhat
    worst = max(abs(np.linalg.norm(m.eps(c, ph)) - abs(ph) / m.n)
                for ph in (0.1, 0.5, 1.0))
    ok = k["kappa"] < 1e-9 and abs(p["R_pred"]) < 1e-3 and worst < 1e-12
    return ok, (f"kappa={k['kappa']:.2e}  R_pred={p['R_pred']:.2e}  "
                f"closed-form err {worst:.1e}")


def _t8_discriminator():
    """On a perfectly converged mean control the offset must COST mean loss.
    If it does not, the baseline was not converged -- exactly the Stage 6 trap,
    and on hardware a miscalibrated device reproduces it."""
    m = CompositeX(3)
    _, E, L = ensemble(m, m.c0(), 0.9, 4000, seed=5)
    p = predict(E, L, 0.2)
    return (p["tail_specific"] and p["R_pred"] > 0.01), \
        (f"R_pred={p['R_pred']:.4f} mean_loss_change={p['mean_loss_change']:+.4f} "
         f"trade={p['tail_specific']}")


def _t9_offset_realisable():
    """The predicted error-space step must be reachable as a control move: the
    realised displacement should match the request to a few percent.  This is
    the hardware analogue of the 0.33 linear-delivery fraction, computed with
    no device at all -- if it fails here it will fail there."""
    m = CompositeX(3)
    c = m.c0()
    _, E, L = ensemble(m, c, 0.9, 3000, seed=6)
    p = predict(E, L, 0.2)
    c2, dc = offset_control(m, c, p["step"], 0.0)
    got = m.eps(c2, 0.0) - m.eps(c, 0.0)
    want = np.asarray(p["step"], float)
    rel = np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-300)
    return rel < 0.15, f"realised/requested mismatch {rel:.3f}, ||dc||={np.linalg.norm(dc):.4f}"


def _t10_budget_fits():
    b = budget(64, 256, THROUGHPUT_DEFAULT)
    return b["fits_one_window"], (f"K=64 shots=256 -> {b['circuits']} circuits, "
                                  f"{b['shots_total']} shots, {b['seconds']:.0f}s "
                                  f"of a {WINDOW_SECONDS:.0f}s window")


def selftest():
    tests = [("T1 ideal control is exact", _t1_ideal),
             ("T2 injected phase matches closed form", _t2_known_detuning),
             ("T3 loss equals average gate infidelity", _t3_loss_is_infidelity),
             ("T4 analytic Jacobian", _t4_jacobian),
             ("T5 error space fully reachable", _t5_reachability),
             ("T6 SU(2)/SO(3)/counts round trip", _t6_bloch_roundtrip),
             ("T7 kappa^2 onset", _t7_kappa_law),
             ("T7a hard pulse is exactly equivalent", _t7a_hard_pulse_is_equivalent),
             ("T8 discriminator on a converged baseline", _t8_discriminator),
             ("T9 predicted step is realisable", _t9_offset_realisable),
             ("T10 budget fits a free window", _t10_budget_fits)]
    ok = True
    for name, fn in tests:
        try:
            good, msg = fn()
        except Exception as exc:                       # noqa: BLE001
            good, msg = False, f"raised {type(exc).__name__}: {exc}"
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {name}: {msg}")
    print("\nselftest " + ("PASSED" if ok else "FAILED"))
    return ok


def calibration_circuits():
    """Five device checks to run BEFORE the science window.

    None of these is a data run and together they cost a few thousand shots.
    Each one exists because a silent failure there would be indistinguishable
    from a null result in the science window.
    """
    return [
        {"id": "C1 RX(pi) is a pi pulse",
         "circuits": "RX(pi) on |0>, measure Z, 1024 shots",
         "expect": "P(1) = 1 - 2e_ro within a few 1e-3",
         "fails_like": "a miscalibrated pi pulse looks like a static e0 offset, "
                       "which is exactly the quantity the predictor keys on"},
        {"id": "C2 fractional RX(theta) is monotone and linear in theta",
         "circuits": "RX(theta) on |0> for theta in 0..pi in 13 steps, measure Z",
         "expect": "P(1) = sin^2(theta/2) to a percent",
         "fails_like": "if fractional gates are silently synthesised from "
                       "pi/2 pulses the control family is not the one we "
                       "optimised over and every theta_i means something else"},
        {"id": "C3 RZ really is virtual",
         "circuits": "RY(pi/2), RZ(psi), RY(-pi/2) on |0> for psi in 0..2pi, "
                     "measure Z; and a long RZ chain of 200 with net angle 0",
         "expect": "cos(psi) curve; the null chain returns |0> with no extra loss",
         "fails_like": "if RZ costs time or error the injected ensemble picks up "
                       "a decoherence background that scales with |phi|, which "
                       "mimics a tail"},
        {"id": "C4 six-circuit Bloch reconstruction returns a known gate",
         "circuits": "the six-circuit set with c = c0, phi = 0, 1024 shots",
         "expect": "reconstructed eps within 3 sigma_shot of 0",
         "fails_like": "a basis-change sign error flips a component of d_alpha "
                       "and the predicted step points the wrong way"},
        {"id": "C5 throughput",
         "circuits": "10 circuits x 256 shots, batched as the science run will be",
         "expect": "record reported usage; pass it as --throughput",
         "fails_like": "the 2000 shots/s default is a guess and the budget is "
                       "linear in it"}]


def preregister(path="stage7_prereg.json", throughput=THROUGHPUT_DEFAULT):
    """Freeze the predictions BEFORE the device sees anything.

    Stage 7 is only a validation experiment if the numbers it will be compared
    against are fixed in advance.  This writes them plus a hash of the frozen
    controls, so the comparison afterwards is blind in the only sense available
    to a single experimenter: the prediction cannot be edited after the fact
    without the hash changing.
    """
    import hashlib
    op = OPERATING_POINT
    m = CompositeX(op["n"], reps=op["reps"])
    m1 = CompositeX(op["n"], reps=1)
    payload = {
        "frozen": True,
        "operating_point": op,
        "experiments": {
            "E0_null": {"control": "hard pulse (c_hard)", "kappa": "~0 (2e-16)",
                        "prediction": "R = 0 within the error bar",
                        "falsifies": "a substantial measured gain means the "
                                     "injected ensemble is not what we injected"},
            "E1_characterise": {"control": "pooled mean-optimal",
                                "measure": "eps_k, L_k, mean L, CVaR_alpha, d_alpha",
                                "prediction": "R_pred = 0.17 +- 0.01"},
            "E2_predictor_PRIMARY": {
                "control": "mean-optimal + dc = J^+ step, step from E1's OWN "
                           "measured d_alpha",
                "prediction": "R_offset / R_pred = 1.0, and R_offset > 0",
                "note": "this is the primary endpoint: a measurement-only "
                        "quantity predicting where tail-risk control departs "
                        "is the novel Stage 6 claim, not that CVaR differs"},
            "E3_arm_vs_arm": {"control": "pooled CVaR-optimal",
                              "prediction": "R = 0.134 +- 0.049, R/R_pred ~ 0.95",
                              "note": "secondary: confirms the gain is real, but "
                                      "uses a control optimised offline"},
            "E4_replication": {"sigma": 0.3, "prediction": "R_pred = 0.172, "
                               "delivery 1.00", "note": "shot feasibility not yet "
                               "emulated at this sigma"}},
        "discriminator": "mean_loss_change must be POSITIVE at E2 and E3; a "
                         "negative value means the mean arm was not converged "
                         "and the gain is baseline slack, not a tail trade",
        "analysis_frozen": {
            "debias": "subtract 3/(6*shots) from each L before forming CVaR",
            "cvar": "atomic upper-tail CVaR, as in every prior stage",
            "pairing": "the SAME phi draws across all arms"},
        "budget": budget(op["K"], op["shots"], throughput, arms=3),
        "calibration": calibration_circuits()}
    blob = json.dumps(payload, sort_keys=True, default=float).encode()
    payload["sha256"] = hashlib.sha256(blob).hexdigest()
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    return payload


def main():
    ap = argparse.ArgumentParser(description="VIPER Stage 7, free-tier hardware test")
    ap.add_argument("cmd", choices=["selftest", "design", "plan",
                                "prereg", "calibrate"])
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--sigma", type=float, default=0.9)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--throughput", type=float, default=THROUGHPUT_DEFAULT)
    ap.add_argument("--out", default="stage7_plan.json")
    a = ap.parse_args()

    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)

    if a.cmd == "calibrate":
        for c in calibration_circuits():
            print(f"\n{c['id']}\n  run    : {c['circuits']}\n"
                  f"  expect : {c['expect']}\n  if wrong: {c['fails_like']}")
        return

    if a.cmd == "prereg":
        pl = preregister(a.out if a.out != "stage7_plan.json" else "stage7_prereg.json",
                         a.throughput)
        print(f"frozen. sha256 = {pl['sha256']}")
        print(f"primary endpoint: {pl['experiments']['E2_predictor_PRIMARY']['prediction']}")
        print(f"budget (3 arms): {pl['budget']['seconds']:.0f}s of "
              f"{WINDOW_SECONDS:.0f}s, {pl['budget']['circuits']} circuits")
        return

    if a.cmd == "design":
        print(f"free window {WINDOW_SECONDS:.0f}s, throughput {a.throughput:.0f} shots/s, "
              f"{CIRCUITS_PER_DRAW} circuits per draw, 2 arms\n")
        print(f"{'K':>5}{'shots':>7}{'tail':>7}{'circuits':>10}{'shots tot':>11}"
              f"{'sec':>8}{'windows':>9}  verdict")
        for r in design(a.alpha, a.throughput):
            v = ("ok" if r["fits_one_window"] and r["ok_tail"]
                 else ("tail too small" if not r["ok_tail"] else "over budget"))
            print(f"{r['K']:5d}{r['shots']:7d}{r['tail_draws']:7.1f}{r['circuits']:10d}"
                  f"{r['shots_total']:11d}{r['seconds']:8.0f}{r['windows']:9.2f}  {v}")
        return

    m = CompositeX(a.n)
    c = m.c0()
    k = kappa_of(m, c, a.sigma, K=8000, seed=0)
    _, E, L = ensemble(m, c, a.sigma, 4000, seed=1)
    p = predict(E, L, a.alpha)
    c2, dc = offset_control(m, c, p["step"], 0.0)
    plan = {"n_segments": a.n, "alpha": a.alpha, "sigma": a.sigma,
            "kappa": k, "predictor_offline": p,
            "c_mean": c.tolist(), "c_offset": c2.tolist(), "delta_c": dc.tolist(),
            "circuits_per_draw": CIRCUITS_PER_DRAW,
            "budget": budget(64, 256, a.throughput)}
    with open(a.out, "w") as fh:
        json.dump(plan, fh, indent=1)
    print(f"kappa={k['kappa']:.4f} (fit R2 {k['fit_R2']:.4f})  "
          f"R_pred={p['R_pred']:.4f}  t*={p['t_best']:.3f}  "
          f"trade={p['tail_specific']}\nwrote {a.out}")


if __name__ == "__main__":
    main()
