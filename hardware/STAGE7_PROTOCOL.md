# VIPER Stage 7 — hardware test on ₹0

## The constraint that reshaped it

Pulse-level control on IBM hardware was **removed in early 2025**; `qiskit.pulse`
is deprecated for removal in Qiskit 2.0. There is no free platform on which you
can send a 52-coefficient shaped pulse to a transmon. IQM Resonance advertises
pulse-level access (IQM Pulla) and a free Starter tier of 30 credits/month, but
does not say the two overlap — worth one email, not worth planning around.

So Stage 7 does not ship a pulse. It re-instantiates the *same mathematics* on
the smallest control family a free QPU will execute.

## The design

| | |
|---|---|
| **control** | `RZ(ψ₃)RX(θ₃) · RZ(ψ₂)RX(θ₂) · RZ(ψ₁)RX(θ₁)`, 6 real parameters |
| **hardware** | IBM **fractional gates** give continuous `RX(θ)` on Heron; `RZ` is a frame change — exact, zero duration, zero error |
| **noise** | quasi-static dephasing **injected** as virtual `RZ(φ/n)` kicks, φ ~ N(0, σ²) |
| **error coords** | 3 (SU(2)) — the same `n_coord: 3` as every Stage 6 cell |
| **loss** | `L = εᵀWε` with `W = I/6`, which *is* the average gate infidelity to leading order |
| **κ** | fitted in the standardised noise `z = φ/σ` — but see the correction below: it does **not** land on the paper's axis |

Injection is the point, not a compromise: σ is set exactly, so **κ is a dial**
rather than a measurement. The device's own T1, T2, control and readout errors
remain as an uncontrolled background — that is the part that makes it a test.

## Correction: κ at this operating point is 3.63, not 0.13

An earlier draft of this document quoted κ = 0.13 and claimed the hardware point
sits on the same axis as Fig. 16. Both halves were wrong, and not through a
convention clash — both numbers use the standardised `z = φ/σ`.

κ = 0.13 is the **split ansatz** `c0`. The experiment runs at the **pooled
mean-optimal** control, where κ = 3.63. The gap is physics: optimising the mean
over a 3-segment composite finds a dephasing-**refocused** pulse, collapsing the
first-order response |A| from 0.388 to 0.0046 — an 84× cancellation — while |Q|
barely moves. κ = |Q|/|A| diverges precisely because the mean arm is good.

At σ = 0.6 the three contributions to ε in the standardised variable are

```
|e0| = 0.0159     |A| = 0.0046     |Q| = 0.0167
```

a static offset and a quadratic term with almost no linear part between them.
**That triple is the honest description of the regime**; κ is a badly
conditioned way to say it, being a ratio of a small number to a smaller one.
Report the triple, quote κ second, and state plainly that the point sits
**outside** the paper's perturbative band (grid 0.025–0.5, validity capped at 1.0).

This also forces `n = 3`: at n = 2 and n = 4 the mean optimiser drives the gate
to machine-exact (|A| = |e0| = 0, surrogate fit R² = 0.02) and there is nothing
left to measure.

## Three things the offline model already establishes

**1. One hard pulse is exactly equivalent.** Put the whole π in one segment and
the kicks commute through, giving `ε(φ) = −(φ/n)ẑ` to machine precision — perfectly
linear, so **κ = 2×10⁻¹⁶** and Theorem 1 forces mean ≡ CVaR. Measured `R_pred =
1.8×10⁻⁵`. This is a free, sharp null: a *measured* gain here would mean the
injected ensemble is not the one we think we injected.

**2. Sharing the π across segments puts κ on the paper's grid.** The kicks stop
commuting and κ runs 0.043 → 0.38 as σ runs 0.2 → 1.6, with the κ² onset holding
to a spread of 1.14× (`R_pred/κ² = 2.56, 2.86, 2.91, 2.82`).

**3. The mean arm self-refocuses, and that is where the gain comes from.** The
pooled mean optimisation finds a dephasing-**refocused** composite pulse:
first-order response cancelled, infidelity 9.7×10⁻⁵. The residual error is then
purely quadratic, which is exactly the regime the theorem says CVaR should
depart from the mean. Result at σ = 0.6:

```
R = +0.158    R_pred = +0.166    R/R_pred = 0.95    mean-loss ratio 1.23 (genuine trade)
```

`R/R_pred = 0.95` against the simulation's `s = 0.588` is itself a finding — the
delivery fraction is not universal, which the paper already declines to claim.

## Two failures the emulator caught before any device time

Both would have burned a free window.

**The linear-offset move fails at `c0` and works at the optimised control.**
At the split ansatz, `δc = J⁺·step` gives a *negative* R at every σ, even exactly
and noiselessly, with no rescaling that rescues it — changing θᵢ moves the offset
and the dephasing sensitivity together, and the tilt costs more in the tail than
the shift saves. At the **pooled mean-optimal** control the same move delivers
**1.00, 1.01, 1.02** of the predicted gain at σ = 0.3, 0.6, 0.9.

So the delivery fraction is a property of *where you stand*, not of the method —
which is worth carrying back into the paper, where `s = 0.588` is quoted as a
single constant. It also means the predictor test is the stronger hardware
experiment, not the weaker one, and it is now the primary endpoint.

**A refocused gate is invisible at 256 shots.** `|ε| = 0.024` sits under the
`1/√shots = 0.06` Bloch-component floor: `R = −0.03 ± 0.22` against a truth of
`+0.16`. Fix is **amplification** — repeat the sequence 16× at *fixed* φ, which
is legitimate precisely because the noise is quasi-static by construction.

Then the second surprise: **more shots did not help.** 256 → 2048 shots/draw left
the error bar flat. The dominant variance is the CVaR estimator's own, not
binomial. Spending the budget on *draws* instead cut it fourfold.

## Operating point

```
n = 3 segments    σ = 0.6 rad    α = 0.2    reps = 16
K = 256 draws     shots = 128    6 circuits/draw    2 arms
→ 3072 circuits, 393k shots, ~197 s of a 600 s window
```

Emulated outcome, 16 repeats: **R = 0.134 ± 0.049**, true value 0.150, positive
in **16/16**. The ~11% low bias is residual shot bias on `|ε|²` and is correctable.

## The frozen experiment

```bash
python3 stage7.py selftest     # 11 checks, no hardware, no Qiskit, no network
python3 stage7.py calibrate    # the five device checks, with what each failure mimics
python3 stage7.py prereg       # freezes the predictions + sha256
python3 stage7.py design       # budget table
```

**Nothing is tuned on the QPU.** Predictions are frozen and hashed before the
device sees anything; afterwards the only permitted operation is comparison.
Analysis choices are frozen too — debias by `3/(6·shots)`, atomic upper-tail
CVaR, same φ draws across every arm.

| | experiment | prediction |
|---|---|---|
| **E0** | hard pulse, κ ≈ 0 | **R = 0.** A substantial gain here falsifies the injected ensemble, not the theory |
| **E1** | mean-optimal control, measure ε_k, L_k, d_α | R_pred = 0.17 ± 0.01 |
| **E2** | **PRIMARY** — offset by `δc = J⁺·step`, step from E1's *own measured* d_α | **R_offset/R_pred = 1.0**, R_offset > 0 |
| **E3** | CVaR-optimal control (secondary) | R = 0.134 ± 0.049, R/R_pred ≈ 0.95 |
| **E4** | replication at σ = 0.3 | R_pred = 0.172, delivery 1.00 — *shot feasibility not yet emulated* |

E2 is primary because the novel Stage 6 claim is not "CVaR gives a different
pulse" but "a measurement-only quantity predicts when and in which direction it
departs". E2 tests exactly that, and it needs no offline CVaR optimisation — the
step comes from the hardware's own measurements through an analytic Jacobian.

**The discriminator gates both.** `mean_loss_change` must come out *positive* at
E2 and E3. Negative means the mean arm was not converged and the "gain" is
baseline slack — the Stage 6 trap, which a miscalibrated device reproduces.

Three arms at K = 256, shots = 128: **4608 circuits, 295 s of a 600 s window.**
Reuse the φ draws across all arms — it is paired, so the ensemble draw cancels
and the error bar sits on the difference.

## Claim boundary — keep this in the first sentence of the section

What is tested: **the Stage 6 mechanism, on a hardware-executable control family,
under synthetically injected quasi-static dephasing.**

What is *not* tested: the 52-coefficient VIPER pulse on a real transmon; and the
device's own naturally occurring CVaR distribution. The ensemble is injected, not
inherited. Saying so first is what makes the rest of the section credible.

## Account setup

IBM Quantum Open Plan — free, no card, **10 minutes per 28-day rolling window**,
us-east region only. Sign up at `quantum.cloud.ibm.com`, take the API token, and
load the backend with `use_fractional_gates=True` (required — `RX(θ)` and
`RZZ(θ)` only appear in the target with that flag).

**Calibrate the throughput before trusting the budget.** The 2000 shots/s default
in `stage7.py` is a guess. Submit 10 circuits × 256 shots, read the reported
usage, and pass the measured number as `--throughput`. My ETAs have been wrong by
8× before; this one is a parameter for that reason.
