# Reference Implementation Map

This repository contains the reference implementation accompanying:

"When does tail-risk optimization change a quantum gate?
An equivalence boundary for mean and CVaR control"

## Structure

- `paper/` - manuscript and Supplemental Material
- `src/` - core simulation and experimental-stage implementations
- `analysis/` - figure and analysis scripts
- `analysis/predictor/` - Stage 6 measurement-only predictor analysis
- `hardware/` - Stage 3 hardware validation implementation
- `hardware/raw/` - canonical raw hardware datasets
- `figures/` - canonical figure inputs and figure assets

## Research stages

- Stage 0: equivalence-regime validation
- Stage 1: channel/mechanism experiments
- Stage 2: nonlinear and leakage-aware experiments
- Stage 3: colored-noise and mechanism analysis
- Stage 4: relaxation/rare-channel analysis
- Stage 5: predictor development
- Stage 6: predictor calibration, Ktrain control, offset decomposition,
  and low-kappa analysis
- Stage 7: hardware validation

The release intentionally excludes caches, Python bytecode, smoke-test
outputs, superseded experiment archives, and historical development artifacts.
