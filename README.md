# Reference Code

Reference implementation for:

"When does tail-risk optimization change a quantum gate?
An equivalence boundary for mean and CVaR control"

The repository contains the final research implementation, analysis
scripts, canonical datasets, figure-generation material, and hardware
validation code associated with the manuscript.

See `REFERENCE_MAP.md` for the correspondence between the code and
the research stages described in the paper.

## Main components

- `src/` contains the core simulation and experiment implementations.
- `analysis/predictor/` contains the measurement-only predictor analysis.
- `hardware/` contains the Stage 3 hardware implementation and raw datasets.
- `figures/` contains canonical figure inputs and assets.
- `paper/` contains the manuscript and Supplemental Material.

Historical development artifacts, caches, smoke tests, superseded
experiment configurations, and intermediate optimization outputs are
not included in this release.

The Stage 3 hardware experiment is a validation attempt of the
measurement/prediction protocol and does not constitute a positive
hardware demonstration of a CVaR advantage; the manuscript reports
the hardware null and background-floor diagnosis explicitly.
