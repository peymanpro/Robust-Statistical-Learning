# Project State

## Repository

peymanpro/Robust-Statistical-Learning

## Current Phase

Core MVP + Robust Regression extension; awaiting CI-backed verification of the
current public source.

## Current Implementation

### Numerical Core

- finite-value and shape validation
- L2 and Frobenius norms
- 2-norm condition number with numerical-rank handling
- residual calculation
- forward-error measurement
- normwise backward-error indicator
- residual-orthogonality diagnostic

### Least Squares

- Normal Equations
- reduced QR
- reduced SVD
- rank-deficient minimum-norm behavior for SVD
- numerical-rank truncation
- reference comparison against NumPy

### Regularization

- SVD-based Ridge regression
- generalized Tikhonov via an augmented SVD system
- generalized cross-validation (GCV)
- candidate-grid alpha selection

### PCA

- centered-data SVD
- immutable PCAResult
- transform / inverse_transform
- explained variance
- explained-variance ratio
- reconstruction error

### Robust Regression

- Huber loss
- IRLS weighting
- SVD-backed weighted least squares
- explicit convergence configuration and result object

### Experiments

- well-conditioned comparison
- ill-conditioned comparison
- Hilbert matrix
- Vandermonde matrix
- perturbation sensitivity
- noise sensitivity
- forward/backward/residual diagnostics
- regularization comparison
- PCA reconstruction
- real-data validation

### Quality / CI

- Python 3.12
- Ruff
- strict mypy
- pytest
- representative experiment execution
- SciPy / scikit-learn reference validation

## Development Commits

| Commit | Purpose |
|---|---|
| 24cdfe7 | least-squares forward/backward/residual error diagnostics |
| dac36eb | Ridge, Tikhonov, and PCA learning layer |
| 2774c61 | Huber regression, experiments, and CI workflow |

Earlier commits contain the validated Normal Equations, QR, SVD, conditioning,
and controlled stability studies.

## Verification

A fresh local clone could not be performed in this session because external
repository access from the local runtime was unavailable.

The CI workflow in .github/workflows/ci.yml is intended to be the authoritative
fresh verification path. After it executes, this document should record the run
status and any failures before claiming final completion.

## Next

1. Verify the CI workflow for the current source.
2. Reconcile the Git Projects Phase control file with the actual repository.
3. Keep future extensions focused on numerical reliability, robust statistics,
   optimization, uncertainty, or statistically grounded decision support.

## Constraints

- Never hide failures.
- Never guess past failures.
- Numerical claims require reproducible evidence.
- Keep the numerical core independent from infrastructure.
- Keep commits atomic and meaningful.
- Update documentation when implementation state changes.
