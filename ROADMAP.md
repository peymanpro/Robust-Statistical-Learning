# Roadmap

Legend:

- [ ] Not started
- [~] In progress
- [x] Completed and verified
- [!] Blocked
- [-] Intentionally out of scope

## Phase 0 — Foundation

- [x] Repository structure
- [x] Python package configuration
- [x] Test structure
- [x] Ruff configuration
- [x] mypy configuration
- [x] Documentation system
- [x] Foundation tests

## Phase 1 — Least Squares & Numerical Stability

### Mathematical and Numerical Core

- [x] Least-squares formulation and residuals
- [x] Matrix/vector validation
- [x] L2 and Frobenius norms
- [x] Condition-number calculation
- [x] Normal Equations solver
- [x] QR-based solver
- [x] SVD-based solver
- [x] Numerical-rank and minimum-norm behavior

### Stability Analysis

- [x] Well-conditioned comparison
- [x] Ill-conditioned comparison
- [x] Hilbert-matrix study
- [x] Vandermonde study
- [x] Perturbation sensitivity
- [x] Observation-noise sensitivity
- [x] Forward-error diagnostics
- [x] Backward-error indicator
- [x] Residual orthogonality diagnostics
- [x] Consolidated error-analysis experiment
- [x] Phase findings documented

## Phase 2 — Regularization

### Mathematical and Numerical Core

- [x] Ridge formulation
- [x] Stable SVD-based Ridge solver
- [x] Generalized Tikhonov solver
- [x] SVD shrinkage interpretation
- [x] Generalized cross-validation score
- [x] GCV candidate selection

### Experiments

- [x] Controlled ill-conditioned noisy comparison
- [x] Regularization-path evidence
- [x] Reference-oriented interpretation
- [x] Statistical-vs-numerical trade-off documented

## Phase 3 — Principal Component Analysis

### Learning Core

- [x] Centering
- [x] SVD-based PCA
- [x] Component extraction
- [x] Explained variance
- [x] Explained-variance ratio
- [x] Transform
- [x] Inverse transform
- [x] Reconstruction error

### Experiments

- [x] Structured low-dimensional synthetic dataset
- [x] Reconstruction by component count
- [x] Numerical/invariant tests
- [x] PCA documentation

## Phase 4 — Real Data & Validation

- [x] Select regression dataset
- [x] Select PCA dataset
- [x] Reproducible reference validation
- [x] Compare least-squares coefficients
- [x] Compare PCA explained-variance ratios
- [x] Add validation to CI
- [x] Document limitations of reference comparisons

## Extension — Robust Regression

- [x] Huber loss
- [x] IRLS weighting
- [x] SVD-backed weighted least squares
- [x] Convergence contract
- [x] Outlier-sensitivity test
- [x] Controlled robustness documentation

## Engineering Completion

- [x] Error-diagnostic API
- [x] Representative reproducible experiments
- [x] Automated CI workflow
- [x] Public-facing README synchronized with implementation

## Explicitly Not Active

The following remain future scope unless explicitly promoted:

- sparse numerical methods
- iterative Krylov solvers
- LASSO / Elastic Net
- Bayesian uncertainty quantification
- streaming/incremental PCA
- GPU/CUDA implementation
- distributed execution
- REST serving layer
- NumPy/SciPy compatibility reimplementation

These are not required for the project's current portfolio role.
