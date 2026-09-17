# Roadmap

Legend:

- [ ] Not started
- [~] In progress
- [x] Completed
- [!] Blocked

## Phase 0 ? Foundation

- [x] Repository created
- [x] Python package structure created
- [x] Test structure created
- [x] `pyproject.toml` configured
- [x] `.gitignore` configured
- [x] pytest available
- [x] Ruff configured
- [x] mypy configured
- [x] Documentation system created
- [x] MVP boundary defined
- [x] Architecture defined
- [x] Foundation test implemented
- [x] Foundation validation passed
- [x] Foundation committed

---

# Phase 1 ? Least Squares & Numerical Stability

## 1.1 Mathematical Foundation

- [x] Define the least-squares optimization problem
- [x] Define residual and residual norm
- [x] Define required vector and matrix conventions
- [x] Define numerical tolerances
- [x] Add mathematical invariant tests

## 1.2 Numerical Core

- [x] Establish core package boundaries
- [x] Add required numerical utility functions
- [x] Add input validation
- [x] Add norm calculations
- [x] Add condition-number calculation
- [x] Add residual calculation

## 1.3 Normal Equations

- [x] Implement Normal Equations solver
- [x] Test exact and recoverable systems
- [x] Test overdetermined systems
- [x] Test failure conditions
- [x] Document conditioning implications

## 1.4 QR-based Least Squares

- [x] Define QR requirements
- [x] Implement or wrap the selected QR strategy
- [x] Test reconstruction and invariants
- [x] Implement QR-based least squares
- [x] Compare against reference behavior

## 1.5 SVD-based Least Squares

- [x] Define SVD requirements and solver contract
- [ ] Implement or wrap the selected SVD strategy
- [ ] Test reconstruction and invariants
- [ ] Implement SVD-based least squares
- [ ] Test rank-deficient and numerical-rank behavior
- [ ] Compare against reference behavior

## 1.6 Numerical Stability Analysis

- [ ] Well-conditioned cases
- [ ] Ill-conditioned cases
- [ ] Hilbert-matrix experiment
- [ ] Vandermonde-matrix experiment
- [ ] Perturbation experiments
- [ ] Noise experiments
- [ ] Residual vs forward-error analysis
- [ ] Backward-error analysis
- [ ] Normal Equations vs QR vs SVD comparison

## 1.7 Phase 1 Findings

- [ ] Evidence summary
- [ ] Failure-case summary
- [ ] Practical solver-selection guidance
- [ ] Phase 1 documentation update

---

# Phase 2 ? Regularization

## 2.1 Mathematical Foundations

- [ ] Ridge formulation
- [ ] Tikhonov formulation
- [ ] Conditioning effect
- [ ] Bias/stability trade-off
- [ ] Parameter-selection concepts

## 2.2 Numerical Core

- [ ] Ridge solver
- [ ] Tikhonov solver
- [ ] SVD interpretation
- [ ] Parameter-selection implementation

## 2.3 Experiments

- [ ] Noise sensitivity
- [ ] Regularization path
- [ ] Ill-conditioned improvement
- [ ] Reference comparison
- [ ] Practical recommendations

---

# Phase 3 ? Principal Component Analysis

## 3.1 Mathematical Foundations

- [ ] Centering
- [ ] PCA formulation
- [ ] SVD interpretation
- [ ] Explained variance
- [ ] Reconstruction error

## 3.2 Numerical Core

- [ ] Centering operation
- [ ] SVD-based PCA
- [ ] Component extraction
- [ ] Explained variance
- [ ] Reconstruction

## 3.3 Experiments

- [ ] Synthetic structured data
- [ ] Reconstruction experiments
- [ ] Noise sensitivity
- [ ] Reference comparison
- [ ] Real small dataset

---

# Phase 4 ? Real Data & Validation

- [ ] Select a small regression dataset
- [ ] Select a small PCA dataset
- [ ] Define reproducible preprocessing
- [ ] Run regression experiments
- [ ] Run regularization experiments
- [ ] Run PCA experiments
- [ ] Compare accuracy
- [ ] Compare numerical stability
- [ ] Compare runtime
- [ ] Analyze failure cases
- [ ] Publish practical recommendations

---

# Post-MVP ? Future Scope

## Numerical Extensions

- [ ] Additional matrix decompositions
- [ ] Iterative linear solvers
- [ ] Eigenvalue methods
- [ ] Sparse numerical methods

## Optimization

- [ ] Gradient Descent
- [ ] Newton
- [ ] Quasi-Newton
- [ ] Coordinate Descent
- [ ] Convex Optimization
- [ ] Constrained Optimization

## Deep Learning

- [ ] Neural-network numerical foundations
- [ ] Backpropagation
- [ ] Automatic differentiation foundations
- [ ] Neural-network optimization
- [ ] GPU execution

## Platform

- [ ] REST API
- [ ] FastAPI and/or Django
- [ ] Docker
- [ ] Distributed computation
- [ ] Microservices

## Compatibility / Reimplementation

- [ ] Selected NumPy-compatible functionality
- [ ] Selected SciPy-compatible functionality
- [ ] Reference performance comparison

Post-MVP items are not active commitments until explicitly promoted.
