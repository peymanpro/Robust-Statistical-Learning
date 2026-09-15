# Roadmap

Operational roadmap using GitHub task-list syntax:
- `[ ]` Not started
- `[~]` In progress
- `[x]` Completed
- `[!]` Blocked

## Phase 0 — Foundation

### Project Setup
- [ ] Python package structure (`src/robust_statistical_learning/`)
- [ ] `pyproject.toml` configuration
- [ ] `.gitignore` (Python standard)
- [ ] `pytest.ini` configuration
- [ ] ruff linter configuration
- [ ] mypy type checker configuration
- [ ] Initial commit

### Testing Infrastructure
- [ ] pytest setup with basic test structure
- [ ] First test file (mathematical invariant tests)
- [ ] Test utilities for numerical comparisons
- [ ] CI/CD hooks (local validation before commit)

### Documentation
- [x] README.md
- [x] ROADMAP.md
- [x] PROJECT-STATE.md
- [x] HANDOFF.md
- [x] docs/architecture.md
- [x] docs/mathematical-foundations.md
- [x] docs/numerical-stability.md
- [x] docs/design-decisions.md
- [x] docs/testing-strategy.md
- [x] docs/experiments.md
- [x] docs/future-scope.md

## Phase 1 — Least Squares & Numerical Stability

### Mathematical Foundations
- [ ] Least-squares formulation and definition
- [ ] Vector and matrix norms (L2, Frobenius)
- [ ] Residual definition and calculation
- [ ] Condition number definition and measurement
- [ ] Forward error definition
- [ ] Backward error definition
- [ ] Ill-conditioning concepts and examples
- [ ] Floating-point arithmetic effects

### Numerical Core — Algorithms
- [ ] Vector and matrix types (or numpy.ndarray with type aliases)
- [ ] Norm calculations (L2, Frobenius, operator norms)
- [ ] Condition number calculations
- [ ] Residual calculations
- [ ] Forward/backward error measurements
- [ ] Normal Equations solver
- [ ] QR decomposition (via numpy or custom)
- [ ] QR-based least-squares solver
- [ ] SVD (via numpy or custom)
- [ ] SVD-based least-squares solver
- [ ] Solver abstraction/interface
- [ ] Mathematical invariant tests

### Experiments
- [ ] Well-conditioned test cases
- [ ] Ill-conditioned test cases (Hilbert matrix, vandermonde)
- [ ] Perturbation experiments (noise on A and b)
- [ ] Noise sensitivity analysis
- [ ] Error analysis: residual vs forward error
- [ ] Comparison with NumPy/SciPy implementations
- [ ] Performance measurements (small to medium scale)
- [ ] Practical recommendations based on evidence

### Documentation Update
- [ ] Phase 1 results and findings
- [ ] Experimental evidence summary
- [ ] Recommendations document

## Phase 2 — Regularization

### Mathematical Foundations
- [ ] Ridge regression formulation
- [ ] Tikhonov regularization formulation
- [ ] Connection to condition number improvement
- [ ] Bias-variance tradeoff concept
- [ ] Regularization parameter selection methods

### Numerical Core — Algorithms
- [ ] Ridge regression solver
- [ ] Tikhonov regularization solver
- [ ] SVD-based regularization parameter interpretation
- [ ] GCV (Generalized Cross-Validation) for parameter selection
- [ ] L-curve method (optional)
- [ ] Noise robustness tests

### Experiments
- [ ] Regularization path visualization
- [ ] Parameter selection accuracy
- [ ] Noise sensitivity vs unregularized
- [ ] Ill-conditioned problem improvement
- [ ] Comparison with scikit-learn Ridge implementation
- [ ] Performance measurements
- [ ] Practical recommendations (when and why to regularize)

### Documentation Update
- [ ] Phase 2 results and findings
- [ ] Parameter selection guidance

## Phase 3 — Principal Component Analysis

### Mathematical Foundations
- [ ] PCA formulation (centering, covariance, SVD)
- [ ] Principal components and eigenvalues
- [ ] Explained variance ratio
- [ ] Reconstruction error
- [ ] Dimensionality reduction concept
- [ ] Numerical sensitivity in PCA

### Numerical Core — Algorithms
- [ ] Centering operation
- [ ] Covariance matrix calculation (careful numerics)
- [ ] SVD-based PCA implementation
- [ ] Principal component extraction
- [ ] Explained variance calculation
- [ ] Reconstruction from components
- [ ] Noise effect on principal components

### Experiments
- [ ] Synthetic data with known structure
- [ ] Variance explanation experiments
- [ ] Reconstruction error vs components kept
- [ ] Noise sensitivity on small datasets
- [ ] Centering necessity verification
- [ ] Comparison with scikit-learn PCA
- [ ] Performance measurements
- [ ] Real small dataset (synthetic or standard)

### Documentation Update
- [ ] Phase 3 results and findings

## Phase 4 — Real Data & Validation

### Dataset Selection
- [ ] Choose 2-3 small, well-understood datasets
- [ ] Verify reproducibility (provide seeds, preprocessing steps)
- [ ] Document dimensions, features, target variable
- [ ] Identify expected challenges (conditioning, noise, scale)

### Experiments
- [ ] Least Squares on real data
- [ ] Regularization parameter selection on real data
- [ ] Cross-validation setup and evaluation
- [ ] Prediction accuracy measurements
- [ ] Stability under small perturbations
- [ ] Numerical precision assessment
- [ ] Failure case identification and analysis
- [ ] Runtime measurements
- [ ] Comparison with scikit-learn implementations

### Validation
- [ ] Reproducible pipeline (fixed seeds, documented splits)
- [ ] Performance metrics (MSE, MAE, R²)
- [ ] Stability scores
- [ ] Robustness summary

### Documentation Update
- [ ] Findings and recommendations
- [ ] Final summary of Phase 1-4
- [ ] Practical guidance for practitioners

## Post-MVP Backlog

**Note**: These are possible future directions. They are NOT current implementation commitments.

### Numerical Extensions
- [ ] Additional matrix decompositions (LU, Cholesky)
- [ ] Iterative linear solvers (GMRES, CG)
- [ ] Eigenvalue methods
- [ ] Sparse numerical methods

### Optimization
- [ ] Gradient Descent (basic, with line search)
- [ ] Newton method
- [ ] Quasi-Newton methods (BFGS)
- [ ] Coordinate Descent
- [ ] Convex Optimization framework
- [ ] Constrained Optimization

### Deep Learning
- [ ] Neural network layers (linear, activation)
- [ ] Backpropagation implementation
- [ ] Automatic differentiation basics
- [ ] Optimization for neural networks
- [ ] Regularization techniques
- [ ] Small neural network on standard benchmark

### Platform
- [ ] REST API (FastAPI or similar)
- [ ] Docker containerization (optional)
- [ ] API documentation
- [ ] Web dashboard for experiments
- [ ] Distributed computation exploration

### Compatibility / Reimplementation
- [ ] Selected NumPy functions (not all)
- [ ] Selected SciPy functions (not all)
- [ ] API compatibility for educational purposes
- [ ] Performance comparison with NumPy/SciPy

All post-MVP work is exploratory and extensible.
