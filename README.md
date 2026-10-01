# Robust Statistical Learning

## Project Overview

A numerical and statistical learning project for studying a practical question:

> When mathematically valid learning methods run on finite-precision, noisy, or
> ill-conditioned data, what fails, why does it fail, and what makes the result
> more reliable?

The project is deliberately small enough to inspect mathematically while using
engineering practices expected from production-oriented machine-learning work:
clear contracts, tests, reproducible experiments, reference validation, and CI.

This project is for understanding and investigation. It is not a replacement
for NumPy, SciPy, or scikit-learn.

## What It Demonstrates

- Least-squares solvers using Normal Equations, QR, and SVD
- Numerical-rank detection and minimum-norm solutions
- Conditioning, perturbation sensitivity, forward error, backward error, and
  residual orthogonality
- Ridge and generalized Tikhonov regularization
- Data-driven Ridge parameter selection with generalized cross-validation
- SVD-based PCA, explained variance, transformation, and reconstruction
- Huber regression through iteratively reweighted least squares (IRLS)
- Controlled synthetic experiments and small real-data reference checks
- Automated quality checks with Ruff, mypy, pytest, and reproducible experiments

## Core Mathematical Foundation

Least squares:

    x_hat = arg min_x ||Ax - b||_2^2

Normal equations:

    A^T A x = A^T b

QR:

    A = QR
    Rx = Q^T b

SVD:

    A = U Sigma V^T

The central numerical warning is:

    kappa(A^T A) ~= kappa(A)^2

Ridge:

    x_alpha = arg min_x ||Ax - b||_2^2 + alpha ||x||_2^2

Huber loss:

    rho_delta(r) =
        0.5 r^2                         if |r| <= delta
        delta (|r| - 0.5 delta)         otherwise

## Architecture

    Experiments / validation
            |
            v
    Learning layer
    Least Squares | Ridge/Tikhonov | PCA | Robust Regression
            |
            v
    Numerical core
    Validation | norms | conditioning | residuals | error diagnostics
            |
            v
    NumPy

The numerical core remains independent from infrastructure and experiment code.

## Project Status

The original four-phase MVP is implemented:

| Phase | Scope | Status |
|---|---|---|
| 1 | Least Squares & Numerical Stability | Implemented |
| 2 | Regularization | Implemented |
| 3 | PCA | Implemented |
| 4 | Real Data & Validation | Implemented |

A focused robustness extension is also implemented:

| Extension | Scope | Status |
|---|---|---|
| Robust Regression | Huber loss + IRLS | Implemented |

## Experiments

The repository contains reproducible studies for:

- well-conditioned solver comparison
- ill-conditioned solver comparison
- Hilbert matrices
- Vandermonde matrices
- perturbations in A and b
- additive observation noise
- forward/backward/residual error diagnostics
- Ridge regularization under ill-conditioning and noise
- PCA reconstruction
- real-data reference validation

Each experiment is intended to separate mathematical claims from numerical
evidence. Results depend on the tested dimensions, data, floating-point
environment, and reference implementation.

## Documentation

- README and project-state documents in the repository root
- docs/architecture.md
- docs/mathematical-foundations.md
- docs/numerical-stability.md
- docs/error-analysis.md
- docs/regularization.md
- docs/pca.md
- docs/robust-regression.md
- docs/real-data-validation.md
- docs/testing-strategy.md
- docs/experiments.md
- docs/future-scope.md

## Engineering Standards

The repository uses:

- Python 3.12+
- NumPy 2+
- pytest
- Ruff
- mypy in strict mode
- optional SciPy/scikit-learn reference dependencies

CI verifies code quality, the automated test suite, and representative
experiments.

## Scope Boundary

This repository intentionally does not attempt to become:

- a general-purpose scientific-computing replacement;
- a high-performance GPU library;
- a production model-serving platform;
- an arbitrary collection of machine-learning algorithms.

Future additions should deepen the relationship between mathematics,
statistical behavior, numerical reliability, or learning-system design.

## Repository

https://github.com/peymanpro/Robust-Statistical-Learning
