# Handoff

## Repository

https://github.com/peymanpro/Robust-Statistical-Learning

## Local Path

`E:\git-public-projects\Robust-Statistical-Learning`

## Branch

`main`

## Verified Commit

`ed2cdc2` — `test: complete QR solver contract coverage`

## Project Goal

Study the relationship between mathematical formulation, numerical algorithms,
finite-precision computation, statistical learning, and practical reliability.

The project is intended for understanding and investigation, not replacement of
NumPy, SciPy, or scikit-learn.

## MVP

1. Least Squares & Numerical Stability
2. Regularization
3. PCA
4. Real Data & Validation

## Current Phase

**Phase 1 — Least Squares & Numerical Stability**

## Current Task

Phase 1.5 — SVD-based Least Squares.

## Completed

- Phase 0 Foundation
- Phase 1.1 Least-squares mathematical foundation
- Phase 1.2 Numerical Core
- Phase 1.3 Normal Equations
- Phase 1.4 QR-based Least Squares

## Current State

The project now contains two validated least-squares solvers:

- Normal Equations
- QR-based least squares

QR has explicit tests for:
- reconstruction and orthogonality invariants
- residual orthogonality
- reference agreement with `numpy.linalg.lstsq`
- agreement with Normal Equations
- rejection of rank-deficient matrices

The latest verified commit is:

`ed2cdc2` — `test: complete QR solver contract coverage`

## Exact Next Action

Start Phase 1.5 by defining the SVD requirements and contract before writing
the implementation.

## Engineering Rules

- Never hide failures.
- Never guess past failures.
- Never continue past unresolved failures.
- Inspect status and diff before commits.
- Keep commits atomic and meaningful.
- Keep numerical core independent from infrastructure.
- Use SOLID and design patterns only when they solve real problems.
- Mathematical claims require tests and reproducible evidence.

## Handoff Rule

Before ending a substantial session:

1. Update `ROADMAP.md`.
2. Update `PROJECT-STATE.md`.
3. Update `HANDOFF.md`.
4. Record validation results.
5. Record the last verified commit.
6. Leave the working tree clean when possible.