# Handoff

## Repository

https://github.com/peymanpro/Robust-Statistical-Learning

## Local Path

`E:\git-public-projects\Robust-Statistical-Learning`

## Branch

`main`

## Verified Commit

`371e1e7` — `experiment: add ill-conditioned solver comparison`

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

Phase 1.6 — Numerical Stability Analysis.

Well-conditioned and ill-conditioned solver comparison experiments
are complete and recorded. The remaining Phase 1.6 items are the
Hilbert-matrix experiment, the Vandermonde-matrix experiment,
perturbation and noise experiments, residual vs forward-error
analysis, backward-error analysis, and a consolidated cross-solver
comparison.

## Completed

- Phase 0 Foundation
- Phase 1.1 Least-squares mathematical foundation
- Phase 1.2 Numerical Core
- Phase 1.3 Normal Equations
- Phase 1.4 QR-based Least Squares
- Phase 1.5 SVD-based Least Squares

## Current State

The project now contains three validated least-squares solvers:

- Normal Equations
- QR-based least squares
- SVD-based least squares

SVD has explicit tests for:
- reduced shape and orthogonality invariants
- reconstruction and singular-value ordering
- reference agreement with `numpy.linalg.lstsq`
- residual orthogonality
- rank-deficient minimum-norm behavior
- numerical-rank truncation
- `rcond` validation

The latest verified commit is:

`371e1e7` — `experiment: add ill-conditioned solver comparison`

The branch contains local commits that have not been pushed to
`origin/main` (currently at `1e173aa`). Run `git status --short --branch`
for the current count. No push is performed without explicit instruction.

## Exact Next Action

Continue Phase 1.6 — Numerical Stability Analysis. The next experiment
is a Hilbert-matrix study, followed by a Vandermonde-matrix study, to
test the same solvers on matrices whose structure produces strong
ill-conditioning without relying on the controlled-singular-value
family used so far.

Each experiment follows the structure:

Mathematical Question -> Hypothesis -> Controlled Setup -> Measurement ->
Evidence -> Interpretation -> Limitation -> Recommendation.

## Engineering Rules

- Never hide failures.
- Never guess past failures.
- Never continue past unresolved failures.
- Keep commits atomic and meaningful.
- Keep numerical core independent from infrastructure.
- Use SOLID and design patterns only when they solve real problems.
- Mathematical claims require tests and reproducible evidence.

## Environment Notes

- Windows PowerShell 5.1's `Set-Content -Encoding UTF8` writes a UTF-8 BOM.
  When writing files in this repository, prefer Python's
  `Path.write_text(..., encoding="utf-8")` or
  `[System.IO.File]::WriteAllText($path, $text, [System.Text.UTF8Encoding]::new($false))`.
  See commit `ba8c07c` for the BOM removal fix that motivated this rule.

## Handoff Rule

Before ending a substantial session:

1. Update `ROADMAP.md`.
2. Update `PROJECT-STATE.md`.
3. Update `HANDOFF.md`.
4. Record validation results.
5. Record the last verified commit.
6. Leave the working tree clean when possible.
