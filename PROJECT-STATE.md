# Project State

## Repository

`peymanpro/Robust-Statistical-Learning`

## Current Phase

**Phase 1 — Least Squares & Numerical Stability**

## Current Task

Phase 1.6 — Numerical Stability Analysis.

Phase 1.5 (SVD-based Least Squares) is complete and validated.

## Verified Commit

`8f5ef1a` — `experiment: add well-conditioned solver comparison`

## Completed

Phase 0 — Foundation.

Phase 1.1 — Least-squares mathematical foundation:
- Least-squares optimization formulation
- Residual and residual norm
- Matrix/vector conventions
- Numerical tolerances
- Mathematical invariant tests

Phase 1.2 — Numerical Core:
- Core package boundaries
- Numerical validation
- L2 and Frobenius norms
- 2-norm condition number with numerical-rank handling
- Residual calculation
- Least-squares dimension validation

Phase 1.3 — Normal Equations:
- Normal Equations solver
- Exact and recoverable systems
- Overdetermined systems
- Failure conditions
- Conditioning experiment
- Normal Equations conditioning documentation

Phase 1.4 — QR-based Least Squares:
- QR requirements defined
- Reduced QR strategy selected
- QR least-squares solver implemented
- Reconstruction and orthogonality invariants tested
- Residual orthogonality tested
- Full-column-rank failure contract tested
- Comparison with Normal Equations
- Comparison with `numpy.linalg.lstsq`
- Ill-conditioning comparison documented

Phase 1.5 — SVD-based Least Squares:
- SVD requirements and solver contract defined
- Reduced SVD strategy selected
- `compute_svd` implemented with `full_matrices=False`
- `solve_svd` implemented using the SVD pseudoinverse
- Rank-deficient systems supported via numerical-rank truncation
- Reconstruction, orthogonality, and ordering invariants tested
- Residual orthogonality tested
- Minimum-norm behavior tested
- Comparison with `numpy.linalg.lstsq`
- Numerical-rank stability experiment recorded in `docs/experiments.md`

Phase 1.6 - Numerical Stability Analysis (partial):
- Well-conditioned solver comparison experiment added
- Evidence recorded in `docs/experiments.md`
- Findings: NE forward error grows as eps * kappa^2; QR and SVD
  forward error grows as eps * kappa; small residual does not
  imply accurate solution (kappa = 1e8 case)

## Current Implementation

The Numerical Core provides:
- Matrix and vector conventions
- Numerical tolerances
- Input validation
- Norm calculations
- Condition-number calculation
- Residual calculations

The learning layer provides:
- Least-squares mathematical model
- Normal Equations solver
- QR-based least-squares solver
- SVD-based least-squares solver

The SVD solver uses reduced SVD:

    A = U @ diag(s) @ V.T
    x = V @ diag(1/s) @ U.T @ b

Singular values below the numerical-rank threshold are truncated. Rank-deficient systems return the minimum-norm solution.

The QR solver uses reduced QR factorization:

    A = QR
    Rx = Q.T @ b

The QR solver requires full column rank and raises an explicit `ValueError` for rank-deficient systems.

## Validation

| Check | Status |
|---|---|
| pytest | PASS (55 tests) |
| Ruff | PASS |
| mypy | PASS (19 source files) |
| git diff --check | PASS |

## Next

Begin Phase 1.6 — Numerical Stability Analysis.

1. Well-conditioned controlled cases.
2. Ill-conditioned controlled cases.
3. Hilbert-matrix experiment.
4. Vandermonde-matrix experiment.
5. Perturbation experiments.
6. Noise experiments.
7. Residual vs forward-error analysis.
8. Backward-error analysis.
9. Normal Equations vs QR vs SVD comparison.

## Constraints

- Never hide failures.
- Never guess past failures.
- Never continue past unresolved failures.
- Inspect status and diff before commits.
- Keep commits atomic and meaningful.
- Keep numerical core independent from infrastructure.
- Use SOLID and design patterns only when they solve real problems.
- Mathematical claims require tests and reproducible evidence.
- Documentation must reflect the actual repository state.
