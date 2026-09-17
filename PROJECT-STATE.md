# Project State

## Repository

`peymanpro/Robust-Statistical-Learning`

## Current Phase

**Phase 1 — Least Squares & Numerical Stability**

## Current Task

Phase 1.5 — SVD-based Least Squares.

Requirements and solver contract are defined before implementation.

## Verified Commit

`ed2cdc2` — `test: complete QR solver contract coverage`

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

The QR solver uses reduced QR factorization:

    A = QR
    Rx = Q.T @ b

The QR solver currently requires full column rank and raises an explicit `ValueError` for rank-deficient systems.

## Validation

| Check | Status |
|---|---|
| pytest | PASS |
| Ruff | PASS |
| mypy | PASS |
| git diff --check | PASS |

## Next

1. Implement `compute_svd` using reduced SVD.
2. Implement `solve_svd` using the SVD pseudoinverse.
3. Support rank-deficient systems through numerical-rank truncation.
4. Test reconstruction, orthogonality, ordering, and solver invariants.
5. Compare against `numpy.linalg.lstsq`.

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