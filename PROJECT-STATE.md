# Project State

## Repository

`peymanpro/Robust-Statistical-Learning`

## Current Phase

**Phase 1 — Least Squares & Numerical Stability**

## Current Task

Phase 1.3 — Normal Equations.

## Verified Commit

`32adadb` — `feat: validate least squares dimensions`

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

## Current Implementation

The Numerical Core now provides:
- Matrix and vector conventions
- Numerical tolerances
- Input validation
- Norm calculations
- Condition-number calculation
- Residual calculations

The learning layer keeps a compatibility-facing least-squares API and delegates residual calculations to the Numerical Core.

No least-squares solver has been implemented yet.

## Next

1. Define Normal Equations solver requirements.
2. Implement a solver for valid full-column-rank systems.
3. Test exact and recoverable systems.
4. Test overdetermined systems.
5. Define failure conditions.
6. Document conditioning implications.

## Validation

| Check | Status |
|---|---|
| pytest | PASS |
| ruff | PASS |
| mypy | PASS |
| git diff --check | PASS |

## Constraints

- Never hide failures.
- Never guess past failures.
- Inspect status and diff before commits.
- Keep numerical core independent from infrastructure.
- Use SOLID and design patterns only when justified.