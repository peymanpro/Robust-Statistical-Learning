# Project State

## Repository

`peymanpro/Robust-Statistical-Learning`

## Current Phase

**Phase 1 — Least Squares & Numerical Stability**

## Current Task

Phase 1.2 — Numerical Core.

## Verified Commit

`073b052` — `feat: define numerical array conventions`

## Completed

Phase 0 — Foundation.

Phase 1.1 — Least-squares mathematical foundation:
- Least-squares optimization formulation
- Residual and residual norm
- Matrix/vector conventions
- Numerical tolerances
- Mathematical invariant tests

## Current Implementation

- Least-squares residual/objective model
- Shared numerical array type conventions
- Numerical comparison tolerances
- Tests covering core mathematical identities

No least-squares solver has been implemented yet.

## Next

1. Establish Numerical Core boundaries.
2. Add input validation.
3. Add norm utilities.
4. Add condition-number calculation.
5. Add residual utilities where appropriate.
6. Then implement Normal Equations, QR, and SVD solvers.

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