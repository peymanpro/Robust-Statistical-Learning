# Handoff Document

Read this to understand the project state before continuing work.

## Repository

https://github.com/peymanpro/Robust-Statistical-Learning

## Local Path

`E:\git-public-projects\Robust-Statistical-Learning` (Windows)

or

`/path/to/Robust-Statistical-Learning` (Unix)

## Branch

`main`

## Project Goal

Investigate numerical reliability in statistical learning methods.

**Core question**: When a mathematically valid learning method is executed using finite-precision arithmetic, when can the result become unreliable, and how can we make it more robust?

## MVP Scope

Four phases:

1. **Least Squares & Numerical Stability**: Normal Equations, QR, SVD; norms; condition numbers; residuals; forward/backward error; ill-conditioning; perturbations; reference validation
2. **Regularization**: Ridge/Tikhonov regularization; conditioning; noise sensitivity; bias; parameter selection
3. **PCA**: Principal Component Analysis; centering; variance; reconstruction; noise effects
4. **Real Data & Validation**: Dataset selection; reproducibility; accuracy; stability; performance; recommendations

Not in MVP scope: GPU, distributed computing, deep learning, REST API, microservices, Docker.

## Current Phase

Phase 0 — Foundation (documentation created; implementation not started)

## Current Task

Implement Phase 0 infrastructure:
1. Python package structure
2. Testing framework (pytest)
3. Code quality tooling (ruff, mypy)
4. Project configuration
5. First mathematical invariant tests
6. First commit

## Completed Work

- Repository created on GitHub (empty)
- All documentation files created:
  - README.md
  - ROADMAP.md
  - PROJECT-STATE.md
  - HANDOFF.md
  - docs/architecture.md
  - docs/mathematical-foundations.md
  - docs/numerical-stability.md
  - docs/design-decisions.md
  - docs/testing-strategy.md
  - docs/experiments.md
  - docs/future-scope.md

No source code implementation yet.

## Current State

```
Robust-Statistical-Learning/
├── README.md (public entry point)
├── ROADMAP.md (operational roadmap)
├── PROJECT-STATE.md (authoritative state)
├── HANDOFF.md (this file)
└── docs/
    ├── architecture.md
    ├── mathematical-foundations.md
    ├── numerical-stability.md
    ├── design-decisions.md
    ├── testing-strategy.md
    ├── experiments.md
    └── future-scope.md
```

Git status: clean (no commits yet; files staged for initial commit pending your review)

## Engineering Rules

### Code Quality
- Type hints required throughout
- Clean code principles: focused functions, meaningful names, explicit dependencies
- SOLID principles applied pragmatically (no unnecessary interfaces)
- Design patterns used only where they solve real problems

### Testing
- Four layers: mathematical invariants, numerical behavior, reference validation, engineering tests
- Never hide or skip unexplained test failures
- Numerical tolerances depend on scale, precision, algorithm, and conditioning
- No universal arbitrary tolerance threshold

### Numerical Responsibility
- Condition numbers and perturbation analysis explicit
- Residual errors calculated
- Forward and backward error measured
- Finite-precision effects documented
- Ill-conditioning distinguished from algorithmic instability

### Documentation
- All documentation reflects real repository state
- No invented progress; completed items require code proof
- STATUS fields are source of truth
- Documentation supports software, not replace it

### Git Discipline
- Inspect diff before commit
- No silent failures
- No destructive operations
- Meaningful commit messages
- Repository remains reproducible

## Architecture Constraints

### Numerical Core (must be pristine)
Responsibilities:
- Vectors and matrices (where actually needed)
- Decompositions (QR, SVD)
- Linear solvers
- Norms, condition numbers, error calculations
- Mathematical invariants

Must NOT depend on:
- HTTP, REST, databases
- Docker, microservices, deployment
- Experiment-specific code
- GPU libraries (post-MVP)

### Separation
Infrastructure → Application Services → Statistical Learning → Numerical Core

Each layer depends only on layers below.

### Key Design Decisions
- MVP before platform (build core first)
- Selected reimplementation (not full NumPy clone)
- Reference libraries for validation, not replacement
- Pragmatic SOLID (clarity over excessive abstraction)
- Documentation is operational (not decorative)

## Reference Libraries

- **NumPy**: reference behavior, validation, benchmarking
- **SciPy**: matrix operations, decompositions, validation
- **scikit-learn**: comparison, performance baseline

These are validation partners. The project does not attempt to replace them.

## Exact Next Action

1. Create `.gitignore` (Python standard)
2. Create `pyproject.toml` with package metadata
3. Create directory structure:
   ```
   src/robust_statistical_learning/
   ├── __init__.py
   ├── core/
   │   ├── __init__.py
   │   ├── types.py (type aliases, matrix/vector definitions)
   │   ├── norms.py
   │   ├── condition.py
   │   └── solvers.py
   └── learning/
       ├── __init__.py
       └── least_squares.py
   ```
4. Create `tests/` with pytest configuration
5. Write first test file: mathematical invariant tests for norms and matrix operations
6. Add `pytest.ini` and `pyproject.toml` test configuration
7. Run `pytest` (expect passing tests)
8. Add ruff and mypy configuration to `pyproject.toml`
9. Run `ruff check .` and `mypy src/` (expect success)
10. Review all files (no ChatGPT syntax, no fake commits, valid code)
11. Make first commit: "Initialize project structure with documentation and testing infrastructure"

Do not implement numerical algorithms yet. Focus on structure, configuration, and testing infrastructure.

## Handoff Rules

- This is a reference document; update it only when major milestones change
- Keep it short enough for a new AI agent to read in one minute
- Never include imaginary future progress
- Only reference real code and real commits
- When handing off again: update CURRENT PHASE, COMPLETED WORK, and NEXT ACTION
- Always verify Git status before closing a session
