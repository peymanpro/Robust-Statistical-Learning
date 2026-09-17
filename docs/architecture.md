# Architecture

## Architectural Intent

Architecture grows outward from a pristine mathematical core. The core contains numerically accurate, mathematically sound implementations with no external infrastructure dependencies. Layers above add statistical learning, orchestration, experiments, and (eventually) deployment infrastructure.

Each layer depends only on layers below. Lower layers never depend on upper layers.

> **Status note (operational).** The layered structure below describes the
> target architecture. The current repository implements a subset of it. In
> particular, least-squares solvers currently live under
> `src/robust_statistical_learning/learning/` (`normal_equations.py`,
> `qr.py`, `svd.py`) rather than under the Numerical Core as the diagram
> suggests, and the module layout under `core/` is consolidated
> (`metrics.py`, `residuals.py`, `validation.py`, `types.py`,
> `tolerances.py`) rather than decomposed into `norms.py`, `condition.py`,
> `decomposition.py`, `solvers.py`, `errors.py`, and `invariants.py`. The
> current implementation is the source of truth; this document is updated
> as the architecture evolves.


## Conceptual Architecture

```text
┌─────────────────────────────────────────────────┐
│  Infrastructure Layer                           │
│  (REST API, FastAPI, Docker, deployment, etc.) │
│  [POST-MVP]                                     │
└─────────────┬───────────────────────────────────┘
              │
              v
┌─────────────────────────────────────────────────┐
│  Application Services Layer                      │
│  (orchestration, dataset management,            │
│   experiment runners, reproducibility)          │
└─────────────┬───────────────────────────────────┘
              │
              v
┌─────────────────────────────────────────────────┐
│  Statistical Learning Layer                      │
│  (Least Squares, Ridge Regression, PCA)         │
└─────────────┬───────────────────────────────────┘
              │
              v
┌─────────────────────────────────────────────────┐
│  Numerical Core Layer                            │
│  (solvers, decompositions, norms,               │
│   condition numbers, error calculations)        │
└─────────────────────────────────────────────────┘
```

## 1. Numerical Core

### Purpose
Provide mathematically sound, numerically careful implementations of linear algebra and numerical methods.

### Responsibilities
- Vector and matrix types (or type aliases to `numpy.ndarray`)
- Norms: L2 norm, Frobenius norm, operator norms
- Condition number calculations: \(\kappa(A)\), \(\kappa(A^T A)\)
- Decompositions: QR, SVD
- Linear solvers: normal equations, QR-based, SVD-based
- Residual calculations: \(r = A\hat{x} - b\)
- Forward and backward error measurements
- Numerical stability checks and diagnostics

### Requirements
- **Type Safety**: all functions use type hints
- **Numerical Precision**: explicit handling of floating-point arithmetic
- **No Silent Failures**: all errors must be explicit
- **Mathematical Correctness**: algorithms match published formulations
- **Testability**: every function is covered by mathematical invariant tests

### What It Must NOT Depend On
- HTTP or REST frameworks
- Databases
- File I/O (except configuration)
- Docker or deployment infrastructure
- GPU libraries (post-MVP)
- Experiment-specific code
- Application orchestration

### Example Structure
```text
src/robust_statistical_learning/
├── core/
│   ├── types.py           # type aliases, matrix/vector definitions
│   ├── norms.py           # vector and matrix norms
│   ├── condition.py       # condition number calculations
│   ├── decomposition.py   # QR, SVD
│   ├── solvers.py         # least squares solvers
│   ├── errors.py          # forward error, backward error, residual
│   └── invariants.py      # mathematical invariant checks
```

## 2. Statistical Learning Layer

### Purpose
Implement concrete learning methods (Least Squares, Ridge Regression, PCA) using the Numerical Core.

### Responsibilities
- **Least Squares**: formulation, solver selection, error analysis
- **Ridge Regression**: regularization, parameter selection, noise robustness
- **PCA**: centering, dimensionality reduction, variance explanation
- Algorithm selection (which decomposition for which problem)
- Hyperparameter defaults and guidance
- Validation against mathematical formulations

### Interchangeable Solvers
Different solvers (normal equations, QR, SVD) may implement the same interface:

```python
class LeastSquaresSolver(Protocol):
    def solve(self, A: ndarray, b: ndarray) -> ndarray:
        """Solve ||Ax - b||_2^2"""
        ...

    def residual(self) -> float:
        """Return the residual norm"""
        ...
```

Multiple implementations can be swapped without changing client code.

### Example Structure
```text
src/robust_statistical_learning/
├── learning/
│   ├── least_squares.py   # Least Squares formulation
│   ├── ridge.py           # Ridge/Tikhonov formulation
│   ├── pca.py             # PCA formulation
│   └── solvers/           # solver implementations
│       ├── normal_eqn.py
│       ├── qr_based.py
│       └── svd_based.py
```

## 3. Application Services

### Purpose
Orchestrate experiments, manage datasets, ensure reproducibility, coordinate validation.

### Responsibilities
- Dataset loading and preprocessing
- Experiment runners (train/test splits, cross-validation)
- Reproducibility management (seeds, dimensions, parameter recording)
- Benchmarking (runtime, memory, numerical accuracy)
- Reference validation (comparison with NumPy/SciPy/scikit-learn)
- Result collection and analysis

### Constraints
- Must not introduce experiment-specific parameters into the core
- Must not hide or skip numerical failures
- Must record seeds, dimensions, and versions
- Must support both synthetic and real datasets

### Example Structure
```text
src/robust_statistical_learning/
├── experiments/
│   ├── runners.py          # experiment orchestration
│   ├── datasets.py         # dataset loading and generation
│   ├── reproducibility.py  # seed and state management
│   └── validation.py       # reference comparisons
```

## 4. Experiments & Evaluation

### Purpose
Validate implementations, gather evidence, guide recommendations.

### Methodology
- **Mathematical Invariants**: verify algorithm properties (e.g., \(Q^T Q \approx I\))
- **Numerical Behavior**: test perturbations, ill-conditioning, floating-point effects
- **Reference Validation**: compare to NumPy, SciPy, scikit-learn
- **Engineering Tests**: input validation, deterministic behavior, public API

### Evidence vs Interpretation
Every recommendation must be supported by reproducible experiments:

```
[Evidence] → [Interpretation] → [Recommendation]

Example:
Ill-conditioned matrices cause forward error to grow
  → This is a numerical stability problem
  → Use QR or SVD when condition number is high
```

### Reproducibility
Record:
- Random seed
- Problem dimensions
- Data scale and range
- Algorithm parameters
- Software versions (NumPy, SciPy, Python)
- System (CPU model, RAM)

## 5. Infrastructure (Post-MVP)

### Purpose
Deployment, scaling, and API exposure.

### Not in MVP

The following are future possibilities and are **NOT** part of the current implementation:

- **REST API**: FastAPI or Django REST Framework
- **Containerization**: Docker and container orchestration
- **Distributed Computing**: Dask, Ray, or MPI
- **GPU Acceleration**: CUDA, PyTorch, TensorFlow
- **Microservices**: service decomposition, inter-service communication
- **Caching and Persistence**: Redis, databases
- **Monitoring and Logging**: structured logging, metrics collection

These are considered only after the four-phase MVP is complete and validated.

## Design Principles

### SOLID (Applied Pragmatically)

#### S — Single Responsibility
Each function has one purpose. A solver solves. A norm calculates a norm. An experiment runner runs experiments.

#### O — Open/Closed
Code is open for extension (new solvers, new algorithms) but closed for modification of core interfaces.

#### L — Liskov Substitution
Different solver implementations can substitute for each other without breaking client code.

#### I — Interface Segregation
Interfaces are focused. Solvers are not required to implement unrelated methods.

#### D — Dependency Inversion
High-level statistical learning code depends on abstractions (solver interfaces), not concrete solver implementations.

**Note**: Not every module requires an interface. SOLID is applied where it solves real problems (e.g., interchangeable solvers), not everywhere.

### Design Patterns

Patterns are used **only where they solve real problems**:

- **Strategy**: different least-squares solvers (normal equations, QR, SVD) can be swapped
- **Factory**: solver construction based on problem properties (condition number, size)
- **Builder**: experiment configuration and setup
- **Template Method**: common experimental pipeline (setup → run → validate → report)

No pattern is used for its own sake. The goal is clarity and flexibility, not pattern count.

### Clean Code

- **Focused Functions**: functions do one thing well
- **Meaningful Names**: `condition_number()` not `cn()`; `solve_least_squares()` not `slsq()`
- **Explicit Dependencies**: functions receive what they need; no hidden globals
- **Type Hints**: all public functions have type hints
- **Testability**: functions are pure (or effects are explicit)
- **Low Coupling**: modules can be understood independently
- **High Cohesion**: related functionality is grouped
- **Clear Errors**: errors provide context; no silent failures

## Dependency Direction

```
Infrastructure
    ↑
    │
    depends on
    │
    ↓
Application Services
    ↑
    │
    depends on
    │
    ↓
Statistical Learning
    ↑
    │
    depends on
    │
    ↓
Numerical Core
```

**Core Rule**: The Numerical Core NEVER depends on anything above it.

## Future Extensibility

### Adding a New Algorithm (Example: Least Squares)

1. Define the mathematical formulation in `docs/mathematical-foundations.md`
2. Implement in `src/robust_statistical_learning/learning/`
3. Create or reuse appropriate solvers in `src/robust_statistical_learning/core/`
4. Add mathematical invariant tests in `tests/test_core_invariants.py`
5. Add numerical behavior tests in `tests/test_numerical_behavior.py`
6. Add reference validation tests in `tests/test_reference_validation.py`
7. Document results in experiment files
8. Update ROADMAP.md when complete

### Adding a New Decomposition (Example: LU)

1. Define mathematical formulation
2. Implement in `src/robust_statistical_learning/core/decomposition.py`
3. Write invariant tests
4. Update solver options in Statistical Learning layer
5. Benchmark against NumPy/SciPy
6. Document when and why to use it

## Summary

The architecture is **layered, dependency-ordered, and mathematically grounded**. It prioritizes:

1. **Mathematical Correctness** over performance optimization
2. **Numerical Stability** over arbitrary speed
3. **Clear Code** over clever code
4. **Explicit Dependencies** over hidden globals
5. **Testability** over convenience
6. **Evidence** over assumptions
