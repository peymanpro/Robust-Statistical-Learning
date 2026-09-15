# Design Decisions

Key decisions guiding the project, documented as lightweight ADRs (Architecture Decision Records).

## DD-001 — Project Purpose: Understanding, Not Replacement

**Decision**

This project investigates numerical reliability in statistical learning methods. It is not intended to replace NumPy, SciPy, or scikit-learn.

**Reason**

Established scientific libraries exist, are heavily optimized, and solve practical problems at scale. This project answers a different question: **when does a mathematically valid method become numerically unreliable?** and **how can we make it more robust?**

By focusing on understanding rather than replacement, the project can:
- Study algorithms in depth without pressure to optimize for speed
- Investigate numerical properties that production libraries abstract away
- Use reference implementations for validation without competing against them
- Maintain mathematical clarity without deployment complexity

**Consequence**

- Reference libraries (NumPy, SciPy, scikit-learn) are trusted partners for validation
- Selected algorithms are implemented explicitly when they provide mathematical or numerical value
- The project is not positioned as a general-purpose linear algebra library
- Performance is secondary to clarity and correctness

## DD-002 — MVP Before Platform

**Decision**

Implement four focused phases (Least Squares, Regularization, PCA, Real Data) before considering deployment infrastructure (REST API, Docker, GPU, microservices).

**Reason**

Premature infrastructure is a common failure mode:
- Infrastructure decisions constrain algorithm design
- Deployment complexity obscures mathematical decisions
- API design for unstable code leads to poor abstraction
- Hardware-specific optimization (GPU) can't be done cleanly without stable algorithms

**Consequence**

- Phase 0-4 deliver a complete, validated, mathematically sound project on its own
- Infrastructure is post-MVP and optional
- The core remains deployable in many ways later
- Evaluation is based on correctness and stability, not performance

## DD-003 — Reference Libraries for Validation

**Decision**

Use NumPy, SciPy, and scikit-learn as:
- Reference behavior for correctness checks
- Benchmark targets for performance
- Source of validation datasets
- Comparison partners in experiments

Not as:
- Libraries to replace
- Performance targets (we will not outperform them)
- Design constraints

**Reason**

These libraries are well-tested, widely used, and implement algorithms conservatively. Comparing to them:
- Builds confidence in correctness
- Identifies potential numerical issues
- Establishes realistic expectations
- Validates experimental methodology

**Consequence**

- Every non-trivial algorithm compares to reference implementations
- Implementation differences must be understood and documented
- Forward error comparisons are standard practice
- Performance comparisons acknowledge that we are not optimized for speed

## DD-004 — Selected Reimplementation

**Decision**

Selected numerical capabilities will be implemented explicitly when doing so provides **mathematical and numerical value**.

Examples of justifiable reimplementation:
- QR decomposition with different algorithmic approaches (to understand stability)
- SVD with clear pedagogical presentation
- Condition number calculation with numerical precision tracking

Examples of unjustifiable reimplementation:
- Linear algebra that simply duplicates NumPy with no new insight
- Optimization of core loops without numerical investigation
- API cloning without understanding the underlying methods

**Reason**

Reimplementation must have purpose. Understanding the mathematical properties of an algorithm is a valid reason; performance optimization is not a valid reason for this project.

**Consequence**

- Not every algorithm is reimplemented
- Where reimplementation occurs, it serves educational and investigative purposes
- Some code may use NumPy/SciPy operations directly when that is appropriate
- The project is not a tutorial that reimplements everything

## DD-005 — Pragmatic SOLID

**Decision**

Apply SOLID principles where they solve real problems (interface segregation for interchangeable solvers), not everywhere.

**Reason**

SOLID is a guide to good design, not a checklist. Overzealous application creates unnecessary abstraction layers and obscures the mathematics.

Examples of justified SOLID application:
- Strategy pattern for interchangeable least-squares solvers (normal equations, QR, SVD)
- Interface segregation for solver implementations
- Dependency inversion so high-level code does not depend on low-level solver details

Examples of unjustified SOLID application:
- A NormCalculator interface with multiple implementations (unnecessary)
- A FactoryFactory for creating solvers (over-engineered)
- Abstract base classes for every concept (overhead without benefit)

**Consequence**

- Code is clean but pragmatic
- Interfaces exist where they solve real coupling problems
- Simple direct code is preferred to over-engineered abstractions
- SOLID is applied to the extent it improves clarity, not as dogma

## DD-006 — Documentation Is Operational

**Decision**

All documentation is operational. It reflects the actual state of the repository and supports the software project.

Documentation includes:
- ROADMAP with precise task lists that match implementation
- PROJECT-STATE with current implementation status
- Architecture that aligns with actual code structure
- Mathematical foundations that directly map to implementations
- Testing strategy that describes what is actually tested

Documentation does **not** include:
- Marketing language or exaggeration
- Imaginary future progress
- Completed items without code proof
- Redundant textbook-style explanation
- Abandoned ideas presented as if they are active

**Reason**

The documentation must be usable by:
1. A recruiter/engineer viewing the GitHub repository
2. An AI agent continuing the project in a future session

Both need accurate, current information. Outdated or misleading documentation destroys trust and creates confusion.

**Consequence**

- Documentation must be updated whenever implementation changes significantly
- No item is marked completed unless code proves it
- Architecture documents describe actual structure, not aspirational structure
- ROADMAP is a living operational document, not a vision statement

## DD-007 — Type Hints for Clarity

**Decision**

All public functions must include type hints. Internal/private functions should also include type hints when not obvious.

**Reason**

- Enables static type checking (mypy)
- Clarifies function contracts
- Supports IDE autocompletion and refactoring
- Makes numerical types explicit

**Consequence**

- All contributions must pass mypy
- Code is longer (5-10% more lines) but clearer
- Type hints are verified in CI

## DD-008 — No Silent Failures

**Decision**

Every error must be explicit. No exceptions are silently caught and ignored.

**Reason**

Numerical computing is error-prone. Silent failures propagate bad results downstream and are extremely difficult to debug.

Examples of what NOT to do:

```python
# Don't do this
try:
    result = solver.solve(A, b)
except:
    return None  # Silent failure!

# Don't do this
if error > tolerance:
    # Just ignore
    pass
```

Examples of what TO do:

```python
# Do this
if error > tolerance:
    warnings.warn(f"Error {error} exceeds tolerance {tolerance}")

# Do this
if condition_number > threshold:
    raise ValueError(f"Problem ill-conditioned: κ={condition_number}")
```

**Consequence**

- All errors propagate with context
- Tests fail loudly when something is wrong
- Numerical warnings are explicit
- Debugging is easier

---

## Decision Review

These decisions form the foundation of the project:

1. **Purpose** (DD-001): Understand, don't replace
2. **Scope** (DD-002): MVP first, platform later
3. **Validation** (DD-003): Reference libraries as partners
4. **Implementation** (DD-004): Reimplement only with purpose
5. **Design** (DD-005): Pragmatic SOLID
6. **Documentation** (DD-006): Operational accuracy
7. **Code Quality** (DD-007): Type hints for clarity
8. **Reliability** (DD-008): No silent failures

These are not constraints; they are guides to maintaining clarity and focus.
