# Numerical Stability

This document describes what numerical stability means in this project, how
conditioning and numerical rank are handled operationally, and what the
experiments have measured so far.

For the underlying mathematical definitions, see
[mathematical-foundations.md](mathematical-foundations.md). This document is
about the operational side: which quantities are computed, which tolerances
are used, and which failure modes have been observed.

## Scope

Numerical stability here refers to the behavior of least-squares solvers and
related operations when:

- input data are stored in IEEE 754 binary64 (float64)
- conditioning may be poor
- noise is present in observations
- the numerical rank of the design matrix may be lower than its algebraic rank

The project does not attempt to prove universal stability theorems. It
measures behavior on controlled matrix families and records evidence.

## Key Quantities

### Condition Number

The 2-norm condition number of a matrix A is defined as

    kappa_2(A) = sigma_max / sigma_min

The implementation is `core/metrics.py::condition_number`.

If the matrix is numerically rank deficient, `condition_number` returns
`inf` rather than a finite estimate. This is deliberate: a truncated singular
value means the mathematical problem has effectively lost rank at the tested
precision, and reporting a large finite number would misrepresent that fact.

### Numerical Rank

Numerical rank is computed from singular values using the threshold

    tau = sigma_max * max(m, n) * eps

where `eps` is the machine epsilon of the working dtype. The SVD least-squares
solver in `learning/svd.py` uses this same threshold.

Numerical rank is not the same concept as the global test tolerances `ATOL`
and `RTOL` in `core/tolerances.py`. Those tolerances govern comparison in test
assertions. The numerical-rank threshold governs whether a singular value
contributes to the pseudoinverse.

### Forward Error and Backward Error

Forward error measures the difference between a computed solution and the
true (or reference) solution. Backward error measures how much the input
problem would have to be perturbed to make the computed solution exact.

The project compares forward error to the NumPy reference
`numpy.linalg.lstsq`. This is not proof of correctness against the true
mathematical solution, but it provides an independent, externally implemented
reference point.

## Observed Behavior

### Normal Equations

From `experiments/normal_equations_conditioning.py` and the recorded results
in [experiments.md](experiments.md):

- Squaring of the condition number is visible for moderate conditioning.
- At very large conditioning, forward error can become order-unity while the
  residual remains small.
- A small residual does not imply an accurate parameter vector.

### QR

From the comparison experiment recorded in [experiments.md](experiments.md):

- QR maintains substantially smaller forward error than Normal Equations on
  the tested matrix family.
- QR requires full column rank and rejects rank-deficient inputs explicitly.
- Residual orthogonality is preserved across the tested conditioning regimes.

### SVD

From `experiments/svd_rank_stability.py` and the recorded results:

- On full-rank systems up to kappa approximately 1e14, the SVD solver tracks
  the NumPy reference within the expected `eps * kappa` scale.
- At kappa approximately 1e16, numerical rank can drop below algebraic rank
  and the solver falls back to the minimum-norm solution.
- The rank-deficient case returns the minimum-norm solution, matching the
  NumPy reference.

## What This Document Does Not Claim

This document does not establish universal condition-number thresholds for
solver selection. The recorded numbers are evidence for specific matrix
families, dimensions, seeds, and floating-point environments.

Practical guidance on solver selection belongs in experiment reports and in
the future Phase 1.7 findings document, not here. This document is intended
to be durable across experiments; specific cutoffs and recommendations are
not.

## References

- Mathematical definitions: [mathematical-foundations.md](mathematical-foundations.md)
- Experiment methodology and evidence: [experiments.md](experiments.md)
- Design decisions: [design-decisions.md](design-decisions.md)
- Core metrics: `src/robust_statistical_learning/core/metrics.py`
- Solver implementations: `src/robust_statistical_learning/learning/`
