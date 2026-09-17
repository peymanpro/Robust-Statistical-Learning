# Experiments

Experiments validate implementations, gather evidence, and guide recommendations.

## Experimental Methodology

Every experiment follows a structure:

```
[Hypothesis] → [Setup] → [Execution] → [Measurement] → [Evidence] → [Interpretation] → [Recommendation]
```

## Key Dimensions

### 1. Accuracy

Does the algorithm compute the correct result?

**Measurement**:
- Forward error: \(\frac{\|\hat{x}_{\text{ours}} - \hat{x}_{\text{reference}}\|}{\|\hat{x}_{\text{reference}}\|}\)
- Residual norm: \(\|A\hat{x} - b\|\)
- Comparison to NumPy/SciPy/scikit-learn

**Example**:

```
Experiment: QR Decomposition Accuracy
Hypothesis: Our QR implementation matches NumPy reconstruction
Setup:
  - Matrix sizes: (50,20), (100,50), (500,100), (1000,500)
  - Condition numbers: κ(A) ≈ 1, 10, 100, 1000
Execution:
  - For each matrix A:
    - Compute Q, R = our_qr(A)
    - Compute Q_np, R_np = numpy.linalg.qr(A)
Measurement:
  - Reconstruction error: ||A - Q R||_F
  - Orthogonality: ||Q'Q - I||_F
Evidence:
  - Reconstruction error < 1e-12 for all well-conditioned cases
  - Reconstruction error < 1e-8 for condition numbers up to 1e12
Interpretation:
  - Our implementation is backward stable
  - Forward error grows with condition number as expected
Recommendation:
  - Use our QR for well-conditioned problems
  - Use our QR as reference for algorithm correctness
```

### 2. Stability

How does the algorithm respond to small perturbations?

**Measurement**:
- Perturbation sensitivity: \(\frac{\|\Delta x\|}{\|x\|} / \epsilon\) where \(\epsilon\) is perturbation size
- Backward error: how much must the problem be perturbed to make our solution exact?
- Condition number estimation accuracy

**Example**:

```
Experiment: QR Stability Under Perturbations
Hypothesis: QR is backward stable; forward error depends on κ(A)
Setup:
  - Test matrix A (various condition numbers)
  - Perturbation sizes: 1e-14, 1e-12, 1e-10, 1e-8
Execution:
  - Compute x = QR_solve(A, b)
  - Perturb: A_pert = A + δ*noise
  - Compute x_pert = QR_solve(A_pert, b)
Measurement:
  - Relative forward error: ||x_pert - x|| / ||x||
  - Expected bound: ε * κ(A)
Evidence:
  - Forward error matches expected bound
  - Error is proportional to κ(A)
Interpretation:
  - QR is backward stable (backward error is small)
  - Problem conditioning dominates forward error
Recommendation:
  - For ill-conditioned problems, forward error will be large
  - This is not a flaw in QR; it reflects problem conditioning
  - Use regularization or truncated SVD for ill-conditioned problems
```

### 3. Performance

How fast is the algorithm? What is the memory usage?

**Measurement**:
- Runtime as a function of problem size
- Memory usage
- Comparison to NumPy/SciPy

**Important**: Performance is **not** a primary goal. We will not be as fast as NumPy.

**Example**:

```
Experiment: Least Squares Solver Performance
Setup:
  - Problem sizes: (100,10), (1000,100), (10000,500), (100000,1000)
  - Solvers: Normal Equations, QR, SVD
Measurement:
  - Runtime for each solver
  - Memory peak
  - Scaling: is it O(n²)? O(n³)?
Evidence:
  - All solvers scale as O(n²m) (expected for m >> n)
  - Our solvers are 10-100x slower than NumPy (expected)
  - Normal Equations is fastest but least stable
Recommendation:
  - Choose solver based on stability, not speed
  - For learning/investigation: use any solver
  - For production: use NumPy/SciPy
```

### 4. Failure Cases

Where does the algorithm break down?

**Measurement**:
- Condition number threshold where forward error becomes unacceptable
- Numerical rank deficiency
- Near-singular matrices

**Example**:

```
Experiment: Least Squares on Nearly Singular Matrices
Hypothesis: Normal equations fail; QR/SVD remain stable
Setup:
  - Hilbert matrices (notoriously ill-conditioned)
  - Vandermonde matrices
  - Random matrices with repeated columns
Execution:
  - For each matrix, solve Ax = b using all three methods
  - Compare to reference solution (high precision)
Measurement:
  - Forward error for each solver
  - κ(A) for each matrix
Evidence:
  - Normal Equations fails (forward error > 1e-2) for κ > 1e8
  - QR fails (forward error > 1e-2) for κ > 1e14
  - SVD with truncation handles all cases
Interpretation:
  - Normal equations unsuitable for ill-conditioned problems
  - QR is much better but still limited
  - SVD with truncation is best for ill-conditioned systems
Recommendation:
  - Never use Normal Equations for unknown problems
  - Use QR by default
  - Use SVD with truncation for ill-conditioned problems
```

## Reproducibility

Every experiment must be reproducible:

**Record**:
- Random seed(s)
- Problem dimensions (m, n)
- Data scale and range (e.g., A ∈ [-1, 1])
- Condition number (theoretical and computed)
- Algorithm parameters (regularization λ, truncation threshold, etc.)
- Software versions:
  - Python version
  - NumPy version
  - SciPy version
  - Our library version
- System information (optional):
  - CPU model
  - RAM
  - OS

**Example**:

```python
# experiments/least_squares_stability.py

import numpy as np
from robust_statistical_learning.learning import least_squares_solve
from robust_statistical_learning.core import condition_number
import scipy.linalg

# Reproducibility
SEED = 42
np.random.seed(SEED)

# Problem specification
m, n = 100, 20
condition_numbers = [1e1, 1e5, 1e10, 1e15]

results = {}

for target_kappa in condition_numbers:
    # Create ill-conditioned matrix
    A = create_ill_conditioned_matrix(m, n, target_kappa, seed=SEED)
    b = np.random.randn(m)

    # Compute condition number
    kappa_actual = condition_number(A)

    # Solve
    x_ours = least_squares_solve(A, b)
    x_scipy = scipy.linalg.lstsq(A, b)[0]

    # Measure
    forward_error = np.linalg.norm(x_ours - x_scipy) / np.linalg.norm(x_scipy)

    results[target_kappa] = {
        'kappa_actual': kappa_actual,
        'forward_error': forward_error,
        'seed': SEED,
        'm': m,
        'n': n,
        'versions': {
            'numpy': np.__version__,
            'scipy': scipy.__version__,
        }
    }
```

## Evidence vs Interpretation vs Recommendation

### Evidence

**Raw Data**
- Measurements from experiments
- Numbers, plots, statistics
- No interpretation

**Example**:
```
Forward error (Our QR vs SciPy):
  κ(A) = 1e2:   error = 2.3e-15
  κ(A) = 1e6:   error = 4.1e-10
  κ(A) = 1e12:  error = 3.2e-4
  κ(A) = 1e16:  error = 0.87
```

### Interpretation

**What the evidence means**
- Relationship to theory
- Comparison to expectations
- Discussion of causes

**Example**:
```
Interpretation:
- For well-conditioned problems (κ < 1e8), forward error is near machine precision
- For ill-conditioned problems (κ > 1e10), forward error grows linearly with κ
- This matches backward stability theory: forward error ≈ ε·κ·(backward error)
- The backward error is small (≈ 1e-16), so conditioning dominates
```

### Recommendation

**What to do based on evidence and interpretation**
- Practical guidance
- When to use which algorithm
- Caveats and limitations

**Example**:
```
Recommendations:
1. For general problems (unknown conditioning): use QR
   - It is backward stable
   - Suitable for κ(A) up to ~1e14
   - Clearly better than Normal Equations

2. For known ill-conditioned problems: use SVD with truncation
   - Explicit control over numerical rank
   - Can handle near-singular systems
   - More expensive than QR but worth it

3. Do not use Normal Equations
   - No advantage over QR
   - Squares the condition number
   - Fails earlier for ill-conditioned problems
   - Only keep in code for pedagogical comparison

4. For unknown problem conditioning:
   - Compute κ(A) first
   - If κ < 1e8: QR is sufficient
   - If κ > 1e10: consider SVD or regularization
   - If κ > 1e14: problem may be fundamentally ill-posed
```

## Experiment Categories

### Mathematical Invariants

Verify algorithm properties (covered in Testing Strategy).

### Noise Robustness

How noise in data affects solutions:

```
- Add Gaussian noise to b
- Vary noise level
- Compare unregularized vs regularized solution
- Measure stability
```

### Ill-Conditioning

Test algorithms on ill-conditioned problems:

```
- Hilbert matrices
- Vandermonde matrices
- Condition numbers ranging from 1 to 1e16
- Measure forward error
```

### Perturbation Analysis

How algorithms respond to perturbations:

```
- Fix problem size and conditioning
- Vary perturbation size
- Measure output change
- Compare to theoretical bounds
```

### Real Data Validation

Test on actual datasets (Phase 4):

```
- Choose representative datasets
- Preprocess according to documented procedure
- Train/test split (with seed)
- Measure predictive accuracy
- Compare to scikit-learn implementations
```

## Example: Full Experiment Report

```
# Experiment: Condition Number Impact on QR Least Squares

## Hypothesis
QR decomposition is backward stable. Forward error grows with condition number
according to theory: forward_error ≈ ε·κ(A)·backward_error

## Setup
- Matrix dimensions: m=100, n=20
- Problem types: random dense, Hilbert, Vandermonde
- Condition numbers: 1e1, 1e5, 1e10, 1e15
- Seed: 42 (for reproducibility)
- Software: NumPy 1.24.3, SciPy 1.11.2, Python 3.11.4

## Method
For each problem type and condition number:
1. Create matrix A with target κ(A)
2. Create random b, compute true x = lstsq(A, b) using high precision
3. Compute x_qr using our QR solver
4. Compute x_scipy using SciPy's lstsq
5. Measure forward error: ||x_qr - x_true|| / ||x_true||

## Results

| Problem Type | κ(A) | Forward Error (QR) | Forward Error (SciPy) | Ratio |
|---|---|---|---|---|
| Random Dense | 1e1 | 1.2e-15 | 8.3e-16 | 1.4 |
| Random Dense | 1e5 | 3.4e-11 | 2.8e-11 | 1.2 |
| Random Dense | 1e10 | 2.1e-6 | 1.9e-6 | 1.1 |
| Random Dense | 1e15 | 0.41 | 0.38 | 1.1 |
| Hilbert | 1e3 | 5.2e-13 | 4.1e-13 | 1.3 |
| Hilbert | 1e13 | 0.0012 | 0.0010 | 1.2 |
| Vandermonde | 1e4 | 2.8e-12 | 2.1e-12 | 1.3 |
| Vandermonde | 1e14 | 0.58 | 0.52 | 1.1 |

## Evidence
1. QR and SciPy lstsq produce nearly identical results (ratio ≈ 1.1-1.4)
2. Forward error closely follows condition number: error ≈ 10^(-16) * κ(A)
3. For κ > 1e14, forward error becomes large (> 0.1) across all solvers

## Interpretation
- Our QR implementation is backward stable (small backward error)
- Forward error is dominated by problem conditioning, not algorithmic issues
- The forward error ≈ ε·κ(A) relationship is confirmed empirically
- Agreement with SciPy builds confidence in implementation correctness

## Recommendation
1. QR is suitable for problems with κ(A) < 1e12 (or higher with understanding of error)
2. Do not expect better than ε·κ(A) forward error; it's a law of nature for this problem
3. For κ > 1e14, consider regularization or reformulation instead of expecting high precision
4. Our QR is numerically equivalent to SciPy; use it for learning, NumPy for production
```

## Summary

Good experiments:
- Have clear hypotheses
- Are fully reproducible
- Separate evidence from interpretation
- Lead to actionable recommendations
- Document all relevant parameters
- Compare to established references where appropriate

---

# Experiment: Normal Equations and Conditioning

## Hypothesis

Forming the normal equations

\[
A^T A x = A^T b
\]

amplifies conditioning approximately as

\[
\kappa(A^T A) \approx \kappa(A)^2
\]

As conditioning worsens, forward error may become large even when the residual remains small.

## Setup

- Dimensions: \(m=40,\ n=5\)
- Seed: `42`
- True solution: \(x=[1,2,3,4,5]^T\)
- \(b=Ax\)
- Target condition numbers: \(10^2,10^6,10^{10},10^{12},10^{14}\)
- Solver: Normal Equations
- Reference solution: constructed `x_true`
## Measurements

| Target \(\kappa(A)\) | Actual \(\kappa(A)\) | \(\kappa(A^T A)\) | Forward Error | Residual |
|---:|---:|---:|---:|---:|
| \(10^2\) | \(1.000\times10^2\) | \(1.000\times10^4\) | \(3.519\times10^{-13}\) | \(2.653\times10^{-14}\) |
| \(10^6\) | \(1.000\times10^6\) | \(1.000\times10^{12}\) | \(8.723\times10^{-6}\) | \(6.470\times10^{-11}\) |
| \(10^{10}\) | \(1.000\times10^{10}\) | \(6.589\times10^{15}\) | \(2.354\times10^{-2}\) | \(4.126\times10^{-9}\) |
| \(10^{12}\) | \(1.000\times10^{12}\) | \(1.015\times10^{17}\) | \(7.114\times10^{-1}\) | \(5.276\times10^{-9}\) |
| \(10^{14}\) | \(1.001\times10^{14}\) | \(1.936\times10^{17}\) | \(2.300\times10^{1}\) | \(2.539\times10^{-8}\) |

## Interpretation

The experiment demonstrates that a small residual does not guarantee an accurate parameter vector. For the approximately 1e14 condition-number case, the relative forward error is about 23 while the residual remains about 2.5e-8.

For moderate condition numbers, kappa(A^T A) follows the expected squaring relationship. At very large condition numbers, finite-precision effects limit the computed value.

## Recommendation

Normal Equations should remain in this project as a pedagogical and comparative solver. This experiment does not establish a universal condition-number cutoff for all problems.

## Reproducibility

Script: experiments/normal_equations_conditioning.py

---

# Experiment: Normal Equations vs QR under Ill-Conditioning

This experiment compares forward error and residuals for Normal Equations and QR on the same controlled matrix family.

| Target kappa(A) | Normal Error | QR Error | Normal Residual | QR Residual |
|---:|---:|---:|---:|---:|
| 1e2 | 3.519e-13 | 2.468e-15 | 2.653e-14 | 1.020e-15 |
| 1e6 | 8.723e-06 | 1.596e-11 | 6.470e-11 | 5.718e-16 |
| 1e10 | 2.354e-02 | 3.174e-09 | 4.126e-09 | 7.453e-16 |
| 1e12 | 7.114e-01 | 1.727e-06 | 5.276e-09 | 6.761e-16 |
| 1e14 | 2.300e+01 | 4.507e-03 | 2.539e-08 | 6.780e-16 |

## Interpretation

On this controlled matrix family, QR maintains substantially smaller forward error than Normal Equations as conditioning worsens. At kappa(A) approximately 1e14, Normal Equations has forward error approximately 23, while QR remains approximately 4.5e-3.

Both methods can still produce small residuals. Therefore, a small residual alone is not sufficient evidence of an accurate parameter vector.

This experiment does not establish a universal superiority threshold; it is evidence for this matrix family, size, seed, scaling, and floating-point environment.


---

# Experiment: SVD Numerical Rank and Reference Agreement

## Hypothesis

The SVD least-squares solver should:

- agree with numpy.linalg.lstsq on full-rank systems across conditioning regimes
- preserve residual orthogonality even under ill-conditioning
- correctly truncate singular values below the numerical-rank threshold
- return the minimum-norm solution for rank-deficient systems

## Setup

- Dimensions: m = 40, n = 5
- Seed: 42
- True solution: x = [1, 2, 3, 4, 5]^T
- b = A x
- Target condition numbers: 1e2, 1e6, 1e10, 1e14, 1e16
- Solver: solve_svd (SVD pseudoinverse)
- Reference solver: numpy.linalg.lstsq(rcond=None)
- Numerical-rank threshold: sigma_max * max(m, n) * eps

## Measurements

| Target kappa(A) | Actual kappa(A) | Rank | Reference Error | Residual |
|---:|---:|---:|---:|---:|
| 1e2  | 1.000e2  | 5 | 1.029e-15 | 4.279e-15 |
| 1e6  | 1.000e6  | 5 | 6.845e-12 | 3.817e-15 |
| 1e10 | 1.000e10 | 5 | 3.767e-08 | 3.446e-15 |
| 1e14 | 9.996e13 | 5 | 3.604e-04 | 1.595e-15 |
| 1e16 | inf      | 4 | 6.067e-06 | 1.923e-15 |

Rank-deficient case:

    computed        = [1. 1.]
    reference       = [1. 1.]
    reference_error = 2.483e-16
    residual        = 8.882e-16
    solution_norm   = 1.414e+00

## Interpretation

- For condition numbers up to 1e14, the SVD solver closely tracks the NumPy reference. The forward error grows roughly like kappa(A) * eps.
- At kappa(A) = 1e16, the computed condition number is inf and the numerical rank drops from 5 to 4. Both our solver and numpy.linalg.lstsq operate on the same numerical-rank interpretation. Agreement with NumPy in this case does not imply accuracy relative to the original x_true.
- The residual stays near machine precision across all tested conditioning regimes, consistent with backward stability.
- The rank-deficient case returns the minimum-norm solution, matching the NumPy reference.

## Recommendation

- Use the SVD solver when numerical rank deficiency is possible or when explicit control over rcond is required.
- The numerical-rank threshold used here is not a universal cutoff. It is specific to this matrix family, dimensions, seed, scaling, and float64 environment.
- Agreement with numpy.linalg.lstsq does not by itself establish correctness against the true mathematical solution. Both solvers are finite-precision interpretations of the same problem.

## Reproducibility

Script: experiments/svd_rank_stability.py

Environment: Python 3.12.x, NumPy float64, seed 42.


---

# Experiment: Well-Conditioned Solver Agreement

## Hypothesis

For well-conditioned systems (small kappa), all three least-squares solvers
(Normal Equations, QR, SVD) should return forward errors on the order of
machine epsilon times the condition number, and residuals near machine
precision. If this holds, the experiment establishes a baseline for
comparison against ill-conditioned cases.

## Setup

- Dimensions: m = 40, n = 5
- Seed: 42
- True solution: x = [1, 2, 3, 4, 5]^T
- b = A x
- Target condition numbers: 1, 1e2, 1e4, 1e6, 1e8
- Solvers: solve_normal_equations, solve_qr, solve_svd
- Reference: the constructed x_true (exact mathematical solution)
- dtype: float64

## Measurements

| Target kappa | Actual kappa | NE fwd | QR fwd | SVD fwd | NE res | QR res | SVD res |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0e0 | 1.000e0 | 2.760e-16 | 2.744e-16 | 4.828e-16 | 2.271e-15 | 2.386e-15 | 3.781e-15 |
| 1.0e2 | 1.000e2 | 5.324e-13 | 2.552e-15 | 2.221e-15 | 4.120e-14 | 1.492e-15 | 5.777e-15 |
| 1.0e4 | 1.000e4 | 1.020e-09 | 3.067e-14 | 1.363e-13 | 7.740e-13 | 3.210e-16 | 2.247e-14 |
| 1.0e6 | 1.000e6 | 3.421e-07 | 3.968e-12 | 2.085e-12 | 4.717e-12 | 3.486e-16 | 7.739e-16 |
| 1.0e8 | 1.000e8 | 2.701e-01 | 7.339e-09 | 1.329e-09 | 2.003e-08 | 7.390e-16 | 1.882e-15 |

Cross-solver solution differences:

| Target kappa | ||NE-QR|| | ||QR-SVD|| | ||NE-SVD|| |
|---:|---:|---:|---:|
| 1.0e0 | 1.676e-15 | 4.021e-15 | 3.179e-15 |
| 1.0e2 | 3.957e-12 | 1.443e-14 | 3.944e-12 |
| 1.0e4 | 7.566e-09 | 1.222e-12 | 7.565e-09 |
| 1.0e6 | 2.537e-06 | 1.398e-11 | 2.537e-06 |
| 1.0e8 | 2.003e+00 | 4.458e-08 | 2.003e+00 |

## Interpretation

Four patterns emerge on this matrix family, dimensions, seed, and dtype:

1. At kappa = 1, all three solvers agree to machine precision. Forward
   errors are on the order of eps and residuals are on the order of
   eps * ||b||.

2. For QR and SVD, forward error grows approximately as eps * kappa across
   the tested range. At kappa = 1e8 their forward errors remain around
   7e-9 and 1e-9 respectively.

3. For Normal Equations, forward error grows approximately as eps * kappa^2
   and diverges from QR/SVD as kappa increases. At kappa = 1e8 the Normal
   Equations forward error reaches 2.7e-1 while QR and SVD remain below
   1e-8. The condition-number-squared behavior of the normal equations is
   therefore visible on this family.

4. The residual remains small for all three solvers at all tested condition
   numbers, including the kappa = 1e8 case where Normal Equations is
   effectively incorrect (forward error 2.7e-1, residual 2.0e-8). A small
   residual is not sufficient evidence of an accurate parameter vector.

The cross-solver differences confirm the picture: ||QR-SVD|| stays below
5e-8 across the tested range, while ||NE-QR|| grows from 1.7e-15 to 2.0e0
as kappa increases. Normal Equations is the outlier.

The original hypothesis that all three solvers behave similarly for
kappa up to 1e8 is rejected on this family. The well-conditioned region
for Normal Equations is narrower than for QR and SVD.

## Recommendation

- For kappa up to roughly 1e4, all three solvers provide forward errors
  below 1e-9 on this family and may be used interchangeably.
- For kappa above roughly 1e6, Normal Equations should not be relied upon
  for forward accuracy even though its residual remains small.
- QR and SVD remain numerically reliable across the entire tested range
  (kappa up to 1e8) on this family, with forward errors consistent with
  the eps * kappa scale.
- These numbers do not establish universal cutoffs. They are evidence for
  this specific matrix family, dimensions, seed, and float64 environment.

## Reproducibility

Script: experiments/well_conditioned_comparison.py

Environment: Python 3.12.x, NumPy float64, seed 42.


---

# Experiment: Ill-Conditioned Solver Comparison

## Hypothesis

Extending the well-conditioned comparison toward larger condition numbers
should reveal the region where Normal Equations becomes unusable, where
QR begins to lose forward accuracy, and where the SVD numerical-rank
truncation becomes the deciding mechanism. Residuals are expected to stay
small across all solvers even when the solution is wrong.

## Setup

- Dimensions: m = 40, n = 5
- Seed: 42
- True solution: x = [1, 2, 3, 4, 5]^T
- b = A x
- Target condition numbers: 1e10, 1e12, 1e14, 1e16
- Solvers: solve_normal_equations, solve_qr, solve_svd
- Reference: the constructed x_true (exact mathematical solution)
- Numerical rank: computed from singular values with the
  sigma_max * max(m, n) * eps threshold
- dtype: float64

## Measurements

| Target kappa | Actual kappa | Rank | NE fwd | QR fwd | SVD fwd | NE res | QR res | SVD res |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0e10 | 1.000e10 | 5 | 2.179e+00 | 1.050e-07 | 1.472e-07 | 1.233e-08 | 6.783e-16 | 3.818e-15 |
| 1.0e12 | 1.000e12 | 5 | 1.198e+01 | 1.454e-05 | 1.296e-05 | 6.761e-08 | 8.144e-16 | 2.676e-15 |
| 1.0e14 | 9.999e13 | 5 | 5.036e-01 | 1.325e-03 | 2.752e-03 | 3.883e-09 | 6.656e-16 | 2.035e-14 |
| 1.0e16 | inf | 4 | ValueError | ValueError | 3.363e-02 | ValueError | ValueError | 3.159e-16 |

Cross-solver solution differences:

| Target kappa | ||NE-QR|| | ||QR-SVD|| | ||NE-SVD|| |
|---:|---:|---:|---:|
| 1.0e10 | 1.616e+01 | 3.131e-07 | 1.616e+01 |
| 1.0e12 | 8.882e+01 | 1.172e-05 | 8.882e+01 |
| 1.0e14 | 3.729e+00 | 1.058e-02 | 3.724e+00 |
| 1.0e16 | n/a | n/a | n/a |

## Interpretation

Five patterns emerge on this matrix family, dimensions, seed, and dtype:

1. Normal Equations is unusable across the entire tested range. Forward
   errors exceed 1 at kappa = 1e10 and 1e12. The value at kappa = 1e12
   (approximately 12) is larger than at kappa = 1e10 (approximately 2.2),
   consistent with the eps * kappa^2 scaling combined with numerical
   amplification. Normal Equations has no usable forward-accuracy regime
   beyond the well-conditioned cases already recorded.

2. QR forward error grows from 1.05e-07 at kappa = 1e10 to 1.33e-03 at
   kappa = 1e14, following the eps * kappa scale observed in the
   well-conditioned experiment.

3. SVD forward error is within a factor of a few of QR across the
   range where QR succeeds: 1.47e-07, 1.30e-05, 2.75e-03. The two solvers
   are numerically equivalent on this family when both succeed.

4. At kappa = 1e16, the numerical rank drops from 5 to 4. Both QR and
   Normal Equations raise ValueError because both implementations
   explicitly require full column rank. SVD does not require full column
   rank; it truncates the singular value below the numerical-rank
   threshold and returns the minimum-norm solution with forward error
   3.36e-02. This is the deciding difference between the solvers at the
   boundary of numerical rank deficiency.

5. The residual remains small for every solver at every tested condition
   number, including the cases where Normal Equations returns a solution
   with forward error above 1 and the case where SVD operates in a
   truncated-rank regime. A small residual continues to be insufficient
   evidence of an accurate parameter vector.

## Recommendation

- Normal Equations should not be used beyond the well-conditioned
  region identified previously (roughly kappa <= 1e4 on this family).
  The ill-conditioned results here reinforce that recommendation directly.
- QR is reliable on this family up to kappa approximately 1e14, with
  forward error consistent with the eps * kappa scale. At kappa = 1e16
  it rejects the system by contract, which is correct behavior: it does
  not silently produce a rank-truncated solution.
- SVD is the only solver of the three that produces a well-defined
  answer when numerical rank drops. Its forward error at kappa = 1e16
  (3.36e-02) reflects the truncated-rank problem, not an algorithmic
  failure.
- These numbers do not establish universal cutoffs. They are evidence
  for this specific matrix family, dimensions, seed, and float64
  environment.

## Reproducibility

Script: experiments/ill_conditioned_comparison.py

Environment: Python 3.12.x, NumPy float64, seed 42.


---

# Experiment: Hilbert Matrix Solver Comparison

## Hypothesis

The Hilbert matrix H_n with entries H[i, j] = 1 / (i + j - 1) produces
exponentially growing condition numbers as n increases. On this family
we expect to observe the crossing-over point for each solver: the size
at which Normal Equations loses accuracy, the size at which QR follows,
and the size at which the matrix becomes numerically rank deficient so
that SVD truncation and rejection behavior become the deciding factor.

## Setup

- Matrix: H_n with H[i, j] = 1 / (i + j - 1) for i, j = 1..n
- Sizes: n = 4, 6, 8, 10, 12
- True solution: x_true = ones(n)
- b = H_n @ x_true
- Solvers: solve_normal_equations, solve_qr, solve_svd
- Direct reference: numpy.linalg.solve (LU)
- Numerical rank: sigma_max * max(m, n) * eps threshold
- dtype: float64

## Measurements

Main comparison:

| n | kappa | rank | NE fwd | QR fwd | SVD fwd | NE res | QR res | SVD res |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.551e04 | 4 | 9.445e-10 | 3.108e-13 | 2.204e-13 | 1.827e-13 | 3.140e-16 | 2.455e-15 |
| 6 | 1.495e07 | 6 | 5.404e-03 | 5.340e-10 | 1.349e-10 | 1.433e-09 | 5.207e-16 | 8.742e-16 |
| 8 | 1.526e10 | 8 | 4.895e+00 | 1.494e-07 | 4.668e-07 | 7.106e-09 | 1.024e-15 | 2.539e-15 |
| 10 | 1.602e13 | 10 | 3.243e+00 | 1.421e-04 | 1.208e-04 | 3.776e-09 | 6.844e-16 | 2.329e-15 |
| 12 | inf | 11 | ValueError | ValueError | 9.074e-04 | ValueError | ValueError | 1.558e-15 |

Cross-check vs numpy.linalg.solve (LU):

| n | LU fwd | NE vs LU | QR vs LU | SVD vs LU |
|---:|---:|---:|---:|---:|
| 4 | 4.137e-14 | 1.889e-09 | 5.389e-13 | 3.584e-13 |
| 6 | 1.424e-10 | 1.324e-02 | 1.657e-09 | 1.860e-11 |
| 8 | 6.124e-08 | 1.384e+01 | 5.959e-07 | 1.494e-06 |
| 10 | 8.670e-05 | 1.026e+01 | 1.753e-04 | 1.079e-04 |
| 12 | 3.249e-01 | n/a | n/a | 1.126e+00 |

## Interpretation

Six patterns emerge on this family:

1. Normal Equations loses forward accuracy at n = 6 (kappa = 1.5e7,
   forward error 5.4e-3) and is unusable from n = 8 onward (forward
   error above 1). The crossing-over from "usable" to "unusable" occurs
   earlier on the Hilbert family than on the controlled
   singular-value family, where Normal Equations still produced forward
   errors below 1e-6 at kappa = 1e6. The eps * kappa^2 scaling makes the
   crossing-over point sensitive to the specific conditioning growth rate.

2. QR and SVD remain numerically equivalent to each other across the
   entire Hilbert family where both succeed: 3.1e-13 / 2.2e-13 at n = 4,
   1.5e-07 / 4.7e-07 at n = 8, 1.4e-04 / 1.2e-04 at n = 10. Their forward
   errors follow the eps * kappa scale observed previously.

3. At n = 12, the computed condition number is inf and the numerical
   rank drops from 12 to 11. Normal Equations and QR both raise
   ValueError because both implementations require full column rank
   and reject the numerically rank-deficient input. SVD does not
   require full column rank; it truncates the smallest singular value
   and returns a minimum-norm solution with forward error 9.1e-4.

4. The LU reference (numpy.linalg.solve) at n = 12 has forward error
   3.2e-01. This is not a solver failure but a property of the problem:
   on H_12, no direct solver can recover x_true reliably at float64
   precision. SVD and LU disagree by 1.13 in this case, confirming that
   the answer is genuinely ambiguous.

5. The residual remains small (on the order of eps * ||b||) for every
   solver at every size, including n = 8 and n = 10 where Normal
   Equations returns forward errors above 1. A small residual is again
   insufficient evidence of an accurate parameter vector.

6. The Hilbert family demonstrates that the "well-conditioned" region
   for a solver depends on how kappa grows with problem size. On the
   controlled singular-value family, kappa was a direct input. On the
   Hilbert family, kappa grows exponentially with n, so the usable
   problem sizes are much smaller.

## Recommendation

- On exponentially ill-conditioned families like Hilbert, Normal
  Equations should be avoided beyond the smallest sizes (n <= 4 here).
  This is a stronger statement than the well-conditioned experiment
  produced, and it is consistent with the eps * kappa^2 mechanism.
- QR and SVD track each other closely on this family and remain usable
  until numerical rank deficiency occurs. QR rejects rank-deficient
  input explicitly, which is correct behavior; SVD falls back to the
  minimum-norm solution, which is the appropriate response when the
  problem has genuinely lost rank.
- When the numerical rank drops, no solver returns x_true accurately.
  At n = 12 the "true" solution is not recoverable at float64 precision
  on this family. Reporting this as a solver limitation would be
  incorrect; it is a property of the problem.
- These numbers do not establish universal cutoffs. They are evidence
  for the Hilbert family and the specific sizes tested.

## Reproducibility

Script: experiments/hilbert_comparison.py

Environment: Python 3.12.x, NumPy float64, no random seed required.


---

# Experiment: Vandermonde Matrix Solver Comparison

## Hypothesis

A Vandermonde matrix V[i, j] = x_i^(j-1) with uniformly spaced nodes in
(0, 1) produces conditioning that grows with the number of columns n but
more slowly than the Hilbert family at comparable sizes. We expect the
same solver ordering observed on the previous families: Normal Equations
loses forward accuracy first, QR and SVD remain equivalent to each other
until numerical rank deficiency occurs, and residual stays small
regardless of forward error.

## Setup

- Rows: m = 40
- Columns: n = 4, 6, 8, 10, 12, 15
- Nodes: x_i = i / (m + 1) for i = 1..m, uniform in (0, 1)
- True solution: x_true = ones(n)
- b = V @ x_true
- Solvers: solve_normal_equations, solve_qr, solve_svd
- Cross-check reference: numpy.linalg.lstsq (rcond=None)
- Numerical rank: sigma_max * max(m, n) * eps threshold
- dtype: float64

## Measurements

Main comparison:

| n | kappa | rank | NE fwd | QR fwd | SVD fwd | NE res | QR res | SVD res |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 1.343e02 | 4 | 1.242e-13 | 3.951e-15 | 4.818e-15 | 2.004e-14 | 1.601e-15 | 1.632e-15 |
| 6 | 4.433e03 | 6 | 1.866e-10 | 9.089e-14 | 7.430e-14 | 8.254e-13 | 2.719e-15 | 8.685e-15 |
| 8 | 1.528e05 | 8 | 5.317e-07 | 1.082e-11 | 1.300e-11 | 8.021e-11 | 4.094e-15 | 1.844e-14 |
| 10 | 5.456e06 | 10 | 7.672e-05 | 2.906e-10 | 2.915e-10 | 3.706e-10 | 3.546e-15 | 1.109e-14 |
| 12 | 2.018e08 | 12 | 1.459e-01 | 3.525e-09 | 1.445e-09 | 2.096e-08 | 5.908e-15 | 3.022e-14 |
| 15 | 4.889e10 | 15 | LinAlgError | 2.995e-07 | 1.941e-07 | LinAlgError | 6.339e-15 | 1.350e-14 |

Cross-check vs numpy.linalg.lstsq:

| n | ref fwd | NE vs ref | QR vs ref | SVD vs ref |
|---:|---:|---:|---:|---:|
| 4 | 3.360e-15 | 2.418e-13 | 1.450e-14 | 1.624e-14 |
| 6 | 2.621e-13 | 4.565e-10 | 8.555e-13 | 8.124e-13 |
| 8 | 4.367e-12 | 1.504e-06 | 1.833e-11 | 2.452e-11 |
| 10 | 5.890e-11 | 2.426e-04 | 1.105e-09 | 1.108e-09 |
| 12 | 1.701e-09 | 5.055e-01 | 1.810e-08 | 1.089e-08 |
| 15 | 9.144e-07 | n/a | 4.628e-06 | 2.959e-06 |

## Interpretation

Six patterns emerge on this family:

1. The condition number grows much more slowly with n than on the
   Hilbert family. At n = 15 the Vandermonde matrix reaches
   kappa = 4.9e10, whereas the Hilbert matrix reached kappa = inf at
   n = 12. This reflects the geometric growth of Vandermonde
   conditioning with uniform nodes versus the exponential growth of
   Hilbert conditioning.

2. Normal Equations maintains usable forward accuracy through n = 10
   (forward error 7.7e-05 at kappa = 5.5e6) and begins to fail at n = 12
   (forward error 1.5e-01 at kappa = 2.0e8). At n = 15 it raises
   LinAlgError, consistent with kappa(A)^2 exceeding 1/eps at that
   size.

3. QR and SVD remain numerically equivalent to each other across the
   entire tested range. Their forward errors track the eps * kappa
   scale: at n = 12 (kappa = 2.0e8) both are around 1e-9, and at n = 15
   (kappa = 4.9e10) both are around 2e-7.

4. The numerical rank stays full (rank = n) for every tested size. No
   rank deficiency is observed on this Vandermonde family at these
   dimensions, unlike the Hilbert family at n = 12. The SVD solver
   therefore does not enter its rank-truncated regime on this family.

5. The cross-check against numpy.linalg.lstsq confirms the picture. The
   reference itself has forward error 9.1e-07 at n = 15. QR differs
   from that reference by 4.6e-06 and SVD by 3.0e-06, consistent with
   the eps * kappa scale of the family. Normal Equations diverges from
   the reference by 5.1e-01 at n = 12 and rejects the problem at n = 15.

6. The residual remains small for every solver at every size, including
   n = 12 where Normal Equations has forward error above 1. A small
   residual continues to be insufficient evidence of an accurate
   parameter vector.

## Recommendation

- On Vandermonde families with uniform nodes, Normal Equations is
  usable up to kappa approximately 1e6 and should not be relied upon
  beyond that. This is consistent with the well-conditioned and
  ill-conditioned findings.
- QR and SVD remain reliable across the entire tested range on this
  family. They agree with each other to within one order of magnitude
  and track the eps * kappa scale. On this family, the choice between
  them is a matter of requirements (QR rejects rank-deficient input
  explicitly; SVD returns a minimum-norm solution), not accuracy.
- The condition number of the Vandermonde family with uniform nodes
  grows slowly enough that rank deficiency is not observed at the
  tested sizes. On other node distributions (clustered, Chebyshev,
  endpoints-heavy), the condition number can be much larger; those
  distributions are not tested here.
- These numbers do not establish universal cutoffs. They are evidence
  for the tested node distribution, dimensions, and float64
  environment.

## Reproducibility

Script: experiments/vandermonde_comparison.py

Environment: Python 3.12.x, NumPy float64, no random seed required.
