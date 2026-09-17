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
