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
