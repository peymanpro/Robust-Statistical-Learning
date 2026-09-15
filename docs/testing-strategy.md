# Testing Strategy

Testing in this project operates across four layers:

1. **Mathematical Invariants** — verify algorithm properties
2. **Numerical Behavior** — test under realistic/challenging conditions
3. **Reference Validation** — compare to established libraries
4. **Engineering Tests** — verify public API and contracts

## Layer 1: Mathematical Invariants

Mathematical invariants are properties that **must** hold if an algorithm is correct, independent of conditioning or floating-point arithmetic.

### Purpose
Catch algorithmic errors early, before numerical effects can obscure them.

### Examples

#### QR Decomposition

The mathematical property is:

$$A = QR$$

Numerical tests verify (to floating-point tolerance):

```python
def test_qr_reconstruction(A):
    Q, R = qr_decomposition(A)
    # Check: ||A - QR|| is small
    assert_allclose(A, Q @ R, atol=1e-10)
```

#### QR Orthogonality

$$Q^T Q = I$$

```python
def test_qr_orthogonality(A):
    Q, R = qr_decomposition(A)
    # Check: ||Q'Q - I|| is small
    assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-10)
```

#### SVD Reconstruction

$$A = U\Sigma V^T$$

```python
def test_svd_reconstruction(A):
    U, sigma, VT = svd(A)
    # Check: ||A - U Σ V'|| is small
    reconstructed = U @ np.diag(sigma) @ VT
    assert_allclose(A, reconstructed, atol=1e-10)
```

#### SVD Orthogonality

$$U^T U = I, \quad V^T V = I$$

```python
def test_svd_orthogonality(A):
    U, sigma, VT = svd(A)
    assert_allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-10)
    assert_allclose(VT @ VT.T, np.eye(VT.shape[0]), atol=1e-10)
```

#### Least Squares: Residual Orthogonality

For the least-squares solution, the residual \(r = Ax - b\) should be orthogonal to the column space of \(A\):

$$A^T r \approx 0$$

```python
def test_least_squares_residual_orthogonality(A, b):
    x = least_squares_solve(A, b)
    r = A @ x - b
    # Check: ||A'r|| is small
    assert_allclose(A.T @ r, np.zeros(A.shape[1]), atol=1e-10)
```

#### Normal Equations Consistency

$$A^T A x = A^T b$$

```python
def test_normal_equations_consistency(A, b):
    x = normal_equations_solve(A, b)
    # Check: ||A'Ax - A'b|| is small
    lhs = A.T @ A @ x
    rhs = A.T @ b
    assert_allclose(lhs, rhs, atol=1e-10)
```

### Tolerance in Invariant Tests

Tolerances depend on:
- **Machine precision**: \(\epsilon \approx 10^{-16}\) for float64
- **Problem scale**: larger matrices accumulate more rounding error
- **Condition number**: ill-conditioned problems have larger numerical errors
- **Algorithm complexity**: more operations → more rounding

A rough guideline:

$$\text{tolerance} \approx \max(\epsilon \cdot \|A\| \cdot n^{1.5}, 10^{-10})$$

where \(n\) is problem dimension and \(\|A\|\) is the matrix norm.

**DO NOT use a universal tolerance.** Justify each tolerance.

## Layer 2: Numerical Behavior

Numerical behavior tests verify that algorithms behave correctly under challenging conditions: ill-conditioning, noise, perturbations, near-singularity.

### Ill-Conditioning Tests

Test on matrices with large condition numbers:

```python
def test_qr_ill_conditioned():
    # Hilbert matrix: notoriously ill-conditioned
    A = hilbert(20)  # κ(A) ≈ 10^18
    Q, R = qr_decomposition(A)
    # QR should still satisfy mathematical invariants
    # even though forward error may be large
    assert_allclose(A, Q @ R, atol=1e-8)
    assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8)
```

### Perturbation Tests

Test stability under input perturbations:

```python
def test_least_squares_perturbation():
    A = np.random.randn(50, 10)
    b = np.random.randn(50)
    x1 = least_squares_solve(A, b)

    # Perturb slightly
    epsilon = 1e-8
    A_pert = A + epsilon * np.random.randn(*A.shape)
    b_pert = b + epsilon * np.random.randn(50)
    x2 = least_squares_solve(A_pert, b_pert)

    # Solution change should be proportional to perturbation size
    # times condition number
    relative_change = np.linalg.norm(x2 - x1) / np.linalg.norm(x1)
    kappa = condition_number(A)
    expected_change = epsilon * kappa

    assert relative_change < 10 * expected_change  # Loose bound
```

### Noise Sensitivity Tests

Test how noise in data affects the solution:

```python
def test_regularization_noise_robustness():
    # Generate problem with known solution
    x_true = np.array([1.0, 2.0, 3.0])
    A = np.random.randn(30, 3)
    b = A @ x_true

    # Add noise
    noise_level = 1e-3
    b_noisy = b + noise_level * np.random.randn(30)

    # Unregularized solution may overfit to noise
    x_unreg = least_squares_solve(A, b_noisy)
    error_unreg = np.linalg.norm(x_unreg - x_true) / np.linalg.norm(x_true)

    # Regularized solution should be more stable
    x_reg = ridge_solve(A, b_noisy, lambda_=1e-2)
    error_reg = np.linalg.norm(x_reg - x_true) / np.linalg.norm(x_true)

    # Regularization should help (in this noisy scenario)
    assert error_reg < 2 * error_unreg  # Loose bound
```

### Near-Singular Tests

Test on matrices close to singular:

```python
def test_qr_near_singular():
    # Create a near-singular matrix
    A = np.eye(20)
    A[-1, :] = A[-2, :]  # Duplicate rows
    A[-1, :] += 1e-10 * np.random.randn(20)  # Small perturbation

    Q, R = qr_decomposition(A)

    # Mathematical properties still hold
    assert_allclose(A, Q @ R, atol=1e-8)
    assert_allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-8)

    # But R should have very small diagonal entries
    assert np.min(np.abs(np.diag(R))) < 1e-8
```

## Layer 3: Reference Validation

Compare implementations to established, trusted libraries: NumPy, SciPy, scikit-learn.

### Purpose
Build confidence that our implementations are correct and identify implementation-specific differences.

### Examples

#### QR Against NumPy

```python
def test_qr_vs_numpy():
    A = np.random.randn(50, 30)

    # Our implementation
    Q_ours, R_ours = our_qr_decomposition(A)

    # NumPy
    Q_numpy, R_numpy = np.linalg.qr(A)

    # Reconstruction should match
    reconstruction_ours = Q_ours @ R_ours
    reconstruction_numpy = Q_numpy @ R_numpy

    assert_allclose(reconstruction_ours, reconstruction_numpy, atol=1e-12)
```

#### Least Squares Against SciPy

```python
def test_least_squares_vs_scipy():
    A = np.random.randn(100, 20)
    b = np.random.randn(100)

    # Our implementation
    x_ours = our_least_squares_solve(A, b)

    # SciPy
    x_scipy = scipy.linalg.lstsq(A, b)[0]

    # Forward error should be small
    assert_allclose(x_ours, x_scipy, rtol=1e-10)
```

#### Ridge Against scikit-learn

```python
def test_ridge_vs_sklearn():
    from sklearn.linear_model import Ridge

    X = np.random.randn(100, 10)
    y = np.random.randn(100)

    # Our implementation
    x_ours = our_ridge_solve(X, y, lambda_=1.0)

    # scikit-learn
    model = Ridge(alpha=1.0, fit_intercept=False)
    model.fit(X, y)
    x_sklearn = model.coef_

    # Solutions should be very close
    assert_allclose(x_ours, x_sklearn, rtol=1e-8)
```

### Interpreting Differences

When our implementation differs from a reference:

1. **Investigate**: is it a precision difference, algorithm difference, or implementation error?
2. **Document**: explain any intended differences
3. **Benchmark**: if differences are systematic, measure the impact
4. **Decide**: accept differences, improve implementation, or investigate further

## Layer 4: Engineering Tests

Standard engineering tests ensure the public API works correctly and contracts are honored.

### Input Validation

```python
def test_least_squares_invalid_shapes():
    A = np.random.randn(10, 5)
    b = np.random.randn(20)  # Wrong size

    with pytest.raises(ValueError):
        least_squares_solve(A, b)
```

### Output Properties

```python
def test_qr_output_types():
    A = np.random.randn(10, 5)
    Q, R = qr_decomposition(A)

    assert isinstance(Q, np.ndarray)
    assert isinstance(R, np.ndarray)
    assert Q.shape == (10, 5)
    assert R.shape == (5, 5)
```

### Determinism

```python
def test_svd_determinism():
    A = np.random.randn(20, 10)

    U1, sigma1, VT1 = svd(A)
    U2, sigma2, VT2 = svd(A)

    assert_allclose(sigma1, sigma2)  # Singular values always in same order
    # (eigenvectors may differ by sign/permutation)
```

### API Contracts

```python
def test_condition_number_properties():
    A = np.random.randn(10, 10)

    kappa = condition_number(A)

    # Properties of condition number
    assert kappa >= 1.0  # Always at least 1
    assert kappa == condition_number(A)  # Deterministic
    assert np.isfinite(kappa)  # Never NaN or Inf
```

## Failure Contract

**No test failure is silently skipped or ignored.**

If a test fails:

1. Investigate the root cause
2. Determine if it reflects a real problem
3. Either fix the code or update the test (with justification)
4. Document unexpected behaviors
5. Make a conscious decision before moving forward

**Example**: If a test shows QR has larger-than-expected error on an ill-conditioned matrix:

```python
# DON'T do this:
if condition_number(A) > 1e10:
    return  # Silently skip!

# DO this:
if condition_number(A) > 1e10:
    # Document: we know QR has large forward error on
    # severely ill-conditioned problems. This is
    # inherent to the problem, not the algorithm.
    # Still verify backward stability.
    pass
```

## Test Organization

Tests are organized into files:

```text
tests/
├── test_core_invariants.py     # Layer 1: invariants
├── test_numerical_behavior.py  # Layer 2: perturbations, noise, etc.
├── test_reference_validation.py # Layer 3: NumPy/SciPy/scikit-learn
├── test_engineering.py          # Layer 4: API, contracts, determinism
└── conftest.py                  # Shared fixtures and utilities
```

## Testing Infrastructure

- **Framework**: pytest
- **Assertions**: numpy.testing.assert_allclose for floating-point comparisons
- **Utilities**: custom tolerance functions that depend on scale and condition
- **Fixtures**: standard problem matrices (well-conditioned, ill-conditioned, etc.)

## Conclusion

A test is **good** if:

1. It verifies something meaningful (a mathematical property, a numerical behavior, or a contract)
2. It fails when something is wrong
3. It passes when the code is correct
4. It communicates what is being tested

A test is **bad** if:

1. It passes when the code has bugs (false negative)
2. It fails when the code is correct (false positive)
3. It obscures what is being tested
4. It is brittle or dependent on implementation details
