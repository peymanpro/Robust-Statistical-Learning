# Mathematical Foundations

This document contains the core mathematical formulations underlying the project. All equations are the foundation for implementation, testing, and validation.

## Least Squares Problem

The fundamental least-squares problem:

$$\hat{x} = \operatorname*{arg\,min}_x \|Ax - b\|_2^2$$

where:
- \(A \in \mathbb{R}^{m \times n}\) is the data/design matrix
- \(b \in \mathbb{R}^m\) is the observation vector
- \(x \in \mathbb{R}^n\) is the solution vector
- \(\|\cdot\|_2\) is the Euclidean (L2) norm

**Assumption**: \(m \geq n\) (overdetermined or square system).

## Vectors and Norms

### L2 Norm (Euclidean Norm)

$$\|v\|_2 = \sqrt{\sum_{i=1}^n v_i^2}$$

for \(v \in \mathbb{R}^n\).

### Frobenius Norm (Matrix L2 Norm)

$$\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{ij}^2}$$

for \(A \in \mathbb{R}^{m \times n}\).

### Operator Norm (Spectral Norm)

$$\|A\|_2 = \max_{x \neq 0} \frac{\|Ax\|_2}{\|x\|_2} = \sigma_{\max}(A)$$

where \(\sigma_{\max}(A)\) is the largest singular value of \(A\).

## Residual

The residual vector at a candidate solution \(x\):

$$r = Ax - b$$

The residual norm:

$$\|r\| = \|Ax - b\|$$

At the least-squares solution \(\hat{x}\):

$$r = A\hat{x} - b$$

The minimal residual norm is:

$$\|r_{\min}\| = \min_x \|Ax - b\|$$

## Normal Equations Formulation

The normal equations are derived from setting the gradient of the objective to zero:

$$\nabla_x \|Ax - b\|_2^2 = 2A^T(Ax - b) = 0$$

This yields:

$$A^T A x = A^T b$$

The solution is:

$$\hat{x} = (A^T A)^{-1} A^T b$$

**Condition Number Warning**: \(\kappa(A^T A) \approx \kappa(A)^2\), which can be problematic for ill-conditioned problems.

## QR Decomposition

Any matrix \(A \in \mathbb{R}^{m \times n}\) with \(m \geq n\) can be decomposed as:

$$A = QR$$

where:
- \(Q \in \mathbb{R}^{m \times n}\) has orthonormal columns (\(Q^T Q = I_n\))
- \(R \in \mathbb{R}^{n \times n}\) is upper triangular

The least-squares solution via QR:

$$Rx = Q^T b$$

Solve this triangular system to get \(\hat{x}\).

**Advantage**: Better numerical stability than normal equations. \(\kappa(R) \approx \kappa(A)\).

## Singular Value Decomposition (SVD)

Any matrix \(A \in \mathbb{R}^{m \times n}\) can be decomposed as:

$$A = U\Sigma V^T$$

where:
- \(U \in \mathbb{R}^{m \times m}\) is orthogonal (columns are left singular vectors)
- \(\Sigma \in \mathbb{R}^{m \times n}\) is diagonal with non-negative entries (singular values) in decreasing order
- \(V \in \mathbb{R}^{n \times n}\) is orthogonal (columns are right singular vectors)

The singular values: \(\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_n \geq 0\)

### Pseudoinverse via SVD

$$A^+ = V\Sigma^+ U^T$$

where \(\Sigma^+\) is the pseudoinverse of \(\Sigma\) (transpose, reciprocate nonzero diagonal entries).

The least-squares solution via SVD:

$$\hat{x} = A^+ b = V\Sigma^+ U^T b$$

## Condition Number

### Definition

$$\kappa(A) = \|A\|\|A^{-1}\|$$

In terms of singular values (for full-rank \(A\)):

$$\kappa(A) = \frac{\sigma_{\max}(A)}{\sigma_{\min}(A)}$$

### Interpretation

The condition number measures sensitivity of the linear system to perturbations:

- **\(\kappa(A) \approx 1\)**: well-conditioned; small perturbations cause small changes
- **\(\kappa(A) \gg 1\)**: ill-conditioned; small perturbations can cause large changes
- **\(\kappa(A) \approx 10^{k}\)**: expect to lose approximately \(k\) digits of precision

### Normal Equations Condition Number

$$\kappa(A^T A) = \kappa(A)^2$$

**Critical Issue**: Squaring the condition number can double the impact of ill-conditioning. This is why QR and SVD methods are numerically preferable to normal equations for ill-conditioned problems.

## Forward Error

The forward error (or absolute forward error) at an approximate solution \(\tilde{x}\):

$$e = \tilde{x} - \hat{x}$$

The relative forward error:

$$\frac{\|e\|}{\|\hat{x}\|} = \frac{\|\tilde{x} - \hat{x}\|}{\|\hat{x}\|}$$

**Interpretation**: How far is the computed solution from the true solution?

## Backward Error

Backward error measures how much the problem must be perturbed to make the computed solution exact.

For a computed solution \(\tilde{x}\), the backward error asks: what is the smallest \(\Delta A\) and \(\Delta b\) such that:

$$(A + \Delta A)\tilde{x} = b + \Delta b$$

**Interpretation**: How close is the computed solution to the true solution of a nearby problem?

A small backward error with a well-conditioned problem implies a small forward error.

## Regularization: Ridge Regression / Tikhonov

The regularized least-squares problem:

$$\hat{x}_\lambda = \operatorname*{arg\,min}_x \left(\|Ax - b\|_2^2 + \lambda \|x\|_2^2\right)$$

where \(\lambda > 0\) is the regularization parameter.

### Normal Equations Form

$$(A^T A + \lambda I)\hat{x}_\lambda = A^T b$$

### SVD Form

$$\hat{x}_\lambda = V\Sigma_\lambda^+ U^T b$$

where \(\Sigma_\lambda^+\) has diagonal entries:

$$\frac{\sigma_i}{\sigma_i^2 + \lambda}$$

### Effect on Conditioning

The condition number of the regularized system:

$$\kappa(A^T A + \lambda I) \leq \kappa(A^T A)$$

Regularization improves conditioning by suppressing small singular values.

### Bias-Variance Tradeoff

- **\(\lambda \to 0\)**: solution approaches unregularized least squares (high variance, low bias)
- **\(\lambda \to \infty\)**: solution approaches zero (low variance, high bias)

The optimal \(\lambda\) balances these effects.

## Principal Component Analysis (PCA)

### SVD Formulation

Given a data matrix \(X \in \mathbb{R}^{m \times n}\) (rows are observations, columns are features):

$$X = U\Sigma V^T$$

### Centering

Typically, center the data first:

$$X_c = X - \frac{1}{m}\mathbf{1}\mathbf{1}^T X$$

where \(\mathbf{1}\) is the vector of ones.

### Principal Components

The right singular vectors \(V\) are the principal component directions. The left singular vectors \(U\) (scaled by singular values) give the principal component scores.

### Explained Variance

The fraction of variance explained by the first \(k\) components:

$$\text{FVE}(k) = \frac{\sum_{i=1}^k \sigma_i^2}{\sum_{i=1}^n \sigma_i^2}$$

This guides dimensionality reduction decisions.

### Reconstruction from Components

Using the first \(k\) components:

$$\tilde{X} = U_k \Sigma_k V_k^T$$

where \(U_k\), \(\Sigma_k\), \(V_k^T\) use only the first \(k\) singular values and vectors.

The reconstruction error:

$$\|X - \tilde{X}\|_F = \sum_{i=k+1}^n \sigma_i^2$$

## Mathematical Traceability

Every algorithm in this project follows a standard traceability pattern:

1. **Mathematical Definition** — The equation(s) from this document
2. **Algorithm** — Step-by-step procedure in pseudocode
3. **Assumptions** — Problem size, conditioning, precision requirements
4. **Numerical Properties** — Stability, conditioning, error behavior
5. **Invariants** — Mathematical properties that must hold (e.g., \(Q^T Q \approx I\))
6. **Tests** — Concrete checks that verify correctness

Example traceability chain for QR decomposition:

```
Mathematical Definition (QR = A)
    ↓
Algorithm (Householder / Gram-Schmidt)
    ↓
Assumptions (full rank, m >= n)
    ↓
Numerical Properties (backward stable)
    ↓
Invariants (Q'Q ≈ I, A ≈ QR)
    ↓
Tests (numerical + reference)
```

Every implementation connects back to this document.

## Reference

These formulations are standard in numerical linear algebra. See:

- Golub & van Loan, *Matrix Computations*, 4th edition
- Trefethen & Bau, *Numerical Linear Algebra*
- Boyd & Vandenberghe, *Convex Optimization*
