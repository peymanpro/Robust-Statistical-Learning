# Principal Component Analysis

PCA is implemented with centered-data SVD.

Given X, the implementation:

1. computes the feature-wise mean;
2. centers X;
3. computes reduced SVD;
4. retains the requested right singular vectors;
5. reports explained variance and explained-variance ratio.

For n observations and centered singular values sigma_i:

    explained_variance_i = sigma_i^2 / (n - 1).

The retained components are orthonormal. Transform and inverse_transform use
the same fitted mean and component basis, making the learned transformation
explicit and reproducible.

## Reconstruction

With k retained components:

    Z = (X - mean) V_k^T
    X_hat = Z V_k + mean.

The reconstruction diagnostic is

    ||X - X_hat||_F / ||X||_F.

## Numerical Choice

Computing PCA from X.T X is algebraically convenient but squares the condition
number of the centered data matrix. This implementation stays in the SVD basis.

## API

fit_pca(X, n_components) returns an immutable PCAResult containing:

- mean
- components
- singular values
- explained variance
- explained variance ratio

PCAResult provides transform and inverse_transform.

## Scope

The implementation is dense and intentionally small. Sparse, incremental,
randomized, and kernel PCA are separate possible extensions.
