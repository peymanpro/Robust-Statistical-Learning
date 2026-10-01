# Real-Data Validation

The project uses small datasets bundled with scikit-learn as reproducible
reference cases.

## Regression

Dataset: scikit-learn diabetes.

The project solves the same no-intercept least-squares problem with the SVD
solver and compares the coefficient vector against
sklearn.linear_model.LinearRegression(fit_intercept=False).

The purpose is formulation and implementation validation, not a claim of
superiority over scikit-learn.

## PCA

Dataset: scikit-learn iris.

The project compares explained-variance ratios from its SVD-based PCA
implementation against sklearn.decomposition.PCA.

Component-vector sign is not used as the primary criterion because singular
vectors are sign-indeterminate: v and -v represent the same principal direction.

## Reproducibility

Run:

    python experiments/real_data_validation.py

Exact numerical differences depend on NumPy, scikit-learn, BLAS/LAPACK, and the
floating-point environment.

## Scope

This is a reference-validation exercise, not a benchmark of library performance
or predictive superiority.
