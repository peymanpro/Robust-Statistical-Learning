"""SVD-based least-squares solver."""

from __future__ import annotations

import numpy as np

from robust_statistical_learning.core.types import Matrix, Vector
from robust_statistical_learning.core.validation import (
    validate_least_squares_system,
    validate_matrix,
)


def compute_svd(
    matrix: Matrix,
) -> tuple[Matrix, Vector, Matrix]:
    """Compute the reduced singular value decomposition of a matrix.

    Returns:

        A = U @ diag(singular_values) @ Vt

    using ``full_matrices=False``.
    """
    validate_matrix(matrix)

    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix must be non-empty")

    u, singular_values, vt = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    return u, singular_values, vt


def solve_svd(
    matrix: Matrix,
    observations: Vector,
    rcond: float | None = None,
) -> Vector:
    """Solve least squares using the SVD pseudoinverse.

    Solves:

        A = U @ Sigma @ V.T
        x = V @ Sigma_plus @ U.T @ b

    Singular values below the numerical-rank threshold are treated as zero.
    Rank-deficient systems are therefore supported.
    """
    validate_least_squares_system(matrix, observations)

    if rcond is not None and (
        not np.isfinite(rcond) or rcond < 0.0
    ):
        raise ValueError("rcond must be finite and non-negative")

    u, singular_values, vt = compute_svd(matrix)

    if rcond is None:
        tolerance = (
            singular_values[0]
            * max(matrix.shape)
            * np.finfo(matrix.dtype).eps
        )
    else:
        tolerance = rcond * singular_values[0]

    inverse_singular_values = np.zeros_like(singular_values)
    active = singular_values > tolerance
    inverse_singular_values[active] = 1.0 / singular_values[active]

    transformed = u.T @ observations
    return vt.T @ (inverse_singular_values * transformed)
