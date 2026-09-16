"""Numerical metrics used by the numerical core."""

import numpy as np

from .types import Matrix, Vector
from .validation import validate_matrix, validate_vector


def l2_norm(vector: Vector) -> float:
    """Return the Euclidean norm of a vector."""
    validate_vector(vector)
    return float(np.linalg.norm(vector, ord=2))


def frobenius_norm(matrix: Matrix) -> float:
    """Return the Frobenius norm of a matrix."""
    validate_matrix(matrix)
    return float(np.linalg.norm(matrix, ord="fro"))


def condition_number(
    matrix: Matrix,
    *,
    tol: float | None = None,
) -> float:
    """Return the 2-norm condition number using numerical rank detection.

    A numerically rank-deficient matrix has infinite condition number.
    When ``tol`` is None, NumPy's default SVD-based rank threshold is used.
    """
    validate_matrix(matrix)

    rank = np.linalg.matrix_rank(matrix, tol=tol)
    full_rank = rank == min(matrix.shape)

    if not full_rank:
        return float(np.inf)

    return float(np.linalg.cond(matrix, p=2))