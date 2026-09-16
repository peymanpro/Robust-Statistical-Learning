"""QR-based least-squares solver."""

import numpy as np

from robust_statistical_learning.core.types import Matrix, Vector
from robust_statistical_learning.core.validation import (
    validate_least_squares_system,
)


def solve_qr(
    matrix: Matrix,
    observations: Vector,
) -> Vector:
    """Solve least squares using reduced QR factorization.

    Solves:

        A = QR
        Rx = Q.T @ b

    The input matrix must have full column rank.
    """
    validate_least_squares_system(matrix, observations)

    rank = np.linalg.matrix_rank(matrix)

    if rank < matrix.shape[1]:
        raise ValueError("QR solver requires full column rank")

    q, r = np.linalg.qr(matrix, mode="reduced")
    return np.linalg.solve(r, q.T @ observations)