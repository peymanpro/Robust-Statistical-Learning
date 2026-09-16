"""Normal-equations least-squares solver."""

import numpy as np

from robust_statistical_learning.core.types import Matrix, Vector
from robust_statistical_learning.core.validation import validate_least_squares_dimensions


def solve_normal_equations(
    matrix: Matrix,
    observations: Vector,
) -> Vector:
    """Solve least squares using the normal equations.

    Solves:

        (A.T @ A) x = A.T @ b

    The input matrix must have full column rank.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    if observations.ndim != 1:
        raise ValueError("observations must be one-dimensional")

    solution_placeholder = np.zeros(matrix.shape[1], dtype=np.float64)

    validate_least_squares_dimensions(
        matrix,
        solution_placeholder,
        observations,
    )

    rank = np.linalg.matrix_rank(matrix)

    if rank < matrix.shape[1]:
        raise ValueError("normal equations require full column rank")

    normal_matrix = matrix.T @ matrix
    normal_rhs = matrix.T @ observations

    return np.linalg.solve(normal_matrix, normal_rhs)