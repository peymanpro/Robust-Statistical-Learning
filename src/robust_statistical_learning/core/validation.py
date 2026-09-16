"""Validation helpers for numerical arrays."""

import numpy as np

from .types import Matrix, Vector


def validate_matrix(matrix: Matrix) -> None:
    """Validate that an input is a finite 2D float64 matrix."""
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("matrix must contain only finite values")


def validate_vector(vector: Vector) -> None:
    """Validate that an input is a finite 1D float64 vector."""
    if vector.ndim != 1:
        raise ValueError("vector must be one-dimensional")
    if not np.all(np.isfinite(vector)):
        raise ValueError("vector must contain only finite values")


def validate_least_squares_system(
    matrix: Matrix,
    observations: Vector,
) -> None:
    """Validate the A,b dimensions used by least-squares solvers."""
    validate_matrix(matrix)
    validate_vector(observations)

    rows, columns = matrix.shape

    if rows == 0 or columns == 0:
        raise ValueError("least-squares matrix must be non-empty")

    if rows != observations.shape[0]:
        raise ValueError("matrix rows must match observation dimension")

    if rows < columns:
        raise ValueError(
            "least-squares matrix must have at least as many rows as columns"
        )


def validate_least_squares_dimensions(
    matrix: Matrix,
    solution: Vector,
    observations: Vector,
) -> None:
    """Validate dimensions for Ax - b."""
    validate_least_squares_system(matrix, observations)
    validate_vector(solution)

    if matrix.shape[1] != solution.shape[0]:
        raise ValueError("matrix columns must match solution dimension")