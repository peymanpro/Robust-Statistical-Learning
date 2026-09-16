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


def validate_least_squares_dimensions(
    matrix: Matrix,
    solution: Vector,
    observations: Vector,
) -> None:
    """Validate dimensions for Ax - b."""
    validate_matrix(matrix)
    validate_vector(solution)
    validate_vector(observations)

    if matrix.shape[1] != solution.shape[0]:
        raise ValueError("matrix columns must match solution dimension")

    if matrix.shape[0] != observations.shape[0]:
        raise ValueError("matrix rows must match observation dimension")