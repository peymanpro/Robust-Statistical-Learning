"""Residual calculations for numerical linear models."""

from .metrics import l2_norm
from .types import Matrix, Vector
from .validation import validate_least_squares_dimensions


def residual(
    matrix: Matrix,
    solution: Vector,
    observations: Vector,
) -> Vector:
    """Return the residual r = Ax - b."""
    validate_least_squares_dimensions(matrix, solution, observations)
    return matrix @ solution - observations


def residual_norm(
    matrix: Matrix,
    solution: Vector,
    observations: Vector,
) -> float:
    """Return ||Ax - b||_2."""
    return l2_norm(residual(matrix, solution, observations))