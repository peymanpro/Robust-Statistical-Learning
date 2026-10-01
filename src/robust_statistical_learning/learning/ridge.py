"""Ridge and Tikhonov regularized least-squares solvers."""

from collections.abc import Iterable

import numpy as np

from robust_statistical_learning.core.types import Matrix, Vector
from robust_statistical_learning.core.validation import (
    validate_least_squares_system,
    validate_matrix,
)
from robust_statistical_learning.learning.svd import compute_svd, solve_svd


def _validate_alpha(alpha: float) -> None:
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive")


def solve_ridge(matrix: Matrix, observations: Vector, alpha: float) -> Vector:
    """Solve Ridge regression using the SVD filter.

    Minimizes ||Ax - b||_2^2 + alpha ||x||_2^2 without forming A.T @ A.
    """
    validate_least_squares_system(matrix, observations)
    _validate_alpha(alpha)

    u, singular_values, vt = compute_svd(matrix)
    factors = singular_values / (singular_values**2 + alpha)
    return vt.T @ (factors * (u.T @ observations))


def solve_tikhonov(
    matrix: Matrix,
    observations: Vector,
    regularization: Matrix,
    alpha: float,
) -> Vector:
    """Solve generalized Tikhonov regression with an augmented SVD system.

    Minimizes ||Ax - b||_2^2 + alpha ||Lx||_2^2.
    """
    validate_least_squares_system(matrix, observations)
    validate_matrix(regularization)
    _validate_alpha(alpha)

    if regularization.shape[1] != matrix.shape[1]:
        raise ValueError("regularization columns must match matrix columns")

    augmented_matrix = np.vstack(
        [matrix, np.sqrt(alpha) * regularization]
    )
    augmented_observations = np.concatenate(
        [observations, np.zeros(regularization.shape[0])]
    )
    return solve_svd(augmented_matrix, augmented_observations)


def ridge_gcv_score(
    matrix: Matrix,
    observations: Vector,
    alpha: float,
) -> float:
    """Return generalized cross-validation score for Ridge."""
    validate_least_squares_system(matrix, observations)
    _validate_alpha(alpha)

    u, singular_values, vt = compute_svd(matrix)
    projected = u.T @ observations
    factors = singular_values / (singular_values**2 + alpha)
    solution = vt.T @ (factors * projected)

    residual = matrix @ solution - observations
    effective_dof = float(
        np.sum(singular_values**2 / (singular_values**2 + alpha))
    )
    denominator = matrix.shape[0] - effective_dof

    if denominator <= 0.0:
        return float(np.inf)

    return float(np.linalg.norm(residual) ** 2 / denominator**2)


def select_ridge_alpha_gcv(
    matrix: Matrix,
    observations: Vector,
    candidates: Iterable[float],
) -> float:
    """Select the Ridge parameter with the smallest GCV score."""
    validate_least_squares_system(matrix, observations)

    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("candidates must contain at least one alpha")

    for alpha in candidate_list:
        _validate_alpha(alpha)

    scores = [
        ridge_gcv_score(matrix, observations, alpha)
        for alpha in candidate_list
    ]
    return float(candidate_list[int(np.argmin(scores))])
