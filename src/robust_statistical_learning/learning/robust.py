"""Robust regression methods."""

from dataclasses import dataclass

import numpy as np

from robust_statistical_learning.core.types import Matrix, Vector
from robust_statistical_learning.core.validation import validate_least_squares_system
from robust_statistical_learning.learning.svd import solve_svd


@dataclass(frozen=True)
class HuberResult:
    """Outcome of an IRLS Huber regression fit."""

    solution: Vector
    iterations: int
    converged: bool
    objective: float


def huber_loss(residual: Vector, delta: float) -> float:
    """Return the summed Huber loss for a residual vector."""
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be finite and positive")

    absolute = np.abs(residual)
    quadratic = 0.5 * residual**2
    linear = delta * (absolute - 0.5 * delta)
    return float(np.sum(np.where(absolute <= delta, quadratic, linear)))


def solve_huber(
    matrix: Matrix,
    observations: Vector,
    *,
    delta: float = 1.345,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> HuberResult:
    """Fit a Huber-regression model with iteratively reweighted least squares.

    The Huber influence function is implemented through weights

        w_i = 1                         if |r_i| <= delta
              delta / |r_i|             otherwise.

    Each weighted least-squares subproblem is solved by the SVD solver.
    """
    validate_least_squares_system(matrix, observations)

    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be finite and positive")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")

    solution = solve_svd(matrix, observations)

    for iteration in range(1, max_iter + 1):
        residual = matrix @ solution - observations
        absolute = np.abs(residual)

        weights = np.ones_like(residual)
        outliers = absolute > delta
        weights[outliers] = delta / absolute[outliers]

        sqrt_weights = np.sqrt(weights)
        weighted_matrix = sqrt_weights[:, None] * matrix
        weighted_observations = sqrt_weights * observations

        updated = solve_svd(weighted_matrix, weighted_observations)
        denominator = max(1.0, float(np.linalg.norm(solution)))
        change = float(np.linalg.norm(updated - solution) / denominator)
        solution = updated

        if change <= tol:
            return HuberResult(
                solution=solution,
                iterations=iteration,
                converged=True,
                objective=huber_loss(matrix @ solution - observations, delta),
            )

    return HuberResult(
        solution=solution,
        iterations=max_iter,
        converged=False,
        objective=huber_loss(matrix @ solution - observations, delta),
    )
