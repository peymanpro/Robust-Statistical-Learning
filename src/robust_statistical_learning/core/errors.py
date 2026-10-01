"""Forward, residual, and normwise backward-error diagnostics."""

import numpy as np

from .metrics import l2_norm
from .types import Matrix, Vector
from .validation import validate_least_squares_system, validate_vector


def relative_forward_error(computed: Vector, reference: Vector) -> float:
    """Return ||computed - reference||_2 / ||reference||_2.

    For a zero reference, the result is 0 when the computed vector is also
    zero and infinity otherwise.
    """
    validate_vector(computed)
    validate_vector(reference)

    reference_norm = l2_norm(reference)
    difference_norm = l2_norm(computed - reference)

    if reference_norm == 0.0:
        return 0.0 if difference_norm == 0.0 else float(np.inf)

    return difference_norm / reference_norm


def normwise_backward_error(
    matrix: Matrix,
    solution: Vector,
    observations: Vector,
) -> float:
    """Return a normwise residual-based backward-error indicator.

    The quantity

        ||Ax - b||_2 / (||A||_F ||x||_2 + ||b||_2)

    is the relative size of a joint perturbation bound for the computed
    least-squares relation. It measures whether the computed solution
    nearly solves a nearby problem; it is not claimed to be the exact
    minimal perturbation in every norm.
    """
    validate_least_squares_system(matrix, observations)
    validate_vector(solution)

    if matrix.shape[1] != solution.shape[0]:
        raise ValueError("matrix columns must match solution dimension")

    residual = matrix @ solution - observations
    denominator = (
        np.linalg.norm(matrix, ord="fro") * np.linalg.norm(solution, ord=2)
        + np.linalg.norm(observations, ord=2)
    )

    residual_norm = np.linalg.norm(residual, ord=2)
    if denominator == 0.0:
        return 0.0 if residual_norm == 0.0 else float(np.inf)

    return float(residual_norm / denominator)


def residual_orthogonality_error(
    matrix: Matrix,
    solution: Vector,
    observations: Vector,
) -> float:
    """Measure relative violation of the least-squares normality condition.

    For an exact least-squares solution, A.T @ (Ax-b) = 0. The returned
    value is normalized by ||A||_2 ||r||_2 and is zero for a zero residual.
    """
    validate_least_squares_system(matrix, observations)
    validate_vector(solution)

    if matrix.shape[1] != solution.shape[0]:
        raise ValueError("matrix columns must match solution dimension")

    residual = matrix @ solution - observations
    residual_norm = np.linalg.norm(residual, ord=2)

    if residual_norm == 0.0:
        return 0.0

    denominator = np.linalg.norm(matrix, ord=2) * residual_norm
    if denominator == 0.0:
        return float(np.inf)

    return float(np.linalg.norm(matrix.T @ residual, ord=2) / denominator)
