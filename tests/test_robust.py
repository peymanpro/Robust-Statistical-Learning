import numpy as np
import pytest
from numpy.testing import assert_allclose

from robust_statistical_learning.learning.robust import (
    huber_loss,
    solve_huber,
)
from robust_statistical_learning.learning.svd import solve_svd


def test_huber_loss_matches_quadratic_region() -> None:
    residual = np.array([-1.0, 0.0, 1.0])
    assert huber_loss(residual, 2.0) == pytest.approx(1.0)


def test_huber_loss_is_linear_beyond_delta() -> None:
    residual = np.array([3.0])
    expected = 2.0 * (3.0 - 1.0)
    assert huber_loss(residual, 2.0) == pytest.approx(expected)


def test_huber_solver_recovers_exact_linear_data() -> None:
    matrix = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    true_solution = np.array([2.0, 3.0])
    observations = matrix @ true_solution

    result = solve_huber(matrix, observations)

    assert result.converged
    assert result.iterations < 20
    assert_allclose(result.solution, true_solution, rtol=1e-10, atol=1e-10)
    assert result.objective == pytest.approx(0.0, abs=1e-10)


def test_huber_is_less_sensitive_to_a_large_outlier_than_least_squares() -> None:
    matrix = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ]
    )
    true_solution = np.array([1.0, 2.0])
    observations = matrix @ true_solution
    observations[-1] += 50.0

    least_squares_error = np.linalg.norm(
        solve_svd(matrix, observations) - true_solution
    )
    huber_result = solve_huber(
        matrix,
        observations,
        delta=1.0,
    )

    assert huber_result.converged
    huber_error = np.linalg.norm(huber_result.solution - true_solution)
    assert huber_error < least_squares_error


def test_huber_rejects_invalid_configuration() -> None:
    matrix = np.eye(2)
    observations = np.ones(2)

    with pytest.raises(ValueError, match="delta"):
        solve_huber(matrix, observations, delta=0.0)

    with pytest.raises(ValueError, match="max_iter"):
        solve_huber(matrix, observations, max_iter=0)

    with pytest.raises(ValueError, match="tol"):
        solve_huber(matrix, observations, tol=0.0)
