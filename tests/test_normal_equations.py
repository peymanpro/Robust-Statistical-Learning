import numpy as np
import pytest
from numpy.testing import assert_allclose

from robust_statistical_learning.learning.normal_equations import (
    solve_normal_equations,
)


def test_normal_equations_recovers_exact_square_system() -> None:
    matrix = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
    ])
    expected = np.array([4.0, -2.0])
    observations = matrix @ expected

    actual = solve_normal_equations(matrix, observations)

    assert_allclose(actual, expected)


def test_normal_equations_solves_overdetermined_system() -> None:
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ])
    expected = np.array([2.0, 3.0])
    observations = matrix @ expected

    actual = solve_normal_equations(matrix, observations)

    assert_allclose(actual, expected)


def test_normal_equations_returns_least_squares_solution_for_consistent_data() -> None:
    matrix = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 4.0],
    ])
    expected = np.array([5.0, -2.0])
    observations = matrix @ expected

    actual = solve_normal_equations(matrix, observations)

    assert_allclose(actual, expected)


def test_normal_equations_rejects_rank_deficient_matrix() -> None:
    matrix = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])
    observations = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="full column rank"):
        solve_normal_equations(matrix, observations)
def test_normal_equations_solution_minimizes_residual_for_noisy_data() -> None:
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ])
    observations = np.array([2.0, 5.0, 7.0, 11.0])

    solution = solve_normal_equations(matrix, observations)

    candidate_residual = matrix @ solution - observations

    assert np.linalg.norm(candidate_residual) <= np.linalg.norm(
        matrix @ np.array([2.0, 3.0]) - observations
    )


def test_normal_equations_rejects_wrong_observation_dimension() -> None:
    matrix = np.eye(2)
    observations = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="observation dimension"):
        solve_normal_equations(matrix, observations)