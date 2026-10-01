import numpy as np
import pytest
from numpy.testing import assert_allclose

from robust_statistical_learning.core.errors import (
    normwise_backward_error,
    relative_forward_error,
    residual_orthogonality_error,
)


def test_relative_forward_error_is_zero_for_exact_match() -> None:
    vector = np.array([1.0, 2.0, 3.0])
    assert relative_forward_error(vector, vector) == pytest.approx(0.0)


def test_relative_forward_error_matches_definition() -> None:
    computed = np.array([1.0, 3.0])
    reference = np.array([1.0, 2.0])

    expected = np.linalg.norm(computed - reference) / np.linalg.norm(reference)

    assert relative_forward_error(computed, reference) == pytest.approx(expected)


def test_relative_forward_error_handles_zero_reference() -> None:
    zero = np.zeros(2)

    assert relative_forward_error(zero, zero) == pytest.approx(0.0)
    assert np.isinf(relative_forward_error(np.ones(2), zero))


def test_relative_forward_error_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        relative_forward_error(np.zeros((2, 1)), np.zeros(2))


def test_normwise_backward_error_is_zero_for_exact_solution() -> None:
    matrix = np.eye(2)
    solution = np.array([2.0, 3.0])
    observations = matrix @ solution

    assert normwise_backward_error(matrix, solution, observations) == pytest.approx(0.0)


def test_normwise_backward_error_is_small_for_a_small_residual() -> None:
    matrix = np.eye(2)
    solution = np.array([2.0, 3.0])
    observations = np.array([2.0, 3.0 + 1e-8])

    expected = np.linalg.norm(matrix @ solution - observations) / (
        np.linalg.norm(matrix, ord="fro") * np.linalg.norm(solution)
        + np.linalg.norm(observations)
    )

    assert_allclose(
        normwise_backward_error(matrix, solution, observations),
        expected,
    )


def test_normwise_backward_error_rejects_solution_dimension() -> None:
    with pytest.raises(ValueError, match="solution dimension"):
        normwise_backward_error(
            np.eye(2),
            np.array([1.0]),
            np.array([1.0, 2.0]),
        )


def test_residual_orthogonality_error_is_zero_for_exact_fit() -> None:
    matrix = np.array([[1.0], [2.0], [3.0]])
    solution = np.array([2.0])
    observations = matrix @ solution

    assert residual_orthogonality_error(matrix, solution, observations) == pytest.approx(0.0)


def test_residual_orthogonality_error_is_small_for_least_squares_solution() -> None:
    matrix = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )
    observations = np.array([2.1, 4.9, 8.2, 10.8])

    solution = np.linalg.lstsq(matrix, observations, rcond=None)[0]

    assert residual_orthogonality_error(
        matrix,
        solution,
        observations,
    ) < 1e-12
