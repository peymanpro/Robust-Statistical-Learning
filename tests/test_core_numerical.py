import numpy as np
import pytest
from numpy.testing import assert_allclose

from robust_statistical_learning.core.metrics import (
    condition_number,
    frobenius_norm,
    l2_norm,
)
from robust_statistical_learning.core.residuals import (
    residual,
    residual_norm,
)
from robust_statistical_learning.core.validation import (
    validate_least_squares_dimensions,
    validate_matrix,
    validate_vector,
)


def test_l2_norm() -> None:
    vector = np.array([3.0, 4.0])

    assert l2_norm(vector) == pytest.approx(5.0)


def test_frobenius_norm() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]])

    assert frobenius_norm(matrix) == pytest.approx(5.0)


def test_condition_number_identity_is_one() -> None:
    matrix = np.eye(3)

    assert condition_number(matrix) == pytest.approx(1.0)


def test_condition_number_singular_matrix_is_infinite() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 4.0]])

    assert np.isinf(condition_number(matrix))


def test_condition_number_tolerance_can_detect_effective_rank_deficiency() -> None:
    matrix = np.diag([1.0, 1e-8])

    assert np.isfinite(condition_number(matrix))
    assert np.isinf(condition_number(matrix, tol=1e-6))


def test_condition_number_custom_tolerance_can_preserve_full_rank() -> None:
    matrix = np.diag([1.0, 1e-8])

    assert np.isfinite(condition_number(matrix, tol=1e-12))


def test_validate_matrix_rejects_one_dimensional_input() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        validate_matrix(np.array([1.0, 2.0]))


def test_validate_matrix_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_matrix(np.array([[1.0, np.nan]]))


def test_validate_vector_rejects_two_dimensional_input() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_vector(np.array([[1.0, 2.0]]))


def test_validate_vector_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_vector(np.array([1.0, np.inf]))


def test_validate_least_squares_dimensions_accepts_matching_shapes() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    solution = np.array([1.0, 2.0])
    observations = np.array([5.0, 11.0, 17.0])

    validate_least_squares_dimensions(matrix, solution, observations)


def test_validate_least_squares_dimensions_rejects_solution_mismatch() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    solution = np.array([1.0])
    observations = np.array([5.0, 11.0])

    with pytest.raises(ValueError, match="solution dimension"):
        validate_least_squares_dimensions(matrix, solution, observations)


def test_validate_least_squares_dimensions_rejects_observation_mismatch() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    solution = np.array([1.0, 2.0])
    observations = np.array([5.0])

    with pytest.raises(ValueError, match="observation dimension"):
        validate_least_squares_dimensions(matrix, solution, observations)


def test_norms_match_direct_euclidean_calculation() -> None:
    vector = np.array([1.0, -2.0, 3.0])

    assert_allclose(l2_norm(vector), np.sqrt(vector @ vector))


def test_core_residual_matches_ax_minus_b() -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    solution = np.array([2.0, -1.0])
    observations = np.array([4.0, 5.0])

    assert_allclose(
        residual(matrix, solution, observations),
        np.array([-4.0, -3.0]),
    )


def test_core_residual_norm_matches_euclidean_norm() -> None:
    matrix = np.eye(2)
    solution = np.array([3.0, 4.0])
    observations = np.zeros(2)

    assert residual_norm(matrix, solution, observations) == pytest.approx(5.0)


def test_core_residual_rejects_dimension_mismatch() -> None:
    matrix = np.eye(2)
    solution = np.array([1.0])
    observations = np.zeros(2)

    with pytest.raises(ValueError, match="solution dimension"):
        residual(matrix, solution, observations)
def test_validate_least_squares_dimensions_rejects_empty_matrix() -> None:
    matrix = np.empty((0, 2))
    solution = np.array([1.0, 2.0])
    observations = np.empty(0)

    with pytest.raises(ValueError, match="non-empty"):
        validate_least_squares_dimensions(matrix, solution, observations)


def test_validate_least_squares_dimensions_rejects_underdetermined_system() -> None:
    matrix = np.eye(2, 3)
    solution = np.array([1.0, 2.0, 3.0])
    observations = np.array([1.0, 2.0])

    with pytest.raises(ValueError, match="at least as many rows"):
        validate_least_squares_dimensions(matrix, solution, observations)