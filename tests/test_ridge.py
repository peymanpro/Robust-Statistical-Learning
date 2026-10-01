import numpy as np
import pytest
from numpy.testing import assert_allclose

from robust_statistical_learning.learning.ridge import (
    ridge_gcv_score,
    select_ridge_alpha_gcv,
    solve_ridge,
    solve_tikhonov,
)


def test_ridge_recovers_exact_solution_for_small_alpha() -> None:
    matrix = np.eye(2)
    observations = np.array([2.0, 4.0])

    solution = solve_ridge(matrix, observations, 1e-12)

    assert_allclose(solution, observations, rtol=1e-10, atol=1e-10)


def test_ridge_shrinks_solution() -> None:
    matrix = np.eye(2)
    observations = np.array([2.0, 4.0])

    assert_allclose(
        solve_ridge(matrix, observations, 1.0),
        np.array([1.0, 2.0]),
    )


def test_ridge_matches_singular_vector_formula() -> None:
    matrix = np.array([[3.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    observations = np.array([3.0, 1.0, 0.0])
    alpha = 2.0

    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    expected = vt.T @ (
        (singular_values / (singular_values**2 + alpha))
        * (u.T @ observations)
    )

    assert_allclose(
        solve_ridge(matrix, observations, alpha),
        expected,
    )


def test_ridge_rejects_non_positive_alpha() -> None:
    with pytest.raises(ValueError, match="positive"):
        solve_ridge(np.eye(2), np.ones(2), 0.0)

    with pytest.raises(ValueError, match="positive"):
        solve_ridge(np.eye(2), np.ones(2), -1.0)


def test_tikhonov_reduces_to_ridge_for_identity_regularizer() -> None:
    matrix = np.array([[2.0], [1.0], [3.0]])
    observations = np.array([4.0, 2.0, 6.0])
    alpha = 0.5

    assert_allclose(
        solve_tikhonov(matrix, observations, np.eye(1), alpha),
        solve_ridge(matrix, observations, alpha),
    )


def test_tikhonov_rejects_wrong_regularization_shape() -> None:
    with pytest.raises(ValueError, match="regularization columns"):
        solve_tikhonov(
            np.eye(2),
            np.ones(2),
            np.ones((2, 3)),
            1.0,
        )


def test_gcv_score_is_finite_for_valid_alpha() -> None:
    matrix = np.vstack([np.eye(2), np.eye(2)])
    observations = np.array([2.0, 4.0, 2.0, 4.0])

    assert np.isfinite(ridge_gcv_score(matrix, observations, 1.0))


def test_gcv_selects_from_candidates() -> None:
    matrix = np.vstack([np.eye(2), np.eye(2)])
    observations = np.array([2.0, 4.0, 2.0, 4.0])
    candidates = [1e-3, 1e-2, 1e-1, 1.0]

    selected = select_ridge_alpha_gcv(matrix, observations, candidates)

    assert selected in candidates


def test_gcv_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_ridge_alpha_gcv(np.eye(2), np.ones(2), [])


def test_gcv_rejects_invalid_candidate() -> None:
    with pytest.raises(ValueError, match="positive"):
        select_ridge_alpha_gcv(np.eye(2), np.ones(2), [0.0])


def test_ridge_gcv_is_finite_for_an_ill_conditioned_system() -> None:
    matrix = np.diag([1.0, 1e-6])
    observations = np.array([1.0, 1e-6])

    assert np.isfinite(ridge_gcv_score(matrix, observations, 1e-4))
