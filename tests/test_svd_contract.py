import numpy as np
from numpy.testing import assert_allclose

from robust_statistical_learning.core.tolerances import ATOL, RTOL
from robust_statistical_learning.learning.svd import compute_svd, solve_svd


def test_compute_svd_returns_reduced_shapes() -> None:
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    u, singular_values, vt = compute_svd(matrix)

    assert u.shape == (3, 2)
    assert singular_values.shape == (2,)
    assert vt.shape == (2, 2)


def test_compute_svd_reconstructs_matrix() -> None:
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    u, singular_values, vt = compute_svd(matrix)
    reconstructed = u @ np.diag(singular_values) @ vt

    assert_allclose(reconstructed, matrix, rtol=RTOL, atol=ATOL)


def test_compute_svd_left_vectors_are_orthonormal() -> None:
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    u, _, _ = compute_svd(matrix)

    assert_allclose(
        u.T @ u,
        np.eye(u.shape[1]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_compute_svd_right_vectors_are_orthonormal() -> None:
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    _, _, vt = compute_svd(matrix)

    assert_allclose(
        vt @ vt.T,
        np.eye(vt.shape[0]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_compute_svd_singular_values_are_sorted_descending() -> None:
    matrix = np.array([
        [3.0, 0.0],
        [0.0, 2.0],
        [0.0, 0.0],
    ])

    _, singular_values, _ = compute_svd(matrix)

    assert np.all(singular_values[:-1] >= singular_values[1:])


def test_solve_svd_exact_full_rank_system() -> None:
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
    ])
    expected = np.array([2.0, 3.0])
    observations = matrix @ expected

    actual = solve_svd(matrix, observations)

    assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_solve_svd_matches_numpy_lstsq_reference() -> None:
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ])
    observations = np.array([2.0, 5.0, 8.0, 11.0])

    actual = solve_svd(matrix, observations)
    expected = np.linalg.lstsq(matrix, observations, rcond=None)[0]

    assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_solve_svd_residual_is_orthogonal_to_column_space() -> None:
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ])
    observations = np.array([2.1, 4.9, 8.2, 10.8])

    solution = solve_svd(matrix, observations)
    residual = matrix @ solution - observations

    assert_allclose(
        matrix.T @ residual,
        np.zeros(matrix.shape[1]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_solve_svd_supports_rank_deficient_systems() -> None:
    matrix = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])
    observations = np.array([2.0, 4.0, 6.0])

    actual = solve_svd(matrix, observations)
    expected = np.linalg.lstsq(matrix, observations, rcond=None)[0]

    assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def test_solve_svd_uses_numerical_rank_threshold() -> None:
    matrix = np.array([
        [1.0, 0.0],
        [0.0, 1e-16],
        [0.0, 0.0],
    ])
    observations = np.array([1.0, 1.0, 0.0])

    actual = solve_svd(matrix, observations)

    assert_allclose(
        actual,
        np.array([1.0, 0.0]),
        rtol=RTOL,
        atol=ATOL,
    )

def test_solve_svd_returns_minimum_norm_solution() -> None:
    matrix = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])
    observations = np.array([2.0, 4.0, 6.0])

    actual = solve_svd(matrix, observations)

    assert_allclose(
        actual,
        np.array([1.0, 1.0]),
        rtol=RTOL,
        atol=ATOL,
    )


def test_solve_svd_rejects_negative_rcond() -> None:
    matrix = np.eye(2)
    observations = np.array([1.0, 2.0])

    with np.testing.assert_raises(ValueError):
        solve_svd(matrix, observations, rcond=-1.0)


def test_solve_svd_rejects_non_finite_rcond() -> None:
    matrix = np.eye(2)
    observations = np.array([1.0, 2.0])

    with np.testing.assert_raises(ValueError):
        solve_svd(matrix, observations, rcond=np.nan)

    with np.testing.assert_raises(ValueError):
        solve_svd(matrix, observations, rcond=np.inf)
