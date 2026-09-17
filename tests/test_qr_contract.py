import numpy as np
from numpy.testing import assert_allclose

from robust_statistical_learning.core.tolerances import ATOL, RTOL
from robust_statistical_learning.learning.normal_equations import (
    solve_normal_equations,
)
from robust_statistical_learning.learning.qr import solve_qr


def test_qr_reference_shapes() -> None:
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
    ])

    q, r = np.linalg.qr(matrix, mode="reduced")

    assert q.shape == (3, 2)
    assert r.shape == (2, 2)


def test_qr_reconstructs_matrix() -> None:
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    q, r = np.linalg.qr(matrix, mode="reduced")

    assert_allclose(q @ r, matrix)


def test_qr_columns_are_orthonormal() -> None:
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    q, _ = np.linalg.qr(matrix, mode="reduced")

    assert_allclose(q.T @ q, np.eye(2), rtol=RTOL, atol=ATOL)


def test_qr_transformed_rhs_is_consistent() -> None:
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
    ])
    expected = np.array([2.0, 3.0])
    observations = matrix @ expected

    q, r = np.linalg.qr(matrix, mode="reduced")
    solution = np.linalg.solve(r, q.T @ observations)

    assert_allclose(solution, expected)


def test_qr_solution_matches_normal_equations() -> None:
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ])
    observations = np.array([2.0, 5.0, 8.0, 11.0])

    q, r = np.linalg.qr(matrix, mode="reduced")
    qr_solution = np.linalg.solve(r, q.T @ observations)
    normal_solution = solve_normal_equations(matrix, observations)

    assert_allclose(qr_solution, normal_solution)


def test_solve_qr_matches_numpy_lstsq_reference() -> None:
    matrix = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
    observations = np.array([2.0, 5.0, 8.0, 11.0])
    actual = solve_qr(matrix, observations)
    expected = np.linalg.lstsq(matrix, observations, rcond=None)[0]
    assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)
