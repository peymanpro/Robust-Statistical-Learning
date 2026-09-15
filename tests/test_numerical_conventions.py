import numpy as np
from numpy.testing import assert_allclose

from robust_statistical_learning.core.tolerances import ATOL, RTOL
from robust_statistical_learning.core.types import Matrix, Vector


def test_matrix_convention_is_two_dimensional_float64() -> None:
    matrix: Matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    assert matrix.ndim == 2
    assert matrix.dtype == np.dtype(np.float64)


def test_vector_convention_is_one_dimensional_float64() -> None:
    vector: Vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    assert vector.ndim == 1
    assert vector.dtype == np.dtype(np.float64)


def test_numerical_tolerances_are_positive() -> None:
    assert ATOL > 0.0
    assert RTOL > 0.0


def test_residual_objective_identity() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.array([2.0, -1.0])
    b = np.array([4.0, 5.0])

    residual_vector = a @ x - b
    objective_value = float(residual_vector @ residual_vector)
    expected = float(np.linalg.norm(residual_vector, ord=2) ** 2)

    assert_allclose(objective_value, expected, rtol=RTOL, atol=ATOL)