import numpy as np
import pytest
from numpy.testing import assert_allclose

from robust_statistical_learning.learning.least_squares import (
    objective,
    residual,
    residual_norm,
)


def test_residual_matches_ax_minus_b() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.array([2.0, -1.0])
    b = np.array([4.0, 5.0])

    assert_allclose(residual(a, x, b), np.array([-4.0, -3.0]))


def test_residual_norm_is_euclidean_norm() -> None:
    a = np.eye(2)
    x = np.array([3.0, 4.0])
    b = np.zeros(2)

    assert residual_norm(a, x, b) == pytest.approx(5.0)


def test_objective_is_squared_residual_norm() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    x = np.array([2.0, -1.0])
    b = np.array([4.0, 5.0])

    assert objective(a, x, b) == pytest.approx(
        residual_norm(a, x, b) ** 2
    )


def test_exact_solution_has_zero_residual() -> None:
    a = np.array([[2.0, 0.0], [0.0, 3.0]])
    x = np.array([4.0, 5.0])
    b = a @ x

    assert_allclose(residual(a, x, b), np.zeros(2))
    assert residual_norm(a, x, b) == pytest.approx(0.0)
    assert objective(a, x, b) == pytest.approx(0.0)
