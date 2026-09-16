"""Least-squares mathematical model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from robust_statistical_learning.core import residuals as core_residuals

FloatArray = NDArray[np.float64]


def residual(
    a: FloatArray,
    x: FloatArray,
    b: FloatArray,
) -> FloatArray:
    """Return the least-squares residual r = Ax - b."""
    return core_residuals.residual(a, x, b)


def residual_norm(
    a: FloatArray,
    x: FloatArray,
    b: FloatArray,
) -> float:
    """Return the Euclidean norm ||Ax - b||_2."""
    return core_residuals.residual_norm(a, x, b)


def objective(
    a: FloatArray,
    x: FloatArray,
    b: FloatArray,
) -> float:
    """Return the least-squares objective ||Ax - b||_2^2."""
    r = residual(a, x, b)
    return float(r @ r)