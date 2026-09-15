"""Least-squares mathematical model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def residual(
    a: FloatArray,
    x: FloatArray,
    b: FloatArray,
) -> FloatArray:
    """Return the least-squares residual r = Ax - b."""
    return np.asarray(a @ x - b, dtype=np.float64)


def residual_norm(
    a: FloatArray,
    x: FloatArray,
    b: FloatArray,
) -> float:
    """Return the Euclidean norm ||Ax - b||_2."""
    return float(np.linalg.norm(residual(a, x, b), ord=2))


def objective(
    a: FloatArray,
    x: FloatArray,
    b: FloatArray,
) -> float:
    """Return the least-squares objective ||Ax - b||_2^2."""
    r = residual(a, x, b)
    return float(r @ r)
