"""Learning-level mathematical models."""

from .pca import PCAResult, fit_pca
from .ridge import select_ridge_alpha_gcv, solve_ridge, solve_tikhonov

__all__ = [
    "PCAResult",
    "fit_pca",
    "select_ridge_alpha_gcv",
    "solve_ridge",
    "solve_tikhonov",
]
