"""Learning-level mathematical models."""

from .pca import PCAResult, fit_pca
from .ridge import select_ridge_alpha_gcv, solve_ridge, solve_tikhonov
from .robust import HuberResult, huber_loss, solve_huber

__all__ = [
    "HuberResult",
    "PCAResult",
    "fit_pca",
    "huber_loss",
    "select_ridge_alpha_gcv",
    "solve_huber",
    "solve_ridge",
    "solve_tikhonov",
]
