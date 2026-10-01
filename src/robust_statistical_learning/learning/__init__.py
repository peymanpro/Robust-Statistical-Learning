"""Learning-level mathematical models."""

from .least_squares import objective, residual, residual_norm
from .normal_equations import solve_normal_equations
from .pca import PCAResult, fit_pca, reconstruction_error
from .qr import solve_qr
from .ridge import select_ridge_alpha_gcv, solve_ridge, solve_tikhonov
from .robust import HuberResult, huber_loss, solve_huber
from .svd import compute_svd, solve_svd

__all__ = [
    "HuberResult",
    "PCAResult",
    "compute_svd",
    "fit_pca",
    "huber_loss",
    "objective",
    "reconstruction_error",
    "residual",
    "residual_norm",
    "select_ridge_alpha_gcv",
    "solve_huber",
    "solve_normal_equations",
    "solve_qr",
    "solve_ridge",
    "solve_svd",
    "solve_tikhonov",
]
