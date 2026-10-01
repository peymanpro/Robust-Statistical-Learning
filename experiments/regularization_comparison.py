"""Controlled experiment for Ridge under noisy, ill-conditioned data."""

import numpy as np

from robust_statistical_learning.core.errors import relative_forward_error
from robust_statistical_learning.learning.ridge import (
    select_ridge_alpha_gcv,
    solve_ridge,
)
from robust_statistical_learning.learning.svd import solve_svd


def make_matrix(
    rng: np.random.Generator,
    rows: int,
    columns: int,
    target_kappa: float,
) -> np.ndarray:
    """Create a matrix with controlled singular values."""
    q_left, _ = np.linalg.qr(rng.normal(size=(rows, columns)))
    q_right, _ = np.linalg.qr(rng.normal(size=(columns, columns)))
    singular_values = np.geomspace(1.0, 1.0 / target_kappa, columns)
    return q_left @ np.diag(singular_values) @ q_right.T


def main() -> None:
    """Compare unregularized SVD, GCV-selected Ridge, and fixed Ridge."""
    rng = np.random.default_rng(42)
    matrix = make_matrix(rng, 60, 6, 1e8)
    true_solution = np.arange(1.0, 7.0)
    clean = matrix @ true_solution

    candidate_alphas = np.logspace(-12, 0, 25)
    noise_level = 1e-3
    trials = 20

    errors: dict[str, list[float]] = {
        "SVD": [],
        "Ridge-GCV": [],
        "Ridge-1e-4": [],
    }

    selected: list[float] = []

    for trial in range(trials):
        trial_rng = np.random.default_rng(1000 + trial)
        noise = trial_rng.normal(size=matrix.shape[0])
        noise = noise / np.linalg.norm(noise)
        observations = clean + noise_level * np.linalg.norm(clean) * noise

        errors["SVD"].append(
            relative_forward_error(
                solve_svd(matrix, observations),
                true_solution,
            )
        )

        alpha = select_ridge_alpha_gcv(matrix, observations, candidate_alphas)
        selected.append(alpha)
        errors["Ridge-GCV"].append(
            relative_forward_error(
                solve_ridge(matrix, observations, alpha),
                true_solution,
            )
        )
        errors["Ridge-1e-4"].append(
            relative_forward_error(
                solve_ridge(matrix, observations, 1e-4),
                true_solution,
            )
        )

    print("Ridge regularization comparison")
    print("dimensions: m=60, n=6")
    print("target condition number: 1e8")
    print("noise level: 1e-3 relative to ||b||")
    print("trials: 20")
    print()
    print(f"{'method':>14} {'mean_forward':>16} {'std_forward':>16}")

    for name, values in errors.items():
        print(
            f"{name:>14} "
            f"{np.mean(values):>16.3e} "
            f"{np.std(values):>16.3e}"
        )

    print()
    print(f"GCV selected alpha range: {min(selected):.3e} .. {max(selected):.3e}")
