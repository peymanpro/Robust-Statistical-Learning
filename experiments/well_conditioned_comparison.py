"""Well-conditioned solver comparison experiment."""

import numpy as np

from robust_statistical_learning.core.metrics import condition_number
from robust_statistical_learning.learning.normal_equations import (
    solve_normal_equations,
)
from robust_statistical_learning.learning.qr import solve_qr
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

    singular_values = np.geomspace(
        1.0,
        1.0 / target_kappa,
        columns,
    )

    return (
        q_left[:, :columns]
        @ np.diag(singular_values)
        @ q_right.T
    )


def main() -> None:
    """Run the well-conditioned solver comparison."""
    rng = np.random.default_rng(42)

    rows, columns = 40, 5
    x_true = np.arange(1.0, columns + 1.0)
    x_true_norm = np.linalg.norm(x_true)

    targets = (1.0, 1e2, 1e4, 1e6, 1e8)
    cases = [
        (target_kappa, make_matrix(rng, rows, columns, target_kappa))
        for target_kappa in targets
    ]

    print("Well-conditioned solver comparison")
    print(f"dimensions: m={rows}, n={columns}")
    print("seed: 42")
    print()

    print(
        f"{'target':>8} {'actual':>10} "
        f"{'NE_fwd':>10} {'QR_fwd':>10} {'SVD_fwd':>10} "
        f"{'NE_res':>10} {'QR_res':>10} {'SVD_res':>10}"
    )

    for target_kappa, matrix in cases:
        observations = matrix @ x_true
        actual_kappa = condition_number(matrix)

        x_ne = solve_normal_equations(matrix, observations)
        x_qr = solve_qr(matrix, observations)
        x_svd = solve_svd(matrix, observations)

        ne_fwd = np.linalg.norm(x_ne - x_true) / x_true_norm
        qr_fwd = np.linalg.norm(x_qr - x_true) / x_true_norm
        svd_fwd = np.linalg.norm(x_svd - x_true) / x_true_norm

        ne_res = np.linalg.norm(matrix @ x_ne - observations)
        qr_res = np.linalg.norm(matrix @ x_qr - observations)
        svd_res = np.linalg.norm(matrix @ x_svd - observations)

        print(
            f"{target_kappa:>8.1e} {actual_kappa:>10.3e} "
            f"{ne_fwd:>10.3e} {qr_fwd:>10.3e} {svd_fwd:>10.3e} "
            f"{ne_res:>10.3e} {qr_res:>10.3e} {svd_res:>10.3e}"
        )

    print()
    print("Cross-solver solution differences")
    print(
        f"{'target':>8} {'||NE-QR||':>12} "
        f"{'||QR-SVD||':>12} {'||NE-SVD||':>12}"
    )

    for target_kappa, matrix in cases:
        observations = matrix @ x_true

        x_ne = solve_normal_equations(matrix, observations)
        x_qr = solve_qr(matrix, observations)
        x_svd = solve_svd(matrix, observations)

        d_ne_qr = np.linalg.norm(x_ne - x_qr)
        d_qr_svd = np.linalg.norm(x_qr - x_svd)
        d_ne_svd = np.linalg.norm(x_ne - x_svd)

        print(
            f"{target_kappa:>8.1e} {d_ne_qr:>12.3e} "
            f"{d_qr_svd:>12.3e} {d_ne_svd:>12.3e}"
        )


if __name__ == "__main__":
    main()