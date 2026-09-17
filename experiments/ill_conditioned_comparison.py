"""Ill-conditioned solver comparison experiment."""

from collections.abc import Callable

import numpy as np

from robust_statistical_learning.core.metrics import condition_number
from robust_statistical_learning.learning.normal_equations import (
    solve_normal_equations,
)
from robust_statistical_learning.learning.qr import solve_qr
from robust_statistical_learning.learning.svd import solve_svd

Solver = Callable[[np.ndarray, np.ndarray], np.ndarray]


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


def numerical_rank(matrix: np.ndarray) -> int:
    """Compute the numerical rank using the sigma_max * max(m, n) * eps rule."""
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    threshold = (
        singular_values[0]
        * max(matrix.shape)
        * np.finfo(matrix.dtype).eps
    )
    return int(np.count_nonzero(singular_values > threshold))


def safe_solve(
    solver: Solver,
    matrix: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    """Call a solver, returning (solution, error_name_or_None)."""
    try:
        return solver(matrix, observations), None
    except ValueError as exc:
        return None, type(exc).__name__


def main() -> None:
    """Run the ill-conditioned solver comparison."""
    rng = np.random.default_rng(42)

    rows, columns = 40, 5
    x_true = np.arange(1.0, columns + 1.0)
    x_true_norm = np.linalg.norm(x_true)

    targets = (1e10, 1e12, 1e14, 1e16)
    cases = [
        (target_kappa, make_matrix(rng, rows, columns, target_kappa))
        for target_kappa in targets
    ]

    print("Ill-conditioned solver comparison")
    print(f"dimensions: m={rows}, n={columns}")
    print("seed: 42")
    print()

    header = (
        f"{'target':>8} {'actual':>10} {'rank':>4} "
        f"{'NE_fwd':>10} {'QR_fwd':>10} {'SVD_fwd':>10} "
        f"{'NE_res':>10} {'QR_res':>10} {'SVD_res':>10}"
    )
    print(header)

    for target_kappa, matrix in cases:
        observations = matrix @ x_true
        actual_kappa = condition_number(matrix)
        rank = numerical_rank(matrix)

        def measure(
            solver: Solver,
            matrix: np.ndarray,
            observations: np.ndarray,
        ) -> tuple[str, str]:
            x, err = safe_solve(solver, matrix, observations)
            if err is not None or x is None:
                return err or "error", err or "error"
            fwd = np.linalg.norm(x - x_true) / x_true_norm
            res = np.linalg.norm(matrix @ x - observations)
            return f"{fwd:.3e}", f"{res:.3e}"

        ne_fwd, ne_res = measure(
            solve_normal_equations, matrix, observations
        )
        qr_fwd, qr_res = measure(solve_qr, matrix, observations)
        svd_fwd, svd_res = measure(solve_svd, matrix, observations)

        actual_str = (
            f"{actual_kappa:.3e}" if np.isfinite(actual_kappa) else "inf"
        )

        print(
            f"{target_kappa:>8.1e} {actual_str:>10} {rank:>4} "
            f"{ne_fwd:>10} {qr_fwd:>10} {svd_fwd:>10} "
            f"{ne_res:>10} {qr_res:>10} {svd_res:>10}"
        )

    print()
    print("Cross-solver solution differences")
    print(
        f"{'target':>8} {'||NE-QR||':>12} "
        f"{'||QR-SVD||':>12} {'||NE-SVD||':>12}"
    )

    for target_kappa, matrix in cases:
        observations = matrix @ x_true

        x_ne, _ = safe_solve(solve_normal_equations, matrix, observations)
        x_qr, _ = safe_solve(solve_qr, matrix, observations)
        x_svd, _ = safe_solve(solve_svd, matrix, observations)

        def diff(a: np.ndarray | None, b: np.ndarray | None) -> str:
            if a is None or b is None:
                return "n/a"
            return f"{np.linalg.norm(a - b):.3e}"

        print(
            f"{target_kappa:>8.1e} "
            f"{diff(x_ne, x_qr):>12} "
            f"{diff(x_qr, x_svd):>12} "
            f"{diff(x_ne, x_svd):>12}"
        )


if __name__ == "__main__":
    main()