"""Hilbert matrix solver comparison experiment."""

from collections.abc import Callable

import numpy as np

from robust_statistical_learning.core.metrics import condition_number
from robust_statistical_learning.learning.normal_equations import (
    solve_normal_equations,
)
from robust_statistical_learning.learning.qr import solve_qr
from robust_statistical_learning.learning.svd import solve_svd

Solver = Callable[[np.ndarray, np.ndarray], np.ndarray]


def hilbert_matrix(n: int) -> np.ndarray:
    """Return the n-by-n Hilbert matrix."""
    indices = np.arange(1, n + 1, dtype=np.float64)
    return 1.0 / (indices[:, None] + indices[None, :] - 1.0)


def numerical_rank(matrix: np.ndarray) -> int:
    """Compute numerical rank with the sigma_max * max(m, n) * eps rule."""
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


def lu_reference(
    matrix: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    """Solve with numpy.linalg.solve, returning (solution, error_name)."""
    try:
        return np.linalg.solve(matrix, observations), None
    except np.linalg.LinAlgError as exc:
        return None, type(exc).__name__


def measure_solver(
    solver: Solver,
    matrix: np.ndarray,
    observations: np.ndarray,
    x_true: np.ndarray,
) -> tuple[str, str]:
    """Return (forward_error_str, residual_str) or error names."""
    x, err = safe_solve(solver, matrix, observations)
    if x is None:
        tag = err or "error"
        return tag, tag
    fwd = np.linalg.norm(x - x_true) / np.linalg.norm(x_true)
    res = np.linalg.norm(matrix @ x - observations)
    return f"{fwd:.3e}", f"{res:.3e}"


def diff_to_reference(
    solver: Solver,
    matrix: np.ndarray,
    observations: np.ndarray,
    reference: np.ndarray,
) -> str:
    """Return ||solver(matrix, observations) - reference|| or 'n/a'."""
    x, _ = safe_solve(solver, matrix, observations)
    if x is None:
        return "n/a"
    return f"{np.linalg.norm(x - reference):.3e}"


def main() -> None:
    """Run the Hilbert matrix solver comparison."""
    sizes = (4, 6, 8, 10, 12)
    cases = [(n, hilbert_matrix(n)) for n in sizes]

    print("Hilbert matrix solver comparison")
    print("x_true = ones(n)")
    print()

    print(
        f"{'n':>4} {'kappa':>10} {'rank':>4} "
        f"{'NE_fwd':>10} {'QR_fwd':>10} {'SVD_fwd':>10} "
        f"{'NE_res':>10} {'QR_res':>10} {'SVD_res':>10}"
    )

    for n, matrix in cases:
        x_true = np.ones(n)
        observations = matrix @ x_true
        actual_kappa = condition_number(matrix)
        rank = numerical_rank(matrix)

        ne_fwd, ne_res = measure_solver(
            solve_normal_equations, matrix, observations, x_true
        )
        qr_fwd, qr_res = measure_solver(
            solve_qr, matrix, observations, x_true
        )
        svd_fwd, svd_res = measure_solver(
            solve_svd, matrix, observations, x_true
        )

        kappa_str = (
            f"{actual_kappa:.3e}" if np.isfinite(actual_kappa) else "inf"
        )

        print(
            f"{n:>4} {kappa_str:>10} {rank:>4} "
            f"{ne_fwd:>10} {qr_fwd:>10} {svd_fwd:>10} "
            f"{ne_res:>10} {qr_res:>10} {svd_res:>10}"
        )

    print()
    print("Cross-check vs numpy.linalg.solve (LU)")
    print(
        f"{'n':>4} {'LU_fwd':>10} "
        f"{'NE_vs_LU':>12} {'QR_vs_LU':>12} {'SVD_vs_LU':>12}"
    )

    for n, matrix in cases:
        x_true = np.ones(n)
        observations = matrix @ x_true

        x_lu, lu_err = lu_reference(matrix, observations)
        if x_lu is None:
            tag = lu_err or "error"
            print(
                f"{n:>4} {tag:>10} "
                f"{'n/a':>12} {'n/a':>12} {'n/a':>12}"
            )
            continue

        lu_fwd = np.linalg.norm(x_lu - x_true) / np.linalg.norm(x_true)

        ne_diff = diff_to_reference(
            solve_normal_equations, matrix, observations, x_lu
        )
        qr_diff = diff_to_reference(
            solve_qr, matrix, observations, x_lu
        )
        svd_diff = diff_to_reference(
            solve_svd, matrix, observations, x_lu
        )

        print(
            f"{n:>4} {lu_fwd:>10.3e} "
            f"{ne_diff:>12} {qr_diff:>12} {svd_diff:>12}"
        )


if __name__ == "__main__":
    main()