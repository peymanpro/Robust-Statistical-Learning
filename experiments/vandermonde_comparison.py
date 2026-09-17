"""Vandermonde matrix solver comparison experiment."""

from collections.abc import Callable

import numpy as np

from robust_statistical_learning.core.metrics import condition_number
from robust_statistical_learning.learning.normal_equations import (
    solve_normal_equations,
)
from robust_statistical_learning.learning.qr import solve_qr
from robust_statistical_learning.learning.svd import solve_svd

Solver = Callable[[np.ndarray, np.ndarray], np.ndarray]


def vandermonde_matrix(rows: int, columns: int) -> np.ndarray:
    """Return a Vandermonde matrix with uniform nodes in (0, 1)."""
    nodes = np.arange(1, rows + 1, dtype=np.float64) / (rows + 1.0)
    powers = np.arange(columns, dtype=np.float64)
    return nodes[:, None] ** powers[None, :]


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


def reference_lstsq(
    matrix: np.ndarray,
    observations: np.ndarray,
) -> tuple[np.ndarray | None, str | None]:
    """Solve with numpy.linalg.lstsq, returning (solution, error_name)."""
    try:
        return np.linalg.lstsq(matrix, observations, rcond=None)[0], None
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
    """Run the Vandermonde matrix solver comparison."""
    rows = 40
    columns_list = (4, 6, 8, 10, 12, 15)
    cases = [
        (n, vandermonde_matrix(rows, n)) for n in columns_list
    ]

    print("Vandermonde matrix solver comparison")
    print(f"rows = {rows}, uniform nodes in (0, 1)")
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
    print("Cross-check vs numpy.linalg.lstsq")
    print(
        f"{'n':>4} {'ref_fwd':>10} "
        f"{'NE_vs_ref':>12} {'QR_vs_ref':>12} {'SVD_vs_ref':>12}"
    )

    for n, matrix in cases:
        x_true = np.ones(n)
        observations = matrix @ x_true

        x_ref, ref_err = reference_lstsq(matrix, observations)
        if x_ref is None:
            tag = ref_err or "error"
            print(
                f"{n:>4} {tag:>10} "
                f"{'n/a':>12} {'n/a':>12} {'n/a':>12}"
            )
            continue

        ref_fwd = np.linalg.norm(x_ref - x_true) / np.linalg.norm(x_true)

        ne_diff = diff_to_reference(
            solve_normal_equations, matrix, observations, x_ref
        )
        qr_diff = diff_to_reference(
            solve_qr, matrix, observations, x_ref
        )
        svd_diff = diff_to_reference(
            solve_svd, matrix, observations, x_ref
        )

        print(
            f"{n:>4} {ref_fwd:>10.3e} "
            f"{ne_diff:>12} {qr_diff:>12} {svd_diff:>12}"
        )


if __name__ == "__main__":
    main()