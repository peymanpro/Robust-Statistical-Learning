"""Cross-solver forward, backward, and residual diagnostics."""

from collections.abc import Callable

import numpy as np

from robust_statistical_learning.core.errors import (
    normwise_backward_error,
    relative_forward_error,
    residual_orthogonality_error,
)
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
    """Create a full-column-rank matrix with controlled singular values."""
    q_left, _ = np.linalg.qr(rng.normal(size=(rows, columns)))
    q_right, _ = np.linalg.qr(rng.normal(size=(columns, columns)))

    singular_values = np.geomspace(1.0, 1.0 / target_kappa, columns)

    return q_left @ np.diag(singular_values) @ q_right.T


def solvers() -> list[tuple[str, Solver]]:
    """Return solvers in a stable reporting order."""
    return [
        ("NE", solve_normal_equations),
        ("QR", solve_qr),
        ("SVD", solve_svd),
    ]


def analyze(
    matrix: np.ndarray,
    observations: np.ndarray,
    reference: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Return comparable diagnostics for all solvers."""
    report: dict[str, dict[str, float]] = {}

    for name, solver in solvers():
        solution = solver(matrix, observations)
        report[name] = {
            "forward_error": relative_forward_error(solution, reference),
            "backward_error": normwise_backward_error(
                matrix,
                solution,
                observations,
            ),
            "orthogonality_error": residual_orthogonality_error(
                matrix,
                solution,
                observations,
            ),
            "residual_norm": float(
                np.linalg.norm(matrix @ solution - observations, ord=2)
            ),
        }

    return report


def main() -> None:
    """Run a deterministic conditioning/error diagnostic experiment."""
    rng = np.random.default_rng(42)
    matrix = make_matrix(rng, 40, 5, 1e10)
    reference = np.arange(1.0, 6.0)
    observations = matrix @ reference

    print("Least-squares error analysis")
    print(f"condition number: {condition_number(matrix):.3e}")
    print(
        f"{'solver':>8} {'forward':>14} {'backward':>14} "
        f"{'orthogonality':>16} {'residual':>14}"
    )

    for name, values in analyze(matrix, observations, reference).items():
        print(
            f"{name:>8} "
            f"{values['forward_error']:>14.3e} "
            f"{values['backward_error']:>14.3e} "
            f"{values['orthogonality_error']:>16.3e} "
            f"{values['residual_norm']:>14.3e}"
        )


if __name__ == "__main__":
    main()
