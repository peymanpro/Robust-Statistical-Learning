"""Perturbation sensitivity study for least-squares solvers."""

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


def relative_error(
    computed: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Return the relative L2 error."""
    return float(
        np.linalg.norm(computed - reference) / np.linalg.norm(reference)
    )


def measure_all(
    matrix: np.ndarray,
    observations: np.ndarray,
    x_true: np.ndarray,
) -> dict[str, float]:
    """Return relative forward error for each solver."""
    return {
        "NE": relative_error(
            solve_normal_equations(matrix, observations), x_true
        ),
        "QR": relative_error(
            solve_qr(matrix, observations), x_true
        ),
        "SVD": relative_error(
            solve_svd(matrix, observations), x_true
        ),
    }


def format_cell(value: float) -> str:
    """Format a positive float for the table."""
    return f"{value:.3e}"


def main() -> None:
    """Run the perturbation sensitivity study."""
    rng = np.random.default_rng(42)

    rows, columns = 40, 5
    target_kappa = 1e4

    matrix = make_matrix(rng, rows, columns, target_kappa)
    x_true = np.arange(1.0, columns + 1.0)
    observations = matrix @ x_true

    actual_kappa = condition_number(matrix)
    matrix_norm_f = np.linalg.norm(matrix, "fro")
    observations_norm = np.linalg.norm(observations)

    magnitudes = (1e-14, 1e-12, 1e-10, 1e-8, 1e-6)

    print("Perturbation sensitivity study")
    print(f"dimensions: m={rows}, n={columns}")
    print(f"target kappa: {target_kappa:.1e}")
    print(f"actual kappa: {actual_kappa:.3e}")
    print("seed: 42")
    print()

    # ---------------------------------------------------------
    # Part A: perturb observations b
    # ---------------------------------------------------------

    print("Part A - perturb b")
    print(
        f"{'magnitude':>10} {'bound':>10} "
        f"{'NE':>10} {'QR':>10} {'SVD':>10}"
    )

    for magnitude in magnitudes:
        direction = rng.normal(size=rows)
        direction = direction / np.linalg.norm(direction)
        delta_b = direction * (observations_norm * magnitude)

        perturbed = observations + delta_b
        errors = measure_all(matrix, perturbed, x_true)
        bound = actual_kappa * magnitude

        print(
            f"{magnitude:>10.1e} {bound:>10.3e} "
            f"{format_cell(errors['NE']):>10} "
            f"{format_cell(errors['QR']):>10} "
            f"{format_cell(errors['SVD']):>10}"
        )

    print()

    # ---------------------------------------------------------
    # Part B: perturb design matrix A
    # ---------------------------------------------------------

    print("Part B - perturb A")
    print(
        f"{'magnitude':>10} {'bound':>10} "
        f"{'NE':>10} {'QR':>10} {'SVD':>10}"
    )

    for magnitude in magnitudes:
        direction = rng.normal(size=(rows, columns))
        direction = direction / np.linalg.norm(direction, "fro")
        delta_a = direction * (matrix_norm_f * magnitude)

        perturbed = matrix + delta_a
        errors = measure_all(perturbed, observations, x_true)
        bound = actual_kappa * magnitude

        print(
            f"{magnitude:>10.1e} {bound:>10.3e} "
            f"{format_cell(errors['NE']):>10} "
            f"{format_cell(errors['QR']):>10} "
            f"{format_cell(errors['SVD']):>10}"
        )


if __name__ == "__main__":
    main()