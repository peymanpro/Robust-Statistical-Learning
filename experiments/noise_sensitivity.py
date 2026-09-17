"""Noise sensitivity study for least-squares solvers."""

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


def solvers() -> list[tuple[str, Solver]]:
    """Return the three solvers in a stable order."""
    return [
        ("NE", solve_normal_equations),
        ("QR", solve_qr),
        ("SVD", solve_svd),
    ]


def main() -> None:
    """Run the noise sensitivity study."""
    base_rng = np.random.default_rng(42)

    rows, columns = 40, 5
    target_kappa = 1e4

    matrix = make_matrix(base_rng, rows, columns, target_kappa)
    x_true = np.arange(1.0, columns + 1.0)
    observations_clean = matrix @ x_true

    actual_kappa = condition_number(matrix)
    observations_norm = np.linalg.norm(observations_clean)

    noise_levels = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
    trials = 20

    print("Noise sensitivity study")
    print(f"dimensions: m={rows}, n={columns}")
    print(f"target kappa: {target_kappa:.1e}")
    print(f"actual kappa: {actual_kappa:.3e}")
    print(f"trials per level: {trials}")
    print("seed: 42")
    print()

    # ---------------------------------------------------------
    # Baseline (no noise)
    # ---------------------------------------------------------

    print("Baseline - no noise")
    print(f"{'solver':>8} {'fwd':>12}")
    for name, solver in solvers():
        solution = solver(matrix, observations_clean)
        fwd = relative_error(solution, x_true)
        print(f"{name:>8} {fwd:>12.3e}")

    print()

    # ---------------------------------------------------------
    # Noise sweep
    # ---------------------------------------------------------

    header = (
        f"{'noise':>10} {'bound':>10} "
        f"{'NE_mean':>10} {'NE_std':>10} "
        f"{'QR_mean':>10} {'QR_std':>10} "
        f"{'SVD_mean':>10} {'SVD_std':>10}"
    )
    print(header)

    for noise_level in noise_levels:
        # Per-trial forward errors, keyed by solver name
        errors: dict[str, list[float]] = {
            name: [] for name, _ in solvers()
        }

        # Deterministic per-level rng so each level is reproducible
        rng = np.random.default_rng(42 + int(round(noise_level * 1e9)))

        for _ in range(trials):
            direction = rng.normal(size=rows)
            direction = direction / np.linalg.norm(direction)
            delta_b = direction * (observations_norm * noise_level)
            observations = observations_clean + delta_b

            for name, solver in solvers():
                solution = solver(matrix, observations)
                errors[name].append(
                    relative_error(solution, x_true)
                )

        bound = actual_kappa * noise_level

        ne_mean = float(np.mean(errors["NE"]))
        ne_std = float(np.std(errors["NE"]))
        qr_mean = float(np.mean(errors["QR"]))
        qr_std = float(np.std(errors["QR"]))
        svd_mean = float(np.mean(errors["SVD"]))
        svd_std = float(np.std(errors["SVD"]))

        print(
            f"{noise_level:>10.1e} {bound:>10.3e} "
            f"{ne_mean:>10.3e} {ne_std:>10.3e} "
            f"{qr_mean:>10.3e} {qr_std:>10.3e} "
            f"{svd_mean:>10.3e} {svd_std:>10.3e}"
        )


if __name__ == "__main__":
    main()