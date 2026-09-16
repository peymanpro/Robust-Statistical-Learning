"""Normal-equations conditioning experiment."""

import numpy as np

from robust_statistical_learning.core.metrics import condition_number
from robust_statistical_learning.learning.normal_equations import (
    solve_normal_equations,
)


def main() -> None:
    """Run a reproducible conditioning experiment."""
    rng = np.random.default_rng(42)

    m, n = 40, 5
    x_true = np.arange(1.0, n + 1.0)

    print("Normal Equations conditioning experiment")
    print(f"dimensions: m={m}, n={n}")
    print("seed: 42")
    print()

    for target_kappa in (1e2, 1e6, 1e10, 1e12, 1e14):
        q_left, _ = np.linalg.qr(rng.normal(size=(m, n)))
        q_right, _ = np.linalg.qr(rng.normal(size=(n, n)))

        singular_values = np.geomspace(1.0, 1.0 / target_kappa, n)

        matrix = (
            q_left[:, :n]
            @ np.diag(singular_values)
            @ q_right.T
        )

        observations = matrix @ x_true

        actual_kappa = condition_number(matrix)
        normal_kappa = np.linalg.cond(matrix.T @ matrix, p=2)

        try:
            computed = solve_normal_equations(matrix, observations)

            forward_error = (
                np.linalg.norm(computed - x_true)
                / np.linalg.norm(x_true)
            )

            residual = np.linalg.norm(
                matrix @ computed - observations
            )

            print(
                f"target_kappa={target_kappa:.1e} "
                f"actual_kappa={actual_kappa:.3e} "
                f"kappa_AtA={normal_kappa:.3e} "
                f"forward_error={forward_error:.3e} "
                f"residual={residual:.3e}"
            )
        except ValueError as exc:
            print(
                f"target_kappa={target_kappa:.1e} "
                f"actual_kappa={actual_kappa:.3e} "
                f"kappa_AtA={normal_kappa:.3e} "
                f"status=REJECTED "
                f"reason={exc}"
            )


if __name__ == "__main__":
    main()