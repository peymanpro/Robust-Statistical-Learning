"""SVD numerical-rank and least-squares reference experiment."""

import numpy as np

from robust_statistical_learning.core.metrics import condition_number
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
    """Run the SVD numerical-rank experiment."""
    rng = np.random.default_rng(42)

    rows, columns = 40, 5
    x_true = np.arange(1.0, columns + 1.0)
    eps = np.finfo(np.float64).eps

    print("SVD numerical-rank and reference experiment")
    print(f"dimensions: m={rows}, n={columns}")
    print("seed: 42")
    print()

    for target_kappa in (1e2, 1e6, 1e10, 1e14, 1e16):
        matrix = make_matrix(rng, rows, columns, target_kappa)
        observations = matrix @ x_true

        singular_values = np.linalg.svd(
            matrix,
            compute_uv=False,
        )
        threshold = (
            singular_values[0]
            * max(matrix.shape)
            * eps
        )
        numerical_rank = int(
            np.count_nonzero(singular_values > threshold)
        )

        actual_kappa = condition_number(matrix)

        computed = solve_svd(matrix, observations)
        reference = np.linalg.lstsq(
            matrix,
            observations,
            rcond=None,
        )[0]

        residual = np.linalg.norm(
            matrix @ computed - observations
        )
        reference_error = (
            np.linalg.norm(computed - reference)
            / max(np.linalg.norm(reference), 1.0)
        )

        solution_norm = np.linalg.norm(computed)
        reference_norm = np.linalg.norm(reference)

        print(
            f"target_kappa={target_kappa:.1e} "
            f"actual_kappa={actual_kappa:.3e} "
            f"rank={numerical_rank} "
            f"threshold={threshold:.3e} "
            f"reference_error={reference_error:.3e} "
            f"residual={residual:.3e} "
            f"solution_norm={solution_norm:.3e} "
            f"reference_norm={reference_norm:.3e}"
        )

    print()
    print("Rank-deficient case")

    vector = np.array([1.0, 2.0, 3.0, 4.0])
    matrix = np.column_stack((vector, vector))
    observations = matrix @ np.array([1.0, 1.0])

    computed = solve_svd(matrix, observations)
    reference = np.linalg.lstsq(
        matrix,
        observations,
        rcond=None,
    )[0]

    residual = np.linalg.norm(
        matrix @ computed - observations
    )

    print(f"computed={computed}")
    print(f"reference={reference}")
    print(f"reference_error={np.linalg.norm(computed - reference):.3e}")
    print(f"residual={residual:.3e}")
    print(f"solution_norm={np.linalg.norm(computed):.3e}")


if __name__ == "__main__":
    main()
