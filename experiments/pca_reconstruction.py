"""Controlled PCA reconstruction experiment."""

import numpy as np

from robust_statistical_learning.learning.pca import fit_pca, reconstruction_error


def main() -> None:
    """Measure reconstruction as the retained component count changes."""
    rng = np.random.default_rng(42)

    latent = rng.normal(size=(200, 2))
    loadings = np.array(
        [
            [3.0, 0.1],
            [0.1, 2.0],
            [1.0, 0.2],
            [0.2, 0.5],
            [0.5, 1.0],
        ]
    )
    noise = 0.15 * rng.normal(size=(200, 5))
    matrix = latent @ loadings.T + noise

    print("PCA reconstruction experiment")
    print("samples: 200")
    print("features: 5")
    print()
    print(f"{'k':>4} {'cum_explained':>16} {'reconstruction':>16}")

    for components in range(1, 6):
        result = fit_pca(matrix, n_components=components)
        reconstructed = result.inverse_transform(result.transform(matrix))
        cumulative = float(np.sum(result.explained_variance_ratio))
        error = reconstruction_error(matrix, reconstructed)
        print(f"{components:>4} {cumulative:>16.3f} {error:>16.3e}")


if __name__ == "__main__":
    main()
