"""SVD-based Principal Component Analysis."""

from dataclasses import dataclass

import numpy as np

from robust_statistical_learning.core.types import FloatArray
from robust_statistical_learning.core.validation import validate_matrix


@dataclass(frozen=True)
class PCAResult:
    """Fitted PCA model and statistics."""

    mean: FloatArray
    components: FloatArray
    singular_values: FloatArray
    explained_variance: FloatArray
    explained_variance_ratio: FloatArray

    @property
    def n_components(self) -> int:
        """Return the retained component count."""
        return int(self.components.shape[0])

    @property
    def n_features(self) -> int:
        """Return the original feature count."""
        return int(self.components.shape[1])

    def transform(self, matrix: FloatArray) -> FloatArray:
        """Project observations into component space."""
        validate_matrix(matrix)
        if matrix.shape[1] != self.n_features:
            raise ValueError("matrix columns must match fitted feature dimension")
        return (matrix - self.mean) @ self.components.T

    def inverse_transform(self, transformed: FloatArray) -> FloatArray:
        """Reconstruct observations from component coordinates."""
        validate_matrix(transformed)
        if transformed.shape[1] != self.n_components:
            raise ValueError("transformed columns must match component count")
        return transformed @ self.components + self.mean


def fit_pca(matrix: FloatArray, n_components: int | None = None) -> PCAResult:
    """Fit PCA by centering observations and applying reduced SVD."""
    validate_matrix(matrix)

    if matrix.shape[0] < 2:
        raise ValueError("PCA requires at least two observations")

    max_components = min(matrix.shape)
    if n_components is None:
        component_count = max_components
    elif not isinstance(n_components, int) or isinstance(n_components, bool):
        raise ValueError("n_components must be an integer")
    elif not 1 <= n_components <= max_components:
        raise ValueError("n_components must be between 1 and min(matrix.shape)")
    else:
        component_count = n_components

    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)

    explained_variance_all = singular_values**2 / (matrix.shape[0] - 1.0)
    total_variance = float(np.sum(explained_variance_all))
    if total_variance == 0.0:
        explained_ratio_all = np.zeros_like(explained_variance_all)
    else:
        explained_ratio_all = explained_variance_all / total_variance

    return PCAResult(
        mean=np.asarray(mean, dtype=np.float64),
        components=np.asarray(vt[:component_count], dtype=np.float64),
        singular_values=np.asarray(
            singular_values[:component_count],
            dtype=np.float64,
        ),
        explained_variance=np.asarray(
            explained_variance_all[:component_count],
            dtype=np.float64,
        ),
        explained_variance_ratio=np.asarray(
            explained_ratio_all[:component_count],
            dtype=np.float64,
        ),
    )


def reconstruction_error(original: FloatArray, reconstructed: FloatArray) -> float:
    """Return relative Frobenius reconstruction error."""
    validate_matrix(original)
    validate_matrix(reconstructed)

    if original.shape != reconstructed.shape:
        raise ValueError("original and reconstructed shapes must match")

    original_norm = np.linalg.norm(original, ord="fro")
    reconstructed_norm = np.linalg.norm(reconstructed, ord="fro")
    if original_norm == 0.0:
        return 0.0 if reconstructed_norm == 0.0 else float(np.inf)

    return float(
        np.linalg.norm(original - reconstructed, ord="fro") / original_norm
    )
