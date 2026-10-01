import numpy as np
import pytest
from numpy.testing import assert_allclose

from robust_statistical_learning.learning.pca import (
    PCAResult,
    fit_pca,
    reconstruction_error,
)


def test_fit_pca_returns_expected_shapes() -> None:
    matrix = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ]
    )

    result = fit_pca(matrix, n_components=2)

    assert isinstance(result, PCAResult)
    assert result.mean.shape == (3,)
    assert result.components.shape == (2, 3)
    assert result.singular_values.shape == (2,)
    assert result.explained_variance.shape == (2,)
    assert result.explained_variance_ratio.shape == (2,)


def test_pca_components_are_orthonormal() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(20, 4))

    result = fit_pca(matrix)

    assert_allclose(
        result.components @ result.components.T,
        np.eye(4),
        rtol=1e-12,
        atol=1e-12,
    )


def test_pca_transform_is_centered_projection() -> None:
    matrix = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [4.0, 8.0],
        ]
    )

    result = fit_pca(matrix, n_components=1)
    transformed = result.transform(matrix)

    assert_allclose(transformed, (matrix - result.mean) @ result.components.T)


def test_pca_full_fit_reconstructs_data() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(12, 3))

    result = fit_pca(matrix)
    reconstructed = result.inverse_transform(result.transform(matrix))

    assert_allclose(reconstructed, matrix, rtol=1e-12, atol=1e-12)


def test_pca_explained_variance_ratio_sums_to_one() -> None:
    matrix = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 3.0],
            [2.0, 1.0, 1.0],
            [3.0, 2.0, 4.0],
            [4.0, 3.0, 2.0],
        ]
    )

    result = fit_pca(matrix)

    assert result.explained_variance_ratio.sum() == pytest.approx(1.0)


def test_pca_truncated_reconstruction_has_nonzero_error() -> None:
    rng = np.random.default_rng(42)
    matrix = rng.normal(size=(30, 5))

    result = fit_pca(matrix, n_components=2)
    reconstructed = result.inverse_transform(result.transform(matrix))

    assert reconstruction_error(matrix, reconstructed) > 0.0


def test_pca_rejects_single_observation() -> None:
    with pytest.raises(ValueError, match="at least two"):
        fit_pca(np.ones((1, 3)))


def test_pca_rejects_invalid_component_count() -> None:
    matrix = np.ones((5, 3))

    with pytest.raises(ValueError, match="between 1"):
        fit_pca(matrix, n_components=4)

    with pytest.raises(ValueError, match="integer"):
        fit_pca(matrix, n_components=1.5)  # type: ignore[arg-type]


def test_pca_transform_rejects_wrong_feature_count() -> None:
    result = fit_pca(np.ones((5, 3)))

    with pytest.raises(ValueError, match="feature dimension"):
        result.transform(np.ones((2, 2)))


def test_pca_reconstruction_error_validates_shapes() -> None:
    with pytest.raises(ValueError, match="shapes"):
        reconstruction_error(np.ones((3, 2)), np.ones((3, 3)))
