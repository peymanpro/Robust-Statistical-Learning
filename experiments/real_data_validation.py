"""Validate the project's learning layer on bundled sklearn datasets."""

import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LinearRegression

from robust_statistical_learning.learning.pca import fit_pca
from robust_statistical_learning.learning.svd import solve_svd


def main() -> None:
    """Compare selected outputs against sklearn references."""
    diabetes = load_diabetes()
    reference_regressor = LinearRegression(fit_intercept=False)
    reference_regressor.fit(diabetes.data, diabetes.target)

    computed_solution = solve_svd(diabetes.data, diabetes.target)
    coefficient_max_abs_error = float(
        np.max(np.abs(computed_solution - reference_regressor.coef_))
    )

    iris = load_iris()
    our_pca = fit_pca(iris.data)
    reference_pca = SklearnPCA(n_components=4)
    reference_pca.fit(iris.data)

    variance_ratio_max_abs_error = float(
        np.max(
            np.abs(
                our_pca.explained_variance_ratio
                - reference_pca.explained_variance_ratio_
            )
        )
    )

    print("Real-data validation")
    print("Regression dataset: sklearn diabetes")
    print(
        "maximum absolute coefficient difference: "
        f"{coefficient_max_abs_error:.3e}"
    )
    print("PCA dataset: sklearn iris")
    print(
        "maximum absolute explained-variance-ratio difference: "
        f"{variance_ratio_max_abs_error:.3e}"
    )


if __name__ == "__main__":
    main()
