import robust_statistical_learning


def test_package_foundation() -> None:
    """Verify that the package foundation is importable."""
    assert robust_statistical_learning.__version__ == "0.1.0"