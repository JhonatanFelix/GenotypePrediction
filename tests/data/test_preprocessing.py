"""Tests for imputations, scaling, and phenotype centering."""

from __future__ import annotations

import numpy as np

from genotypeprediction.data.preprocessing import GenotypeStandardizer


def test_preprocessing_uses_statistics_and_removes_markers() -> None:
    X_train = np.array(
        [[0.0, np.nan, 5.0, 1.0], [2.0, 1.0, 5.0, 2.0], [1.0, 2.0, 5.0, 3.0]]
    )
    X_test = np.array([[np.nan, 0.0, 5.0, 4.0]])
    y_train = np.array([10.0, 12.0, 14.0])

    standardizer = GenotypeStandardizer()
    X_train_standardized = standardizer.fit_transform(
        X_train,
        feature_names=["snp_0", "snp_1", "snp_zero_variance", "snp_to_finally_end"],
    )
    standardizer.fit_y(y_train)
    X_test_standardized = standardizer.transform(X_test)

    assert standardizer.keep_mask_.tolist() == [True, True, False, True]
    assert standardizer.kept_feature_names_ == ["snp_0", "snp_1", "snp_to_finally_end"]

    expected_train = np.array(
        [
            [-1.22474487, 0.0, -1.22474487],
            [1.22474487, -1.22474487, 0.0],
            [0.0, 1.22474487, 1.22474487],
        ]
    )
    expected_test = np.array([[0.0, -3.67423461, 2.44948974]])

    np.testing.assert_allclose(X_train_standardized, expected_train, atol=1e-6)
    np.testing.assert_allclose(X_test_standardized, expected_test, atol=1e-6)
    np.testing.assert_allclose(
        standardizer.center_y(y_train), np.array([-2.0, 0.0, 2.0])
    )
    np.testing.assert_allclose(
        standardizer.restore_y(standardizer.center_y(y_train)),
        y_train,
    )
