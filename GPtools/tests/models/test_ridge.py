"""Tests for the primal ridge estimator."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from GPtools.models.ridge import RidgePrimal


def _linear_regression_data() -> tuple[np.ndarray, np.ndarray]:
    X_train = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [3.0, 2.0],
        ]
    )
    y_train = 5.0 + 2.0 * X_train[:, 0] - 3.0 * X_train[:, 1]
    return X_train, y_train


def test_predict_requires_a_fitted_model() -> None:
    model = RidgePrimal()

    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict(np.array([[0.0, 0.0]]))


def test_fit_recovers_a_linear_signal_with_tiny_regularization() -> None:
    X_train, y_train = _linear_regression_data()
    model = RidgePrimal(random_state=0)

    model.fit(X_train, y_train, lambda_value=1e-8)
    predictions = model.predict(X_train)

    assert predictions.shape == y_train.shape
    np.testing.assert_allclose(predictions, y_train, atol=1e-6)
    assert model.score(X_train, y_train) == pytest.approx(1.0, abs=1e-6)
    assert model.beta_hat is not None
    assert model.beta_hat.shape == (X_train.shape[1],)


def test_fit_selects_the_best_lambda_from_the_grid() -> None:
    X_train, y_train = _linear_regression_data()
    model = RidgePrimal(validation_size=0.3, random_state=0)

    model.fit(X_train, y_train, lambda_grid=[1e-8, 1.0, 1000.0])

    assert model.selected_lambda_ == pytest.approx(1e-8)
    assert model.lambda_value_ == pytest.approx(1e-8)


def test_metric_report_uses_standard_regression_definitions() -> None:
    X_train, y_train = _linear_regression_data()
    X_test = np.array(
        [
            [1.5, 0.5],
            [2.5, 1.5],
            [0.5, 2.5],
        ]
    )
    y_test = 5.0 + 2.0 * X_test[:, 0] - 3.0 * X_test[:, 1]

    model = RidgePrimal(random_state=0)
    model.fit(X_train, y_train, lambda_value=0.1)
    predictions = model.predict(X_test)
    report = model.metric_report(X_test, y_test)

    expected_mse = mean_squared_error(y_test, predictions)
    expected_mae = mean_absolute_error(y_test, predictions)
    expected_r2 = r2_score(y_test, predictions)
    expected_pearson = np.corrcoef(y_test, predictions)[0, 1]

    assert set(report) == {"mse", "mae", "r2", "pearson"}
    assert expected_mse != pytest.approx(expected_mae)
    assert report["mse"] == pytest.approx(expected_mse)
    assert report["mae"] == pytest.approx(expected_mae)
    assert report["r2"] == pytest.approx(expected_r2)
    assert report["pearson"] == pytest.approx(expected_pearson)
