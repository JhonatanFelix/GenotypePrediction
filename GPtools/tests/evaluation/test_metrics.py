"""Tests for regression metrics."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import r2_score

from GPtools.evaluation.metrics import pearson_corr, r2


def test_r2_matches_reference_value_on_nontrivial_predictions() -> None:
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 4.0])

    assert r2(y_true, y_pred) == pytest.approx(0.5)


def test_r2_treats_perfect_predictions_on_constant_targets_as_perfect() -> None:
    y_true = np.array([5.0, 5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 5.0, 5.0, 5.0])

    assert r2(y_true, y_pred) == 0.0 


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 6.0]), 1.0),
        (np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0]), -1.0),
    ],
)
def test_pearson_corr_matches_expected_direction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    expected: float,
) -> None:
    assert pearson_corr(y_true, y_pred) == pytest.approx(expected)


@pytest.mark.parametrize("metric", [r2, pearson_corr])
def test_metrics_require_matching_shapes(metric) -> None:
    with pytest.raises(ValueError, match="same shape"):
        metric(np.array([1.0, 2.0]), np.array([1.0]))
