"""Estimator-wrapper tests for BayesR and BayesRC."""

from __future__ import annotations

import numpy as np
import pytest

from genotypeprediction.models.bayesr import BayesR
from genotypeprediction.models.bayesrc import BayesRC


def _training_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = np.array(
        [
            [0.0, 1.0, 2.0, 0.0, 5.0],
            [1.0, 0.0, 2.0, 1.0, 5.0],
            [2.0, 1.0, 1.0, 0.0, 5.0],
            [0.0, 2.0, 0.0, 1.0, 5.0],
            [1.0, 1.0, 1.0, 0.0, 5.0],
            [2.0, 0.0, 0.0, 1.0, 5.0],
        ],
        dtype=float,
    )
    y = np.array([1.2, 1.5, 2.2, 1.1, 1.7, 2.4], dtype=float)
    feature_names = ["snp_0", "snp_1", "snp_2", "snp_3", "constant_snp"]
    return X, y, feature_names


def test_bayesr_fit_predict_and_summary_shapes() -> None:
    X, y, feature_names = _training_data()
    model = BayesR(n_iter=12, burn_in=4, thin=2, random_state=321)

    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(X[:2])

    assert model.fit(X, y, feature_names=feature_names) is model
    predictions = model.predict(X[:2])
    summary = model.get_posterior_summary()
    traces = model.get_trace_summary()
    mixture_summary = model.get_mixture_summary()
    top_snps = model.get_top_snps_by_pip(top_n=2)

    assert predictions.shape == (2,)
    assert np.all(np.isfinite(predictions))
    assert model.feature_names_ == ["snp_0", "snp_1", "snp_2", "snp_3"]
    assert model.posterior_sample_count_ == 4
    assert summary["beta_mean"].shape == (4,)
    assert summary["class_probabilities"].shape == (4, 4)
    assert traces["pi_trace"].shape == (4, 4)
    assert len(mixture_summary) == 4
    assert [row["rank"] for row in top_snps] == [1, 2]
    assert all(row["feature_name"] in model.feature_names_ for row in top_snps)
    np.testing.assert_allclose(
        summary["class_probabilities"].sum(axis=1),
        np.ones(4),
    )
    np.testing.assert_allclose(
        summary["pip"],
        1.0 - summary["class_probabilities"][:, 0],
    )


def test_bayesrc_fit_aligns_annotations_after_marker_filtering() -> None:
    X, y, feature_names = _training_data()
    annotation_categories = ["genic", "genic", "rare", "rare", "dropped"]
    model = BayesRC(
        n_iter=12,
        burn_in=4,
        thin=2,
        min_category_size_warning=1,
        random_state=321,
    )

    assert (
        model.fit(
            X,
            y,
            annotation_categories=annotation_categories,
            feature_names=feature_names,
        )
        is model
    )
    predictions = model.predict(X[:3])
    summary = model.get_posterior_summary()
    traces = model.get_trace_summary()
    annotation_summary = model.get_annotation_summary()
    top_snps = model.get_top_snps_by_pip(top_n=2)

    assert predictions.shape == (3,)
    assert np.all(np.isfinite(predictions))
    assert model.feature_names_ == ["snp_0", "snp_1", "snp_2", "snp_3"]
    assert list(model.annotation_categories_) == ["genic", "genic", "rare", "rare"]
    assert model.category_labels_ == ["genic", "rare"]
    assert model.category_sizes_.tolist() == [2, 2]
    assert model.posterior_sample_count_ == 4
    assert summary["beta_mean"].shape == (4,)
    assert summary["class_probabilities"].shape == (4, 4)
    assert traces["pi_by_category_trace"].shape == (4, 2, 4)
    assert traces["category_class_count_trace"].shape == (4, 2, 4)
    assert set(annotation_summary["category"]) == {"genic", "rare"}
    assert set(annotation_summary["n_snps"]) == {2}
    assert [row["rank"] for row in top_snps] == [1, 2]
    assert all(row["annotation_category"] in {"genic", "rare"} for row in top_snps)
    np.testing.assert_allclose(
        summary["pi_by_category_mean"].sum(axis=1),
        np.ones(2),
    )
    np.testing.assert_allclose(
        summary["pip"],
        1.0 - summary["class_probabilities"][:, 0],
    )


def test_bayesrc_validates_annotation_categories() -> None:
    X, y, _ = _training_data()
    model = BayesRC(n_iter=6, burn_in=2, thin=2, random_state=321)

    with pytest.raises(ValueError, match="length equal"):
        model.fit(X, y, annotation_categories=["genic"])

    with pytest.raises(ValueError, match="missing values"):
        model.fit(
            X,
            y,
            annotation_categories=["genic", "genic", np.nan, "rare", "dropped"],
        )
