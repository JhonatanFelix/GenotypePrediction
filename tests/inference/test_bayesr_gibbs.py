"""Tests for BayesR and BayesRC Gibbs sampler helpers."""

from __future__ import annotations

import numpy as np
import pytest

from genotypeprediction.inference.bayesr_gibbs import (
    DEFAULT_BAYESR_ALPHA_PI,
    DEFAULT_BAYESR_GAMMA,
    compute_bayesr_sigma_beta2_scale_term,
    run_bayesr_fixed_prior_gibbs,
    run_bayesr_gibbs,
    run_bayesrc_gibbs,
    validate_bayesr_prior,
    validate_bayesr_prior_probabilities,
)


def _sampler_data() -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [-1.2, 0.4, 1.0, -0.7],
            [-0.4, -1.1, 0.2, 0.5],
            [0.1, 0.8, -1.3, -0.2],
            [0.7, -0.3, 0.4, 1.0],
            [1.1, 0.2, -0.6, -0.6],
        ],
        dtype=float,
    )
    y = np.array([-1.0, -0.2, 0.3, 0.9, 1.1], dtype=float)
    return X, y


def _assert_common_posterior(
    posterior: dict[str, object],
    n_markers: int,
    n_classes: int,
    sample_count: int,
) -> None:
    class_probabilities = np.asarray(posterior["class_probabilities"], dtype=float)
    pip = np.asarray(posterior["pip"], dtype=float)

    assert np.asarray(posterior["beta_mean"]).shape == (n_markers,)
    assert np.asarray(posterior["beta2_mean"]).shape == (n_markers,)
    assert class_probabilities.shape == (n_markers, n_classes)
    assert np.asarray(posterior["posterior_expected_class"]).shape == (n_markers,)
    assert np.asarray(posterior["posterior_snp_variance_mean"]).shape == (n_markers,)
    assert int(posterior["posterior_sample_count"]) == sample_count
    assert list(posterior["class_labels"]) == ["zero", "small", "medium", "large"]

    np.testing.assert_allclose(class_probabilities.sum(axis=1), np.ones(n_markers))
    np.testing.assert_allclose(pip, 1.0 - class_probabilities[:, 0])
    assert np.all((class_probabilities >= 0.0) & (class_probabilities <= 1.0))
    assert np.all((pip >= 0.0) & (pip <= 1.0))


def test_bayesr_prior_validation_and_scale_term() -> None:
    gamma, alpha_pi = validate_bayesr_prior(
        gamma=DEFAULT_BAYESR_GAMMA,
        alpha_pi=DEFAULT_BAYESR_ALPHA_PI,
    )
    np.testing.assert_allclose(gamma, DEFAULT_BAYESR_GAMMA)
    np.testing.assert_allclose(alpha_pi, DEFAULT_BAYESR_ALPHA_PI)

    with pytest.raises(ValueError, match="first BayesR variance"):
        validate_bayesr_prior(gamma=[0.1, 1.0], alpha_pi=[1.0, 1.0])

    normalized = validate_bayesr_prior_probabilities(
        np.array([[2.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 1.0]])
    )
    np.testing.assert_allclose(normalized.sum(axis=1), np.ones(2))

    with pytest.raises(ValueError, match="strictly positive"):
        validate_bayesr_prior_probabilities(np.array([[0.5, 0.5, 0.0, 0.1]]))

    scale_term = compute_bayesr_sigma_beta2_scale_term(
        beta=np.array([1.0, 2.0, 3.0]),
        class_assignments=np.array([0, 1, 3]),
        gamma=np.array([0.0, 0.5, 1.0, 2.0]),
    )
    assert scale_term == pytest.approx(12.5)


def test_run_bayesr_gibbs_returns_expected_traces() -> None:
    X, y = _sampler_data()
    posterior = run_bayesr_gibbs(
        X=X,
        y=y,
        n_iter=10,
        burn_in=4,
        thin=2,
        random_state=123,
    )

    _assert_common_posterior(
        posterior=posterior,
        n_markers=X.shape[1],
        n_classes=DEFAULT_BAYESR_GAMMA.shape[0],
        sample_count=3,
    )
    assert np.asarray(posterior["pi_trace"]).shape == (3, 4)
    assert np.asarray(posterior["class_count_trace"]).shape == (3, 4)
    np.testing.assert_allclose(
        np.asarray(posterior["pi_trace"], dtype=float).sum(axis=1),
        np.ones(3),
    )
    np.testing.assert_allclose(
        np.asarray(posterior["class_count_trace"], dtype=int).sum(axis=1),
        np.full(3, X.shape[1]),
    )


def test_run_bayesr_gibbs_verbose_supports_custom_class_count() -> None:
    X, y = _sampler_data()
    posterior = run_bayesr_gibbs(
        X=X,
        y=y,
        n_iter=6,
        burn_in=2,
        thin=2,
        gamma=np.array([0.0, 1e-4, 1e-3, 1e-2, 1e-1]),
        alpha_pi=np.ones(5),
        random_state=123,
        verbose=True,
    )

    assert np.asarray(posterior["class_probabilities"]).shape == (X.shape[1], 5)
    assert list(posterior["class_labels"]) == [
        "class_0",
        "class_1",
        "class_2",
        "class_3",
        "class_4",
    ]


def test_run_bayesr_fixed_prior_normalizes_snp_priors() -> None:
    X, y = _sampler_data()
    prior_probabilities = np.array(
        [
            [2.0, 1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0, 1.0],
            [1.0, 1.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 2.0],
        ]
    )
    posterior = run_bayesr_fixed_prior_gibbs(
        X=X,
        y=y,
        prior_probabilities=prior_probabilities,
        n_iter=8,
        burn_in=2,
        thin=2,
        random_state=123,
    )

    _assert_common_posterior(
        posterior=posterior,
        n_markers=X.shape[1],
        n_classes=DEFAULT_BAYESR_GAMMA.shape[0],
        sample_count=3,
    )
    np.testing.assert_allclose(
        np.asarray(posterior["prior_probabilities"], dtype=float).sum(axis=1),
        np.ones(X.shape[1]),
    )


def test_run_bayesrc_gibbs_returns_category_specific_traces() -> None:
    X, y = _sampler_data()
    category_codes = np.array([0, 0, 1, 1])
    posterior = run_bayesrc_gibbs(
        X=X,
        y=y,
        category_codes=category_codes,
        n_iter=10,
        burn_in=4,
        thin=2,
        random_state=123,
    )

    _assert_common_posterior(
        posterior=posterior,
        n_markers=X.shape[1],
        n_classes=DEFAULT_BAYESR_GAMMA.shape[0],
        sample_count=3,
    )
    assert np.asarray(posterior["pi_by_category_trace"]).shape == (3, 2, 4)
    assert np.asarray(posterior["category_class_count_trace"]).shape == (3, 2, 4)
    assert np.asarray(posterior["n_nonzero_by_category_trace"]).shape == (3, 2)
    np.testing.assert_allclose(
        np.asarray(posterior["pi_by_category_trace"], dtype=float).sum(axis=2),
        np.ones((3, 2)),
    )
    np.testing.assert_allclose(
        np.asarray(posterior["category_class_count_trace"], dtype=int).sum(axis=2),
        np.full((3, 2), 2),
    )


def test_run_bayesrc_gibbs_validates_category_codes() -> None:
    X, y = _sampler_data()
    with pytest.raises(ValueError, match="one entry per SNP"):
        run_bayesrc_gibbs(
            X=X,
            y=y,
            category_codes=np.array([0, 1]),
            n_iter=6,
            burn_in=2,
            thin=2,
        )
