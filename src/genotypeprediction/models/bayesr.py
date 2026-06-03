"""Global BayesR with a four-class mixture prior on SNP effects."""

from __future__ import annotations

import numpy as np
import pandas as pd

from genotypeprediction.data.preprocessing import GenotypeStandardizer
from genotypeprediction.evaluation.metrics import r2
from genotypeprediction.inference.bayesr_gibbs import (
    DEFAULT_BAYESR_ALPHA_PI,
    DEFAULT_BAYESR_GAMMA,
    run_bayesr_gibbs,
    validate_bayesr_prior,
)


def _posterior_interval(samples: np.ndarray, alpha: float = 0.05) -> list[float]:
    """Return an equal-tailed posterior interval."""

    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return [lower, upper]


class BayesR:
    """Global BayesR with one shared mixture probability vector ``pi``.

    The BayesR mixture classes are variance classes, not effect-size labels:

    - class 0: zero
    - class 1: small variance ``gamma_1 * sigma_beta2``
    - class 2: medium variance ``gamma_2 * sigma_beta2``
    - class 3: large variance ``gamma_3 * sigma_beta2``
    """

    def __init__(
        self,
        n_iter: int = 3000,
        burn_in: int = 1000,
        thin: int = 5,
        gamma: tuple[float, ...] | list[float] | np.ndarray = tuple(
            DEFAULT_BAYESR_GAMMA.tolist()
        ),
        alpha_pi: tuple[float, ...] | list[float] | np.ndarray = tuple(
            DEFAULT_BAYESR_ALPHA_PI.tolist()
        ),
        a_e: float = 1e-3,
        b_e: float = 1e-3,
        a_beta: float = 1e-3,
        b_beta: float = 1e-3,
        initial_sigma_e2: float | None = None,
        initial_sigma_beta2: float | None = None,
        initial_pi: tuple[float, ...] | list[float] | np.ndarray | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        gamma_array, alpha_pi_array = validate_bayesr_prior(
            gamma=gamma, alpha_pi=alpha_pi
        )

        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thin = thin
        self.gamma = gamma_array
        self.alpha_pi = alpha_pi_array
        self.a_e = a_e
        self.b_e = b_e
        self.a_beta = a_beta
        self.b_beta = b_beta
        self.initial_sigma_e2 = initial_sigma_e2
        self.initial_sigma_beta2 = initial_sigma_beta2
        self.initial_pi = (
            None if initial_pi is None else np.asarray(initial_pi, dtype=float)
        )
        self.random_state = random_state
        self.verbose = verbose

        self.standardizer_: GenotypeStandardizer | None = None
        self.beta_mean_: np.ndarray | None = None
        self.beta2_mean_: np.ndarray | None = None
        self.class_probabilities_: np.ndarray | None = None
        self.posterior_expected_class_: np.ndarray | None = None
        self.posterior_snp_variance_mean_: np.ndarray | None = None
        self.pip_: np.ndarray | None = None
        self.pi_trace_: np.ndarray | None = None
        self.class_count_trace_: np.ndarray | None = None
        self.sigma_e2_trace_: np.ndarray | None = None
        self.sigma_beta2_trace_: np.ndarray | None = None
        self.n_nonzero_trace_: np.ndarray | None = None
        self.class_labels_: list[str] | None = None
        self.feature_names_: list[str] | None = None
        self.gebv_train: np.ndarray | None = None
        self.gebv_test: np.ndarray | None = None
        self.posterior_sample_count_: int | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "BayesR":
        """Fit BayesR by Gibbs sampling."""

        self.standardizer_ = GenotypeStandardizer()
        X_standardized = self.standardizer_.fit_transform(
            X_train, feature_names=feature_names
        )
        self.standardizer_.fit_y(y_train)
        y_centered = self.standardizer_.center_y(y_train)

        posterior = run_bayesr_gibbs(
            X=X_standardized,
            y=y_centered,
            n_iter=self.n_iter,
            burn_in=self.burn_in,
            thin=self.thin,
            gamma=self.gamma,
            alpha_pi=self.alpha_pi,
            a_e=self.a_e,
            b_e=self.b_e,
            a_beta=self.a_beta,
            b_beta=self.b_beta,
            initial_sigma_e2=self.initial_sigma_e2,
            initial_sigma_beta2=self.initial_sigma_beta2,
            initial_pi=self.initial_pi,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self.beta_mean_ = np.asarray(posterior["beta_mean"], dtype=float)
        self.beta2_mean_ = np.asarray(posterior["beta2_mean"], dtype=float)
        self.class_probabilities_ = np.asarray(
            posterior["class_probabilities"], dtype=float
        )
        self.posterior_expected_class_ = np.asarray(
            posterior["posterior_expected_class"], dtype=float
        )
        self.posterior_snp_variance_mean_ = np.asarray(
            posterior["posterior_snp_variance_mean"], dtype=float
        )
        self.pip_ = np.asarray(posterior["pip"], dtype=float)
        self.pi_trace_ = np.asarray(posterior["pi_trace"], dtype=float)
        self.class_count_trace_ = np.asarray(posterior["class_count_trace"], dtype=int)
        self.sigma_e2_trace_ = np.asarray(posterior["sigma_e2_trace"], dtype=float)
        self.sigma_beta2_trace_ = np.asarray(
            posterior["sigma_beta2_trace"], dtype=float
        )
        self.n_nonzero_trace_ = np.asarray(posterior["n_nonzero_trace"], dtype=int)
        self.class_labels_ = list(posterior["class_labels"])
        self.posterior_sample_count_ = int(posterior["posterior_sample_count"])
        self.feature_names_ = self.standardizer_.kept_feature_names_
        self.gebv_train = X_standardized @ self.beta_mean_
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict phenotypes on the original scale using posterior mean effects."""

        if self.standardizer_ is None or self.beta_mean_ is None:
            raise RuntimeError("The model must be fitted before prediction.")

        X_test_standardized = self.standardizer_.transform(X_test)
        self.gebv_test = X_test_standardized @ self.beta_mean_
        return self.standardizer_.restore_y(self.gebv_test)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Return the out-of-sample R-squared."""

        return r2(y_test, self.predict(X_test))

    def get_posterior_summary(
        self,
    ) -> dict[
        str, np.ndarray | float | list[float] | dict[str, list[float]] | list[str]
    ]:
        """Return posterior summaries for BayesR mixture probabilities and SNP effects."""

        if (
            self.beta_mean_ is None
            or self.beta2_mean_ is None
            or self.class_probabilities_ is None
            or self.posterior_expected_class_ is None
            or self.posterior_snp_variance_mean_ is None
            or self.pip_ is None
            or self.pi_trace_ is None
            or self.class_count_trace_ is None
            or self.class_labels_ is None
        ):
            raise RuntimeError(
                "The model must be fitted before requesting posterior summaries."
            )

        pi_mean = np.mean(self.pi_trace_, axis=0)
        pi_sd = np.std(self.pi_trace_, axis=0, ddof=0)
        pi_ci_95 = {
            label: _posterior_interval(self.pi_trace_[:, class_index])
            for class_index, label in enumerate(self.class_labels_)
        }
        class_count_mean = np.mean(self.class_count_trace_, axis=0)
        class_count_sd = np.std(self.class_count_trace_, axis=0, ddof=0)
        summary: dict[
            str, np.ndarray | float | list[float] | dict[str, list[float]] | list[str]
        ] = {
            "beta_mean": self.beta_mean_,
            "beta2_mean": self.beta2_mean_,
            "class_probabilities": self.class_probabilities_,
            "pip": self.pip_,
            "posterior_expected_class": self.posterior_expected_class_,
            "posterior_snp_variance_mean": self.posterior_snp_variance_mean_,
            "pi_mean": pi_mean,
            "pi_sd": pi_sd,
            "pi_ci_95": pi_ci_95,
            "class_count_mean": class_count_mean,
            "class_count_sd": class_count_sd,
            "sigma_e2_mean": float(np.mean(self.sigma_e2_trace_)),
            "sigma_beta2_mean": float(np.mean(self.sigma_beta2_trace_)),
            "n_nonzero_mean": float(np.mean(self.n_nonzero_trace_)),
            "n_nonzero_sd": float(np.std(self.n_nonzero_trace_, ddof=0)),
            "pi_trace": self.pi_trace_,
            "class_count_trace": self.class_count_trace_,
            "sigma_e2_trace": self.sigma_e2_trace_,
            "sigma_beta2_trace": self.sigma_beta2_trace_,
            "n_nonzero_trace": self.n_nonzero_trace_,
            "class_labels": list(self.class_labels_),
            "gamma": self.gamma.copy(),
        }

        for class_index, label in enumerate(self.class_labels_):
            summary[f"p_{label}"] = self.class_probabilities_[:, class_index]

        if self.class_probabilities_.shape[1] >= 4:
            summary["p_medium_or_large"] = np.sum(
                self.class_probabilities_[:, 2:], axis=1
            )
            summary["p_large"] = self.class_probabilities_[:, 3]
        else:
            summary["p_medium_or_large"] = np.sum(
                self.class_probabilities_[:, 1:], axis=1
            )
            summary["p_large"] = self.class_probabilities_[:, -1]

        return summary

    def get_trace_summary(self) -> dict[str, np.ndarray]:
        """Return posterior traces stored after burn-in and thinning."""

        if self.pi_trace_ is None or self.class_count_trace_ is None:
            raise RuntimeError(
                "The model must be fitted before requesting trace summaries."
            )

        return {
            "pi_trace": self.pi_trace_,
            "class_count_trace": self.class_count_trace_,
            "sigma_e2_trace": self.sigma_e2_trace_,
            "sigma_beta2_trace": self.sigma_beta2_trace_,
            "n_nonzero_trace": self.n_nonzero_trace_,
        }

    def get_mixture_summary(self) -> pd.DataFrame:
        """Return a per-class summary table for the BayesR mixture."""

        summary = self.get_posterior_summary()
        rows = []
        for class_index, class_label in enumerate(summary["class_labels"]):
            interval = summary["pi_ci_95"][class_label]
            rows.append(
                {
                    "class_label": class_label,
                    "gamma": float(summary["gamma"][class_index]),
                    "pi_mean": float(summary["pi_mean"][class_index]),
                    "pi_sd": float(summary["pi_sd"][class_index]),
                    "pi_ci_95_lower": float(interval[0]),
                    "pi_ci_95_upper": float(interval[1]),
                    "class_count_mean": float(summary["class_count_mean"][class_index]),
                    "class_count_sd": float(summary["class_count_sd"][class_index]),
                }
            )
        return pd.DataFrame(rows)

    def get_top_snps_by_pip(
        self, top_n: int = 20
    ) -> list[dict[str, float | int | str]]:
        """Return the top markers ranked by posterior inclusion probability."""

        summary = self.get_posterior_summary()
        class_probabilities = np.asarray(summary["class_probabilities"], dtype=float)
        feature_names = (
            self.feature_names_
            if self.feature_names_ is not None
            else [f"snp_{index}" for index in range(class_probabilities.shape[0])]
        )
        top_indices = np.argsort(self.pip_)[::-1][:top_n]
        return [
            {
                "rank": rank,
                "marker_index": int(marker_index),
                "feature_name": feature_names[marker_index],
                "pip": float(self.pip_[marker_index]),
                "p_medium_or_large": float(summary["p_medium_or_large"][marker_index]),
                "p_large": float(summary["p_large"][marker_index]),
                "posterior_expected_class": float(
                    self.posterior_expected_class_[marker_index]
                ),
                "beta_mean": float(self.beta_mean_[marker_index]),
                "beta2_mean": float(self.beta2_mean_[marker_index]),
                "posterior_snp_variance_mean": float(
                    self.posterior_snp_variance_mean_[marker_index]
                ),
            }
            for rank, marker_index in enumerate(top_indices, start=1)
        ]
