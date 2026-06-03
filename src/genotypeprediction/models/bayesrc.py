"""BayesRC with disjoint annotation categories and BayesR variance classes."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from genotypeprediction.data.preprocessing import GenotypeStandardizer
from genotypeprediction.evaluation.metrics import r2
from genotypeprediction.inference.bayesr_gibbs import (
    DEFAULT_BAYESR_ALPHA_PI,
    DEFAULT_BAYESR_GAMMA,
    run_bayesrc_gibbs,
    validate_bayesr_prior,
)


def _posterior_interval(samples: np.ndarray, alpha: float = 0.05) -> list[float]:
    """Return an equal-tailed posterior interval."""

    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return [lower, upper]


def _validate_annotation_categories(
    annotation_categories: np.ndarray | list[object],
    n_markers: int,
) -> np.ndarray:
    """Validate disjoint annotation labels and return a 1D object array."""

    categories = np.asarray(annotation_categories, dtype=object).reshape(-1)
    if categories.shape[0] != n_markers:
        raise ValueError(
            "annotation_categories must have length equal to the number of SNPs."
        )
    if pd.isna(categories).any():
        raise ValueError("annotation_categories contains missing values.")
    return categories


class BayesRC:
    """BayesRC with one class-probability vector ``pi_c`` per annotation category."""

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
        initial_pi_by_category: np.ndarray | None = None,
        min_category_size_warning: int = 5,
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
        self.initial_pi_by_category = (
            None
            if initial_pi_by_category is None
            else np.asarray(initial_pi_by_category, dtype=float)
        )
        self.min_category_size_warning = int(min_category_size_warning)
        self.random_state = random_state
        self.verbose = verbose

        self.standardizer_: GenotypeStandardizer | None = None
        self.beta_mean_: np.ndarray | None = None
        self.beta2_mean_: np.ndarray | None = None
        self.class_probabilities_: np.ndarray | None = None
        self.posterior_expected_class_: np.ndarray | None = None
        self.posterior_snp_variance_mean_: np.ndarray | None = None
        self.pip_: np.ndarray | None = None
        self.pi_by_category_trace_: np.ndarray | None = None
        self.category_class_count_trace_: np.ndarray | None = None
        self.n_nonzero_by_category_trace_: np.ndarray | None = None
        self.sigma_e2_trace_: np.ndarray | None = None
        self.sigma_beta2_trace_: np.ndarray | None = None
        self.n_nonzero_trace_: np.ndarray | None = None
        self.class_labels_: list[str] | None = None
        self.feature_names_: list[str] | None = None
        self.annotation_categories_: np.ndarray | None = None
        self.category_codes_: np.ndarray | None = None
        self.category_labels_: list[object] | None = None
        self.category_sizes_: np.ndarray | None = None
        self.warning_messages_: list[str] = []
        self.gebv_train: np.ndarray | None = None
        self.gebv_test: np.ndarray | None = None
        self.posterior_sample_count_: int | None = None

    def _warn(self, message: str) -> None:
        """Store and emit category-validation warnings."""

        self.warning_messages_.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        annotation_categories: np.ndarray | list[object],
        feature_names: list[str] | None = None,
    ) -> "BayesRC":
        """Fit BayesRC by Gibbs sampling with category-specific class priors."""

        X_train = np.asarray(X_train, dtype=float)
        categories = _validate_annotation_categories(
            annotation_categories=annotation_categories,
            n_markers=X_train.shape[1],
        )
        self.warning_messages_ = []

        self.standardizer_ = GenotypeStandardizer()
        X_standardized = self.standardizer_.fit_transform(
            X_train, feature_names=feature_names
        )
        self.standardizer_.fit_y(y_train)
        y_centered = self.standardizer_.center_y(y_train)

        kept_categories = categories[self.standardizer_.keep_mask_]
        category_codes, unique_labels = pd.factorize(kept_categories, sort=False)
        category_sizes = np.bincount(category_codes, minlength=len(unique_labels))
        small_categories = np.where(category_sizes < self.min_category_size_warning)[0]
        for category_index in small_categories:
            self._warn(
                f"Annotation category {unique_labels[category_index]!r} has only "
                f"{int(category_sizes[category_index])} SNPs after preprocessing."
            )

        posterior = run_bayesrc_gibbs(
            X=X_standardized,
            y=y_centered,
            category_codes=category_codes,
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
            initial_pi_by_category=self.initial_pi_by_category,
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
            posterior["posterior_snp_variance_mean"],
            dtype=float,
        )
        self.pip_ = np.asarray(posterior["pip"], dtype=float)
        self.pi_by_category_trace_ = np.asarray(
            posterior["pi_by_category_trace"], dtype=float
        )
        self.category_class_count_trace_ = np.asarray(
            posterior["category_class_count_trace"],
            dtype=int,
        )
        self.n_nonzero_by_category_trace_ = np.asarray(
            posterior["n_nonzero_by_category_trace"],
            dtype=int,
        )
        self.sigma_e2_trace_ = np.asarray(posterior["sigma_e2_trace"], dtype=float)
        self.sigma_beta2_trace_ = np.asarray(
            posterior["sigma_beta2_trace"], dtype=float
        )
        self.n_nonzero_trace_ = np.asarray(posterior["n_nonzero_trace"], dtype=int)
        self.class_labels_ = list(posterior["class_labels"])
        self.posterior_sample_count_ = int(posterior["posterior_sample_count"])
        self.feature_names_ = self.standardizer_.kept_feature_names_
        self.annotation_categories_ = np.asarray(kept_categories, dtype=object)
        self.category_codes_ = np.asarray(category_codes, dtype=int)
        self.category_labels_ = list(unique_labels)
        self.category_sizes_ = category_sizes
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

    def get_posterior_summary(self) -> dict[str, object]:
        """Return posterior BayesRC summaries at SNP and category levels."""

        if (
            self.beta_mean_ is None
            or self.class_probabilities_ is None
            or self.posterior_expected_class_ is None
            or self.posterior_snp_variance_mean_ is None
            or self.pip_ is None
            or self.pi_by_category_trace_ is None
            or self.category_class_count_trace_ is None
            or self.class_labels_ is None
            or self.category_labels_ is None
        ):
            raise RuntimeError(
                "The model must be fitted before requesting posterior summaries."
            )

        pi_by_category_mean = np.mean(self.pi_by_category_trace_, axis=0)
        pi_by_category_sd = np.std(self.pi_by_category_trace_, axis=0, ddof=0)
        pi_by_category_ci_95 = {
            category_label: {
                class_label: _posterior_interval(
                    self.pi_by_category_trace_[:, category_index, class_index]
                )
                for class_index, class_label in enumerate(self.class_labels_)
            }
            for category_index, category_label in enumerate(self.category_labels_)
        }

        summary: dict[str, object] = {
            "beta_mean": self.beta_mean_,
            "beta2_mean": self.beta2_mean_,
            "class_probabilities": self.class_probabilities_,
            "pip": self.pip_,
            "posterior_expected_class": self.posterior_expected_class_,
            "posterior_snp_variance_mean": self.posterior_snp_variance_mean_,
            "pi_by_category_mean": pi_by_category_mean,
            "pi_by_category_sd": pi_by_category_sd,
            "pi_by_category_ci_95": pi_by_category_ci_95,
            "category_class_count_mean": np.mean(
                self.category_class_count_trace_, axis=0
            ),
            "category_class_count_sd": np.std(
                self.category_class_count_trace_, axis=0, ddof=0
            ),
            "sigma_e2_mean": float(np.mean(self.sigma_e2_trace_)),
            "sigma_beta2_mean": float(np.mean(self.sigma_beta2_trace_)),
            "n_nonzero_mean": float(np.mean(self.n_nonzero_trace_)),
            "n_nonzero_sd": float(np.std(self.n_nonzero_trace_, ddof=0)),
            "pi_by_category_trace": self.pi_by_category_trace_,
            "category_class_count_trace": self.category_class_count_trace_,
            "n_nonzero_by_category_trace": self.n_nonzero_by_category_trace_,
            "sigma_e2_trace": self.sigma_e2_trace_,
            "sigma_beta2_trace": self.sigma_beta2_trace_,
            "n_nonzero_trace": self.n_nonzero_trace_,
            "class_labels": list(self.class_labels_),
            "category_labels": list(self.category_labels_),
            "annotation_categories": self.annotation_categories_,
            "category_codes": self.category_codes_,
            "warning_messages": list(self.warning_messages_),
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
        """Return post-burn-in traces recorded by the sampler."""

        if self.pi_by_category_trace_ is None:
            raise RuntimeError(
                "The model must be fitted before requesting trace summaries."
            )

        return {
            "pi_by_category_trace": self.pi_by_category_trace_,
            "category_class_count_trace": self.category_class_count_trace_,
            "n_nonzero_by_category_trace": self.n_nonzero_by_category_trace_,
            "sigma_e2_trace": self.sigma_e2_trace_,
            "sigma_beta2_trace": self.sigma_beta2_trace_,
            "n_nonzero_trace": self.n_nonzero_trace_,
        }

    def get_annotation_summary(self) -> pd.DataFrame:
        """Return one row per annotation category with BayesRC posterior summaries."""

        summary = self.get_posterior_summary()
        rows: list[dict[str, float | object]] = []
        for category_index, category_label in enumerate(summary["category_labels"]):
            category_mask = self.category_codes_ == category_index
            category_pip = self.pip_[category_mask]
            category_pi_mean = summary["pi_by_category_mean"][category_index]
            category_pi_sd = summary["pi_by_category_sd"][category_index]
            category_interval = summary["pi_by_category_ci_95"][category_label]

            rows.append(
                {
                    "category": category_label,
                    "n_snps": int(self.category_sizes_[category_index]),
                    "pi_zero_mean": float(category_pi_mean[0]),
                    "pi_small_mean": float(category_pi_mean[1])
                    if category_pi_mean.shape[0] > 1
                    else np.nan,
                    "pi_medium_mean": float(category_pi_mean[2])
                    if category_pi_mean.shape[0] > 2
                    else np.nan,
                    "pi_large_mean": float(category_pi_mean[3])
                    if category_pi_mean.shape[0] > 3
                    else np.nan,
                    "pi_zero_sd": float(category_pi_sd[0]),
                    "pi_small_sd": float(category_pi_sd[1])
                    if category_pi_sd.shape[0] > 1
                    else np.nan,
                    "pi_medium_sd": float(category_pi_sd[2])
                    if category_pi_sd.shape[0] > 2
                    else np.nan,
                    "pi_large_sd": float(category_pi_sd[3])
                    if category_pi_sd.shape[0] > 3
                    else np.nan,
                    "pi_zero_ci_95_lower": float(
                        category_interval[self.class_labels_[0]][0]
                    ),
                    "pi_zero_ci_95_upper": float(
                        category_interval[self.class_labels_[0]][1]
                    ),
                    "nonzero_prob_mean": float(1.0 - category_pi_mean[0]),
                    "medium_or_large_prob_mean": float(np.sum(category_pi_mean[2:])),
                    "expected_nonzero_snps_mean": float(
                        np.mean(self.n_nonzero_by_category_trace_[:, category_index])
                    ),
                    "expected_nonzero_snps_sd": float(
                        np.std(
                            self.n_nonzero_by_category_trace_[:, category_index], ddof=0
                        )
                    ),
                    "mean_pip": float(np.mean(category_pip)),
                    "max_pip": float(np.max(category_pip)),
                }
            )

        return pd.DataFrame(rows)

    def get_top_snps_by_pip(
        self, top_n: int = 20
    ) -> list[dict[str, float | int | str | object]]:
        """Return top markers ranked by posterior inclusion probability."""

        if self.pip_ is None or self.beta_mean_ is None or self.beta2_mean_ is None:
            raise RuntimeError("The model must be fitted before requesting top SNPs.")

        feature_names = (
            self.feature_names_
            if self.feature_names_ is not None
            else [f"snp_{index}" for index in range(self.pip_.shape[0])]
        )
        top_indices = np.argsort(self.pip_)[::-1][:top_n]
        posterior_summary = self.get_posterior_summary()
        return [
            {
                "rank": rank,
                "marker_index": int(marker_index),
                "feature_name": feature_names[marker_index],
                "annotation_category": self.annotation_categories_[marker_index],
                "pip": float(self.pip_[marker_index]),
                "p_medium_or_large": float(
                    posterior_summary["p_medium_or_large"][marker_index]
                ),
                "p_large": float(posterior_summary["p_large"][marker_index]),
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
