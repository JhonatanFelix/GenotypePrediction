"""BayesCpi with learned global inclusion probability ``q``."""

from __future__ import annotations

import numpy as np

from neural_pi_bayes.data.preprocessing import GenotypeStandardizer
from neural_pi_bayes.evaluation.metrics import r2
from neural_pi_bayes.inference.gibbs import run_bayescpi_gibbs


def _posterior_interval(samples: np.ndarray, alpha: float = 0.05) -> list[float]:
    """Return an equal-tailed posterior interval."""

    lower = float(np.quantile(samples, alpha / 2.0))
    upper = float(np.quantile(samples, 1.0 - alpha / 2.0))
    return [lower, upper]


class BayesCPi:
    """BayesCpi with global inclusion probability learned from the posterior.

    In this codebase, ``q`` means the probability that a marker is included:

    ``q = P(beta_j != 0)``.

    This differs from the classical ``pi`` notation often used in the BayesCpi
    literature, where ``pi`` denotes the probability of a zero effect.
    """

    def __init__(
        self,
        n_iter: int = 3000,
        burn_in: int = 1000,
        thin: int = 5,
        a_q: float = 1.0,
        b_q: float = 1.0,
        a_e: float = 1e-3,
        b_e: float = 1e-3,
        a_beta: float = 1e-3,
        b_beta: float = 1e-3,
        initial_q: float = 0.01,
        initial_sigma_e2: float | None = None,
        initial_sigma_beta2: float | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ) -> None:
        self.n_iter = n_iter
        self.burn_in = burn_in
        self.thin = thin
        self.a_q = a_q
        self.b_q = b_q
        self.a_e = a_e
        self.b_e = b_e
        self.a_beta = a_beta
        self.b_beta = b_beta
        self.initial_q = initial_q
        self.initial_sigma_e2 = initial_sigma_e2
        self.initial_sigma_beta2 = initial_sigma_beta2
        self.random_state = random_state
        self.verbose = verbose

        self.standardizer_: GenotypeStandardizer | None = None
        self.beta_mean_: np.ndarray | None = None
        self.beta2_mean_: np.ndarray | None = None
        self.pip_: np.ndarray | None = None
        self.q_trace_: np.ndarray | None = None
        self.sigma_e2_trace_: np.ndarray | None = None
        self.sigma_beta2_trace_: np.ndarray | None = None
        self.n_included_trace_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None
        self.gebv_train: np.ndarray | None = None
        self.gebv_test: np.ndarray | None = None
        self.posterior_sample_count_: int | None = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "BayesCPi":
        """Fit BayesCpi by Gibbs sampling with posterior updates for ``q``."""

        self.standardizer_ = GenotypeStandardizer()
        X_standardized = self.standardizer_.fit_transform(X_train, feature_names=feature_names)
        self.standardizer_.fit_y(y_train)
        y_centered = self.standardizer_.center_y(y_train)

        posterior = run_bayescpi_gibbs(
            X=X_standardized,
            y=y_centered,
            n_iter=self.n_iter,
            burn_in=self.burn_in,
            thin=self.thin,
            a_q=self.a_q,
            b_q=self.b_q,
            a_e=self.a_e,
            b_e=self.b_e,
            a_beta=self.a_beta,
            b_beta=self.b_beta,
            initial_q=self.initial_q,
            initial_sigma_e2=self.initial_sigma_e2,
            initial_sigma_beta2=self.initial_sigma_beta2,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        self.beta_mean_ = np.asarray(posterior["beta_mean"], dtype=float)
        self.beta2_mean_ = np.asarray(posterior["beta2_mean"], dtype=float)
        self.pip_ = np.asarray(posterior["pip"], dtype=float)
        self.q_trace_ = np.asarray(posterior["q_trace"], dtype=float)
        self.sigma_e2_trace_ = np.asarray(posterior["sigma_e2_trace"], dtype=float)
        self.sigma_beta2_trace_ = np.asarray(posterior["sigma_beta2_trace"], dtype=float)
        self.n_included_trace_ = np.asarray(posterior["n_included_trace"], dtype=int)
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

    def get_posterior_summary(self) -> dict[str, np.ndarray | float | list[float]]:
        """Return posterior summaries for marker effects and global inclusion ``q``."""

        if self.beta_mean_ is None or self.pip_ is None or self.beta2_mean_ is None:
            raise RuntimeError("The model must be fitted before requesting posterior summaries.")

        q_mean = float(np.mean(self.q_trace_))
        q_sd = float(np.std(self.q_trace_, ddof=0))
        n_included_mean = float(np.mean(self.n_included_trace_))
        n_included_sd = float(np.std(self.n_included_trace_, ddof=0))
        sigma_e2_mean = float(np.mean(self.sigma_e2_trace_))
        sigma_beta2_mean = float(np.mean(self.sigma_beta2_trace_))

        return {
            "beta_mean": self.beta_mean_,
            "beta2_mean": self.beta2_mean_,
            "pip": self.pip_,
            "q_mean": q_mean,
            "q_sd": q_sd,
            "q_median": float(np.median(self.q_trace_)),
            "q_ci_95": _posterior_interval(self.q_trace_),
            "sigma_e2_mean": sigma_e2_mean,
            "sigma_beta2_mean": sigma_beta2_mean,
            "n_included_mean": n_included_mean,
            "n_included_sd": n_included_sd,
            "q_trace": self.q_trace_,
            "sigma_e2_trace": self.sigma_e2_trace_,
            "sigma_beta2_trace": self.sigma_beta2_trace_,
            "n_included_trace": self.n_included_trace_,
            # Convenience aliases matching the existing fixed-q code path.
            "posterior_mean_sigma_e2": sigma_e2_mean,
            "posterior_mean_sigma_beta2": sigma_beta2_mean,
            "posterior_mean_n_included": n_included_mean,
        }

    def get_trace_summary(self) -> dict[str, np.ndarray]:
        """Return the traces recorded after burn-in and thinning."""

        if self.q_trace_ is None:
            raise RuntimeError("The model must be fitted before requesting trace summaries.")

        return {
            "q_trace": self.q_trace_,
            "sigma_e2_trace": self.sigma_e2_trace_,
            "sigma_beta2_trace": self.sigma_beta2_trace_,
            "n_included_trace": self.n_included_trace_,
        }

    def get_top_snps_by_pip(self, top_n: int = 20) -> list[dict[str, float | int | str]]:
        """Return the top markers ranked by posterior inclusion probability."""

        if self.pip_ is None or self.beta_mean_ is None or self.beta2_mean_ is None:
            raise RuntimeError("The model must be fitted before requesting top SNPs.")

        feature_names = (
            self.feature_names_
            if self.feature_names_ is not None
            else [f"snp_{index}" for index in range(self.pip_.shape[0])]
        )
        top_indices = np.argsort(self.pip_)[::-1][:top_n]
        return [
            {
                "rank": rank,
                "marker_index": int(marker_index),
                "feature_name": feature_names[marker_index],
                "pip": float(self.pip_[marker_index]),
                "beta_mean": float(self.beta_mean_[marker_index]),
                "beta2_mean": float(self.beta2_mean_[marker_index]),
            }
            for rank, marker_index in enumerate(top_indices, start=1)
        ]
