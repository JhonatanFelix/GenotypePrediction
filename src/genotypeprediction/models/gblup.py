"""Dual GBLUP implementation with REML lambda estimation"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve

from genotypeprediction.data.preprocessing import GenotypeStandardizer
from genotypeprediction.evaluation.metrics import r2, pearson_corr
from genotypeprediction.inference.reml import estimate_reml_variance_components

from sklearn.exceptions import NotFittedError


class GBLUPDual:
    """Dual genomic BLUP using K = XX^T / p"""

    def __init__(
        self, validation_size: float = 0.2, random_state: int | None = None
    ) -> None:
        self.validation_size = validation_size
        self.random_state = random_state

        self.standardizer_: GenotypeStandardizer | None = None
        self.X_train_: np.ndarray | None = None
        self.K_train_: np.ndarray | None = None
        self.alpha_hat: np.ndarray | None = None
        self.lambda_g: float | None = None
        self.sigma_g2_hat: float | None = None
        self.sigma_e2_hat: float | None = None

    @staticmethod
    def _relationship_matrix(X_standardized: np.ndarray) -> np.ndarray:
        """Compute the genomic _relationship_matrix K = XX^T / p."""
        n_markers = X_standardized.shape[1]
        return (X_standardized @ X_standardized.T) / n_markers

    def _fit_standardized(
        self,
        X_standardized: np.ndarray,
        y_train: np.ndarray,
        lambda_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit the dual coeff for one fixed lambda."""

        y_centered = y_train - np.mean(y_train)
        K_train = self._relationship_matrix(X_standardized)
        alpha_hat = solve(
            K_train + lambda_value * np.eye(K_train.shape[0]),
            y_centered,
            assume_a="pos",
        )
        return K_train, alpha_hat

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        lambda_value: float | None = None,
        estimate_lambda_reml: bool = True,
        feature_names: list[str] | None = None,
    ) -> "GBLUPDual":
        """Fit dual GBLUP.

        The default path estimates =  ``lambda_g = sigma_e2 / sigma_g2`` by REML.
        """

        self.standardizer_ = GenotypeStandardizer()
        self.X_train_ = self.standardizer_.fit_transform(
            X_train, feature_names=feature_names
        )
        self.standardizer_.fit_y(y_train)
        self.K_train_ = self._relationship_matrix(self.X_train_)

        if lambda_value is not None:
            self.lambda_g = float(lambda_value)
            self.sigma_e2_hat = None
            self.sigma_g2_hat = None
        elif estimate_lambda_reml:
            reml_estimates = estimate_reml_variance_components(
                K=self.K_train_, y=np.asarray(y_train, dtype=float)
            )
            self.lambda_g = float(reml_estimates["lambda_g"])
            self.sigma_e2_hat = float(reml_estimates["sigma_e2_hat"])
            self.sigma_g2_hat = float(reml_estimates["sigma_e2_hat"])
        else:
            self.lambda_g = 1.0
            self.sigma_e2_hat = None
            self.sigma_g2_hat = None

        y_centered = self.standardizer_.center_y(y_train)
        self.alpha_hat = solve(
            self.K_train_ + self.lambda_g * np.eye(self.K_train_.shape[0]),
            y_centered,
            assume_a="pos",
        )
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict phenotypes on the original scale."""

        if (
            self.standardizer_ is None
            or self.X_train_ is None
            or self.alpha_hat is None
        ):
            raise NotFittedError("The model must be fitted before prediction.")

        X_test_standardized = self.standardizer_.transform(X_test)
        K_test_train = (X_test_standardized @ self.X_train_T) / self.X_train_.shape[1]
        y_pred_centered = K_test_train @ self.alpha_hat
        return self.standardizer_.restore_y(y_pred_centered)

    def score(
        self, X_test: np.ndarray, y_test: np.ndarray, method: str = "r2"
    ) -> float:
        """Return the out-of-sample method to measure performance.

        method: "r2", "corr"
        """
        map_performance = {"r2": r2, "corr": pearson_corr}
        if method not in map_performance.keys():
            raise ValueError("The method must be one of the options 'corr' or 'r2.")
        performance = map_performance[method]

        return performance(y_true=y_test, y_pred=self.predict(X_test))
