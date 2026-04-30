"""SNP-BLUP implementation / Ridge"""

from __future__ import annotations # Really necessary??

import numpy as np

from GPtools.data.preprocessing import GenotypeStandardizer
from GPtools.evaluation.metrics import pearson_corr, r2

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


class RidgePrimal:
    """Primal Ridge estimator"""
    def __init__(
            self,
            validation_size: float = 0.2,
            random_state: int | None = None,
            solver: str = "auto",
            ) -> None:
        self.validation_size = validation_size
        self.random_state = random_state
        self.solver = solver

        self.beta_hat: np.ndarray | None = None
        self.lambda_value_: float | None = None
        self.selected_lambda_: float | None = None 
        self.standardizer_: GenotypeStandardizer | None = None
        self.ridge_: Ridge | None = None

    def _fit_given_lambda(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        lambda_value: float, 
        feature_names: list[str] | None = None,
    ) -> tuple[GenotypeStandardizer, Ridge]:
        "Fit a ridge model for one fixed lambda"

        standardizer = GenotypeStandardizer()
        X_standardized = standardizer.fit_transform(X_train, feature_names=feature_names)
        standardizer.fit_y(y_train)
        y_centered = standardizer.center_y(y_train)

        ridge = Ridge(
                alpha=float(lambda_value),
                fit_intercept=False,
                solver=self.solver,
                random_state=self.random_state,
                )
        ridge.fit(X_standardized, y_centered)
        return standardizer, ridge

    def _select_lambda(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        lambda_grid: list[float] | np.ndarray,
        feature_names: list[str] | None = None,) -> float:
        """Pick a lambda by validation-set mean squared error."""

        X_inner_train, X_val, y_inner_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=self.validation_size,
                random_state=self.random_state,
                )
        best_lambda = None
        best_mse = np.inf
        for lambda_candidate in lambda_grid:
            standardizer, ridge = self._fit_given_lambda(
                X_train=X_inner_train,
                y_train=y_inner_train,
                lambda_value=float(lambda_candidate),
                feature_names=feature_names,
            )
            y_val_pred = standardizer.restore_y(ridge.predict(standardizer.transform(X_val)))
            validation_mse = float(mean_squared_error(y_val, y_val_pred))
            if validation_mse < best_mse:
                best_mse = validation_mse
                best_lambda = float(lambda_candidate)
        
        if best_lambda is None:
            raise RuntimeError("Lambda selection failed.")
        return best_lambda

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            lambda_value: float | None = None,
            lambda_grid: list[float] | np.ndarray | None = None,
            feature_names: list[str] | None = None
            ) -> "RidgePrimal":

        if lambda_value is None and lambda_grid is not None:
            lambda_value = self._select_lambda(
                    X_train=X_train,
                    y_train=y_train,
                    lambda_grid=lambda_grid,
                    feature_names=feature_names,
                    )
        if lambda_value is None:
            lambda_value = 1.0

        self.standardizer_, self.ridge_ = self._fit_given_lambda(
                X_train=X_train,
                y_train=y_train,
                lambda_value=float(lambda_value),
                feature_names=feature_names,
                )
        self.beta_hat = np.asarray(self.ridge_.coef_, dtype=float).copy()
        self.lambda_value_ = float(lambda_value)
        self.selected_lambda_ = float(lambda_value)
        
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:

        if self.standardizer_ is None or self.ridge_ is None:
            raise RuntimeError("The model must be fitted before...")

        X_test_standardized = self.standardizer_.transform(X_test)
        y_pred_centered = self.ridge_.predict(X_test_standardized)
        return self.standardizer_.restore_y(y_pred_centered)

    def score(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """return the out-of-sample R-squared"""
        return r2(y_test, self.predict(X_test))

    def metric_report(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """Compute a compact regression metric report on test data."""
        predictions = self.predict(X_test)

        return {
                "mse": float(mean_squared_error(y_test, predictions)),
                "mae": float(mean_absolute_error(y_test, predictions)),
                "r2": r2(y_test, predictions),
                "pearson": pearson_corr(y_test, predictions)
                }
                
