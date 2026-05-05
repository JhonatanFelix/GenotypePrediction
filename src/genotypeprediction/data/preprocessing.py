"""..."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GenotypeStandardizer:
    """Impute, filter and standardize SNP markers using training stastistics.

    The workflow is:
        1. Compute SNP-wise means on the training set, ignoring missing values.
        2. Impute missing values with those training means.
        3. Remove markers with zero variance after imputation.
        4. Standardize the remaining markers using the training mean and standard deviation.
        5. Center the phenotype using the training mean only.

        obs.: If you don't want to standardize in this way you can skip or just do the filtering of
        missing values first.
    """

    marker_means: np.ndarray | None = field(default=None, init=False)
    marker_stds_: np.ndarray | None = field(default=None, init=False)
    keep_mask_: np.ndarray | None = field(default=None, init=False)
    kept_feature_names_: list[str] | None = field(default=None, init=False)
    y_mean_: float | None = field(default=None, init=False)

    def fit(
        self, X: np.ndarray, feature_names: list[str] | None = None
    ) -> "GenotypeStandardizer":
        """Fit imputation and scaling stastistics on training markers"""

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        contain_nan = np.any(np.isnan(X))
        if contain_nan:
            print(
                "The array X contain missing values, and they're going to be substituted by the mean of non-missing values."
            )

        marker_means = np.nanmean(X, axis=0)
        marker_means = np.where(np.isnan(marker_means), 0.0, marker_means)

        X_imputed = np.where(np.isnan(X), marker_means, X)
        marker_variances = X_imputed.var(axis=0, ddof=0)
        keep_mask = marker_variances > 0.0

        if not np.any(keep_mask):
            raise ValueError(
                "All genotype columns have zero variance after imputation."
            )

        self.marker_means_ = marker_means[keep_mask]
        self.marker_stds_ = np.sqrt(marker_variances[keep_mask])
        self.keep_mask_ = keep_mask

        if feature_names is None:
            original_names = [f"snp_{index}" for index in range(X.shape[1])]
        else:
            if len(feature_names) != X.shape[1]:
                raise ValueError(
                    "features_names must match the number of genotype columns."
                )
            original_names = list(feature_names)

        self.kept_feature_names_ = [
            feature_name
            for feature_name, keep in zip(original_names, keep_mask, strict=True)
            if keep
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply training-set imputation, filtering, and scaling to new data."""
        if (
            self.keep_mask_ is None
            or self.marker_means_ is None
            or self.marker_stds_ is None
        ):
            raise RuntimeError(
                "The standardizer must be fitted before calling transform."
            )

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if X.shape[1] != self.keep_mask_.shape[0]:
            raise ValueError(
                "X has a different number of comumns than the fitted training data."
            )

        kept = X[:, self.keep_mask_]
        kept = np.where(np.isnan(kept), self.marker_means_, kept)
        standardized = (kept - self.marker_means_) / self.marker_stds_
        return standardized

    def fit_transform(
        self, X: np.ndarray, feature_names: list[str] | None = None
    ) -> np.ndarray:
        """Fit on the training data and return the standardized markers."""
        return self.fit(X, feature_names=feature_names).transform(X)

    def fit_y(self, y: np.ndarray) -> "GenotypeStandardizer":
        """Fit on the training data and return the standardized markers"""
        y = np.asarray(y, dtype=float)
        self.y_mean_ = float(np.mean(y))
        return self

    def center_y(self, y: np.ndarray) -> np.ndarray:
        """Center a phenotype vector using the stored training-set mean."""

        if self.y_mean_ is None:
            raise RuntimeError(
                "The phenotype mean is not available. Call fit_y first please"
            )
        y = np.asarray(y, dtype=float)
        return y - self.y_mean_

    def restore_y(self, y_centered: np.ndarray) -> np.ndarray:
        """Undo phenotype centering."""
        if self.y_mean_ is None:
            raise RuntimeError("The phenotype mean is not available. Call fit_y first.")
        y_centered = np.asarray(y_centered, dtype=float)
        return y_centered + self.y_mean_
