"""Dual GBLUP implementation with REML lambda estimation"""

from __future__ import annotation

import numpy as np
from scipy.linalg import solve

from GPtools.data.preprocessing import GenotypeStandardizer
from GPtools.evaluation.metrics import r2
from GPtools.evaluation.reml import estimation_reml_variance_components

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

class GBLUPDual:
    """Dual genomic BLUP using K = XX^T / p"""

    def __init__(self, validation_size:float = 0.2, random_state: int | None = None) -> None:
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
            ) -> tuple[np.ndarray , np.ndarray]:
        """Fit the dual coeff for one fixed lambda."""

        y_centered = y_train - np.mean(y_train)
        K_train = self._relationship_matrix(X_standardized)
        alpha_hat = solve(
                K_train + lambda_value * np.eye(K_train.shape[0]),
                y_centered,
                assume_a="pos"
                )
        return K_train, alpha_hat

    
