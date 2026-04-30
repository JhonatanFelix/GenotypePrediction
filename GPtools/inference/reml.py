"""REML estimation"""

from __future__ import annotations

import numpy as np 
from scipy.linalg import solve
from scipy.optimize import minimize_scaler

def _prepare_fixed_effects(y: np.ndarray, X_fixed: np.ndarray | None) -> np.ndarray:
    """Return a fixed-effect design matrix, default to an intercept only."""

    if X_fixed is None:
        return np.ones((y.shape[0],1), dtype=float)

    X_fixed = np.asarray(X_fixed, dtype=float)
    if X_fixed.ndim == 1:
        X_fixed = X_fixed[:, None]
    return X_fixed

def _profile_reml_components(
        eigenvalues: np.ndarray,
        eigenvectors: np.ndarray,
        y: np.ndarray,
        X_fixed: np.ndarray,
        delta: float,
        ) -> tuple[float, float, float, float]:
    """Compute the ingredients of REML profile likelihood for a given delta."""

    h = eigenvalues + delta
    if np.any(h <= 0.0):
        return np.inf, np.nan, np.nan, np.nan

    inv_h = 1.0 / h
    uy = eigenvectors.T @ y
    ux = eigenvectors.T @ X_fixed

    h_inv_y = eigenvectors @ (inv_h * uy)
    h_inv_x = eigenvectors @ (inv_h[:, None] * ux)
    xt_hinv_x = X_fixed.T @ h_inv_x

    sign, log_det_xt_hinv_x = np.linalg.slogdet(xt_hinv_x)
    if sign <= 0.0:
        return np.inf, np.inf, np.inf, np.inf

    beta_hat = solve(xt_hinv_x, X_fixed.T @ h_inv_y, assume_a="sym")
    projected_residual = h_inv_y - h_inv_x @ beta_hat
    quad_form = float(y.T @ projected_residual)

    dof = y.shape[0] - X_fixed.shape[1]
    sigma_g2 = max(quad_form / dof, 1e-12)
    log_det_h = float(np.sum(np.log(h)))
    neg_log_reml = 0.5 * (dof * np.log(sigma_g2) + log_det_h + log_det_xt_hinv_x)

    return neg_log_reml, sigma_g2, quad_form, log_det_h


def estimate_reml_variance_components(
    K: np.ndarray,
    y: np.ndarray,
    X_fixed: np.ndarray | None = None,
    log_delta_bounds: tuple[float, float] = (-10.0, 10.0),
) -> dict[str, float]:
    """Estimate ``sigma_g2`` and ``sigma_e2`` for GBLUP via REML.

    The covariance model is

    ``V = sigma_g2 * (K + delta * I)``

    where ``delta = sigma_e2 / sigma_g2``. The restricted likelihood is profiled
    over ``sigma_g2`` and optimized over ``log(delta)``.
    """

    K = np.asarray(K, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    X_fixed = _prepare_fixed_effects(y, X_fixed)

    if K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square relationship matrix.")
    if K.shape[0] != y.shape[0]:
        raise ValueError("K and y must have compatible dimensions.")

    eigenvalues, eigenvectors = np.linalg.eigh(K)
    eigenvalues = np.clip(eigenvalues, 0.0, None)

    def objective(log_delta: float) -> float:
        delta = float(np.exp(log_delta))
        neg_log_reml, _, _, _ = _profile_reml_components(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            y=y,
            X_fixed=X_fixed,
            delta=delta,
        )
        return neg_log_reml

    result = minimize_scalar(
        objective,
        bounds=log_delta_bounds,
        method="bounded",
    )
    if not result.success:
        raise RuntimeError(f"REML optimization failed: {result.message}")

    delta_hat = float(np.exp(result.x))
    _, sigma_g2_hat, _, _ = _profile_reml_components(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        y=y,
        X_fixed=X_fixed,
        delta=delta_hat,
    )
    sigma_e2_hat = delta_hat * sigma_g2_hat

    return {
        "sigma_g2_hat": float(sigma_g2_hat),
        "sigma_e2_hat": float(sigma_e2_hat),
        "lambda_g": float(delta_hat),
        "log_delta_hat": float(result.x),
        "objective_value": float(result.fun),
    }

