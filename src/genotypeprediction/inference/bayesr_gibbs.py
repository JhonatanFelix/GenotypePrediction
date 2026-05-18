
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .gibbs import _kept_sample, _validate_sampler_inputs, sample_inverse_gamma

DEFAULT_BAYESR_GAMMA = np.asarray([0.0, 1e-4, 1e-3, 1e-2], dtype=float)
DEFAULT_BAYES_ALPHA_PI = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=float)

def validate_bayesr_prior(
    gamma : np.ndarray | tuple[float, ...] | list[float],
    alpha_pi: np.ndarray | tuple[float, ...] | list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate BayesR mixture-variance and Dirichlet prior hyperparameters."""

    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    alpha_pi = np.asarray(alpha_pi, dtype=float).reshape(-1)
    if gamma.ndim != 1 or alpha_pi.ndim != 1:
        raise ValueError("gamma and alpha_pi must be one-dimensional.")
    if gamma.shape[0] < 2:
        raise ValueError("BayesR requires at least two mixture classes.")
    if gamma.shape[0] != alpha_pi.shape[0]:
        raise ValueError("gamma and alpha_pi must have the same length.")
    if not np.isclose(gamma[0], 0.0):
        raise ValueError("The first BayesR variance multiplier must be exactly zero.")
    if np.any(gamma[1:] <= 0.0):
        raise ValueError("All nonzero BayesR variance multipliers must be positive.")
    if np.any(alpha_pi <= 0.0):
        raise ValueError("All Dirichlet prior parameters alpha_pi must be positive.")
    return gamma, alpha_pi

def validate_bayesr_prior_probabilities(
    prior_probabilities: np.ndarray,
    n_markers: int | None = None,
    n_classes: int | None = None,
) -> np.ndarray:
    """Validate a SNP-by-class prior-probability matrix."""

    prior_probabilities = np.asarray(prior_probabilities, dtype=float)
    if prior_probabilities.ndim != 2:
        raise ValueError("prior_probabilities must be a 2D array of shape (p, K).")
    if n_markers is not None and prior_probabilities.shape[0] != n_markers:
        raise ValueError("prior_probabilities must have one row per SNP.")
    if n_classes is not None and prior_probabilities.shape[1] != n_classes:
        raise ValueError("prior_probabilities must have one column per BayesR class.")
    if np.any(~np.isfinite(prior_probabilities)):
        raise ValueError("prior_probabilities must be finite.")
    if np.any(prior_probabilities <= 0.0):
        raise ValueError("prior_probabilities must be strictly positive.")

    row_sums = np.sum(prior_probabilities, axis=1, keepdims=True)
    if np.any(np.isclose(row_sums, 0.0)):
        raise ValueError("Every prior-probability row must sum to a positive value.")

    normalized = prior_probabilities / row_sums
    if np.any(np.abs(np.sum(normalized, axis=1) - 1.0) > 1e-8):
        raise ValueError("prior_probabilities rows must sum to one after normalization.")
    return normalized

def compute_bayesr_sigma_beta2_scale_term(
        beta: np.ndarray,
        class_assignments: np.ndaray,
        gamma: np.ndarray,
) -> float:
    """Return ``sum(beta_j^2 / gamma_{z_j})`` for nonzero BayesR classes."""

    beta = np.asarray(beta, dtype = float).reshape(-1)
    class_assignments = np.asarray(class_assignments, dtype = int).reshape(-1)
    gamma = np.asarray(gamma, dtype = float).reshape(-1)
    if beta.shape[0] != class_assignments.shape[0]:
        raise ValueError("beta and class_assigments must have the same length")
    
    included_mask = class_assignments > 0
    if not np.any(included_mask):
        return 0.0
    
    assigned_gamma = gamma[class_assignments[included_mask]]
    return float(np.sum((beta[included_mask] ** 2) / assigned_gamma))

def _default_bayesr_class_labels(n_classes: int) -> list[str]:
    """Return readable class labels for a BayesR mixture."""

    if n_classes == 4:
        return ["zero", "small", "medium", "large"]
    return [f"class_{index}" for index in range(n_classes)]

def _initialize_bayesr_variances(
        y: np.ndarray,
        p: int,
        gamma: np.ndarray,
        average_prior_probabilities: np.ndarray,
        initial_sigma_e2: float | None, 
        initial_sigma_beta2: float | None, 
) -> tuple[float, float]:
    """Choose readable initial values for BayesR variance components."""
    
    y_variance = float(np.var(y, ddof=1)) if y.shape[0] > 1 else float(np.var(y))
    y_variance = max(y_variance, 1e-6)
    sigma_e2 = y_variance if initial_sigma_e2 is None else float(initial_sigma_e2)

    if initial_sigma_e2 is None:
        nonzero_mass = float(np.sum(average_prior_probabilities[1:]))
        if nonzero_mass > 0.0:
            average_nonzero_gamma = float(
                np.sum(average_prior_probabilities[1:] * gamma[1:] / nonzero_mass)
            )
        else:
            average_nonzero_gamma = float(np.mean(gamma[1:]))
        effective_prior_signal = max(
            p * max(nonzero_mass, 1.0 / max(p,1)),
            average_nonzero_gamma
        )
        sigma_beta2 = y_variance / max(
            effective_prior_signal * average_nonzero_gamma,
            1e-4,
        )
    else: 
        sigma_beta2 = float(initial_sigma_beta2)
    
    return max(sigma_e2, 1e-12), max(sigma_beta2, 1e-12)