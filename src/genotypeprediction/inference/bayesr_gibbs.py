"""Gibbs sampling helpers for BayesR, BayesRC, and fixed-prior BayesR variants."""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .gibbs import _kept_sample, _validate_sampler_inputs, sample_inverse_gamma


DEFAULT_BAYESR_GAMMA = np.asarray([0.0, 1e-4, 1e-3, 1e-2], dtype=float)
DEFAULT_BAYESR_ALPHA_PI = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=float)


def validate_bayesr_prior(
    gamma: np.ndarray | tuple[float, ...] | list[float],
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
        raise ValueError(
            "prior_probabilities rows must sum to one after normalization."
        )
    return normalized


def compute_bayesr_sigma_beta2_scale_term(
    beta: np.ndarray,
    class_assignments: np.ndarray,
    gamma: np.ndarray,
) -> float:
    """Return ``sum(beta_j^2 / gamma_{z_j})`` for nonzero BayesR classes."""

    beta = np.asarray(beta, dtype=float).reshape(-1)
    class_assignments = np.asarray(class_assignments, dtype=int).reshape(-1)
    gamma = np.asarray(gamma, dtype=float).reshape(-1)
    if beta.shape[0] != class_assignments.shape[0]:
        raise ValueError("beta and class_assignments must have the same length.")

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

    if initial_sigma_beta2 is None:
        nonzero_mass = float(np.sum(average_prior_probabilities[1:]))
        if nonzero_mass > 0.0:
            average_nonzero_gamma = float(
                np.sum(average_prior_probabilities[1:] * gamma[1:]) / nonzero_mass
            )
        else:
            average_nonzero_gamma = float(np.mean(gamma[1:]))
        effective_prior_signal = max(
            p * max(nonzero_mass, 1.0 / max(p, 1)),
            average_nonzero_gamma,
        )
        sigma_beta2 = y_variance / max(
            effective_prior_signal * average_nonzero_gamma,
            1e-4,
        )
    else:
        sigma_beta2 = float(initial_sigma_beta2)

    return max(sigma_e2, 1e-12), max(sigma_beta2, 1e-12)


def _sample_bayesr_sweep(
    X: np.ndarray,
    residual: np.ndarray,
    beta: np.ndarray,
    class_assignments: np.ndarray,
    prior_probabilities: np.ndarray,
    sigma_e2: float,
    sigma_beta2: float,
    gamma: np.ndarray,
    x_squared_norms: np.ndarray,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Run one residual-update BayesR sweep for fixed SNP-level class priors."""

    n_classes = gamma.shape[0]
    log_weights = np.empty(n_classes, dtype=float)
    conditional_means = np.zeros(n_classes, dtype=float)
    conditional_variances = np.zeros(n_classes, dtype=float)
    posterior_class_probabilities = np.empty((X.shape[1], n_classes), dtype=float)
    posterior_beta_mean = np.zeros(X.shape[1], dtype=float)
    posterior_beta2_mean = np.zeros(X.shape[1], dtype=float)
    posterior_snp_variance_mean = np.zeros(X.shape[1], dtype=float)

    for marker_index in range(X.shape[1]):
        x_j = X[:, marker_index]
        old_beta = beta[marker_index]
        if old_beta != 0.0:
            residual = residual + x_j * old_beta

        log_prior = np.log(
            np.clip(prior_probabilities[marker_index], np.finfo(float).tiny, None)
        )
        log_weights[0] = log_prior[0]
        conditional_means[0] = 0.0
        conditional_variances[0] = 0.0

        x_residual_inner = float(x_j @ residual)
        x_norm = float(x_squared_norms[marker_index])

        for class_index in range(1, n_classes):
            tau_k2 = gamma[class_index] * sigma_beta2
            v_jk = 1.0 / (x_norm / sigma_e2 + 1.0 / tau_k2)
            m_jk = v_jk * x_residual_inner / sigma_e2
            conditional_means[class_index] = m_jk
            conditional_variances[class_index] = v_jk
            log_weights[class_index] = (
                log_prior[class_index]
                + 0.5 * (np.log(v_jk) - np.log(tau_k2))
                + (m_jk**2) / (2.0 * v_jk)
            )

        marker_class_probabilities = np.exp(log_weights - logsumexp(log_weights))
        posterior_class_probabilities[marker_index] = marker_class_probabilities
        posterior_beta_mean[marker_index] = float(
            marker_class_probabilities[1:] @ conditional_means[1:]
        )
        posterior_beta2_mean[marker_index] = float(
            marker_class_probabilities[1:]
            @ (conditional_variances[1:] + conditional_means[1:] ** 2)
        )
        posterior_snp_variance_mean[marker_index] = float(
            marker_class_probabilities @ (gamma * sigma_beta2)
        )

        new_class = int(rng.choice(n_classes, p=marker_class_probabilities))
        class_assignments[marker_index] = new_class

        if new_class == 0:
            beta[marker_index] = 0.0
        else:
            beta[marker_index] = float(
                rng.normal(
                    loc=conditional_means[new_class],
                    scale=np.sqrt(conditional_variances[new_class]),
                )
            )

        residual = residual - x_j * beta[marker_index]

    return (
        residual,
        beta,
        class_assignments,
        posterior_class_probabilities,
        posterior_beta_mean,
        posterior_beta2_mean,
        posterior_snp_variance_mean,
    )


def _update_bayesr_variances(
    residual: np.ndarray,
    beta: np.ndarray,
    class_assignments: np.ndarray,
    gamma: np.ndarray,
    a_e: float,
    b_e: float,
    a_beta: float,
    b_beta: float,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Sample ``sigma_e2`` and ``sigma_beta2`` after a full BayesR sweep."""

    n = residual.shape[0]
    n_nonzero = int(np.sum(class_assignments > 0))
    residual_sum_squares = float(residual @ residual)
    sigma_e2 = sample_inverse_gamma(
        rng=rng,
        shape=a_e + 0.5 * n,
        scale=b_e + 0.5 * residual_sum_squares,
    )
    sigma_e2 = max(sigma_e2, 1e-12)

    if n_nonzero > 0:
        beta_scale_sum = compute_bayesr_sigma_beta2_scale_term(
            beta=beta,
            class_assignments=class_assignments,
            gamma=gamma,
        )
        sigma_beta2 = sample_inverse_gamma(
            rng=rng,
            shape=a_beta + 0.5 * n_nonzero,
            scale=b_beta + 0.5 * beta_scale_sum,
        )
    else:
        sigma_beta2 = sample_inverse_gamma(
            rng=rng,
            shape=a_beta,
            scale=b_beta,
        )
    sigma_beta2 = max(sigma_beta2, 1e-12)
    return sigma_e2, sigma_beta2, n_nonzero


def _initialize_posterior_accumulator(
    p: int, n_classes: int
) -> dict[str, np.ndarray | int]:
    """Create posterior running sums shared by BayesR-class samplers."""

    return {
        "beta_sum": np.zeros(p, dtype=float),
        "beta2_sum": np.zeros(p, dtype=float),
        "class_probability_sum": np.zeros((p, n_classes), dtype=float),
        "posterior_snp_variance_sum": np.zeros(p, dtype=float),
        "sample_count": 0,
    }


def _store_bayesr_sample(
    accumulator: dict[str, np.ndarray | int],
    posterior_class_probabilities: np.ndarray,
    posterior_beta_mean: np.ndarray,
    posterior_beta2_mean: np.ndarray,
    posterior_snp_variance_mean: np.ndarray,
) -> None:
    """Update posterior running sums from one retained Gibbs sample.

    The stored moments are Rao-Blackwellized using the marker-level full
    conditional probabilities from the Gibbs sweep, which substantially reduces
    Monte Carlo noise in PIPs and posterior mixture summaries.
    """

    accumulator["beta_sum"] += posterior_beta_mean
    accumulator["beta2_sum"] += posterior_beta2_mean
    accumulator["class_probability_sum"] += posterior_class_probabilities
    accumulator["posterior_snp_variance_sum"] += posterior_snp_variance_mean
    accumulator["sample_count"] = int(accumulator["sample_count"]) + 1


def _finalize_bayesr_posterior(
    accumulator: dict[str, np.ndarray | int],
    n_classes: int,
    class_labels: list[str],
    gamma: np.ndarray,
) -> dict[str, np.ndarray | float | int | list[str]]:
    """Convert posterior running sums into posterior means."""

    sample_count = int(accumulator["sample_count"])
    if sample_count == 0:
        raise RuntimeError("No posterior samples were stored. Check burn-in and thin.")

    class_probabilities = (
        np.asarray(accumulator["class_probability_sum"], dtype=float) / sample_count
    )
    return {
        "beta_mean": np.asarray(accumulator["beta_sum"], dtype=float) / sample_count,
        "beta2_mean": np.asarray(accumulator["beta2_sum"], dtype=float) / sample_count,
        "class_probabilities": class_probabilities,
        "posterior_expected_class": class_probabilities
        @ np.arange(n_classes, dtype=float),
        "pip": 1.0 - class_probabilities[:, 0],
        "posterior_snp_variance_mean": (
            np.asarray(accumulator["posterior_snp_variance_sum"], dtype=float)
            / sample_count
        ),
        "posterior_sample_count": sample_count,
        "class_labels": class_labels,
        "gamma": gamma.copy(),
    }


def _compute_category_class_counts(
    category_codes: np.ndarray,
    class_assignments: np.ndarray,
    n_categories: int,
    n_classes: int,
) -> np.ndarray:
    """Count class assignments within each annotation category."""

    counts = np.zeros((n_categories, n_classes), dtype=int)
    for category_index in range(n_categories):
        category_mask = category_codes == category_index
        counts[category_index] = np.bincount(
            class_assignments[category_mask],
            minlength=n_classes,
        )
    return counts


def run_bayesr_fixed_prior_gibbs(
    X: np.ndarray,
    y: np.ndarray,
    prior_probabilities: np.ndarray,
    n_iter: int,
    burn_in: int,
    thin: int,
    gamma: np.ndarray | tuple[float, ...] | list[float] = DEFAULT_BAYESR_GAMMA,
    a_e: float = 1e-3,
    b_e: float = 1e-3,
    a_beta: float = 1e-3,
    b_beta: float = 1e-3,
    initial_sigma_e2: float | None = None,
    initial_sigma_beta2: float | None = None,
    random_state: int | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray | float | int | list[str]]:
    """Run BayesR with a fixed SNP-specific prior probability matrix."""

    X, y = _validate_sampler_inputs(X=X, y=y, n_iter=n_iter, burn_in=burn_in, thin=thin)
    gamma, alpha_pi = validate_bayesr_prior(
        gamma=gamma, alpha_pi=np.ones(len(gamma), dtype=float)
    )
    del alpha_pi

    n, p = X.shape
    n_classes = gamma.shape[0]
    class_labels = _default_bayesr_class_labels(n_classes)
    prior_probabilities = validate_bayesr_prior_probabilities(
        prior_probabilities=prior_probabilities,
        n_markers=p,
        n_classes=n_classes,
    )

    rng = np.random.default_rng(random_state)
    beta = np.zeros(p, dtype=float)
    class_assignments = np.zeros(p, dtype=int)
    residual = y.copy()
    sigma_e2, sigma_beta2 = _initialize_bayesr_variances(
        y=y,
        p=p,
        gamma=gamma,
        average_prior_probabilities=np.mean(prior_probabilities, axis=0),
        initial_sigma_e2=initial_sigma_e2,
        initial_sigma_beta2=initial_sigma_beta2,
    )
    x_squared_norms = np.sum(X * X, axis=0)
    accumulator = _initialize_posterior_accumulator(p=p, n_classes=n_classes)

    sigma_e2_trace: list[float] = []
    sigma_beta2_trace: list[float] = []
    n_nonzero_trace: list[int] = []
    progress_every = max(1, n_iter // 10)

    for iteration in range(1, n_iter + 1):
        (
            residual,
            beta,
            class_assignments,
            posterior_class_probabilities,
            posterior_beta_mean,
            posterior_beta2_mean,
            posterior_snp_variance_mean,
        ) = _sample_bayesr_sweep(
            X=X,
            residual=residual,
            beta=beta,
            class_assignments=class_assignments,
            prior_probabilities=prior_probabilities,
            sigma_e2=sigma_e2,
            sigma_beta2=sigma_beta2,
            gamma=gamma,
            x_squared_norms=x_squared_norms,
            rng=rng,
        )
        sigma_e2, sigma_beta2, n_nonzero = _update_bayesr_variances(
            residual=residual,
            beta=beta,
            class_assignments=class_assignments,
            gamma=gamma,
            a_e=a_e,
            b_e=b_e,
            a_beta=a_beta,
            b_beta=b_beta,
            rng=rng,
        )

        if _kept_sample(iteration=iteration, burn_in=burn_in, thin=thin):
            _store_bayesr_sample(
                accumulator=accumulator,
                posterior_class_probabilities=posterior_class_probabilities,
                posterior_beta_mean=posterior_beta_mean,
                posterior_beta2_mean=posterior_beta2_mean,
                posterior_snp_variance_mean=posterior_snp_variance_mean,
            )
            sigma_e2_trace.append(sigma_e2)
            sigma_beta2_trace.append(sigma_beta2)
            n_nonzero_trace.append(n_nonzero)

        if verbose and (
            iteration == 1 or iteration % progress_every == 0 or iteration == n_iter
        ):
            mean_prior = np.mean(prior_probabilities, axis=0)
            prior_preview = ", ".join(
                f"{label}={value:.3f}"
                for label, value in zip(class_labels, mean_prior, strict=True)
            )
            print(
                f"[BayesRFixedPrior] iteration={iteration:>5d} "
                f"n_nonzero={n_nonzero:>5d} "
                f"sigma_e2={sigma_e2:.4f} sigma_beta2={sigma_beta2:.4f} "
                f"mean_prior=[{prior_preview}]"
            )

    posterior = _finalize_bayesr_posterior(
        accumulator=accumulator,
        n_classes=n_classes,
        class_labels=class_labels,
        gamma=gamma,
    )
    posterior.update(
        {
            "sigma_e2_trace": np.asarray(sigma_e2_trace, dtype=float),
            "sigma_beta2_trace": np.asarray(sigma_beta2_trace, dtype=float),
            "n_nonzero_trace": np.asarray(n_nonzero_trace, dtype=int),
            "prior_probabilities": prior_probabilities.copy(),
        }
    )
    return posterior


def run_bayesr_gibbs(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int,
    burn_in: int,
    thin: int,
    gamma: np.ndarray | tuple[float, ...] | list[float] = DEFAULT_BAYESR_GAMMA,
    alpha_pi: np.ndarray | tuple[float, ...] | list[float] = DEFAULT_BAYESR_ALPHA_PI,
    a_e: float = 1e-3,
    b_e: float = 1e-3,
    a_beta: float = 1e-3,
    b_beta: float = 1e-3,
    initial_sigma_e2: float | None = None,
    initial_sigma_beta2: float | None = None,
    initial_pi: np.ndarray | None = None,
    random_state: int | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray | float | int | list[str]]:
    """Run global BayesR with one shared mixture-probability vector ``pi``."""

    X, y = _validate_sampler_inputs(X=X, y=y, n_iter=n_iter, burn_in=burn_in, thin=thin)
    gamma, alpha_pi = validate_bayesr_prior(gamma=gamma, alpha_pi=alpha_pi)

    n, p = X.shape
    n_classes = gamma.shape[0]
    class_labels = _default_bayesr_class_labels(n_classes)

    if initial_pi is None:
        pi = alpha_pi / np.sum(alpha_pi)
    else:
        pi = np.asarray(initial_pi, dtype=float).reshape(-1)
        if pi.shape[0] != n_classes:
            raise ValueError("initial_pi must have the same length as gamma.")
        if np.any(pi <= 0.0):
            raise ValueError("initial_pi must be strictly positive.")
        pi = pi / np.sum(pi)

    rng = np.random.default_rng(random_state)
    beta = np.zeros(p, dtype=float)
    class_assignments = np.zeros(p, dtype=int)
    residual = y.copy()
    sigma_e2, sigma_beta2 = _initialize_bayesr_variances(
        y=y,
        p=p,
        gamma=gamma,
        average_prior_probabilities=pi,
        initial_sigma_e2=initial_sigma_e2,
        initial_sigma_beta2=initial_sigma_beta2,
    )
    x_squared_norms = np.sum(X * X, axis=0)
    accumulator = _initialize_posterior_accumulator(p=p, n_classes=n_classes)

    pi_trace: list[np.ndarray] = []
    class_count_trace: list[np.ndarray] = []
    sigma_e2_trace: list[float] = []
    sigma_beta2_trace: list[float] = []
    n_nonzero_trace: list[int] = []
    progress_every = max(1, n_iter // 10)

    for iteration in range(1, n_iter + 1):
        prior_probabilities = np.repeat(pi.reshape(1, -1), p, axis=0)
        (
            residual,
            beta,
            class_assignments,
            posterior_class_probabilities,
            posterior_beta_mean,
            posterior_beta2_mean,
            posterior_snp_variance_mean,
        ) = _sample_bayesr_sweep(
            X=X,
            residual=residual,
            beta=beta,
            class_assignments=class_assignments,
            prior_probabilities=prior_probabilities,
            sigma_e2=sigma_e2,
            sigma_beta2=sigma_beta2,
            gamma=gamma,
            x_squared_norms=x_squared_norms,
            rng=rng,
        )

        class_counts = np.bincount(class_assignments, minlength=n_classes)
        pi = rng.dirichlet(alpha_pi + class_counts)
        sigma_e2, sigma_beta2, n_nonzero = _update_bayesr_variances(
            residual=residual,
            beta=beta,
            class_assignments=class_assignments,
            gamma=gamma,
            a_e=a_e,
            b_e=b_e,
            a_beta=a_beta,
            b_beta=b_beta,
            rng=rng,
        )

        if _kept_sample(iteration=iteration, burn_in=burn_in, thin=thin):
            _store_bayesr_sample(
                accumulator=accumulator,
                posterior_class_probabilities=posterior_class_probabilities,
                posterior_beta_mean=posterior_beta_mean,
                posterior_beta2_mean=posterior_beta2_mean,
                posterior_snp_variance_mean=posterior_snp_variance_mean,
            )
            pi_trace.append(pi.copy())
            class_count_trace.append(class_counts.copy())
            sigma_e2_trace.append(sigma_e2)
            sigma_beta2_trace.append(sigma_beta2)
            n_nonzero_trace.append(n_nonzero)

        if verbose and (
            iteration == 1 or iteration % progress_every == 0 or iteration == n_iter
        ):
            pi_preview = ", ".join(
                f"{label}={value:.3f}"
                for label, value in zip(
                    class_labels[: min(n_classes, 4)],
                    pi[: min(n_classes, 4)],
                    strict=True,
                )
            )
            print(
                f"[BayesR] iteration={iteration:>5d} "
                f"n_nonzero={n_nonzero:>5d} "
                f"sigma_e2={sigma_e2:.4f} sigma_beta2={sigma_beta2:.4f} "
                f"pi=[{pi_preview}]"
            )

    posterior = _finalize_bayesr_posterior(
        accumulator=accumulator,
        n_classes=n_classes,
        class_labels=class_labels,
        gamma=gamma,
    )
    posterior.update(
        {
            "pi_trace": np.asarray(pi_trace, dtype=float),
            "class_count_trace": np.asarray(class_count_trace, dtype=int),
            "sigma_e2_trace": np.asarray(sigma_e2_trace, dtype=float),
            "sigma_beta2_trace": np.asarray(sigma_beta2_trace, dtype=float),
            "n_nonzero_trace": np.asarray(n_nonzero_trace, dtype=int),
        }
    )
    return posterior


def run_bayesrc_gibbs(
    X: np.ndarray,
    y: np.ndarray,
    category_codes: np.ndarray,
    n_iter: int,
    burn_in: int,
    thin: int,
    gamma: np.ndarray | tuple[float, ...] | list[float] = DEFAULT_BAYESR_GAMMA,
    alpha_pi: np.ndarray | tuple[float, ...] | list[float] = DEFAULT_BAYESR_ALPHA_PI,
    a_e: float = 1e-3,
    b_e: float = 1e-3,
    a_beta: float = 1e-3,
    b_beta: float = 1e-3,
    initial_sigma_e2: float | None = None,
    initial_sigma_beta2: float | None = None,
    initial_pi_by_category: np.ndarray | None = None,
    random_state: int | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray | float | int | list[str]]:
    """Run BayesRC with one class-probability vector ``pi_c`` per annotation category."""

    X, y = _validate_sampler_inputs(X=X, y=y, n_iter=n_iter, burn_in=burn_in, thin=thin)
    gamma, alpha_pi = validate_bayesr_prior(gamma=gamma, alpha_pi=alpha_pi)

    n, p = X.shape
    n_classes = gamma.shape[0]
    class_labels = _default_bayesr_class_labels(n_classes)
    category_codes = np.asarray(category_codes, dtype=int).reshape(-1)
    if category_codes.shape[0] != p:
        raise ValueError("category_codes must have one entry per SNP.")
    if np.any(category_codes < 0):
        raise ValueError("category_codes must be non-negative integers.")

    n_categories = int(np.max(category_codes)) + 1
    if initial_pi_by_category is None:
        pi_by_category = np.repeat(
            (alpha_pi / np.sum(alpha_pi)).reshape(1, -1),
            n_categories,
            axis=0,
        )
    else:
        pi_by_category = np.asarray(initial_pi_by_category, dtype=float)
        if pi_by_category.shape != (n_categories, n_classes):
            raise ValueError(
                "initial_pi_by_category must have shape (n_categories, n_classes)."
            )
        pi_by_category = validate_bayesr_prior_probabilities(
            prior_probabilities=pi_by_category,
            n_markers=n_categories,
            n_classes=n_classes,
        )

    rng = np.random.default_rng(random_state)
    beta = np.zeros(p, dtype=float)
    class_assignments = np.zeros(p, dtype=int)
    residual = y.copy()
    sigma_e2, sigma_beta2 = _initialize_bayesr_variances(
        y=y,
        p=p,
        gamma=gamma,
        average_prior_probabilities=np.mean(pi_by_category, axis=0),
        initial_sigma_e2=initial_sigma_e2,
        initial_sigma_beta2=initial_sigma_beta2,
    )
    x_squared_norms = np.sum(X * X, axis=0)
    accumulator = _initialize_posterior_accumulator(p=p, n_classes=n_classes)

    pi_by_category_trace: list[np.ndarray] = []
    category_class_count_trace: list[np.ndarray] = []
    sigma_e2_trace: list[float] = []
    sigma_beta2_trace: list[float] = []
    n_nonzero_trace: list[int] = []
    n_nonzero_by_category_trace: list[np.ndarray] = []
    progress_every = max(1, n_iter // 10)

    for iteration in range(1, n_iter + 1):
        prior_probabilities = pi_by_category[category_codes]
        (
            residual,
            beta,
            class_assignments,
            posterior_class_probabilities,
            posterior_beta_mean,
            posterior_beta2_mean,
            posterior_snp_variance_mean,
        ) = _sample_bayesr_sweep(
            X=X,
            residual=residual,
            beta=beta,
            class_assignments=class_assignments,
            prior_probabilities=prior_probabilities,
            sigma_e2=sigma_e2,
            sigma_beta2=sigma_beta2,
            gamma=gamma,
            x_squared_norms=x_squared_norms,
            rng=rng,
        )

        category_class_counts = _compute_category_class_counts(
            category_codes=category_codes,
            class_assignments=class_assignments,
            n_categories=n_categories,
            n_classes=n_classes,
        )
        for category_index in range(n_categories):
            pi_by_category[category_index] = rng.dirichlet(
                alpha_pi + category_class_counts[category_index]
            )

        sigma_e2, sigma_beta2, n_nonzero = _update_bayesr_variances(
            residual=residual,
            beta=beta,
            class_assignments=class_assignments,
            gamma=gamma,
            a_e=a_e,
            b_e=b_e,
            a_beta=a_beta,
            b_beta=b_beta,
            rng=rng,
        )

        if _kept_sample(iteration=iteration, burn_in=burn_in, thin=thin):
            _store_bayesr_sample(
                accumulator=accumulator,
                posterior_class_probabilities=posterior_class_probabilities,
                posterior_beta_mean=posterior_beta_mean,
                posterior_beta2_mean=posterior_beta2_mean,
                posterior_snp_variance_mean=posterior_snp_variance_mean,
            )
            pi_by_category_trace.append(pi_by_category.copy())
            category_class_count_trace.append(category_class_counts.copy())
            sigma_e2_trace.append(sigma_e2)
            sigma_beta2_trace.append(sigma_beta2)
            n_nonzero_trace.append(n_nonzero)
            n_nonzero_by_category_trace.append(
                np.sum(category_class_counts[:, 1:], axis=1)
            )

        if verbose and (
            iteration == 1 or iteration % progress_every == 0 or iteration == n_iter
        ):
            mean_nonzero_by_category = 1.0 - pi_by_category[:, 0]
            print(
                f"[BayesRC] iteration={iteration:>5d} "
                f"n_nonzero={n_nonzero:>5d} "
                f"sigma_e2={sigma_e2:.4f} sigma_beta2={sigma_beta2:.4f} "
                f"mean_nonzero_prior={float(np.mean(mean_nonzero_by_category)):.4f}"
            )

    posterior = _finalize_bayesr_posterior(
        accumulator=accumulator,
        n_classes=n_classes,
        class_labels=class_labels,
        gamma=gamma,
    )
    posterior.update(
        {
            "pi_by_category_trace": np.asarray(pi_by_category_trace, dtype=float),
            "category_class_count_trace": np.asarray(
                category_class_count_trace, dtype=int
            ),
            "n_nonzero_by_category_trace": np.asarray(
                n_nonzero_by_category_trace, dtype=int
            ),
            "sigma_e2_trace": np.asarray(sigma_e2_trace, dtype=float),
            "sigma_beta2_trace": np.asarray(sigma_beta2_trace, dtype=float),
            "n_nonzero_trace": np.asarray(n_nonzero_trace, dtype=int),
        }
    )
    return posterior
