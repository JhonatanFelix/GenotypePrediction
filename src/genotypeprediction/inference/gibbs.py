"""Gibbs sampling helpers for BayesC-style spike-and-slab models."""

from __future__ import annotations

import numpy as np
from scipy.special import expit


def sample_inverse_gamma(rng: np.random.Generator, shape: float, scale: float) -> float:
    """Sample from an inverse-gamma distribution with rate-like scale.

    This uses the identity

    ``X ~ InvGamma(shape, scale)``  iff  ``1 / X ~ Gamma(shape, rate=scale)``.
    """

    gamma_sample = rng.gamma(shape=shape, scale=1.0 / scale)
    gamma_sample = max(float(gamma_sample), np.finfo(float).tiny)
    return float(1.0 / gamma_sample)


def _kept_sample(iteration: int, burn_in: int, thin: int) -> bool:
    """Return whether the current iteration should be stored."""

    if iteration <= burn_in:
        return False
    return (iteration - burn_in) % thin == 0


def _validate_sampler_inputs(
    X: np.ndarray, y: np.ndarray, n_iter: int, burn_in: int, thin: int
) -> tuple[np.ndarray, np.ndarray]:
    """Validate basic sampler inputs and return float arrays."""

    if n_iter <= burn_in:
        raise ValueError("n_iter must be larger than burn_in.")
    if thin <= 0:
        raise ValueError("thin must be a positive integer.")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if y.shape[0] != X.shape[0]:
        raise ValueError("X and y must have compatible dimensions.")

    return X, y


def _initialize_variances(
    y: np.ndarray,
    p: int,
    q: float,
    initial_sigma_e2: float | None,
    initial_sigma_beta2: float | None,
) -> tuple[float, float]:
    """Choose readable default initial values for variance components."""

    y_variance = float(np.var(y, ddof=1)) if y.shape[0] > 1 else float(np.var(y))
    y_variance = max(y_variance, 1e-6)

    sigma_e2 = y_variance if initial_sigma_e2 is None else float(initial_sigma_e2)
    if initial_sigma_beta2 is None:
        sigma_beta2 = y_variance / max(q * p, 1.0)
    else:
        sigma_beta2 = float(initial_sigma_beta2)

    sigma_e2 = max(sigma_e2, 1e-12)
    sigma_beta2 = max(sigma_beta2, 1e-12)
    return sigma_e2, sigma_beta2


def _safe_q_for_logs(q: float) -> float:
    """Clip ``q`` only for numerical log calculations."""

    return float(np.clip(q, 1e-12, 1.0 - 1e-12))


def _run_bayesc_gibbs(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int,
    burn_in: int,
    thin: int,
    a_e: float,
    b_e: float,
    a_beta: float,
    b_beta: float,
    random_state: int | None,
    fixed_q: float | None = None,
    a_q: float | None = None,
    b_q: float | None = None,
    initial_q: float | None = None,
    initial_sigma_e2: float | None = None,
    initial_sigma_beta2: float | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray | float | int]:
    """Shared Gibbs sampler for BayesC fixed-q and BayesCpi.

    If ``fixed_q`` is provided, the sampler behaves like BayesC with a fixed
    inclusion probability. Otherwise it behaves like BayesCpi and samples
    ``q | delta ~ Beta(a_q + m, b_q + p - m)`` after each full SNP sweep.
    """

    X, y = _validate_sampler_inputs(X=X, y=y, n_iter=n_iter, burn_in=burn_in, thin=thin)
    n, p = X.shape
    rng = np.random.default_rng(random_state)

    if fixed_q is not None:
        if not 0.0 < fixed_q < 1.0:
            raise ValueError("fixed_q must lie strictly between 0 and 1.")
        q = float(fixed_q)
        learning_q = False
    else:
        if a_q is None or b_q is None or initial_q is None:
            raise ValueError("BayesCpi requires a_q, b_q, and initial_q.")
        if a_q <= 0.0 or b_q <= 0.0:
            raise ValueError("a_q and b_q must be positive.")
        if not 0.0 < initial_q < 1.0:
            raise ValueError("initial_q must lie strictly between 0 and 1.")
        q = float(initial_q)
        learning_q = True

    beta = np.zeros(p, dtype=float)
    delta = np.zeros(p, dtype=int)
    residual = y.copy()
    sigma_e2, sigma_beta2 = _initialize_variances(
        y=y,
        p=p,
        q=q,
        initial_sigma_e2=initial_sigma_e2,
        initial_sigma_beta2=initial_sigma_beta2,
    )
    x_squared_norms = np.sum(X * X, axis=0)

    beta_sum = np.zeros(p, dtype=float)
    beta2_sum = np.zeros(p, dtype=float)
    delta_sum = np.zeros(p, dtype=float)
    sample_count = 0

    q_trace: list[float] = []
    sigma_e2_trace: list[float] = []
    sigma_beta2_trace: list[float] = []
    n_included_trace: list[int] = []

    progress_every = max(1, n_iter // 10)

    for iteration in range(1, n_iter + 1):
        for marker_index in range(p):
            x_j = X[:, marker_index]

            # Add back the current effect so the residual excludes all markers
            # except the one being updated.
            if beta[marker_index] != 0.0:
                residual = residual + x_j * beta[marker_index]

            v_j = 1.0 / (x_squared_norms[marker_index] / sigma_e2 + 1.0 / sigma_beta2)
            m_j = v_j * (x_j @ residual) / sigma_e2

            q_safe = _safe_q_for_logs(q)
            log_w1 = (
                np.log(q_safe)
                + 0.5 * (np.log(v_j) - np.log(sigma_beta2))
                + (m_j**2) / (2.0 * v_j)
            )
            log_w0 = np.log1p(-q_safe)
            p_inclusion = float(expit(log_w1 - log_w0))

            delta[marker_index] = int(rng.binomial(1, p_inclusion))
            if delta[marker_index] == 1:
                beta[marker_index] = float(rng.normal(loc=m_j, scale=np.sqrt(v_j)))
            else:
                beta[marker_index] = 0.0

            residual = residual - x_j * beta[marker_index]

        included_mask = delta == 1
        n_included = int(np.sum(included_mask))

        if learning_q:
            q = float(rng.beta(a_q + n_included, b_q + p - n_included))

        residual_sum_squares = float(residual @ residual)
        sigma_e2 = sample_inverse_gamma(
            rng=rng,
            shape=a_e + 0.5 * n,
            scale=b_e + 0.5 * residual_sum_squares,
        )
        sigma_e2 = max(sigma_e2, 1e-12)

        if n_included > 0:
            beta_sum_squares = float(np.sum(beta[included_mask] ** 2))
            sigma_beta2 = sample_inverse_gamma(
                rng=rng,
                shape=a_beta + 0.5 * n_included,
                scale=b_beta + 0.5 * beta_sum_squares,
            )
        else:
            # When no marker is included, the slab variance is not informed by
            # the current sweep. Sampling from the prior keeps the update fully
            # Bayesian and avoids carrying forward stale information.
            sigma_beta2 = sample_inverse_gamma(
                rng=rng,
                shape=a_beta,
                scale=b_beta,
            )
        sigma_beta2 = max(sigma_beta2, 1e-12)

        if _kept_sample(iteration=iteration, burn_in=burn_in, thin=thin):
            beta_sum += beta
            beta2_sum += beta**2
            delta_sum += delta
            q_trace.append(q)
            sigma_e2_trace.append(sigma_e2)
            sigma_beta2_trace.append(sigma_beta2)
            n_included_trace.append(n_included)
            sample_count += 1

        if verbose and (
            iteration == 1 or iteration % progress_every == 0 or iteration == n_iter
        ):
            mode_name = "BayesCpi" if learning_q else "BayesCFixedQ"
            print(
                f"[{mode_name}] iteration={iteration:>5d} "
                f"q={q:.4f} n_included={n_included:>5d} "
                f"sigma_e2={sigma_e2:.4f} sigma_beta2={sigma_beta2:.4f}"
            )

    if sample_count == 0:
        raise RuntimeError("No posterior samples were stored. Check burn-in and thin.")

    return {
        "beta_mean": beta_sum / sample_count,
        "beta2_mean": beta2_sum / sample_count,
        "pip": delta_sum / sample_count,
        "q_trace": np.asarray(q_trace, dtype=float),
        "sigma_e2_trace": np.asarray(sigma_e2_trace, dtype=float),
        "sigma_beta2_trace": np.asarray(sigma_beta2_trace, dtype=float),
        "n_included_trace": np.asarray(n_included_trace, dtype=int),
        "posterior_sample_count": sample_count,
    }


def run_bayesc_fixed_q_gibbs(
    X: np.ndarray,
    y: np.ndarray,
    q: float,
    n_iter: int,
    burn_in: int,
    thin: int,
    a_e: float,
    b_e: float,
    a_beta: float,
    b_beta: float,
    random_state: int | None = None,
    initial_sigma_e2: float | None = None,
    initial_sigma_beta2: float | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray | float | int]:
    """Run BayesC with user-specified fixed inclusion probability ``q``."""

    return _run_bayesc_gibbs(
        X=X,
        y=y,
        n_iter=n_iter,
        burn_in=burn_in,
        thin=thin,
        a_e=a_e,
        b_e=b_e,
        a_beta=a_beta,
        b_beta=b_beta,
        random_state=random_state,
        fixed_q=q,
        initial_sigma_e2=initial_sigma_e2,
        initial_sigma_beta2=initial_sigma_beta2,
        verbose=verbose,
    )


def run_bayescpi_gibbs(
    X: np.ndarray,
    y: np.ndarray,
    n_iter: int,
    burn_in: int,
    thin: int,
    a_q: float,
    b_q: float,
    a_e: float,
    b_e: float,
    a_beta: float,
    b_beta: float,
    initial_q: float = 0.01,
    initial_sigma_e2: float | None = None,
    initial_sigma_beta2: float | None = None,
    random_state: int | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray | float | int]:
    """Run BayesCpi with learned global inclusion probability ``q``."""

    return _run_bayesc_gibbs(
        X=X,
        y=y,
        n_iter=n_iter,
        burn_in=burn_in,
        thin=thin,
        a_e=a_e,
        b_e=b_e,
        a_beta=a_beta,
        b_beta=b_beta,
        random_state=random_state,
        a_q=a_q,
        b_q=b_q,
        initial_q=initial_q,
        initial_sigma_e2=initial_sigma_e2,
        initial_sigma_beta2=initial_sigma_beta2,
        verbose=verbose,
    )


def run_annotation_bayescpi_gibbs(
    X: np.ndarray,
    y: np.ndarray,
    category_codes: np.ndarray,
    n_iter: int,
    burn_in: int,
    thin: int,
    a_q: float,
    b_q: float,
    a_e: float,
    b_e: float,
    a_beta: float,
    b_beta: float,
    initial_q: float = 0.01,
    initial_sigma_e2: float | None = None,
    initial_sigma_beta2: float | None = None,
    random_state: int | None = None,
    verbose: bool = False,
) -> dict[str, np.ndarray | float | int]:
    """Run annotation-specific BayesCpi with one ``q_c`` per category."""

    X, y = _validate_sampler_inputs(X=X, y=y, n_iter=n_iter, burn_in=burn_in, thin=thin)
    category_codes = np.asarray(category_codes, dtype=int).reshape(-1)
    n, p = X.shape
    if category_codes.shape[0] != p:
        raise ValueError("category_codes must have length equal to the number of SNPs.")
    if np.any(category_codes < 0):
        raise ValueError("category_codes must be non-negative integers.")
    if a_q <= 0.0 or b_q <= 0.0:
        raise ValueError("a_q and b_q must be positive.")
    if not 0.0 < initial_q < 1.0:
        raise ValueError("initial_q must lie strictly between 0 and 1.")

    n_categories = int(np.max(category_codes)) + 1
    category_sizes = np.bincount(category_codes, minlength=n_categories)
    rng = np.random.default_rng(random_state)

    beta = np.zeros(p, dtype=float)
    delta = np.zeros(p, dtype=int)
    residual = y.copy()
    sigma_e2, sigma_beta2 = _initialize_variances(
        y=y,
        p=p,
        q=initial_q,
        initial_sigma_e2=initial_sigma_e2,
        initial_sigma_beta2=initial_sigma_beta2,
    )
    q_by_category = np.full(n_categories, float(initial_q), dtype=float)
    x_squared_norms = np.sum(X * X, axis=0)

    beta_sum = np.zeros(p, dtype=float)
    beta2_sum = np.zeros(p, dtype=float)
    delta_sum = np.zeros(p, dtype=float)
    sample_count = 0

    q_by_category_trace: list[np.ndarray] = []
    sigma_e2_trace: list[float] = []
    sigma_beta2_trace: list[float] = []
    n_included_trace: list[int] = []
    n_included_by_category_trace: list[np.ndarray] = []

    progress_every = max(1, n_iter // 10)

    for iteration in range(1, n_iter + 1):
        for marker_index in range(p):
            x_j = X[:, marker_index]
            category_index = category_codes[marker_index]

            if beta[marker_index] != 0.0:
                residual = residual + x_j * beta[marker_index]

            v_j = 1.0 / (x_squared_norms[marker_index] / sigma_e2 + 1.0 / sigma_beta2)
            m_j = v_j * (x_j @ residual) / sigma_e2

            q_safe = _safe_q_for_logs(q_by_category[category_index])
            log_w1 = (
                np.log(q_safe)
                + 0.5 * (np.log(v_j) - np.log(sigma_beta2))
                + (m_j**2) / (2.0 * v_j)
            )
            log_w0 = np.log1p(-q_safe)
            p_inclusion = float(expit(log_w1 - log_w0))

            delta[marker_index] = int(rng.binomial(1, p_inclusion))
            if delta[marker_index] == 1:
                beta[marker_index] = float(rng.normal(loc=m_j, scale=np.sqrt(v_j)))
            else:
                beta[marker_index] = 0.0

            residual = residual - x_j * beta[marker_index]

        included_mask = delta == 1
        n_included = int(np.sum(included_mask))
        n_included_by_category = np.bincount(
            category_codes[included_mask],
            minlength=n_categories,
        )

        for category_index in range(n_categories):
            q_by_category[category_index] = float(
                rng.beta(
                    a_q + n_included_by_category[category_index],
                    b_q
                    + category_sizes[category_index]
                    - n_included_by_category[category_index],
                )
            )

        residual_sum_squares = float(residual @ residual)
        sigma_e2 = sample_inverse_gamma(
            rng=rng,
            shape=a_e + 0.5 * n,
            scale=b_e + 0.5 * residual_sum_squares,
        )
        sigma_e2 = max(sigma_e2, 1e-12)

        if n_included > 0:
            beta_sum_squares = float(np.sum(beta[included_mask] ** 2))
            sigma_beta2 = sample_inverse_gamma(
                rng=rng,
                shape=a_beta + 0.5 * n_included,
                scale=b_beta + 0.5 * beta_sum_squares,
            )
        else:
            # If no SNP is currently included, the slab variance is not informed
            # by the data in this sweep, so we sample it from the prior.
            sigma_beta2 = sample_inverse_gamma(
                rng=rng,
                shape=a_beta,
                scale=b_beta,
            )
        sigma_beta2 = max(sigma_beta2, 1e-12)

        if _kept_sample(iteration=iteration, burn_in=burn_in, thin=thin):
            beta_sum += beta
            beta2_sum += beta**2
            delta_sum += delta
            q_by_category_trace.append(q_by_category.copy())
            sigma_e2_trace.append(sigma_e2)
            sigma_beta2_trace.append(sigma_beta2)
            n_included_trace.append(n_included)
            n_included_by_category_trace.append(n_included_by_category.copy())
            sample_count += 1

        if verbose and (
            iteration == 1 or iteration % progress_every == 0 or iteration == n_iter
        ):
            q_preview = ", ".join(
                f"{value:.4f}" for value in q_by_category[: min(4, n_categories)]
            )
            print(
                f"[AnnotationBayesCPi] iteration={iteration:>5d} "
                f"n_included={n_included:>5d} q_by_category=[{q_preview}] "
                f"sigma_e2={sigma_e2:.4f} sigma_beta2={sigma_beta2:.4f}"
            )

    if sample_count == 0:
        raise RuntimeError("No posterior samples were stored. Check burn-in and thin.")

    return {
        "beta_mean": beta_sum / sample_count,
        "beta2_mean": beta2_sum / sample_count,
        "pip": delta_sum / sample_count,
        "q_by_category_trace": np.asarray(q_by_category_trace, dtype=float),
        "sigma_e2_trace": np.asarray(sigma_e2_trace, dtype=float),
        "sigma_beta2_trace": np.asarray(sigma_beta2_trace, dtype=float),
        "n_included_trace": np.asarray(n_included_trace, dtype=int),
        "n_included_by_category_trace": np.asarray(
            n_included_by_category_trace, dtype=int
        ),
        "posterior_sample_count": sample_count,
    }
