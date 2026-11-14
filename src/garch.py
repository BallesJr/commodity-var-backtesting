"""GARCH(1,1) with zero mean and normal innovations, estimated by maximum likelihood.

Written from scratch (no arch dependency) and validated against the ``arch`` package
in tests/test_garch.py.

Model, with returns in percent (r * 100):

    eps_t    = r_t
    sigma2_t = omega + alpha * eps_{t-1}^2 + beta * sigma2_{t-1}

Normal log-likelihood:

    l = -0.5 * sum( ln(2*pi) + ln(sigma2_t) + eps_t^2 / sigma2_t )

Parameters are always passed as (omega, alpha, beta).
"""

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

# Backcast settings, identical to arch (GARCH.backcast in arch/univariate/volatility.py)
BACKCAST_DECAY = 0.94
BACKCAST_MAX_OBS = 75

# Starting values for the optimiser
START_ALPHA = 0.05
START_BETA = 0.90

# Numerical slack that turns the strict constraints into ones SLSQP can handle
_EPS = 1e-8


class GarchFit(NamedTuple):
    """Result of ``fit_garch``."""

    omega: float
    alpha: float
    beta: float
    loglik: float
    converged: bool
    backcast: float

    @property
    def params(self) -> tuple[float, float, float]:
        return (self.omega, self.alpha, self.beta)


def compute_backcast(eps) -> float:
    """Pre-sample value used for both eps_{-1}^2 and sigma2_{-1}.

    Same definition as arch: an exponentially weighted average of the first
    min(75, n) squared residuals, with weights 0.94**i normalised to sum to 1
    (the first observation gets the largest weight).
    """
    eps = np.asarray(eps, dtype=float)
    tau = min(BACKCAST_MAX_OBS, eps.size)
    w = BACKCAST_DECAY ** np.arange(tau)
    w = w / w.sum()
    return float(np.sum(eps[:tau] ** 2 * w))


def garch_variance(params, eps, backcast: float) -> np.ndarray:
    """Conditional variances sigma2_0 .. sigma2_{n-1} for the sample ``eps``.

    The first value has no history, so the backcast replaces both the lagged
    squared residual and the lagged variance:

        sigma2_0 = omega + alpha * backcast + beta * backcast
    """
    omega, alpha, beta = params
    eps2 = (np.asarray(eps, dtype=float) ** 2).tolist()  # plain floats: faster loop
    sigma2 = np.empty(len(eps2))
    prev_eps2 = backcast
    prev_sigma2 = backcast
    for t, e2 in enumerate(eps2):
        prev_sigma2 = omega + alpha * prev_eps2 + beta * prev_sigma2
        sigma2[t] = prev_sigma2
        prev_eps2 = e2
    return sigma2


def neg_loglik(params, eps, backcast: float) -> float:
    """Negative normal log-likelihood of the sample (the function to minimise)."""
    eps = np.asarray(eps, dtype=float)
    sigma2 = garch_variance(params, eps, backcast)
    if not np.all(np.isfinite(sigma2)) or np.any(sigma2 <= 0):
        return 1e12  # outside the admissible region
    return float(0.5 * np.sum(np.log(2 * np.pi) + np.log(sigma2) + eps**2 / sigma2))


def fit_garch(returns_pct) -> GarchFit:
    """Estimate (omega, alpha, beta) by maximum likelihood with SLSQP.

    Constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1.
    Starting values: alpha = 0.05, beta = 0.90 and omega from variance targeting,
    omega = var * (1 - alpha - beta).

    ``returns_pct`` must be in percent (returns * 100) so the parameters are well
    scaled for the optimiser. The mean is assumed to be zero (no demeaning).
    """
    eps = np.asarray(returns_pct, dtype=float).ravel()
    if eps.size < 10:
        raise ValueError(f"need at least 10 returns, got {eps.size}")
    if np.isnan(eps).any():
        raise ValueError("returns contain NaN")

    backcast = compute_backcast(eps)
    n = eps.size
    x0 = np.array([np.var(eps) * (1 - START_ALPHA - START_BETA), START_ALPHA, START_BETA])

    result = minimize(
        # Average log-likelihood keeps the objective O(1) regardless of sample size
        lambda p: neg_loglik(p, eps, backcast) / n,
        x0,
        method="SLSQP",
        bounds=[(_EPS, None), (0.0, 1.0), (0.0, 1.0)],
        constraints=[{"type": "ineq", "fun": lambda p: 1 - _EPS - p[1] - p[2]}],
        options={"ftol": 1e-12, "maxiter": 500},
    )

    omega, alpha, beta = (float(v) for v in result.x)
    return GarchFit(
        omega=omega,
        alpha=alpha,
        beta=beta,
        loglik=-neg_loglik(result.x, eps, backcast),
        converged=bool(result.success),
        backcast=backcast,
    )


def forecast_sigma(params, eps, backcast: float) -> float:
    """One-step-ahead volatility: sqrt(omega + alpha * eps_T^2 + beta * sigma2_T).

    Same units as ``eps`` (percent if the returns are in percent).
    """
    omega, alpha, beta = params
    eps = np.asarray(eps, dtype=float)
    sigma2 = garch_variance(params, eps, backcast)
    return float(np.sqrt(omega + alpha * eps[-1] ** 2 + beta * sigma2[-1]))
