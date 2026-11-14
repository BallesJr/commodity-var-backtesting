"""One-day-ahead Value at Risk and Expected Shortfall models.

Conventions shared by every function:
    * Input is a 1-D window of simple returns, oldest first.
    * Loss is defined as L = -return.
    * VaR and ES are returned as POSITIVE numbers (losses), in return units.
    * Parametric models assume a zero mean.
"""

import math
import warnings

import numpy as np
from scipy import stats

from src.garch import fit_garch, forecast_sigma

EWMA_LAMBDA = 0.94  # RiskMetrics daily decay factor
EWMA_INIT_OBS = 30  # observations used to seed the EWMA variance


def _as_returns(returns, alpha: float, min_obs: int = 2) -> np.ndarray:
    """Validate inputs and return the window as a 1-D float array."""
    r = np.asarray(returns, dtype=float).ravel()
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if r.size < min_obs:
        raise ValueError(f"need at least {min_obs} returns, got {r.size}")
    if np.isnan(r).any():
        raise ValueError("returns contain NaN")
    return r


def _normal_var_es(sigma: float, alpha: float) -> tuple[float, float]:
    """VaR and ES of a zero-mean normal loss with standard deviation sigma."""
    z = stats.norm.ppf(alpha)
    var = sigma * z
    es = sigma * stats.norm.pdf(z) / (1 - alpha)
    return float(var), float(es)


# --- 1. Historical simulation ------------------------------------------------
def historical_k(n: int, alpha: float) -> int:
    """Number of tail observations used by historical simulation.

    The rounding is needed because 1 - alpha is not exact in binary floating point:
    1 - 0.99 = 0.010000000000000009, so 500 * (1 - 0.99) = 5.000000000000004 and a
    plain ceil() would give 6 instead of 5. Rounding to 10 decimals removes that
    representation error before taking the ceiling.
    """
    return max(1, math.ceil(round(n * (1 - alpha), 10)))


def historical_var_es(returns, alpha: float) -> tuple[float, float]:
    """Historical-simulation VaR and ES.

    VaR is the k-th largest loss and ES the mean of the k largest losses,
    with k = ceil(n * (1 - alpha)).
    """
    r = _as_returns(returns, alpha, min_obs=1)
    k = historical_k(r.size, alpha)
    worst = np.sort(-r)[::-1][:k]  # k largest losses, largest first
    return float(worst[-1]), float(worst.mean())


# --- 2. Normal ----------------------------------------------------------------
def normal_var_es(returns, alpha: float) -> tuple[float, float]:
    """Normal VaR and ES with zero mean and the window's sample standard deviation."""
    r = _as_returns(returns, alpha)
    return _normal_var_es(np.std(r, ddof=1), alpha)


# --- 3. Student t -------------------------------------------------------------
def student_t_var_es(returns, alpha: float) -> tuple[float, float, float]:
    """Student t VaR and ES from a maximum-likelihood fit with location fixed at 0.

    Returns (var, es, nu).

    The fitted ``scale`` is the t distribution's scale parameter, NOT its standard
    deviation: for nu > 2 the standard deviation is scale * sqrt(nu / (nu - 2)).
    The quantile and ES formulas below are written in terms of the scale, so it is
    used exactly as returned by the fit.

    The t distribution is symmetric and the location is fixed at 0, so fitting the
    returns or the losses gives the same nu and scale.
    """
    r = _as_returns(returns, alpha)
    nu, _, scale = stats.t.fit(r, floc=0)
    if nu <= 1:
        raise ValueError(
            f"Fitted degrees of freedom nu={nu:.4f} <= 1: the t distribution has no "
            "finite mean, so Expected Shortfall is undefined for this window."
        )
    q = stats.t.ppf(alpha, nu)
    var = scale * q
    es = scale * stats.t.pdf(q, nu) / (1 - alpha) * (nu + q**2) / (nu - 1)
    return float(var), float(es), float(nu)


# --- 4. EWMA (RiskMetrics) -----------------------------------------------------
def ewma_sigma(returns, lam: float = EWMA_LAMBDA, n_init: int = EWMA_INIT_OBS) -> float:
    """Next-day EWMA volatility forecast.

    The variance is seeded with the sample variance of the first ``n_init`` returns,
    taken as the forecast for observation ``n_init`` (so no return is used twice).
    It is then updated through the rest of the window with zero-mean squared returns:

        sigma2_{t+1} = lam * sigma2_t + (1 - lam) * r_t ** 2

    and the value after the last return is the forecast for the next day.
    """
    r = np.asarray(returns, dtype=float).ravel()
    if not 0 < lam < 1:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    if r.size <= n_init:
        raise ValueError(f"need more than {n_init} returns, got {r.size}")
    sigma2 = np.var(r[:n_init], ddof=1)
    for r_t in r[n_init:]:
        sigma2 = lam * sigma2 + (1 - lam) * r_t**2
    return float(np.sqrt(sigma2))


def ewma_var_es(returns, alpha: float, lam: float = EWMA_LAMBDA) -> tuple[float, float]:
    """Normal VaR and ES using the EWMA next-day volatility forecast."""
    r = _as_returns(returns, alpha, min_obs=EWMA_INIT_OBS + 1)
    return _normal_var_es(ewma_sigma(r, lam), alpha)


# --- 5. GARCH(1,1) --------------------------------------------------------------
def garch_var_es(returns, alpha: float) -> tuple[float, float]:
    """Normal VaR and ES using the one-day-ahead GARCH(1,1) volatility forecast.

    The GARCH is fitted on returns in percent (x100) for numerical stability, so the
    forecast sigma is divided by 100 to bring VaR and ES back to return fractions,
    like every other model in this module.
    """
    r = _as_returns(returns, alpha, min_obs=10)
    eps = 100 * r
    fit = fit_garch(eps)
    if not fit.converged:
        warnings.warn("GARCH optimiser did not converge on this window", RuntimeWarning)
    sigma = forecast_sigma(fit.params, eps, fit.backcast) / 100
    return _normal_var_es(sigma, alpha)
