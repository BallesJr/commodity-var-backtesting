"""Out-of-sample backtest: rolling VaR / ES forecasts and the statistical tests on them.

Conventions (same as src/models.py): loss L = -return, VaR and ES are positive
numbers, and an exception is a day whose loss exceeds that day's VaR forecast.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import xlog1py, xlogy

from src.config import (
    BACKTEST_MIN_HISTORY,
    CONFIDENCE_LEVELS,
    GARCH_REFIT_EVERY,
    METHODS,
    WINDOW,
)
from src.garch import compute_backcast, fit_garch, garch_variance
from src.models import (
    ewma_var_es,
    historical_var_es,
    normal_var_es,
    normal_var_es_from_sigma,
    student_t_fit,
    student_t_var_es_from_params,
)

# Basel traffic light for 99% VaR over 250 trading days
BASEL_WINDOW = 250
BASEL_YELLOW_FROM = 5
BASEL_RED_FROM = 10
ZONE_ORDER = {"green": 0, "yellow": 1, "red": 2}

# Approximate 5% critical value of the Acerbi-Szekely Z2 statistic (see acerbi_szekely_z2)
Z2_CRITICAL_5PCT = 0.70


def column(kind: str, method: str, alpha: float) -> str:
    """Name of a forecast column, e.g. column('var', 'garch', 0.99) -> 'var_garch_0.99'."""
    return f"{kind}_{method}_{alpha}"


# =============================================================================
# Rolling forecasts
# =============================================================================
def rolling_forecasts(
    returns: pd.Series,
    levels=CONFIDENCE_LEVELS,
    min_history: int = BACKTEST_MIN_HISTORY,
    window: int = WINDOW,
    refit_every: int = GARCH_REFIT_EVERY,
) -> pd.DataFrame:
    """One-day-ahead VaR and ES forecasts for every method, without look-ahead.

    The forecast for day t (position t in ``returns``) only uses returns[:t].
    The first forecast is made once ``min_history`` returns are available.

    * historical, normal, student_t, ewma: the last ``window`` returns.
    * garch, fhs: all returns up to t-1. Parameters are re-estimated every
      ``refit_every`` days with the data available on that day; in between, the
      conditional variance is updated daily through the recursion with the latest
      estimated parameters. FHS residuals z use those same parameters.

    Returns a DataFrame indexed by date with the realised return (``ret``) and the
    columns ``var_<method>_<alpha>`` and ``es_<method>_<alpha>``.
    """
    r = returns.to_numpy(dtype=float)
    n = r.size
    if min_history < window:
        raise ValueError("min_history must be at least as large as window")
    if n <= min_history:
        raise ValueError(f"need more than {min_history} returns, got {n}")

    eps = 100 * r  # the GARCH works in percent
    # The backcast only depends on the first 75 returns, all of them before the
    # first forecast day, so it is the same for every t and uses no future data.
    backcast = compute_backcast(eps[:min_history])

    rows = []
    garch_params = None
    for t in range(min_history, n):
        row = {"ret": r[t]}
        past = r[t - window : t]  # returns t-window .. t-1

        # --- fixed-window models
        nu, scale = student_t_fit(past)  # one fit serves every alpha
        for a in levels:
            row[column("var", "historical", a)], row[column("es", "historical", a)] = (
                historical_var_es(past, a)
            )
            row[column("var", "normal", a)], row[column("es", "normal", a)] = normal_var_es(past, a)
            row[column("var", "student_t", a)], row[column("es", "student_t", a)] = (
                student_t_var_es_from_params(nu, scale, a)
            )
            row[column("var", "ewma", a)], row[column("es", "ewma", a)] = ewma_var_es(past, a)

        # --- GARCH and FHS on the expanding sample eps[:t]
        if (t - min_history) % refit_every == 0:
            garch_params = fit_garch(eps[:t]).params
        omega, alpha_g, beta_g = garch_params
        sigma2 = garch_variance(garch_params, eps[:t], backcast)
        sigma_next = np.sqrt(omega + alpha_g * eps[t - 1] ** 2 + beta_g * sigma2[-1]) / 100
        z = eps[:t] / np.sqrt(sigma2)
        for a in levels:
            row[column("var", "garch", a)], row[column("es", "garch", a)] = (
                normal_var_es_from_sigma(sigma_next, a)
            )
            var_z, es_z = historical_var_es(z, a)
            row[column("var", "fhs", a)] = sigma_next * var_z
            row[column("es", "fhs", a)] = sigma_next * es_z

        rows.append(row)

    out = pd.DataFrame(rows, index=returns.index[min_history:])
    out.index.name = "date"
    return out


# =============================================================================
# Statistical tests
# =============================================================================
def exceptions(returns, var) -> np.ndarray:
    """Indicator I_t = 1 when the loss (-return) is larger than the VaR forecast."""
    return (-np.asarray(returns, dtype=float) > np.asarray(var, dtype=float)).astype(int)


def _bernoulli_loglik(n_zero: float, n_one: float, p_one: float) -> float:
    """n_zero * ln(1 - p) + n_one * ln(p), with the convention 0 * ln(0) = 0."""
    return float(xlog1py(n_zero, -p_one) + xlogy(n_one, p_one))


def kupiec_pof(x: int, n_days: int, p: float) -> tuple[float, float]:
    """Kupiec proportion-of-failures test. Returns (LR_pof, p_value).

    H0: the exception probability is p = 1 - alpha. With x exceptions in T days,

        LR_pof = -2 * [ ln L(p) - ln L(x / T) ]  ~  chi2(1)

    0 * ln(0) is taken as 0, so x = 0 (or x = T) gives a finite statistic.
    """
    if not 0 <= x <= n_days:
        raise ValueError(f"x must be between 0 and {n_days}, got {x}")
    ll_null = _bernoulli_loglik(n_days - x, x, p)
    ll_alt = _bernoulli_loglik(n_days - x, x, x / n_days)
    lr = max(0.0, -2 * (ll_null - ll_alt))
    return lr, float(stats.chi2.sf(lr, 1))


class ChristoffersenResult(NamedTuple):
    lr_ind: float
    p_ind: float
    lr_cc: float
    p_cc: float
    n00: int
    n01: int
    n10: int
    n11: int


def christoffersen(hits, p: float) -> ChristoffersenResult:
    """Christoffersen independence and conditional coverage tests.

    n_ij counts transitions from state i on day t-1 to state j on day t.

    Independence (LR_ind ~ chi2(1)): H0 is that the probability of an exception
    does not depend on whether there was one the day before (pi01 = pi11).

    Conditional coverage (LR_cc = LR_pof + LR_ind ~ chi2(2)): correct exception
    rate and independence jointly.

    Zero counts are handled with 0 * ln(0) = 0. When a row of the transition
    matrix is empty (e.g. no exceptions at all, so pi11 is undefined), that row
    contributes nothing to the likelihood.
    """
    hits = np.asarray(hits).astype(int)
    prev, curr = hits[:-1], hits[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)

    ll_null = _bernoulli_loglik(n00 + n10, n01 + n11, pi)
    ll_alt = _bernoulli_loglik(n00, n01, pi01) + _bernoulli_loglik(n10, n11, pi11)
    lr_ind = max(0.0, -2 * (ll_null - ll_alt))

    lr_pof, _ = kupiec_pof(int(hits.sum()), hits.size, p)
    lr_cc = lr_pof + lr_ind
    return ChristoffersenResult(
        lr_ind=lr_ind,
        p_ind=float(stats.chi2.sf(lr_ind, 1)),
        lr_cc=lr_cc,
        p_cc=float(stats.chi2.sf(lr_cc, 2)),
        n00=n00,
        n01=n01,
        n10=n10,
        n11=n11,
    )


def basel_zone(count: int) -> str:
    """Basel traffic-light zone for a number of 99% VaR exceptions in 250 days."""
    if count >= BASEL_RED_FROM:
        return "red"
    if count >= BASEL_YELLOW_FROM:
        return "yellow"
    return "green"


class TrafficLight(NamedTuple):
    counts: pd.Series  # exceptions in the trailing 250 days, one value per full window
    final_count: int
    final_zone: str
    worst_count: int
    worst_zone: str
    worst_date: pd.Timestamp  # first date on which the worst count was reached


def basel_traffic_light(hits: pd.Series, window: int = BASEL_WINDOW) -> TrafficLight:
    """Basel traffic light on rolling ``window``-day exception counts (99% VaR).

    Green: 0-4 exceptions, yellow: 5-9, red: 10 or more.
    """
    counts = hits.rolling(window).sum().dropna().astype(int)
    if counts.empty:
        raise ValueError(f"need at least {window} days, got {len(hits)}")
    final_count = int(counts.iloc[-1])
    worst_count = int(counts.max())
    return TrafficLight(
        counts=counts,
        final_count=final_count,
        final_zone=basel_zone(final_count),
        worst_count=worst_count,
        worst_zone=basel_zone(worst_count),
        worst_date=counts.idxmax(),
    )


def acerbi_szekely_z2(returns, var, es, alpha: float) -> float:
    """Acerbi-Szekely (2014) Z2 statistic for Expected Shortfall.

        Z2 = sum_t( L_t * I_t / ES_t ) / ( T * (1 - alpha) ) - 1

    with T the number of backtest days.

    Sign convention: the original paper works with returns X (losses negative),
    Z2 = sum(X_t I_t / ES_t) / (T alpha') + 1, and rejects for Z2 < -0.70. Here
    losses and ES are positive, so the sign is flipped: Z2 is 0 in expectation
    when the model is right, Z2 > 0 means losses beyond VaR were more frequent
    and/or larger than predicted (risk underestimated), and we reject at the 5%
    level when Z2 > 0.70. That threshold is the approximate critical value that
    Acerbi and Szekely report as stable across a wide range of distributions.
    """
    losses = -np.asarray(returns, dtype=float)
    var = np.asarray(var, dtype=float)
    es = np.asarray(es, dtype=float)
    hits = losses > var
    return float(np.sum(losses[hits] / es[hits]) / (losses.size * (1 - alpha)) - 1)


def es_ratio(returns, var, es) -> float:
    """Mean of L_t / ES_t on exception days (close to 1 if ES is well calibrated).

    Returns NaN when there are no exceptions.
    """
    losses = -np.asarray(returns, dtype=float)
    hits = losses > np.asarray(var, dtype=float)
    if not hits.any():
        return float("nan")
    return float(np.mean(losses[hits] / np.asarray(es, dtype=float)[hits]))


# =============================================================================
# Summary table
# =============================================================================
def summarize_backtest(
    forecasts: pd.DataFrame, methods=METHODS, levels=CONFIDENCE_LEVELS
) -> pd.DataFrame:
    """One row per (method, alpha) with the exception counts and every test."""
    ret = forecasts["ret"]
    n_days = len(ret)
    rows = []
    for a in levels:
        p = 1 - a
        for m in methods:
            var = forecasts[column("var", m, a)]
            es = forecasts[column("es", m, a)]
            hits = pd.Series(exceptions(ret, var), index=forecasts.index)
            x = int(hits.sum())
            _, kupiec_p = kupiec_pof(x, n_days, p)
            chris = christoffersen(hits, p)
            row = {
                "method": m,
                "alpha": a,
                "days": n_days,
                "exceptions": x,
                "expected": n_days * p,
                "rate": x / n_days,
                "kupiec_p": kupiec_p,
                "ind_p": chris.p_ind,
                "cc_p": chris.p_cc,
                "z2": acerbi_szekely_z2(ret, var, es, a),
                "es_ratio": es_ratio(ret, var, es),
                "final_zone": "",
                "worst_zone": "",
                "worst_date": "",
            }
            if np.isclose(a, 0.99):  # the Basel traffic light is defined for 99% VaR
                light = basel_traffic_light(hits)
                row["final_zone"] = f"{light.final_zone} ({light.final_count})"
                row["worst_zone"] = f"{light.worst_zone} ({light.worst_count})"
                row["worst_date"] = str(light.worst_date.date())
            rows.append(row)
    return pd.DataFrame(rows).set_index(["method", "alpha"])
