"""Tests for the rolling forecasts and the backtesting statistics in src/backtest.py."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.backtest import (
    acerbi_szekely_z2,
    basel_traffic_light,
    basel_zone,
    christoffersen,
    column,
    es_ratio,
    exceptions,
    kupiec_pof,
    rolling_forecasts,
)
from src.config import CONFIDENCE_LEVELS, METHODS

# Small settings so the rolling forecast tests run in a few seconds
SMALL = dict(min_history=150, window=100, refit_every=20)


def simulated_returns(n: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = 0.01 * rng.standard_t(5, size=n)
    return pd.Series(r, index=pd.bdate_range("2020-01-01", periods=n))


# --- Rolling forecasts ---------------------------------------------------------------
def test_forecasts_shape_and_columns():
    returns = simulated_returns(200, seed=1)
    fc = rolling_forecasts(returns, **SMALL)
    assert len(fc) == 200 - SMALL["min_history"]
    assert fc.index[0] == returns.index[SMALL["min_history"]]
    np.testing.assert_array_equal(fc["ret"], returns.iloc[SMALL["min_history"]:])
    for m in METHODS:
        for a in CONFIDENCE_LEVELS:
            var, es = fc[column("var", m, a)], fc[column("es", m, a)]
            assert (var > 0).all() and (es >= var).all()
    assert not fc.isna().any().any()


def test_no_look_ahead():
    """Changing returns from day t onwards must not change any forecast up to day t."""
    returns = simulated_returns(230, seed=2)
    # GARCH refits happen on days 150, 170, 190, 210: test a refit day and a day in between
    for change_from in (190, 197):
        changed = returns.copy()
        changed.iloc[change_from:] = changed.iloc[change_from:] * -3 + 0.02

        base = rolling_forecasts(returns, **SMALL).drop(columns="ret")
        other = rolling_forecasts(changed, **SMALL).drop(columns="ret")

        upto = returns.index[change_from]  # forecast for this day uses data up to the day before
        pd.testing.assert_frame_equal(base.loc[:upto], other.loc[:upto])
        # ... and the change does reach every method afterwards (the test has power)
        after = returns.index[change_from + 1]
        changed_cols = (base.loc[after:] != other.loc[after:]).any()
        assert changed_cols.all()


# --- Kupiec ----------------------------------------------------------------------------
def test_kupiec_known_values():
    lr7, p7 = kupiec_pof(7, 250, 0.01)
    lr6, p6 = kupiec_pof(6, 250, 0.01)
    assert lr7 == pytest.approx(5.50, abs=0.01)
    assert lr6 == pytest.approx(3.56, abs=0.01)
    assert p7 < 0.05 < p6  # 7 exceptions rejects at 5%, 6 does not


def test_kupiec_zero_exceptions_is_finite():
    lr, p = kupiec_pof(0, 250, 0.01)
    assert lr == pytest.approx(-2 * 250 * np.log(0.99))
    assert np.isfinite(lr) and np.isfinite(p)


# --- Christoffersen ----------------------------------------------------------------------
def test_christoffersen_no_nan_with_zero_counts():
    no_hits = np.zeros(250, dtype=int)
    res = christoffersen(no_hits, 0.01)
    assert res.n01 == res.n10 == res.n11 == 0
    assert res.lr_ind == 0.0
    assert all(np.isfinite(v) for v in res)

    isolated = np.zeros(250, dtype=int)
    isolated[[20, 90, 200]] = 1  # exceptions never on consecutive days -> n11 = 0
    res = christoffersen(isolated, 0.01)
    assert res.n11 == 0 and res.n01 == 3
    assert all(np.isfinite(v) for v in res)


def test_christoffersen_clustered_vs_spread():
    clustered = np.zeros(250, dtype=int)
    clustered[100:106] = 1
    spread = np.zeros(250, dtype=int)
    spread[[20, 60, 100, 140, 180, 220]] = 1

    res_clustered = christoffersen(clustered, 0.01)
    res_spread = christoffersen(spread, 0.01)
    assert res_clustered.lr_ind > res_spread.lr_ind
    assert res_clustered.p_ind < 0.05 < res_spread.p_ind
    # Same number of exceptions, so the difference in LR_cc comes only from LR_ind
    assert res_clustered.lr_cc - res_spread.lr_cc == pytest.approx(
        res_clustered.lr_ind - res_spread.lr_ind
    )


# --- Basel traffic light --------------------------------------------------------------------
def test_basel_zone_boundaries():
    assert [basel_zone(c) for c in (0, 4, 5, 9, 10, 15)] == [
        "green", "green", "yellow", "yellow", "red", "red"
    ]


def test_basel_traffic_light_rolling_counts():
    idx = pd.bdate_range("2020-01-01", periods=600)
    hits = pd.Series(0, index=idx)
    hits.iloc[[300, 310, 320, 330, 340, 350]] = 1
    light = basel_traffic_light(hits)
    assert len(light.counts) == 600 - 250 + 1
    assert light.worst_count == 6 and light.worst_zone == "yellow"
    assert light.worst_date == idx[350]  # first day the 250-day count reaches 6
    # The last window covers positions 350..599, so only the exception on day 350 is left
    assert light.final_count == 1 and light.final_zone == "green"


# --- Expected Shortfall -----------------------------------------------------------------------
@pytest.mark.parametrize("alpha", CONFIDENCE_LEVELS)
def test_z2_near_zero_for_correct_model(alpha):
    rng = np.random.default_rng(99)
    sigma, n = 0.01, 1_000_000
    returns = rng.normal(0.0, sigma, size=n)
    z = stats.norm.ppf(alpha)
    var = np.full(n, sigma * z)
    es = np.full(n, sigma * stats.norm.pdf(z) / (1 - alpha))

    assert acerbi_szekely_z2(returns, var, es, alpha) == pytest.approx(0.0, abs=0.03)
    assert es_ratio(returns, var, es) == pytest.approx(1.0, abs=0.02)
    assert exceptions(returns, var).mean() == pytest.approx(1 - alpha, rel=0.05)


def test_z2_positive_when_risk_is_underestimated():
    rng = np.random.default_rng(99)
    sigma, n, alpha = 0.01, 200_000, 0.99
    returns = rng.normal(0.0, 1.5 * sigma, size=n)  # true volatility is 50% higher
    z = stats.norm.ppf(alpha)
    var = np.full(n, sigma * z)
    es = np.full(n, sigma * stats.norm.pdf(z) / (1 - alpha))
    assert acerbi_szekely_z2(returns, var, es, alpha) > 0.70


def test_forecast_uses_most_recent_return():
    """Changing ONLY the return of day t-1 must change the forecast for day t.

    This is the other side of the look-ahead check: the forecast for day t is
    allowed to use day t-1, and the volatility models must actually react to it.
    """
    returns = simulated_returns(230, seed=3)
    base = rolling_forecasts(returns, **SMALL)

    for t in (190, 197):  # a GARCH refit day and a day between refits
        changed = returns.copy()
        changed.iloc[t - 1] = -0.08  # a large loss on day t-1 only
        other = rolling_forecasts(changed, **SMALL)

        day_t = returns.index[t]
        before = returns.index[t - 1]
        for m in ("ewma", "garch", "fhs"):
            for a in CONFIDENCE_LEVELS:
                for kind in ("var", "es"):
                    col = column(kind, m, a)
                    assert other.loc[day_t, col] > base.loc[day_t, col], col
                    # ... while the forecast for day t-1 itself is untouched
                    assert other.loc[before, col] == base.loc[before, col], col
