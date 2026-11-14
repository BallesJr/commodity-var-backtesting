"""Descriptive comparison of the backtest forecasts over crisis and calm sub-periods.

Works on the saved forecasts (results/backtest_forecasts.csv); nothing is re-estimated.

No p-values are reported here on purpose: a sub-period has between ~90 and ~250
days, so at 99% only 1-2.5 exceptions are expected. Kupiec, Christoffersen and Z2
have almost no power with so few observations, and a "not rejected" would say
nothing. The numbers below are meant to be read descriptively (how fast VaR
reacted, how large the losses beyond VaR were compared with the predicted ES).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import column
from src.config import CONFIDENCE_LEVELS, FORECASTS_CSV, METHODS, PERIODS


def load_forecasts(path: str | Path = FORECASTS_CSV) -> pd.DataFrame:
    """Read the saved backtest forecasts, indexed by date."""
    return pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()


def select_period(forecasts: pd.DataFrame, start, end) -> pd.DataFrame:
    """Rows with start <= date <= end (both bounds inclusive)."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end:
        raise ValueError(f"start {start.date()} is after end {end.date()}")
    sub = forecasts.loc[(forecasts.index >= start) & (forecasts.index <= end)]
    if sub.empty:
        raise ValueError(f"no forecasts between {start.date()} and {end.date()}")
    return sub


def period_stats(forecasts: pd.DataFrame, start, end, method: str, alpha: float) -> dict:
    """Descriptive statistics of one method / confidence level over [start, end].

    All VaR, ES and loss figures are in percent of notional, with losses positive.

    Keys:
        days, exceptions, expected
        mean_var_pct                  average VaR forecast over the period
        exc_loss_pct, exc_es_pct      average realised loss and average predicted ES
                                      on exception days (NaN if there are none)
        loss_to_es                    exc_loss_pct / exc_es_pct (about 1 if ES is right)
        var_first_pct                 VaR on the first day of the period
        var_max_pct, var_max_date     highest VaR in the period and the first day it occurred
        var_max_tdays                 trading days from the start to that date (0 = first day)
        worst_loss_date, worst_loss_pct, var_on_worst_pct
                                      largest realised loss and the VaR predicted for that day
    """
    sub = select_period(forecasts, start, end)
    loss = -sub["ret"]
    var = sub[column("var", method, alpha)]
    es = sub[column("es", method, alpha)]
    hits = loss > var

    n_exc = int(hits.sum())
    exc_loss = 100 * loss[hits].mean() if n_exc else np.nan
    exc_es = 100 * es[hits].mean() if n_exc else np.nan

    var_max_date = var.idxmax()
    worst_date = loss.idxmax()

    return {
        "days": len(sub),
        "exceptions": n_exc,
        "expected": len(sub) * (1 - alpha),
        "mean_var_pct": 100 * var.mean(),
        "exc_loss_pct": exc_loss,
        "exc_es_pct": exc_es,
        "loss_to_es": exc_loss / exc_es if n_exc else np.nan,
        "var_first_pct": 100 * var.iloc[0],
        "var_max_pct": 100 * var.max(),
        "var_max_date": var_max_date,
        "var_max_tdays": int(sub.index.get_loc(var_max_date)),
        "worst_loss_date": worst_date,
        "worst_loss_pct": 100 * loss.max(),
        "var_on_worst_pct": 100 * var.loc[worst_date],
    }


def period_table(
    forecasts: pd.DataFrame,
    periods: dict = PERIODS,
    methods=METHODS,
    levels=CONFIDENCE_LEVELS,
) -> pd.DataFrame:
    """``period_stats`` for every period, method and level, indexed by (period, alpha, method)."""
    rows = []
    for name, (start, end) in periods.items():
        for a in levels:
            for m in methods:
                rows.append(
                    {"period": name, "alpha": a, "method": m,
                     **period_stats(forecasts, start, end, m, a)}
                )
    return pd.DataFrame(rows).set_index(["period", "alpha", "method"])
