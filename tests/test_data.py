"""Tests for the data pipeline in src/data.py."""

import duckdb
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import align_dates, compute_returns, load_prices, load_returns, pnl


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    return load_prices()


@pytest.fixture(scope="module")
def returns(prices) -> pd.DataFrame:
    return compute_returns(align_dates(prices))


def test_weights_sum_to_one():
    assert sum(config.WEIGHTS.values()) == pytest.approx(1.0)
    assert set(config.WEIGHTS) == set(config.TICKERS)
    assert all(w > 0 for w in config.WEIGHTS.values())  # long-only


def test_no_nan_after_alignment(prices, returns):
    assert not align_dates(prices).isna().any().any()
    assert not returns.isna().any().any()
    asset_returns, port, port_pnl = load_returns()
    assert not asset_returns.isna().any().any()
    assert not port.isna().any()
    assert not port_pnl.isna().any()


def test_pnl_is_notional_times_return():
    _, port, port_pnl = load_returns()
    np.testing.assert_allclose(port_pnl, config.NOTIONAL * port, rtol=1e-12)


def test_returns_match_duckdb_where_no_gaps(returns):
    """On dates where no ticker skipped a day, our returns equal prices_clean.ret.

    A date t qualifies when, for every ticker, the previous row in DuckDB is the
    same as the previous common date in our aligned series.
    """
    if not config.DEFAULT_DUCKDB_PATH.exists():
        pytest.skip(f"DuckDB file not available: {config.DEFAULT_DUCKDB_PATH}")

    con = duckdb.connect(str(config.DEFAULT_DUCKDB_PATH), read_only=True)
    try:
        db = con.execute(
            """
            SELECT date, ticker, ret,
                   LAG(date) OVER (PARTITION BY ticker ORDER BY date) AS prev_date
            FROM prices_clean
            WHERE ticker IN (SELECT UNNEST(?::VARCHAR[]))
            """,
            [config.TICKERS],
        ).df()
    finally:
        con.close()

    db["date"] = pd.to_datetime(db["date"])
    db["prev_date"] = pd.to_datetime(db["prev_date"])
    db_ret = db.pivot(index="date", columns="ticker", values="ret")
    db_prev = db.pivot(index="date", columns="ticker", values="prev_date")

    # Previous common date for each return date in our aligned series
    common_dates = align_dates(load_prices()).index
    prev_common = pd.Series(common_dates[:-1], index=common_dates[1:])

    db_prev = db_prev.reindex(returns.index)[config.TICKERS]
    no_gap = db_prev.eq(prev_common, axis=0).all(axis=1)

    assert no_gap.sum() > 0.9 * len(returns)  # sanity: gaps should be rare

    ours = returns.loc[no_gap, config.TICKERS]
    theirs = db_ret.reindex(ours.index)[config.TICKERS]
    assert not theirs.isna().any().any()
    np.testing.assert_allclose(ours.to_numpy(), theirs.to_numpy(), rtol=1e-10, atol=1e-14)
