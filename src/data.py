"""Load the price snapshot and turn it into asset returns, portfolio returns and P&L."""

from pathlib import Path

import pandas as pd

from src.config import NOTIONAL, PRICES_CSV, TICKERS, WEIGHTS


def load_prices(path: str | Path = PRICES_CSV, tickers: list[str] = TICKERS) -> pd.DataFrame:
    """Read the wide close-price CSV (one column per ticker) indexed by date.

    Missing prices are kept as NaN; use ``align_dates`` to drop them.
    """
    prices = pd.read_csv(path, parse_dates=["date"], index_col="date")
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        raise KeyError(f"Tickers not found in {path}: {missing}")
    return prices[tickers].sort_index()


def align_dates(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep only the dates on which every ticker has a close price."""
    return prices.dropna(how="any")


def alignment_report(prices: pd.DataFrame) -> pd.DataFrame:
    """Summarise, per ticker, how many dates are lost when aligning on common dates.

    Columns:
        n_obs        prices available for the ticker before alignment
        n_missing    dates in the file where the ticker has no price
        n_dropped    the ticker's own prices removed because another ticker was missing
        n_kept       prices left after alignment (same for every ticker)
    """
    common = prices.notna().all(axis=1)
    has_price = prices.notna()
    return pd.DataFrame(
        {
            "n_obs": has_price.sum(),
            "n_missing": (~has_price).sum(),
            "n_dropped": (has_price & ~common.to_numpy()[:, None]).sum(),
            "n_kept": int(common.sum()),
        }
    )


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns ``close_t / close_{t-1} - 1`` between consecutive rows.

    Call this on already-aligned prices so every return spans the same two dates
    for all tickers. The first row (no previous close) is dropped.
    """
    if prices.isna().any().any():
        raise ValueError("Prices contain NaN; align dates before computing returns.")
    return (prices / prices.shift(1) - 1).iloc[1:]


def portfolio_returns(returns: pd.DataFrame, weights: dict[str, float] = WEIGHTS) -> pd.Series:
    """Weighted sum of asset returns.

    With daily rebalancing back to fixed weights, the portfolio's simple return on
    each day is exactly the weighted sum of the assets' simple returns.
    """
    w = pd.Series(weights, dtype=float)
    return (returns[w.index] * w).sum(axis=1).rename("portfolio")


def pnl(port_returns: pd.Series, notional: float = NOTIONAL) -> pd.Series:
    """Daily profit and loss in USD for a position of size ``notional``."""
    return (notional * port_returns).rename("pnl_usd")


def load_returns(path: str | Path = PRICES_CSV) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Full pipeline: CSV -> aligned prices -> asset returns, portfolio returns, P&L."""
    returns = compute_returns(align_dates(load_prices(path)))
    port = portfolio_returns(returns)
    return returns, port, pnl(port)
