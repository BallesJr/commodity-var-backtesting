"""Run the out-of-sample backtest for every method and print the summary table.

Writes results/backtest_forecasts.csv (daily forecasts, reused by later steps) and
results/backtest_summary.csv (the table printed here).

Usage:
    python scripts/run_backtest.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

# Make the project root importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import Z2_CRITICAL_5PCT, rolling_forecasts, summarize_backtest  # noqa: E402
from src.config import (  # noqa: E402
    BACKTEST_MIN_HISTORY,
    FORECASTS_CSV,
    GARCH_REFIT_EVERY,
    SUMMARY_CSV,
    WINDOW,
)
from src.data import load_returns  # noqa: E402


def main() -> None:
    _, port, _ = load_returns()

    start = time.perf_counter()
    forecasts = rolling_forecasts(port)
    elapsed = time.perf_counter() - start

    FORECASTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(FORECASTS_CSV)
    summary = summarize_backtest(forecasts)
    summary.to_csv(SUMMARY_CSV)

    print(f"Returns: {len(port)} days, {port.index[0].date()} to {port.index[-1].date()}")
    print(f"Backtest: {len(forecasts)} days, "
          f"{forecasts.index[0].date()} to {forecasts.index[-1].date()}")
    print(f"Settings: first forecast after {BACKTEST_MIN_HISTORY} returns, "
          f"{WINDOW}-day window, GARCH refit every {GARCH_REFIT_EVERY} days")
    print(f"Forecasts computed in {elapsed:.1f} s")
    print(f"Saved {FORECASTS_CSV.relative_to(FORECASTS_CSV.parents[1])} "
          f"and {SUMMARY_CSV.relative_to(SUMMARY_CSV.parents[1])}\n")

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(
            summary.to_string(
                formatters={
                    "expected": "{:.1f}".format,
                    "rate": "{:.2%}".format,
                    "kupiec_p": "{:.4f}".format,
                    "ind_p": "{:.4f}".format,
                    "cc_p": "{:.4f}".format,
                    "z2": "{:+.3f}".format,
                    "es_ratio": "{:.3f}".format,
                }
            )
        )
    print(f"\nz2: reject ES at the 5% level when z2 > {Z2_CRITICAL_5PCT:.2f} "
          "(positive = risk underestimated)")
    print("zones: Basel traffic light on rolling 250-day windows, exception count in brackets")


if __name__ == "__main__":
    main()
