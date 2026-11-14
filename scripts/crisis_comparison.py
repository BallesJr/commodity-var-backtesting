"""Compare the VaR / ES models over crisis and calm periods.

Reads the saved forecasts in results/backtest_forecasts.csv (run scripts/run_backtest.py
first), prints one table per period and saves everything to results/crisis_summary.csv.

Usage:
    python scripts/crisis_comparison.py
"""

import sys
from pathlib import Path

import pandas as pd

# Make the project root importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CRISIS_CSV, FORECASTS_CSV, PERIODS  # noqa: E402
from src.periods import load_forecasts, period_table, select_period  # noqa: E402

FORMATTERS = {
    "expected": "{:.2f}".format,
    "mean_var_pct": "{:.3f}".format,
    "exc_loss_pct": "{:.3f}".format,
    "exc_es_pct": "{:.3f}".format,
    "loss_to_es": "{:.3f}".format,
    "var_first_pct": "{:.3f}".format,
    "var_max_pct": "{:.3f}".format,
    "var_max_date": lambda d: str(d.date()),
    "worst_loss_date": lambda d: str(d.date()),
    "worst_loss_pct": "{:.3f}".format,
    "var_on_worst_pct": "{:.3f}".format,
}


def main() -> None:
    if not FORECASTS_CSV.exists():
        sys.exit(f"{FORECASTS_CSV} not found: run scripts/run_backtest.py first")
    forecasts = load_forecasts()
    table = period_table(forecasts)
    table.to_csv(CRISIS_CSV)

    print(f"Forecasts: {FORECASTS_CSV.name}, "
          f"{forecasts.index[0].date()} to {forecasts.index[-1].date()}")
    print("All VaR, ES and loss figures in % of notional; losses are positive.")
    print("var_max_tdays = trading days from the start of the period to the highest VaR.\n")

    with pd.option_context("display.width", 250, "display.max_columns", None):
        for name, (start, end) in PERIODS.items():
            sub = select_period(forecasts, start, end)
            print("=" * 100)
            print(f"{name}: {start} to {end}  ({len(sub)} trading days, "
                  f"{sub.index[0].date()} to {sub.index[-1].date()})")
            print("=" * 100)
            print(table.loc[name].to_string(formatters=FORMATTERS, na_rep="-"))
            print()

    print(f"Saved {CRISIS_CSV.relative_to(CRISIS_CSV.parents[1])}")


if __name__ == "__main__":
    main()
