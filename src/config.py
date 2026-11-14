"""Project-wide configuration: portfolio definition, risk parameters and file paths."""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
PRICES_CSV = DATA_DIR / "prices_close.csv"

# Source database from the commodity-sql-analytics project (always opened read-only)
DEFAULT_DUCKDB_PATH = Path(
    r"C:\Users\JOAN\Desktop\Projectes\Quant\SQL\commodity-sql-analytics\data\commodities.duckdb"
)

# --- Portfolio ---------------------------------------------------------------
# Front-month continuous futures from Yahoo Finance. WTI (CL=F) is excluded because
# its front-month series traded at negative prices in April 2020, which breaks returns.
TICKERS = ["BZ=F", "NG=F", "GC=F", "HG=F", "ZW=F"]

NAMES = {
    "BZ=F": "Brent Crude Oil",
    "NG=F": "Natural Gas",
    "GC=F": "Gold",
    "HG=F": "Copper",
    "ZW=F": "Wheat",
}

# Equal-weight, long-only, rebalanced daily back to these weights
WEIGHTS = {ticker: 1 / len(TICKERS) for ticker in TICKERS}

NOTIONAL = 10_000_000  # USD

# --- Risk parameters ---------------------------------------------------------
CONFIDENCE_LEVELS = (0.99, 0.975)
WINDOW = 500  # trading days in the estimation window

# --- Backtest ----------------------------------------------------------------
METHODS = ("historical", "normal", "student_t", "ewma", "garch", "fhs")
BACKTEST_MIN_HISTORY = 1000  # returns available before the first forecast
GARCH_REFIT_EVERY = 20  # trading days between GARCH re-estimations
FORECASTS_CSV = RESULTS_DIR / "backtest_forecasts.csv"
SUMMARY_CSV = RESULTS_DIR / "backtest_summary.csv"

# --- Sub-period comparison ---------------------------------------------------
# Inclusive (start, end) dates, all inside the backtest sample
PERIODS = {
    "covid_2020": ("2020-02-20", "2020-06-30"),
    "energy_2022": ("2022-02-24", "2022-12-30"),
    "calm_2021": ("2021-01-01", "2021-12-31"),
    "calm_2023": ("2023-01-01", "2023-12-31"),
}
CRISIS_CSV = RESULTS_DIR / "crisis_summary.csv"
