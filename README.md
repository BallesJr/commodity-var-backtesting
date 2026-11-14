# Market Risk for a Commodity Portfolio: VaR, Expected Shortfall & Backtesting

Daily market risk for an equal-weight, long-only portfolio of five commodity futures
(Brent, natural gas, gold, copper, wheat), with USD 10,000,000 notional, rebalanced daily.

## Data

Daily close prices of Yahoo Finance front-month continuous futures, 2016-01-04 to 2025-12-31,
exported from the DuckDB database of the `commodity-sql-analytics` project into
`data/prices_close.csv`. WTI (`CL=F`) is excluded because its front-month series went
negative in April 2020.

Only dates on which all five contracts have a price are kept. Returns are simple daily
returns computed on those common dates.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

python scripts/export_data.py   # optional: rebuild data/prices_close.csv from DuckDB
pytest
```

## Structure

```
data/                  price snapshot (CSV, wide format)
scripts/export_data.py DuckDB -> CSV export
src/config.py          tickers, weights, notional, confidence levels
src/data.py            loading, date alignment, returns, portfolio P&L
tests/                 pytest tests
results/               charts and tables
```
