"""Export close prices from the commodity-sql-analytics DuckDB database to a wide CSV.

Usage:
    python scripts/export_data.py [--db PATH] [--out PATH]

The output has one row per date and one column per portfolio ticker. Dates where a
ticker did not trade are kept, with an empty cell, so the alignment step stays in
src/data.py and is visible in the code.
"""

import argparse
import sys
from pathlib import Path

import duckdb

# Make the project root importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DEFAULT_DUCKDB_PATH, PRICES_CSV, TICKERS  # noqa: E402

QUERY = """
SELECT date, ticker, close
FROM prices
WHERE ticker IN (SELECT UNNEST(?::VARCHAR[]))
ORDER BY date, ticker
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", type=Path, default=DEFAULT_DUCKDB_PATH, help="DuckDB file")
    parser.add_argument("--out", type=Path, default=PRICES_CSV, help="output CSV file")
    args = parser.parse_args()

    if not args.db.exists():
        sys.exit(f"Database not found: {args.db}")

    con = duckdb.connect(str(args.db), read_only=True)
    try:
        long = con.execute(QUERY, [TICKERS]).df()
    finally:
        con.close()

    # pivot() raises if a (date, ticker) pair is duplicated, which doubles as a check
    wide = long.pivot(index="date", columns="ticker", values="close")[TICKERS]
    wide.index = wide.index.strftime("%Y-%m-%d")
    wide.index.name = "date"
    wide.columns.name = None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(args.out)

    print(f"Wrote {len(wide)} dates x {len(TICKERS)} tickers to {args.out}")
    print(f"Range: {wide.index.min()} to {wide.index.max()}")
    print("Missing prices per ticker:")
    print(wide.isna().sum().to_string())


if __name__ == "__main__":
    main()
