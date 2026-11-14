"""Next-day portfolio VaR and ES from the latest window, for every model and level.

Usage:
    python scripts/latest_var.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIDENCE_LEVELS, NOTIONAL, WINDOW  # noqa: E402
from src.data import load_returns  # noqa: E402
from src.models import (  # noqa: E402
    ewma_sigma,
    ewma_var_es,
    historical_var_es,
    normal_var_es,
    student_t_var_es,
)


def main() -> None:
    _, port, _ = load_returns()
    window = port.iloc[-WINDOW:]
    r = window.to_numpy()

    rows = []
    nu = None
    for alpha in CONFIDENCE_LEVELS:
        t_var, t_es, nu = student_t_var_es(r, alpha)
        results = {
            "historical": historical_var_es(r, alpha),
            "normal": normal_var_es(r, alpha),
            "student_t": (t_var, t_es),
            "ewma": ewma_var_es(r, alpha),
        }
        for method, (var, es) in results.items():
            rows.append({"method": method, "alpha": alpha, "var": var, "es": es})

    table = pd.DataFrame(rows).set_index(["method", "alpha"])
    table["var_pct"] = 100 * table["var"]
    table["es_pct"] = 100 * table["es"]
    table["var_usd"] = NOTIONAL * table["var"]
    table["es_usd"] = NOTIONAL * table["es"]
    table = table.drop(columns=["var", "es"])

    print(f"Window: {len(window)} days, {window.index[0].date()} to {window.index[-1].date()}")
    print(f"Forecast for the next trading day after {window.index[-1].date()}")
    print(f"Notional: USD {NOTIONAL:,.0f}\n")
    print(
        table.to_string(
            formatters={
                "var_pct": "{:.3f}%".format,
                "es_pct": "{:.3f}%".format,
                "var_usd": "{:,.0f}".format,
                "es_usd": "{:,.0f}".format,
            }
        )
    )

    print(f"\nStudent t degrees of freedom (nu): {nu:.3f}")
    sigma_window = np.std(r, ddof=1)
    sigma_ewma = ewma_sigma(r)
    print(f"Window sigma (ddof=1): {100 * sigma_window:.4f}%")
    print(f"EWMA sigma (lambda=0.94): {100 * sigma_ewma:.4f}%  "
          f"(ratio EWMA / window = {sigma_ewma / sigma_window:.3f})")


if __name__ == "__main__":
    main()
