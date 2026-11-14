"""Next-day portfolio VaR and ES from the latest window, for every model and level.

Usage:
    python scripts/latest_var.py
"""

import sys
from pathlib import Path

import numpy as np

# Make the project root importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIDENCE_LEVELS, NOTIONAL, WINDOW  # noqa: E402
from src.data import load_returns  # noqa: E402
from src.models import ewma_sigma, student_t_var_es  # noqa: E402
from src.reporting import format_var_es_table, var_es_table  # noqa: E402


def main() -> None:
    _, port, _ = load_returns()
    window = port.iloc[-WINDOW:]
    r = window.to_numpy()

    table = var_es_table(r, methods=["historical", "normal", "student_t", "ewma"])
    _, _, nu = student_t_var_es(r, CONFIDENCE_LEVELS[0])  # nu does not depend on alpha

    print(f"Window: {len(window)} days, {window.index[0].date()} to {window.index[-1].date()}")
    print(f"Forecast for the next trading day after {window.index[-1].date()}")
    print(f"Notional: USD {NOTIONAL:,.0f}\n")
    print(format_var_es_table(table))

    print(f"\nStudent t degrees of freedom (nu): {nu:.3f}")
    sigma_window = np.std(r, ddof=1)
    sigma_ewma = ewma_sigma(r)
    print(f"Window sigma (ddof=1): {100 * sigma_window:.4f}%")
    print(f"EWMA sigma (lambda=0.94): {100 * sigma_ewma:.4f}%  "
          f"(ratio EWMA / window = {sigma_ewma / sigma_window:.3f})")


if __name__ == "__main__":
    main()
