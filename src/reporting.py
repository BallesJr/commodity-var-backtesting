"""Build VaR / ES summary tables shared by the scripts."""

import pandas as pd

from src.config import CONFIDENCE_LEVELS, NOTIONAL
from src.models import (
    ewma_var_es,
    garch_var_es,
    historical_var_es,
    normal_var_es,
    student_t_var_es,
)

# Every model with the common (returns, alpha) -> (var, es) interface
METHODS = {
    "historical": historical_var_es,
    "normal": normal_var_es,
    "student_t": lambda r, a: student_t_var_es(r, a)[:2],  # drop nu
    "ewma": ewma_var_es,
    "garch": garch_var_es,
}


def var_es_table(
    returns,
    methods=tuple(METHODS),
    levels=CONFIDENCE_LEVELS,
    notional: float = NOTIONAL,
) -> pd.DataFrame:
    """Next-day VaR and ES for each method and confidence level.

    Returns a numeric DataFrame indexed by (method, alpha) with columns
    var_pct, es_pct (percent of notional) and var_usd, es_usd. Each row is built
    from a single (var, es) pair, so the percent and USD columns cannot disagree.
    """
    rows = []
    for alpha in levels:
        for method in methods:
            var, es = METHODS[method](returns, alpha)
            rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "var_pct": 100 * var,
                    "es_pct": 100 * es,
                    "var_usd": notional * var,
                    "es_usd": notional * es,
                }
            )
    return pd.DataFrame(rows).set_index(["method", "alpha"])


def format_var_es_table(table: pd.DataFrame) -> str:
    """Render a ``var_es_table`` result as text."""
    return table.to_string(
        formatters={
            "var_pct": "{:.3f}%".format,
            "es_pct": "{:.3f}%".format,
            "var_usd": "{:,.0f}".format,
            "es_usd": "{:,.0f}".format,
        }
    )
