"""Diagnostics for filtered historical simulation on the full portfolio sample.

Checks that the GARCH(1,1) filter removes volatility clustering (Ljung-Box on squared
returns vs squared standardised residuals), describes the residuals, and compares
FHS with the normal GARCH VaR and ES.

Usage:
    python scripts/fhs_diagnostics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make the project root importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONFIDENCE_LEVELS  # noqa: E402
from src.data import load_returns  # noqa: E402
from src.garch import fit_garch, garch_variance, ljung_box  # noqa: E402
from src.models import historical_k  # noqa: E402
from src.reporting import format_var_es_table, var_es_table  # noqa: E402

LAGS = 10


def main() -> None:
    _, port, _ = load_returns()
    r = port.to_numpy()
    eps = 100 * r  # percent

    fit = fit_garch(eps)
    z = eps / np.sqrt(garch_variance(fit.params, eps, fit.backcast))

    print(f"Full sample: {len(port)} days, {port.index[0].date()} to {port.index[-1].date()}")
    print(f"GARCH(1,1): omega={fit.omega:.6f}, alpha={fit.alpha:.6f}, beta={fit.beta:.6f}, "
          f"converged={fit.converged}\n")

    print(f"Ljung-Box test, {LAGS} lags (H0: no autocorrelation)")
    lb = pd.DataFrame(
        [ljung_box(eps**2, LAGS), ljung_box(z**2, LAGS)],
        index=["r^2 (squared returns)", "z^2 (squared std. residuals)"],
        columns=["Q", "p_value"],
    )
    print(lb.to_string(formatters={"Q": "{:.3f}".format, "p_value": "{:.6f}".format}))

    zs = pd.Series(z)
    print("\nStandardised residuals z_t = r_t / sigma_t")
    print(f"  mean:            {zs.mean(): .4f}")
    print(f"  std (ddof=1):    {zs.std(): .4f}")
    print(f"  skewness:        {zs.skew(): .4f}")
    print(f"  excess kurtosis: {zs.kurt(): .4f}")

    print("\nNext-day VaR and ES: FHS vs normal GARCH")
    print(format_var_es_table(var_es_table(r, methods=["fhs", "garch"])))

    print("\nTail observations used by FHS")
    for alpha in CONFIDENCE_LEVELS:
        print(f"  alpha={alpha}: k = {historical_k(len(z), alpha)} of {len(z)}")


if __name__ == "__main__":
    main()
