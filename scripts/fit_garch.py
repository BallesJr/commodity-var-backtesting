"""Fit the hand-written GARCH(1,1) to the portfolio and compare it with arch.

Shown for the full sample and for the latest estimation window.

Usage:
    python scripts/fit_garch.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

# Make the project root importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import WINDOW  # noqa: E402
from src.data import load_returns  # noqa: E402
from src.garch import fit_garch, forecast_sigma, garch_variance  # noqa: E402
from src.reporting import format_var_es_table, var_es_table  # noqa: E402


def report(label: str, returns: pd.Series) -> None:
    eps = 100 * returns.to_numpy()  # percent

    ours = fit_garch(eps)
    ref = arch_model(
        eps, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    ).fit(disp="off")
    ref_params = (ref.params["omega"], ref.params["alpha[1]"], ref.params["beta[1]"])

    table = pd.DataFrame(
        {
            "ours": [*ours.params, ours.alpha + ours.beta, ours.loglik],
            "arch": [*ref_params, ref_params[1] + ref_params[2], ref.loglikelihood],
        },
        index=["omega", "alpha", "beta", "alpha+beta", "loglik"],
    )
    table["diff"] = table["ours"] - table["arch"]

    persistence = ours.alpha + ours.beta
    long_run_vol = np.sqrt(252 * ours.omega / (1 - persistence))
    half_life = np.log(0.5) / np.log(persistence)

    sigma_ours = np.sqrt(garch_variance(ours.params, eps, ours.backcast))
    sigma_arch = np.asarray(ref.conditional_volatility)
    max_sigma_diff = np.max(np.abs(sigma_ours - sigma_arch))

    sigma_next = forecast_sigma(ours.params, eps, ours.backcast)
    sigma_next_arch = np.sqrt(ref.forecast(horizon=1).variance.iloc[-1, 0])

    print("=" * 72)
    print(f"{label}: {len(returns)} days, "
          f"{returns.index[0].date()} to {returns.index[-1].date()}")
    print("=" * 72)
    print(f"Converged: ours={ours.converged}, arch={ref.convergence_flag == 0}"
          f"   |   backcast = {ours.backcast:.6f}")
    print()
    print(table.to_string(float_format=lambda x: f"{x: .8f}"))
    print()
    print(f"Long-run volatility (annualised): {long_run_vol:.3f}%")
    print(f"Half-life of a shock:             {half_life:.1f} days")
    print(f"Max |sigma_ours - sigma_arch|:    {max_sigma_diff:.2e} (in % units)")
    print(f"Sample sigma (ddof=1):            {np.std(eps, ddof=1):.4f}%")
    print(f"Sigma forecast for next day:      {sigma_next:.4f}%  (arch: {sigma_next_arch:.4f}%)")
    print()

    print("GARCH VaR and ES for the next day:")
    print(format_var_es_table(var_es_table(returns.to_numpy(), methods=["garch"])))
    print()


def main() -> None:
    _, port, _ = load_returns()
    report("Full sample", port)
    report(f"Last {WINDOW} days", port.iloc[-WINDOW:])


if __name__ == "__main__":
    main()
