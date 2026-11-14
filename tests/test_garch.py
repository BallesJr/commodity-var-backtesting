"""Tests for the hand-written GARCH(1,1) in src/garch.py, validated against arch."""

import numpy as np
import pytest
from arch import arch_model

from src.data import load_returns
from src.garch import compute_backcast, fit_garch, forecast_sigma, garch_variance, neg_loglik
from src.models import garch_var_es, normal_var_es


@pytest.fixture(scope="module")
def portfolio_pct() -> np.ndarray:
    _, port, _ = load_returns()
    return 100 * port.to_numpy()


def arch_garch(returns_pct):
    """The reference model: zero mean, GARCH(1,1), normal innovations."""
    return arch_model(
        returns_pct, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False
    )


def simulate_garch(omega, alpha, beta, n, seed, burn=500) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n + burn)
    eps = np.empty(n + burn)
    sigma2 = omega / (1 - alpha - beta)  # start at the unconditional variance
    for t in range(n + burn):
        eps[t] = np.sqrt(sigma2) * z[t]
        sigma2 = omega + alpha * eps[t] ** 2 + beta * sigma2
    return eps[burn:]


@pytest.mark.parametrize("params", [(0.02, 0.08, 0.90), (0.10, 0.15, 0.70), (0.05, 0.03, 0.95)])
def test_loglik_matches_arch_at_fixed_params(portfolio_pct, params):
    ours = -neg_loglik(params, portfolio_pct, compute_backcast(portfolio_pct))
    theirs = arch_garch(portfolio_pct).fix(list(params)).loglikelihood
    assert ours == pytest.approx(theirs, abs=1e-8)


def test_conditional_variance_matches_arch_at_fixed_params(portfolio_pct):
    params = (0.02, 0.08, 0.90)
    ours = garch_variance(params, portfolio_pct, compute_backcast(portfolio_pct))
    theirs = arch_garch(portfolio_pct).fix(list(params)).conditional_volatility ** 2
    np.testing.assert_allclose(ours, theirs, rtol=1e-10)


def test_fit_recovers_simulated_parameters():
    eps = simulate_garch(omega=0.02, alpha=0.08, beta=0.90, n=5000, seed=2024)
    fit = fit_garch(eps)
    assert fit.converged
    assert fit.omega == pytest.approx(0.02, abs=0.015)
    assert fit.alpha == pytest.approx(0.08, abs=0.02)
    assert fit.beta == pytest.approx(0.90, abs=0.03)
    assert fit.alpha + fit.beta < 1


@pytest.mark.parametrize("window", [None, 500])
def test_fit_matches_arch_on_portfolio(portfolio_pct, window):
    eps = portfolio_pct if window is None else portfolio_pct[-window:]
    fit = fit_garch(eps)
    res = arch_garch(eps).fit(disp="off")
    assert fit.converged
    assert fit.omega == pytest.approx(res.params["omega"], abs=1e-3)
    assert fit.alpha == pytest.approx(res.params["alpha[1]"], abs=1e-3)
    assert fit.beta == pytest.approx(res.params["beta[1]"], abs=1e-3)
    # Our optimum must be at least as good as arch's, up to optimiser noise
    assert fit.loglik >= res.loglikelihood - 1e-4


def test_forecast_matches_arch(portfolio_pct):
    res = arch_garch(portfolio_pct).fit(disp="off")
    params = tuple(res.params)
    ours = forecast_sigma(params, portfolio_pct, compute_backcast(portfolio_pct))
    theirs = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])
    assert ours == pytest.approx(theirs, rel=1e-10)


@pytest.mark.parametrize("alpha", [0.99, 0.975])
def test_garch_var_es_units_and_ordering(portfolio_pct, alpha):
    r = portfolio_pct[-500:] / 100  # back to fractions, as the models expect
    var, es = garch_var_es(r, alpha)
    assert es >= var > 0
    # Same order of magnitude as the plain normal model: a units mix-up would be x100
    normal_var, _ = normal_var_es(r, alpha)
    assert 0.2 * normal_var < var < 5 * normal_var
