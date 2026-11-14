"""Tests for filtered historical simulation and the Ljung-Box statistic."""

import numpy as np
import pytest

from src.garch import ljung_box
from src.models import fhs_var_es, garch_var_es

OMEGA, ALPHA, BETA = 0.02, 0.08, 0.90
N_OBS = 5000
BURN = 500


def simulate_garch(z: np.ndarray) -> np.ndarray:
    """GARCH(1,1) path in percent driven by the unit-variance innovations ``z``."""
    eps = np.empty(z.size)
    sigma2 = OMEGA / (1 - ALPHA - BETA)  # start at the unconditional variance
    for t in range(z.size):
        eps[t] = np.sqrt(sigma2) * z[t]
        sigma2 = OMEGA + ALPHA * eps[t] ** 2 + BETA * sigma2
    return eps[BURN:]


def test_fhs_close_to_garch_normal_with_normal_innovations():
    rng = np.random.default_rng(2024)
    r = simulate_garch(rng.standard_normal(N_OBS + BURN)) / 100  # fractions
    fhs_var, _ = fhs_var_es(r, 0.99)
    garch_var, _ = garch_var_es(r, 0.99)
    assert fhs_var == pytest.approx(garch_var, rel=0.05)


def test_fhs_above_garch_normal_with_fat_tailed_innovations():
    nu = 4
    rng = np.random.default_rng(2024)
    z = rng.standard_t(nu, size=N_OBS + BURN) * np.sqrt((nu - 2) / nu)  # unit variance
    r = simulate_garch(z) / 100
    fhs_var, fhs_es = fhs_var_es(r, 0.99)
    garch_var, garch_es = garch_var_es(r, 0.99)
    assert fhs_var > garch_var
    assert fhs_es > garch_es


def test_ljung_box_does_not_reject_white_noise():
    rng = np.random.default_rng(11)
    q, p_value = ljung_box(rng.standard_normal(2000), lags=10)
    assert q > 0
    assert p_value > 0.05


def test_ljung_box_rejects_ar1():
    rng = np.random.default_rng(11)
    noise = rng.standard_normal(2000)
    x = np.empty_like(noise)
    x[0] = noise[0]
    for t in range(1, x.size):
        x[t] = 0.5 * x[t - 1] + noise[t]
    _, p_value = ljung_box(x, lags=10)
    assert p_value < 0.05


def test_ljung_box_small_example_by_hand():
    # x = [1, 2, 3, 4]: mean 2.5, deviations [-1.5, -0.5, 0.5, 1.5], sum of squares 5
    # rho_1 = (0.75 - 0.25 + 0.75) / 5 = 0.25 -> Q = 4 * 6 * 0.25**2 / 3 = 0.5
    q, _ = ljung_box([1.0, 2.0, 3.0, 4.0], lags=1)
    assert q == pytest.approx(0.5)
