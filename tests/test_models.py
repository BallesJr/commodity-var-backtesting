"""Tests for the VaR / ES models in src/models.py."""

import math

import numpy as np
import pytest
from scipy import stats

from src import models
from src.models import (
    ewma_var_es,
    historical_k,
    historical_var_es,
    normal_var_es,
    student_t_var_es,
)

SIGMA = 0.01


@pytest.fixture(scope="module")
def normal_returns() -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0.0, SIGMA, size=1_000_000)


# --- Normal --------------------------------------------------------------------
def test_normal_var99_matches_theory(normal_returns):
    var, es = normal_var_es(normal_returns, 0.99)
    assert var == pytest.approx(0.02326, rel=0.02)
    assert es == pytest.approx(SIGMA * stats.norm.pdf(stats.norm.ppf(0.99)) / 0.01, rel=0.02)


# --- Historical -----------------------------------------------------------------
def test_historical_small_hand_made_example():
    # Losses are 5%, 3%, 1%, ... ; order is shuffled on purpose
    r = np.array([0.02, -0.03, 0.04, 0.00, -0.05, 0.01, -0.01, 0.03, 0.05, 0.06])
    # n = 10, alpha = 0.8 -> k = 2 (10 * 0.2 is 1.9999999999999996 before rounding)
    assert historical_k(10, 0.8) == 2
    var, es = historical_var_es(r, 0.8)
    assert var == pytest.approx(0.03)
    assert es == pytest.approx((0.05 + 0.03) / 2)
    # alpha = 0.9 -> k = 1: VaR and ES are both the single worst loss
    assert historical_var_es(r, 0.9) == pytest.approx((0.05, 0.05))


def test_historical_uses_k5_for_500_obs_at_99():
    assert math.ceil(500 * (1 - 0.99)) == 6  # the floating-point trap
    assert historical_k(500, 0.99) == 5

    # Losses 0.001, 0.002, ..., 0.500 in random order
    rng = np.random.default_rng(0)
    r = -rng.permutation(np.arange(1, 501) / 1000)
    var, es = historical_var_es(r, 0.99)
    assert var == pytest.approx(0.496)  # 5th largest loss (k = 6 would give 0.495)
    assert es == pytest.approx(np.mean([0.500, 0.499, 0.498, 0.497, 0.496]))


@pytest.mark.parametrize("alpha", [0.99, 0.975])
def test_historical_matches_normal_formula_on_normal_data(normal_returns, alpha):
    hist_var, hist_es = historical_var_es(normal_returns, alpha)
    z = stats.norm.ppf(alpha)
    assert hist_var == pytest.approx(SIGMA * z, rel=0.02)
    assert hist_es == pytest.approx(SIGMA * stats.norm.pdf(z) / (1 - alpha), rel=0.02)


# --- Student t ------------------------------------------------------------------
def test_student_t_recovers_nu():
    rng = np.random.default_rng(7)
    r = 0.01 * stats.t.rvs(df=5, size=20_000, random_state=rng)
    _, _, nu = student_t_var_es(r, 0.99)
    assert 4 < nu < 6


def test_student_t_rejects_nu_below_one(monkeypatch):
    monkeypatch.setattr(models.stats.t, "fit", lambda *args, **kwargs: (0.8, 0.0, 0.01))
    with pytest.raises(ValueError, match="nu=0.8000 <= 1"):
        student_t_var_es(np.linspace(-0.01, 0.01, 100), 0.99)


# --- All methods ------------------------------------------------------------------
METHODS = {
    "historical": historical_var_es,
    "normal": normal_var_es,
    "student_t": lambda r, a: student_t_var_es(r, a)[:2],
    "ewma": ewma_var_es,
}


@pytest.mark.parametrize("alpha", [0.99, 0.975])
@pytest.mark.parametrize("method", METHODS)
def test_es_at_least_var_and_positive(method, alpha):
    rng = np.random.default_rng(123)
    r = 0.01 * stats.t.rvs(df=4, size=500, random_state=rng)
    var, es = METHODS[method](r, alpha)
    assert var > 0
    assert es >= var
