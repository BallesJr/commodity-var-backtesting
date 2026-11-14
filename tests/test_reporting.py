"""Tests for the VaR / ES summary table in src/reporting.py."""

import pytest

from src.config import CONFIDENCE_LEVELS, NOTIONAL, WINDOW
from src.data import load_returns
from src.reporting import METHODS, var_es_table


@pytest.fixture(scope="module")
def table():
    _, port, _ = load_returns()
    return var_es_table(port.to_numpy()[-WINDOW:])


def test_table_has_every_method_and_level(table):
    expected = {(m, a) for m in METHODS for a in CONFIDENCE_LEVELS}
    assert set(table.index) == expected


@pytest.mark.parametrize("alpha", CONFIDENCE_LEVELS)
@pytest.mark.parametrize("method", METHODS)
def test_usd_equals_pct_times_notional(table, method, alpha):
    row = table.loc[(method, alpha)]
    assert row["var_usd"] == pytest.approx(row["var_pct"] / 100 * NOTIONAL, rel=1e-12)
    assert row["es_usd"] == pytest.approx(row["es_pct"] / 100 * NOTIONAL, rel=1e-12)


@pytest.mark.parametrize("alpha", CONFIDENCE_LEVELS)
@pytest.mark.parametrize("method", METHODS)
def test_table_matches_direct_model_call(table, method, alpha):
    _, port, _ = load_returns()
    var, es = METHODS[method](port.to_numpy()[-WINDOW:], alpha)
    row = table.loc[(method, alpha)]
    assert row["var_pct"] == pytest.approx(100 * var, rel=1e-12)
    assert row["es_usd"] == pytest.approx(NOTIONAL * es, rel=1e-12)
