"""Tests for the sub-period statistics in src/periods.py, on a hand-made forecast table."""

import numpy as np
import pandas as pd
import pytest

from src.periods import period_stats, period_table, select_period

ALPHA = 0.99
VAR_COL, ES_COL = "var_normal_0.99", "es_normal_0.99"


@pytest.fixture
def forecasts() -> pd.DataFrame:
    """Eight business days. Inside 2024-01-03 .. 2024-01-10 (six days):

    date        ret     loss   VaR    ES     exception
    01-03      -0.030   0.030  0.020  0.025  yes
    01-04       0.010  -0.010  0.025  0.030  no
    01-05      -0.020   0.020  0.040  0.050  no   <- highest VaR (day 2 of the period)
    01-08      -0.050   0.050  0.030  0.040  yes  <- worst loss
    01-09       0.000   0.000  0.040  0.050  no   (VaR ties with 01-05: first one counts)
    01-10      -0.021   0.021  0.020  0.030  yes  (last day: tests the inclusive end)

    The first (01-02) and last (01-11) rows are outside the period and both are
    big exceptions, so including them by mistake would change every count.
    """
    idx = pd.to_datetime(
        ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
         "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"]
    )
    return pd.DataFrame(
        {
            "ret": [-0.20, -0.030, 0.010, -0.020, -0.050, 0.000, -0.021, -0.30],
            VAR_COL: [0.01, 0.020, 0.025, 0.040, 0.030, 0.040, 0.020, 0.50],
            ES_COL: [0.02, 0.025, 0.030, 0.050, 0.040, 0.050, 0.030, 0.60],
        },
        index=idx,
    )


def stats(forecasts):
    return period_stats(forecasts, "2024-01-03", "2024-01-10", "normal", ALPHA)


def test_period_bounds_are_inclusive(forecasts):
    sub = select_period(forecasts, "2024-01-03", "2024-01-10")
    assert sub.index[0] == pd.Timestamp("2024-01-03")
    assert sub.index[-1] == pd.Timestamp("2024-01-10")
    assert len(sub) == 6
    # A one-day period works when start == end
    assert len(select_period(forecasts, "2024-01-10", "2024-01-10")) == 1


def test_counts(forecasts):
    s = stats(forecasts)
    assert s["days"] == 6
    assert s["exceptions"] == 3  # 01-03, 01-08, 01-10
    assert s["expected"] == pytest.approx(0.06)


def test_means(forecasts):
    s = stats(forecasts)
    assert s["mean_var_pct"] == pytest.approx(100 * np.mean([0.020, 0.025, 0.040, 0.030, 0.040, 0.020]))
    assert s["exc_loss_pct"] == pytest.approx(100 * (0.030 + 0.050 + 0.021) / 3)
    assert s["exc_es_pct"] == pytest.approx(100 * (0.025 + 0.040 + 0.030) / 3)
    assert s["loss_to_es"] == pytest.approx((0.030 + 0.050 + 0.021) / (0.025 + 0.040 + 0.030))


def test_var_path_and_worst_day(forecasts):
    s = stats(forecasts)
    assert s["var_first_pct"] == pytest.approx(2.0)
    assert s["var_max_pct"] == pytest.approx(4.0)
    assert s["var_max_date"] == pd.Timestamp("2024-01-05")  # first of the two tied maxima
    assert s["var_max_tdays"] == 2
    assert s["worst_loss_date"] == pd.Timestamp("2024-01-08")
    assert s["worst_loss_pct"] == pytest.approx(5.0)
    assert s["var_on_worst_pct"] == pytest.approx(3.0)


def test_no_exceptions_gives_nan_not_error(forecasts):
    s = period_stats(forecasts, "2024-01-04", "2024-01-05", "normal", ALPHA)
    assert s["exceptions"] == 0
    assert np.isnan(s["exc_loss_pct"]) and np.isnan(s["loss_to_es"])


def test_empty_or_reversed_period_raises(forecasts):
    with pytest.raises(ValueError, match="no forecasts"):
        select_period(forecasts, "2024-01-06", "2024-01-07")  # a weekend
    with pytest.raises(ValueError, match="after end"):
        select_period(forecasts, "2024-01-10", "2024-01-03")


def test_period_table_index(forecasts):
    table = period_table(
        forecasts, periods={"p": ("2024-01-03", "2024-01-10")}, methods=["normal"], levels=[ALPHA]
    )
    assert list(table.index) == [("p", ALPHA, "normal")]
    assert table.loc[("p", ALPHA, "normal"), "exceptions"] == 3
