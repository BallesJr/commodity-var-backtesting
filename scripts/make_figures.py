"""Create the README figures from the saved backtest forecasts.

Reads results/backtest_forecasts.csv (run scripts/run_backtest.py first); the only
model fitted here is the full-sample GARCH used for the QQ plot of its residuals.
Writes PNG files to results/figures/.

Usage:
    python scripts/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only, no window

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
from scipy import stats  # noqa: E402

# Make the project root importable when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import BASEL_WINDOW, column, exceptions  # noqa: E402
from src.config import FORECASTS_CSV, METHODS, RESULTS_DIR  # noqa: E402
from src.data import load_returns  # noqa: E402
from src.garch import fit_garch, garch_variance  # noqa: E402
from src.periods import load_forecasts  # noqa: E402

FIG_DIR = RESULTS_DIR / "figures"
DPI = 150
WIDTH = 10  # inches -> 1500 px at 150 dpi, sharp on GitHub at README width

# --- Colours -------------------------------------------------------------------
# One fixed colour per method in every figure (categorical slots in a fixed order,
# checked for colour-blind separation between neighbours).
METHOD_COLORS = {
    "historical": "#2a78d6",  # blue
    "normal": "#eb6834",  # orange
    "student_t": "#1baf7a",  # aqua
    "ewma": "#eda100",  # yellow
    "garch": "#e87ba4",  # magenta
    "fhs": "#008300",  # green
}
METHOD_LABELS = {
    "historical": "Historical",
    "normal": "Normal",
    "student_t": "Student t",
    "ewma": "EWMA",
    "garch": "GARCH",
    "fhs": "FHS",
}
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
LOSS_GRAY = "#a3a29c"
EXCEPTION_RED = "#d03b3b"
# Traffic-light zones: pale washes so the lines on top stay readable
ZONE_FILLS = {"green": "#dff1df", "yellow": "#fdf0cc", "red": "#f8dcdc"}
ZONE_TEXT = {"green": "#2b6e2b", "yellow": "#8a6400", "red": "#a42e2e"}

ALPHA_99 = 0.99
CRASH_DAY = pd.Timestamp("2020-04-21")
LOSS_LABEL = "Loss (% of notional, negative = gain)"


def set_style() -> None:
    """Shared matplotlib style for every figure."""
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.15,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlecolor": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "lines.linewidth": 1.6,
        }
    )


def pct_axis(ax, decimals: int = 0) -> None:
    """Format the y axis of a fraction-valued series as percentages."""
    # Unicode minus, to match matplotlib's default tick labels in the other figures
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{100 * v:.{decimals}f}%".replace("-", "−"))
    )


def year_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def save(fig, name: str) -> Path:
    path = FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


# =============================================================================
# 1. Historical vs FHS over the whole backtest
# =============================================================================
def fig_hist_vs_fhs(fc: pd.DataFrame) -> Path:
    # Whole backtest (from 2019-12-30), so the counts match results/backtest_summary.csv
    loss = -fc["ret"]
    fig, axes = plt.subplots(2, 1, figsize=(WIDTH, 6.6), sharex=True, sharey=True)

    for ax, method in zip(axes, ["historical", "fhs"]):
        var = fc[column("var", method, ALPHA_99)]
        hits = exceptions(fc["ret"], var).astype(bool)

        ax.scatter(fc.index, loss, s=4, color=LOSS_GRAY, linewidths=0, label="Daily loss",
                   zorder=2)
        ax.plot(fc.index, var, color=METHOD_COLORS[method], lw=1.4, label="99% VaR", zorder=3)
        ax.scatter(fc.index[hits], loss[hits], s=26, color=EXCEPTION_RED,
                   edgecolor="white", linewidths=0.8, label="Exception (loss > VaR)", zorder=4)
        ax.axhline(0, color=AXIS, lw=0.8, zorder=1)
        ax.set_title(f"{METHOD_LABELS[method]}: {hits.sum()} exceptions "
                     f"in {len(fc):,} days (expected {0.01 * len(fc):.0f})")
        ax.set_ylabel(LOSS_LABEL)
        pct_axis(ax)

    axes[0].legend(loc="upper right", ncol=3, markerscale=1.2)
    year_axis(axes[1])
    axes[1].set_xlim(pd.Timestamp("2019-12-15"), pd.Timestamp("2026-01-15"))
    fig.tight_layout(h_pad=1.5)
    return save(fig, "backtest_hist_vs_fhs.png")


# =============================================================================
# 2. COVID crash zoom
# =============================================================================
def fig_covid_zoom(fc: pd.DataFrame) -> Path:
    fc = fc.loc["2020-02-01":"2020-06-30"]
    loss = -fc["ret"]
    fig, ax = plt.subplots(figsize=(WIDTH, 5.0))

    ax.bar(fc.index, loss, width=0.8, color=LOSS_GRAY, label="Daily loss", zorder=2)
    for method in ["normal", "historical", "ewma", "fhs"]:
        ax.plot(fc.index, fc[column("var", method, ALPHA_99)], color=METHOD_COLORS[method],
                lw=1.8, label=f"{METHOD_LABELS[method]} 99% VaR", zorder=3)
    ax.axhline(0, color=AXIS, lw=0.8, zorder=1)

    crash_loss = loss.loc[CRASH_DAY]
    ax.annotate(
        f"{CRASH_DAY:%d %b %Y}: loss {100 * crash_loss:.1f}%\n"
        "(Brent −24%, the day the May WTI contract expired)",
        xy=(CRASH_DAY, crash_loss),
        xytext=(pd.Timestamp("2020-05-08"), crash_loss - 0.002),
        fontsize=9.5,
        color=INK,
        va="center",
        arrowprops={"arrowstyle": "-", "color": INK_SECONDARY, "lw": 0.8},
    )

    ax.set_ylabel(LOSS_LABEL)
    pct_axis(ax)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_xlim(pd.Timestamp("2020-01-30"), pd.Timestamp("2020-07-02"))
    # Legend above the plot area so it cannot cover any bar
    handles, labels = ax.get_legend_handles_labels()
    order = [4, 0, 1, 2, 3]  # daily loss first, then the VaR lines
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, borderaxespad=0.3,
              handlelength=1.6, columnspacing=1.4)
    fig.tight_layout()
    return save(fig, "covid_zoom.png")


# =============================================================================
# 3. Basel traffic light
# =============================================================================
def fig_basel(fc: pd.DataFrame) -> Path:
    counts = {}
    for method in ["normal", "historical", "fhs"]:
        hits = pd.Series(exceptions(fc["ret"], fc[column("var", method, ALPHA_99)]),
                         index=fc.index)
        counts[method] = hits.rolling(BASEL_WINDOW).sum().dropna()

    # One panel per method: the counts are small integers and often identical across
    # methods, so lines drawn on one axis would hide each other.
    top = max(int(c.max()) for c in counts.values()) + 1
    first = min(c.index[0] for c in counts.values())
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 7.2), sharex=True, sharey=True)
    bands = [("green", -0.5, 4.5, "Green (0-4)"), ("yellow", 4.5, 9.5, "Yellow (5-9)"),
             ("red", 9.5, top + 0.5, "Red (10+)")]

    for ax, (method, c) in zip(axes, counts.items()):
        ax.grid(False)
        for zone, lo, hi, label in bands:
            ax.axhspan(lo, hi, color=ZONE_FILLS[zone], lw=0, zorder=0)
            ax.text(1.008, (lo + hi) / 2, label, transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=9, color=ZONE_TEXT[zone])
        ax.step(c.index, c, where="post", color=METHOD_COLORS[method], lw=1.8, zorder=3)
        worst = c.max()
        ax.set_title(f"{METHOD_LABELS[method]}: worst {worst:.0f} exceptions "
                     f"(first reached {c.idxmax():%d %b %Y}), {c.iloc[-1]:.0f} at the end")
        ax.set_ylim(-0.5, top + 0.5)
        ax.set_yticks([0, 4, 9, top] if top > 10 else [0, 4, 9])

    axes[1].set_ylabel(f"99% VaR exceptions in the last {BASEL_WINDOW} days")
    year_axis(axes[-1])
    axes[-1].set_xlim(first - pd.Timedelta(days=15), fc.index[-1] + pd.Timedelta(days=15))
    fig.tight_layout(h_pad=1.2)
    return save(fig, "basel_traffic_light.png")


# =============================================================================
# 4. Exceptions per year
# =============================================================================
def fig_exceptions_by_year(fc: pd.DataFrame) -> Path:
    fc = fc.loc["2020-01-01":"2025-12-31"]
    years = np.arange(2020, 2026)
    counts = pd.DataFrame(
        {m: pd.Series(exceptions(fc["ret"], fc[column("var", m, ALPHA_99)]), index=fc.index)
         .groupby(fc.index.year).sum().reindex(years, fill_value=0) for m in METHODS}
    )

    fig, ax = plt.subplots(figsize=(WIDTH, 4.6))
    n = len(METHODS)
    group_width = 0.82
    bar_w = group_width / n
    for i, method in enumerate(METHODS):
        offset = (i - (n - 1) / 2) * bar_w
        ax.bar(years + offset, counts[method], width=bar_w, color=METHOD_COLORS[method],
               edgecolor="white", linewidth=1.0, label=METHOD_LABELS[method], zorder=2)
        # A zero-height bar is invisible: label it so it does not read as missing data
        for year, value in counts[method].items():
            if value == 0:
                ax.text(year + offset, 0.12, "0", ha="center", va="bottom", fontsize=8.5,
                        color=INK_SECONDARY)

    ax.axhline(2.5, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=3)
    ax.text(years[-1] + 0.5, 2.5, "  Expected per year: 2.5", va="center", ha="left",
            fontsize=9.5, color=INK)

    ax.set_xticks(years)
    ax.set_xlim(years[0] - 0.5, years[-1] + 0.5)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel("99% VaR exceptions")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout()
    return save(fig, "exceptions_by_year.png")


# =============================================================================
# 5. QQ plot of the GARCH standardised residuals
# =============================================================================
def fig_qq_residuals() -> Path:
    _, port, _ = load_returns()
    eps = 100 * port.to_numpy()
    fit = fit_garch(eps)
    z = np.sort(eps / np.sqrt(garch_variance(fit.params, eps, fit.backcast)))
    n = z.size
    theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    lim = np.ceil(max(np.abs(z).max(), np.abs(theoretical).max())) + 0.3
    ax.plot([-lim, lim], [-lim, lim], color=INK_SECONDARY, lw=1.2, ls=(0, (5, 3)),
            label="N(0, 1) reference (45° line)", zorder=2)
    ax.scatter(theoretical, z, s=10, color=METHOD_COLORS["garch"], edgecolor="none",
               alpha=0.85, label="GARCH(1,1) residuals z", zorder=3)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Normal quantiles")
    ax.set_ylabel("Standardised residual quantiles")
    ax.text(0.97, 0.04,
            f"Full sample, n = {n:,}\nExcess kurtosis {stats.kurtosis(z):.2f}\n"
            f"Skewness {stats.skew(z):.2f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.5, color=INK_SECONDARY)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, "qq_residuals.png")


def main() -> None:
    if not FORECASTS_CSV.exists():
        sys.exit(f"{FORECASTS_CSV} not found: run scripts/run_backtest.py first")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    fc = load_forecasts()

    paths = [
        fig_hist_vs_fhs(fc),
        fig_covid_zoom(fc),
        fig_basel(fc),
        fig_exceptions_by_year(fc),
        fig_qq_residuals(),
    ]
    for p in paths:
        print(f"Saved {p.relative_to(RESULTS_DIR.parent)}  ({p.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
