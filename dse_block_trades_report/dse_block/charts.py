"""Matplotlib charts (PNG, print resolution) with one consistent visual system.

Palette = validated categorical order (blue, orange, aqua ...), single-hue blue
sequential ramp, blue<->red diverging with a grey midpoint. No dual axes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
BLUE_DARK, BLUE_LIGHT = "#184f95", "#9ec5f4"
RED = "#e34948"
NEUTRAL = "#c3c2b7"
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

W_FULL, W_HALF = 7.1, 3.45  # inches (A4 with ~0.6in margins)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.titlepad": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "ytick.left": False, "legend.frameon": False, "legend.fontsize": 7, "text.color": INK,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})


def _bn(x, _=None):
    return f"{x / 1e9:,.0f}" if abs(x) >= 1e10 or x == 0 else f"{x / 1e9:,.1f}"


def _save(fig, out: Path) -> Path:
    fig.savefig(out, dpi=220, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return out


def _subtitle(ax, text):
    """Subtitle sits between the title and the plot; bump the title up to make room."""
    ax.title.set_y(1.0)
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=17)
    ax.text(0, 1.02, text, transform=ax.transAxes, fontsize=7, color=INK2, va="bottom")


# --------------------------------------------------------------------------- #
def monthly_value(m: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(W_FULL, 2.6))
    x = m.index
    colors = [BLUE] * (len(m) - 1) + [BLUE_DARK]
    ax.bar(x, m["value"], width=22, color=colors, edgecolor="white", linewidth=0.8, label="Monthly block value")
    ma = m["value"].rolling(3, min_periods=1).mean()
    ax.plot(x, ma, color=ORANGE, lw=1.8, label="3-month average")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_bn))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.set_title("Monthly block-trade value (TZS bn)")
    _subtitle(ax, "Latest month highlighted (month-to-date)")
    ax.legend(loc="lower right", ncols=2, bbox_to_anchor=(1, 1.0), fontsize=6.8)
    ax.margins(x=0.01)
    return _save(fig, out)


def monthly_share(m: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(W_HALF, 2.3))
    s = m["share"] * 100
    ax.fill_between(m.index, s, color=BLUE, alpha=0.12, lw=0)
    ax.plot(m.index, s, color=BLUE, lw=1.8)
    avg = s.mean()
    ax.axhline(avg, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(1.0, avg, f"avg\n{avg:.0f}%", transform=ax.get_yaxis_transform(), fontsize=6.3, color=INK2,
            va="center", ha="left")
    ax.scatter([m.index[-1]], [s.iloc[-1]], s=22, color=BLUE, zorder=3, edgecolor="white", lw=1.2)
    ax.annotate(f"{s.iloc[-1]:.0f}%", (m.index[-1], s.iloc[-1]), xytext=(4, 4),
                textcoords="offset points", fontsize=7, color=INK)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.set_ylim(0, max(5, s.max() * 1.2))
    ax.set_title("Blocks as % of market turnover")
    return _save(fig, out)


def annual_stacked(a: pd.DataFrame, as_of, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(W_HALF, 2.3))
    years = a.index.astype(str).tolist()
    years[-1] = f"{years[-1]}\nYTD"
    eq = a.get("Equity", pd.Series(0, index=a.index))
    etf = a.get("ETF", pd.Series(0, index=a.index))
    ax.bar(years, eq, color=BLUE, width=0.62, label="Equity", edgecolor="white", linewidth=0.8)
    ax.bar(years, etf, bottom=eq, color=ORANGE, width=0.62, label="ETF", edgecolor="white", linewidth=0.8)
    tot = eq + etf
    for i, v in enumerate(tot):
        ax.text(i, v, _bn(v), ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_bn))
    ax.set_ylim(0, tot.max() * 1.15)
    ax.set_title("Annual block value (TZS bn)")
    ax.legend(loc="upper left")
    return _save(fig, out)


def waterfall(contrib: pd.DataFrame, prev_total: float, cur_total: float, prev_label: str,
              cur_label: str, out: Path, title: str) -> Path:
    labels = [prev_label] + contrib.index.tolist() + [cur_label]
    deltas = contrib["delta"].tolist()
    fig, ax = plt.subplots(figsize=(W_FULL, 2.7))
    running = prev_total
    ax.bar(0, prev_total, color=MUTED, width=0.6)
    ax.text(0, prev_total, _bn(prev_total), ha="center", va="bottom", fontsize=7, color=INK2)
    for i, d in enumerate(deltas, start=1):
        bottom = running if d >= 0 else running + d
        ax.bar(i, abs(d), bottom=bottom, color=BLUE if d >= 0 else RED, width=0.6)
        ax.text(i, running + max(d, 0), f"{'+' if d >= 0 else '−'}{_bn(abs(d))}", ha="center",
                va="bottom", fontsize=7, color=INK2)
        if i < len(deltas) + 1:
            ax.plot([i - 0.3, i + 1.3], [running + d] * 2, color=AXIS, lw=0.6, zorder=0)
        running += d
    n = len(labels) - 1
    ax.bar(n, cur_total, color=BLUE_DARK, width=0.6)
    ax.text(n, cur_total, _bn(cur_total), ha="center", va="bottom", fontsize=7, color=INK, weight="bold")
    ax.set_xticks(range(len(labels)), labels, fontsize=7, color=INK2)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(_bn))
    ax.set_ylim(0, max(prev_total, cur_total, running) * 1.18)
    ax.set_title(title)
    _subtitle(ax, "TZS bn · blue = security added value, red = security subtracted value")
    return _save(fig, out)


def top_symbols(t: pd.DataFrame, total: float, out: Path) -> Path:
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(W_HALF, 2.9))
    ax.barh(t.index, t["value"], color=BLUE, height=0.62)
    for y, (v, n) in enumerate(zip(t["value"], t["deals"])):
        ax.text(v, y, f"  {_bn(v)} bn · {v / total * 100:.0f}% · {n} deals", va="center", fontsize=6.3, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_bn))
    ax.set_xlim(0, t["value"].max() * 1.75)
    ax.tick_params(axis="y", labelcolor=INK, labelsize=7)
    ax.set_title("Top 10 securities, YTD (TZS bn)")
    return _save(fig, out)


def heatmap(mat: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(W_FULL, 3.0))
    data = mat.to_numpy() / 1e9
    im = ax.imshow(data, aspect="auto", cmap=SEQ, vmin=0)
    ax.set_xticks(range(mat.shape[1]), [p.strftime("%b\n%y") for p in mat.columns], fontsize=6.5)
    ax.set_yticks(range(mat.shape[0]), mat.index, fontsize=7, color=INK)
    ax.grid(False)
    ax.set_xticks(np.arange(-.5, mat.shape[1]), minor=True)
    ax.set_yticks(np.arange(-.5, mat.shape[0]), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.tick_params(which="minor", length=0)
    vmax = data.max() if data.size else 1
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=5.6,
                        color="white" if v > vmax * 0.55 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=6, colors=MUTED)
    ax.set_title("Where the blocks went: top securities × month (TZS bn, trailing 12 months)")
    return _save(fig, out)


def size_mix(cur: pd.DataFrame, prev: pd.DataFrame, cur_lbl: str, prev_lbl: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(W_HALF, 2.4))
    y = np.arange(len(cur))
    h = 0.36
    ax.barh(y + h / 2, prev["value_share"] * 100, height=h, color=NEUTRAL, label=prev_lbl)
    ax.barh(y - h / 2, cur["value_share"] * 100, height=h, color=BLUE, label=cur_lbl)
    for yy, v in zip(y, cur["value_share"] * 100):
        ax.text(v, yy - h / 2, f" {v:.0f}%", va="center", fontsize=6.3, color=INK2)
    ax.set_yticks(y, [f"TZS {l}" for l in cur.index], fontsize=6.8, color=INK)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_title("Share of value by deal size")
    ax.legend(loc="lower right")
    return _save(fig, out)


def premium_bars(pm: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(W_HALF, 2.6))
    v = pm["premium"] * 100
    ax.barh(pm.index, v, color=[BLUE if x >= 0 else RED for x in v], height=0.6)
    ax.axvline(0, color=AXIS, lw=0.8)
    lim = max(1.0, np.nanmax(np.abs(v)) * 1.6)
    for yy, x in enumerate(v):
        ax.text(x + (0.04 if x >= 0 else -0.04) * lim, yy, f"{x:+.1f}%", va="center",
                ha="left" if x >= 0 else "right", fontsize=6.3, color=INK2)
    ax.set_xlim(-lim, lim)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.tick_params(axis="y", labelcolor=INK, labelsize=7)
    ax.set_title("Block price vs prior close, YTD")
    _subtitle(ax, "Value-weighted · top-10 securities by block value")
    return _save(fig, out)


def impact_curve(imp: pd.DataFrame, abnormal: bool, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(W_HALF, 2.6))
    ax.fill_between(imp["t"], imp["p25"] * 100, imp["p75"] * 100, color=BLUE, alpha=0.13, lw=0,
                    label="Inter-quartile range")
    ax.plot(imp["t"], imp["mean"] * 100, color=BLUE, lw=1.8, label="Mean")
    ax.plot(imp["t"], imp["median"] * 100, color=ORANGE, lw=1.4, ls=(0, (4, 2)), label="Median")
    ax.axvline(0, color=INK2, lw=0.7)
    ax.axhline(0, color=AXIS, lw=0.7)
    ax.text(0.2, 0.04, "block day", transform=ax.get_xaxis_transform(), fontsize=6.3, color=INK2)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax.set_xlabel("Trading days relative to block (T)", fontsize=6.8)
    ax.set_title(("Abnormal" if abnormal else "Cumulative") + " return around blocks")
    _subtitle(ax, f"Rebased at T-1 · {int(imp['n'].iloc[0]):,} events, trailing 2 years")
    ax.legend(loc="upper right", fontsize=6.3)
    return _save(fig, out)


def index_context(hi: pd.Series, blocks: pd.DataFrame, name: str, as_of, out: Path) -> Path:
    start = as_of - pd.DateOffset(weeks=104)
    h = hi[(hi.index >= start) & (hi.index <= as_of)]
    wk = blocks[(blocks.date >= start) & (blocks.date <= as_of)].set_index("date")["value"].resample("W-FRI").sum()
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W_FULL, 3.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    a1.plot(h.index, h.values, color=BLUE, lw=1.8)
    a1.set_title(f"Market context: {name} level (top) vs weekly block value, TZS bn (bottom)")
    a1.annotate(f"{h.iloc[-1]:,.0f}", (h.index[-1], h.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                fontsize=7, va="center")
    a2.bar(wk.index, wk.values, width=5, color=BLUE)
    a2.yaxis.set_major_formatter(mtick.FuncFormatter(_bn))
    a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    fig.tight_layout(h_pad=0.6)
    return _save(fig, out)


def broker_flows(t: pd.DataFrame, out: Path) -> Path:
    t = t.iloc[::-1]
    fig, ax = plt.subplots(figsize=(W_FULL, 2.7))
    y = np.arange(len(t))
    h = 0.38
    ax.barh(y + h / 2, t["buy"], height=h, color=BLUE, label="Bought")
    ax.barh(y - h / 2, t["sell"], height=h, color=ORANGE, label="Sold")
    for yy, (b, s, n) in enumerate(zip(t["buy"], t["sell"], t["net"])):
        ax.text(max(b, s), yy, f"   net {'+' if n >= 0 else '−'}{_bn(abs(n))} bn", va="center",
                fontsize=6.5, color=INK2)
    ax.set_yticks(y, t.index, fontsize=7, color=INK)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(_bn))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_xlim(0, t[["buy", "sell"]].max().max() * 1.3)
    ax.set_title("Most active members in block trades, YTD (TZS bn)")
    ax.legend(loc="lower right")
    return _save(fig, out)


def period_bars(labels, cur, prev, out: Path) -> Path:
    """Small multiples: current vs comparison value for each period."""
    fig, axes = plt.subplots(1, len(labels), figsize=(W_FULL, 1.55))
    for ax, l, c, p in zip(axes, labels, cur, prev):
        ax.bar([0, 1], [p, c], color=[NEUTRAL, BLUE], width=0.7)
        ax.set_xticks([0, 1], ["Prior", "Now"], fontsize=6.3)
        ax.set_title(l, fontsize=7.5)
        ax.yaxis.set_visible(False)
        ax.grid(False)
        top = max(c, p, 1)
        ax.set_ylim(0, top * 1.3)
        for x, v in ((0, p), (1, c)):
            ax.text(x, v, _bn(v), ha="center", va="bottom", fontsize=6.3, color=INK2)
    fig.suptitle("Block value by period, TZS bn (prior = same elapsed span of previous period)",
                 x=0.01, ha="left", fontsize=8, color=INK2, y=1.04)
    fig.tight_layout(w_pad=1.2)
    return _save(fig, out)
