"""Analytics layer: period scorecards, drivers, concentration, pricing, impact."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SIZE_BUCKETS = [0, 100e6, 500e6, 1e9, 5e9, np.inf]
SIZE_LABELS = ["< 100 mn", "100-500 mn", "500 mn-1 bn", "1-5 bn", "> 5 bn"]


# --------------------------------------------------------------------------- #
# Formatting helpers (shared with narrative / charts)
# --------------------------------------------------------------------------- #
def fmt_tzs(x: float, unit: bool = True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    a = abs(x)
    for div, suf in ((1e12, "tn"), (1e9, "bn"), (1e6, "mn"), (1e3, "k")):
        if a >= div:
            s = f"{x / div:,.2f} {suf}" if a / div < 100 else f"{x / div:,.1f} {suf}"
            return f"TZS {s}" if unit else s
    return f"TZS {x:,.0f}" if unit else f"{x:,.0f}"


def fmt_pct(x: float, signed: bool = True, dp: int = 1) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x * 100:+.{dp}f}%" if signed else f"{x * 100:.{dp}f}%"


def fmt_num(x: float) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:,.0f}"


def pct_change(cur: float, prev: float) -> float:
    if prev is None or not np.isfinite(prev) or prev == 0:
        return np.nan
    return cur / prev - 1


# --------------------------------------------------------------------------- #
# Periods
# --------------------------------------------------------------------------- #
@dataclass
class Period:
    key: str
    label: str
    start: pd.Timestamp
    end: pd.Timestamp
    prev_start: pd.Timestamp
    prev_end: pd.Timestamp
    compare_label: str


def build_periods(as_of: pd.Timestamp, trading_days: pd.DatetimeIndex) -> list[Period]:
    """Period-to-date windows, each compared with the same elapsed span of the prior period."""
    td = trading_days[trading_days <= as_of]
    prev_day = td[-2] if len(td) > 1 else as_of - pd.Timedelta(days=1)

    def span(start, prev_start, prev_period_end):
        elapsed = as_of - start
        return start, as_of, prev_start, min(prev_start + elapsed, prev_period_end)

    wk = as_of - pd.Timedelta(days=as_of.weekday())
    mo = as_of.replace(day=1)
    q = pd.Timestamp(year=as_of.year, month=3 * ((as_of.month - 1) // 3) + 1, day=1)
    y = pd.Timestamp(year=as_of.year, month=1, day=1)
    prev_q = q - pd.DateOffset(months=3)
    prev_y = y - pd.DateOffset(years=1)
    prev_mo = mo - pd.DateOffset(months=1)
    return [
        Period("D", "Daily", as_of, as_of, prev_day, prev_day, f"vs {prev_day:%d %b}"),
        Period("W", "Week-to-date", *span(wk, wk - pd.Timedelta(days=7), wk - pd.Timedelta(days=1)),
               "vs same span prior week"),
        Period("M", "Month-to-date", *span(mo, prev_mo, mo - pd.Timedelta(days=1)),
               f"vs 1-{min(as_of.day, (mo - pd.Timedelta(days=1)).day)} {prev_mo:%b}"),
        Period("Q", "Quarter-to-date", *span(q, prev_q, q - pd.Timedelta(days=1)),
               f"vs same span Q{(prev_q.month - 1) // 3 + 1} {prev_q.year}"),
        Period("Y", "Year-to-date", *span(y, prev_y, y - pd.Timedelta(days=1)),
               f"vs YTD {prev_y.year}"),
    ]


def _between(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df[(df["date"] >= start) & (df["date"] <= end)]


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def window_metrics(blocks: pd.DataFrame, prices: pd.DataFrame, start, end, incl: bool) -> dict:
    b = _between(blocks, start, end)
    p = _between(prices, start, end)
    val = float(b["value"].sum())
    mkt = float(p["turnover"].sum())
    denom = mkt if incl else mkt + val
    by_sym = b.groupby("symbol")["value"].sum().sort_values(ascending=False)
    shares = by_sym / val if val else by_sym
    prem = b.dropna(subset=["premium"]) if "premium" in b else b.iloc[0:0]
    return {
        "value": val,
        "volume": float(b["qty"].sum()),
        "deals": int(len(b)),
        "avg_deal": val / len(b) if len(b) else np.nan,
        "max_deal": float(b["value"].max()) if len(b) else np.nan,
        "securities": int(b["symbol"].nunique()),
        "market_turnover": mkt,
        "block_share": val / denom if denom else np.nan,
        "top_symbol": by_sym.index[0] if len(by_sym) else "-",
        "top_share": float(shares.iloc[0]) if len(shares) else np.nan,
        "hhi": float((shares ** 2).sum() * 10000) if len(shares) else np.nan,
        "wavg_premium": float(np.average(prem["premium"], weights=prem["value"])) if len(prem) else np.nan,
        "trading_days": int(p["date"].nunique()),
    }


def add_reference_prices(blocks: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach the previous session's close (T-1) as reference price -> premium/discount."""
    px = prices[["date", "symbol", "close"]].sort_values("date")
    px = px.assign(ref_date=px["date"] + pd.Timedelta(days=1))
    b = blocks.sort_values("date").copy()
    out = pd.merge_asof(b, px[["ref_date", "symbol", "close"]].rename(columns={"close": "ref_close"}),
                        left_on="date", right_on="ref_date", by="symbol", direction="backward",
                        tolerance=pd.Timedelta(days=10))
    out["premium"] = out["price"] / out["ref_close"] - 1
    out.loc[out["premium"].abs() > 0.5, "premium"] = np.nan  # guard against corporate-action gaps
    return out.drop(columns=["ref_date"])


def classify_assets(blocks: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    etf_syms = set(prices.loc[prices["asset_class"] == "ETF", "symbol"])
    b = blocks.copy()
    b["asset_class"] = np.where(b["symbol"].isin(etf_syms) | b["symbol"].str.contains("ETF"), "ETF", "Equity")
    return b


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def contribution_by_symbol(blocks, cur: tuple, prev: tuple, top_n: int = 6) -> pd.DataFrame:
    c = _between(blocks, *cur).groupby("symbol")["value"].sum()
    p = _between(blocks, *prev).groupby("symbol")["value"].sum()
    d = pd.concat([c.rename("cur"), p.rename("prev")], axis=1).fillna(0)
    d["delta"] = d["cur"] - d["prev"]
    d = d.reindex(d["delta"].abs().sort_values(ascending=False).index)
    head, tail = d.iloc[:top_n], d.iloc[top_n:]
    if len(tail):
        head = pd.concat([head, pd.DataFrame({"cur": [tail.cur.sum()], "prev": [tail.prev.sum()],
                                              "delta": [tail.delta.sum()]}, index=["Others"])])
    return head


def count_vs_size(cur: dict, prev: dict) -> dict:
    """Log decomposition: Value = Deals x Avg size."""
    if not (cur["deals"] and prev["deals"] and prev["value"]):
        return {"count_effect": np.nan, "size_effect": np.nan, "total": np.nan}
    lc = np.log(cur["deals"] / prev["deals"])
    ls = np.log(cur["avg_deal"] / prev["avg_deal"])
    tot = cur["value"] / prev["value"] - 1
    s = lc + ls
    return {"count_effect": tot * lc / s if s else 0, "size_effect": tot * ls / s if s else 0,
            "total": tot, "deal_growth": cur["deals"] / prev["deals"] - 1,
            "size_growth": cur["avg_deal"] / prev["avg_deal"] - 1}


def monthly_series(blocks, prices, incl: bool, months: int = 36, as_of=None) -> pd.DataFrame:
    b = blocks.set_index("date").resample("MS").agg(value=("value", "sum"), deals=("value", "size"))
    t = prices.set_index("date")["turnover"].resample("MS").sum().rename("turnover")
    m = b.join(t, how="outer").fillna(0)
    denom = m["turnover"] if incl else m["turnover"] + m["value"]
    m["share"] = np.where(denom > 0, m["value"] / denom, np.nan)
    if as_of is not None:
        m = m[m.index <= as_of]
    return m.iloc[-months:]


def annual_by_class(blocks, as_of) -> pd.DataFrame:
    a = blocks[blocks.date <= as_of].assign(year=lambda x: x.date.dt.year)
    return a.pivot_table(index="year", columns="asset_class", values="value", aggfunc="sum", fill_value=0)


def heatmap_matrix(blocks, as_of, n_sym: int = 12) -> pd.DataFrame:
    start = (as_of - pd.DateOffset(months=11)).replace(day=1)
    b = _between(blocks, start, as_of)
    top = b.groupby("symbol")["value"].sum().nlargest(n_sym).index
    mat = (b[b.symbol.isin(top)].assign(month=lambda x: x.date.dt.to_period("M"))
           .pivot_table(index="symbol", columns="month", values="value", aggfunc="sum", fill_value=0))
    mat = mat.reindex(columns=pd.period_range(start, as_of, freq="M"), fill_value=0)
    return mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]


def size_distribution(blocks, start, end) -> pd.DataFrame:
    b = _between(blocks, start, end)
    cut = pd.cut(b["value"], SIZE_BUCKETS, labels=SIZE_LABELS, right=False)
    g = b.groupby(cut, observed=False)["value"].agg(["size", "sum"])
    g["count_share"] = g["size"] / g["size"].sum() if g["size"].sum() else 0
    g["value_share"] = g["sum"] / g["sum"].sum() if g["sum"].sum() else 0
    return g


def premium_by_symbol(blocks, start, end, n: int = 10) -> pd.DataFrame:
    b = _between(blocks, start, end).dropna(subset=["premium"])
    if b.empty:
        return pd.DataFrame(columns=["premium", "value", "deals"])
    g = b.groupby("symbol").apply(lambda x: pd.Series({
        "premium": np.average(x["premium"], weights=x["value"]), "value": x["value"].sum(), "deals": len(x)}),
        include_groups=False)
    return g.nlargest(n, "value").sort_values("premium")


def broker_tables(blocks, start, end, n: int = 8) -> dict | None:
    b = _between(blocks, start, end)
    if b["buyer"].isna().all() or b["seller"].isna().all():
        return None
    buy = b.groupby("buyer")["value"].sum()
    sell = b.groupby("seller")["value"].sum()
    gross = buy.add(sell, fill_value=0) / 2
    top = gross.nlargest(n).index
    t = pd.DataFrame({"buy": buy.reindex(top).fillna(0), "sell": sell.reindex(top).fillna(0)})
    t["net"] = t["buy"] - t["sell"]
    cross = b[b["buyer"] == b["seller"]]["value"].sum()
    tot = b["value"].sum()
    share_top3 = gross.nlargest(3).sum() / gross.sum() if gross.sum() else np.nan
    return {"table": t, "cross_share": cross / tot if tot else np.nan, "top3_share": share_top3,
            "members": int(pd.concat([b["buyer"], b["seller"]]).nunique())}


def price_impact(blocks, prices, index_series: pd.Series | None, start, end,
                 pre: int = 5, post: int = 10) -> pd.DataFrame | None:
    """Event study: average cumulative (abnormal) return from T-pre..T+post, rebased at T-1."""
    events = _between(blocks, start, end).groupby(["symbol", "date"])["value"].sum().reset_index()
    if events.empty:
        return None
    paths = []
    piv = prices.pivot_table(index="date", columns="symbol", values="close").sort_index().ffill()
    dates = piv.index
    idx = index_series.reindex(dates).ffill() if index_series is not None and len(index_series) else None
    for _, e in events.iterrows():
        if e.symbol not in piv.columns:
            continue
        pos = dates.searchsorted(e.date)
        if pos - pre - 1 < 0 or pos + post >= len(dates) or pos >= len(dates) or dates[pos] != e.date:
            continue
        s = piv[e.symbol].iloc[pos - pre - 1: pos + post + 1].to_numpy()
        base = piv[e.symbol].iloc[pos - 1]
        r = s / base - 1
        if idx is not None:
            ib = idx.iloc[pos - 1]
            r = r - (idx.iloc[pos - pre - 1: pos + post + 1].to_numpy() / ib - 1)
        if np.isfinite(r).all():
            paths.append(r)
    if len(paths) < 10:
        return None
    arr = np.vstack(paths)
    rel = np.arange(-pre - 1, post + 1)
    return pd.DataFrame({"t": rel, "mean": arr.mean(0), "median": np.median(arr, 0),
                         "p25": np.percentile(arr, 25, 0), "p75": np.percentile(arr, 75, 0),
                         "n": len(paths)})


def clustering_stats(blocks, start, end) -> dict:
    b = _between(blocks, start, end)
    if b.empty:
        return {}
    qe = b["date"].dt.month.isin([3, 6, 9, 12]) & (b["date"].dt.day >= 15)
    top_day = b.groupby("date")["value"].sum()
    return {"quarter_end_share": b.loc[qe, "value"].sum() / b["value"].sum(),
            "busiest_day": top_day.idxmax(), "busiest_day_value": top_day.max(),
            "active_days": int(b["date"].nunique())}


# --------------------------------------------------------------------------- #
# Bundle everything the report needs
# --------------------------------------------------------------------------- #
@dataclass
class ReportData:
    as_of: pd.Timestamp
    periods: list[Period]
    scorecard: dict[str, dict]
    prev_scorecard: dict[str, dict]
    blocks: pd.DataFrame
    prices: pd.DataFrame
    index: pd.DataFrame
    headline_index: str | None
    notes: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)


def build(blocks, prices, index, notes, headline_index="DSEI", incl=True, as_of=None) -> ReportData:
    blocks = classify_assets(blocks, prices)
    blocks = add_reference_prices(blocks, prices)
    trading_days = pd.DatetimeIndex(sorted(set(prices["date"]) | set(blocks["date"])))
    as_of = pd.Timestamp(as_of) if as_of else max(blocks["date"].max(), prices["date"].max())
    periods = build_periods(as_of, trading_days)
    sc = {p.key: window_metrics(blocks, prices, p.start, p.end, incl) for p in periods}
    psc = {p.key: window_metrics(blocks, prices, p.prev_start, p.prev_end, incl) for p in periods}

    hi_name, hi_series = None, None
    if len(index):
        names = index["index"].unique()
        match = [n for n in names if headline_index.upper() in n.upper()]
        hi_name = match[0] if match else names[0]
        hi_series = index[index["index"] == hi_name].set_index("date")["value"].sort_index()

    Y = next(p for p in periods if p.key == "Y")
    Q = next(p for p in periods if p.key == "Q")
    rd = ReportData(as_of, periods, sc, psc, blocks, prices, index, hi_name, notes)
    rd.extras = {
        "monthly": monthly_series(blocks, prices, incl, 36, as_of),
        "annual": annual_by_class(blocks, as_of),
        "contrib_ytd": contribution_by_symbol(blocks, (Y.start, Y.end), (Y.prev_start, Y.prev_end)),
        "decomp_ytd": count_vs_size(sc["Y"], psc["Y"]),
        "decomp_qtd": count_vs_size(sc["Q"], psc["Q"]),
        "heatmap": heatmap_matrix(blocks, as_of),
        "sizes_ytd": size_distribution(blocks, Y.start, Y.end),
        "sizes_prev": size_distribution(blocks, Y.prev_start, Y.prev_end),
        "premium_ytd": premium_by_symbol(blocks, Y.start, Y.end),
        "brokers_ytd": broker_tables(blocks, Y.start, Y.end),
        "impact": price_impact(blocks, prices, hi_series,
                               as_of - pd.DateOffset(years=2), as_of - pd.Timedelta(days=21)),
        "cluster_ytd": clustering_stats(blocks, Y.start, Y.end),
        "top_symbols_ytd": _between(blocks, Y.start, Y.end).groupby("symbol")
                           .agg(value=("value", "sum"), deals=("value", "size"), qty=("qty", "sum"))
                           .nlargest(10, "value"),
        "top_deals_ytd": _between(blocks, Y.start, Y.end).nlargest(12, "value"),
        "latest_deals": _between(blocks, as_of, as_of).sort_values("value", ascending=False),
        "hi_series": hi_series,
        "qtd_contrib": contribution_by_symbol(blocks, (Q.start, Q.end), (Q.prev_start, Q.prev_end), 4),
    }
    if hi_series is not None and len(hi_series):
        def ret(s, e):
            w = hi_series[(hi_series.index >= s - pd.Timedelta(days=7)) & (hi_series.index <= e)]
            pre = w[w.index < s]
            base = pre.iloc[-1] if len(pre) else w.iloc[0]
            return w.iloc[-1] / base - 1 if len(w) else np.nan
        rd.extras["index_returns"] = {p.key: ret(p.start, p.end) for p in periods}
        rd.extras["index_level"] = float(hi_series[hi_series.index <= as_of].iloc[-1])
    return rd
