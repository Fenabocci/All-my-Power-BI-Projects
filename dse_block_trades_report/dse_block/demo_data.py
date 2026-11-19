"""Synthetic data in the canonical schema, used ONLY to preview the report layout
when the ATS database / APIs are unreachable. Every page of a report built from
this data carries a 'SYNTHETIC DATA' watermark. Numbers are not real.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# symbol: (start price TZS, shares outstanding mn, block popularity weight, is_cross_listed)
EQUITIES = {
    "CRDB": (380, 2612, 9.0, False), "NMB": (2300, 500, 8.0, False), "TBL": (10900, 295, 5.0, False),
    "VODA": (780, 2240, 3.5, False), "TCC": (17000, 100, 3.0, False), "TPCC": (2200, 180, 2.5, False),
    "NICO": (300, 78, 1.5, False), "DSE": (1100, 24, 1.2, False), "SWIS": (1100, 36, 1.2, False),
    "TOL": (500, 58, 0.6, False), "JHL": (5500, 70, 0.8, False), "TTP": (120, 18, 0.3, False),
    "MBP": (600, 30, 0.4, False), "DCB": (200, 70, 0.4, False), "MCB": (400, 61, 0.3, False),
    "TCCL": (1500, 60, 0.6, False), "PAL": (400, 160, 0.2, False), "KCB": (700, 3200, 1.0, True),
    "EABL": (4400, 790, 1.4, True), "JATU": (300, 20, 0.1, False),
}
ETFS = {"VERTEX-ETF": (100, 50, 1.2, False), "ITRUST-ETF": (100, 40, 0.8, False)}
MEMBERS = [f"Member {i:02d}" for i in range(1, 15)]


def generate(as_of: str = "2026-11-19", seed: int = 7):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2019-06-01", as_of)

    # ---------- daily prices --------------------------------------------------
    rows = []
    for sym, (p0, shs, pop, _) in {**EQUITIES, **ETFS}.items():
        is_etf = sym in ETFS
        start = pd.Timestamp("2023-06-01") if is_etf else days[0]
        d = days[days >= start]
        drift = 0.00045 if not is_etf else 0.0003
        # 2024-26 bull run on Tanzanian banks
        regime = np.where(d >= "2024-01-01", drift * 2.2, drift * 0.3)
        rets = rng.normal(regime, 0.012 if not is_etf else 0.004, len(d))
        stale = rng.random(len(d)) < 0.35  # illiquidity: many unchanged closes
        rets[stale] = 0
        close = p0 * np.exp(np.cumsum(rets))
        vol = rng.lognormal(np.log(shs * 1e6 * 0.00004 * pop), 1.0, len(d))
        rows.append(pd.DataFrame({"date": d, "symbol": sym, "close": close.round(0),
                                  "volume": vol.round(0), "asset_class": "ETF" if is_etf else "Equity",
                                  "mcap": close * shs * 1e6}))
    prices = pd.concat(rows, ignore_index=True)
    prices["turnover"] = prices["close"] * prices["volume"]

    # ---------- indices -------------------------------------------------------
    eq = prices[prices.asset_class == "Equity"]
    mc_all = eq.groupby("date")["mcap"].sum()
    dom = eq[~eq.symbol.isin([s for s, v in EQUITIES.items() if v[3]])].groupby("date")["mcap"].sum()
    idx = pd.concat([
        pd.DataFrame({"date": mc_all.index, "index": "DSEI", "value": (mc_all / mc_all.iloc[0] * 1900).round(2)}),
        pd.DataFrame({"date": dom.index, "index": "TSI", "value": (dom / dom.iloc[0] * 3600).round(2)}),
    ])

    # ---------- block trades --------------------------------------------------
    syms = list(EQUITIES) + list(ETFS)
    pops = np.array([({**EQUITIES, **ETFS})[s][2] for s in syms])
    px = prices.set_index(["date", "symbol"])["close"]
    blocks = []
    deal = 100000
    bdays = days[days >= "2020-01-01"]
    for d in bdays:
        year_mult = {2020: 0.55, 2021: 0.7, 2022: 0.8, 2023: 0.9, 2024: 1.15, 2025: 1.35, 2026: 1.6}[d.year]
        qtr_end = d.month in (3, 6, 9, 12) and d.day >= 20
        lam = 0.55 * year_mult * (1.8 if qtr_end else 1.0)
        for _ in range(rng.poisson(lam)):
            w = pops.copy()
            if d < pd.Timestamp("2023-07-01"):
                w[-2:] = 0
            sym = rng.choice(syms, p=w / w.sum())
            ref = px.get((d, sym))
            if ref is None or np.isnan(ref):
                continue
            size_tzs = rng.lognormal(np.log(380e6), 1.25) * year_mult
            if sym in ("CRDB", "NMB") and d.year >= 2025:
                size_tzs *= 1.6  # institutional rotation into banks
            premium = rng.normal(-0.012, 0.025)
            price = max(1.0, round(ref * (1 + premium)))
            qty = max(1000, int(size_tzs / price / 100) * 100)
            b, s = rng.choice(MEMBERS, 2, replace=True, p=_member_weights())
            deal += 1
            blocks.append({"date": d, "symbol": sym, "qty": qty, "price": price, "value": qty * price,
                           "buyer": b, "seller": s, "deal_id": deal,
                           "entered": d + pd.Timedelta(hours=10, minutes=int(rng.integers(0, 330)))})
    blocks = pd.DataFrame(blocks)
    return blocks, prices, idx


def _member_weights():
    w = np.array([14, 12, 10, 9, 8, 7, 7, 6, 6, 5, 5, 4, 4, 3], dtype=float)
    return w / w.sum()
