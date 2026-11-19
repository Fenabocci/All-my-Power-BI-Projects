"""Data extraction + normalisation.

Sources
-------
* Block trades  : SQL Server STT_ATS_MICS (tbDealHistory x tbContractDate x tbInstrument), Origin='P'
* Equity prices : DSE public API (class=EQUITY)
* ETF prices    : DSE public API (class=ETF)
* Indices       : internal index API

Raw column names in the ATS and the APIs are not guaranteed, so every source is
mapped onto a small canonical schema through candidate-name lists. If a column
is not found the report degrades gracefully (e.g. no broker page) and the gap is
listed in the report's data-quality notes.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .config import Settings

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Canonical schemas + candidate raw names (matched case/underscore-insensitive)
# --------------------------------------------------------------------------- #
BLOCK_CANDIDATES = {
    "date": ["TradeDate", "DealDate", "TradingDate"],
    "entered": ["EnteredDate", "EntryDate", "CreatedDate"],
    "symbol": ["Symbol", "SecurityCode", "InstrumentCode", "ShortName", "Code", "Ticker",
               "InstrumentShortName", "InstrumentName", "Name"],
    "qty": ["Quantity", "Qty", "DealQuantity", "TradeQuantity", "Volume", "Shares", "Size"],
    "price": ["Price", "DealPrice", "TradePrice", "ExecPrice"],
    "value": ["Value", "DealValue", "TradeValue", "Consideration", "Amount", "Turnover"],
    "buyer": ["BuyerMember", "BuyMember", "BuyerFirm", "BuyFirm", "BuyBroker", "BuyerBroker",
              "BuyerParticipant", "BuyParticipant", "idBuyerFirm", "idBuyFirm", "BuyerCode"],
    "seller": ["SellerMember", "SellMember", "SellerFirm", "SellFirm", "SellBroker", "SellerBroker",
               "SellerParticipant", "SellParticipant", "idSellerFirm", "idSellFirm", "SellerCode"],
    "deal_id": ["idDealHistory", "idDeal", "DealId", "DealNo", "DealNumber", "TradeId", "TradeNo"],
}
PRICE_CANDIDATES = {
    "date": ["trade_date", "date", "tradeDate", "trading_date", "market_date"],
    "symbol": ["security_code", "symbol", "code", "company", "securityCode", "ticker", "name"],
    "close": ["closing_price", "close", "closingPrice", "close_price", "price", "last_price"],
    "volume": ["volume", "shares_traded", "quantity", "traded_volume"],
    "turnover": ["turnover", "turn_over", "value", "traded_value", "turnOver"],
    "mcap": ["market_cap", "marketcap", "market_capitalisation", "market_capitalization", "mcap"],
}
INDEX_CANDIDATES = {
    "date": ["date", "trade_date", "indexDate", "index_date", "tradeDate", "TradeDate"],
    "index": ["index_name", "indexName", "index", "code", "name", "indexCode", "index_code"],
    "value": ["value", "close", "closing", "index_value", "indexValue", "closing_value", "level"],
}


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def find_col(cols: Iterable[str], candidates: list[str]) -> str | None:
    lookup = {_key(c): c for c in cols}
    for cand in candidates:
        if _key(cand) in lookup:
            return lookup[_key(cand)]
    return None


def map_columns(df: pd.DataFrame, candidates: dict[str, list[str]]) -> dict[str, str | None]:
    return {canon: find_col(df.columns, cands) for canon, cands in candidates.items()}


# --------------------------------------------------------------------------- #
# Block trades (SQL Server)
# --------------------------------------------------------------------------- #
BLOCK_SQL = """
SELECT DH.*, CD.*, IM.*
FROM [dbo].[tbDealHistory] AS DH
INNER JOIN [dbo].[tbContractDate] AS CD ON DH.idContractDate = CD.idContractDate
INNER JOIN [dbo].[tbInstrument]   AS IM ON CD.idInstrument   = IM.idInstrument
WHERE IM.IdInstrumentType IN ({types})
  AND DH.TradeDate >= ?
  AND DH.TradeDate <  DATEADD(DAY, 1, CAST(GETDATE() AS DATE))
  AND DH.Origin = ?
ORDER BY DH.EnteredDate DESC;
"""


def fetch_block_trades_raw(cfg: Settings) -> pd.DataFrame:
    import pyodbc  # imported lazily so demo mode works without the ODBC driver

    sql = BLOCK_SQL.format(types=",".join(str(int(t)) for t in cfg.instrument_types))
    with pyodbc.connect(cfg.connection_string, timeout=30) as conn:
        cur = conn.cursor()
        cur.execute(sql, cfg.block_start, cfg.block_origin)
        cols, seen = [], {}
        # SELECT DH.*, CD.*, IM.* returns duplicated join keys -> make names unique
        for d in cur.description:
            name = d[0]
            seen[name] = seen.get(name, 0) + 1
            cols.append(name if seen[name] == 1 else f"{name}__{seen[name]}")
        rows = cur.fetchall()
    df = pd.DataFrame.from_records([tuple(r) for r in rows], columns=cols)
    log.info("Block trades extracted: %s rows, %s columns", len(df), len(cols))
    return df


def normalise_blocks(raw: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    m = map_columns(raw, BLOCK_CANDIDATES)
    log.info("Block column mapping: %s", m)
    for req in ("date", "symbol"):
        if not m[req]:
            raise ValueError(f"Block trades: could not find a '{req}' column. "
                             f"Add the real name to BLOCK_CANDIDATES in extract.py. Columns: {list(raw.columns)}")
    out = pd.DataFrame({"date": pd.to_datetime(raw[m["date"]]).dt.normalize(),
                        "symbol": raw[m["symbol"]].astype(str).str.strip().str.upper()})
    out["qty"] = pd.to_numeric(raw[m["qty"]], errors="coerce") if m["qty"] else np.nan
    out["price"] = pd.to_numeric(raw[m["price"]], errors="coerce") if m["price"] else np.nan
    if m["value"]:
        out["value"] = pd.to_numeric(raw[m["value"]], errors="coerce")
        # fall back to price x qty where the value column is empty
        out["value"] = out["value"].fillna(out["qty"] * out["price"])
    else:
        out["value"] = out["qty"] * out["price"]
        notes.append("Deal value not present in ATS extract; computed as price x quantity.")
    for c in ("buyer", "seller"):
        out[c] = raw[m[c]].astype(str).str.strip() if m[c] else pd.NA
    if not (m["buyer"] and m["seller"]):
        notes.append("Buyer/seller member columns not identified; broker analysis omitted.")
    out["deal_id"] = raw[m["deal_id"]] if m["deal_id"] else range(len(raw))
    out["entered"] = pd.to_datetime(raw[m["entered"]]) if m["entered"] else out["date"]

    before = len(out)
    out = out.drop_duplicates(subset=["deal_id"]) if m["deal_id"] else out
    if len(out) < before:
        notes.append(f"Removed {before - len(out):,} duplicate deal rows (same deal id).")
    bad = out["value"].isna() | (out["value"] <= 0)
    if bad.any():
        notes.append(f"Excluded {int(bad.sum()):,} block rows with missing/zero value.")
    return out.loc[~bad].sort_values("entered").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# APIs
# --------------------------------------------------------------------------- #
def _records_from_json(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for k in ("data", "results", "result", "prices", "items", "records", "rows", "indices"):
            v = payload.get(k)
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                inner = _records_from_json(v)
                if inner:
                    return inner
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
    return []


def fetch_json_records(url: str, timeout: int = 120) -> pd.DataFrame:
    r = requests.get(url, timeout=timeout, headers={"Accept": "application/json"})
    r.raise_for_status()
    recs = _records_from_json(r.json())
    if not recs:
        raise ValueError(f"No tabular records found in response from {url}")
    return pd.json_normalize(recs)


def normalise_prices(raw: pd.DataFrame, asset_class: str, notes: list[str]) -> pd.DataFrame:
    m = map_columns(raw, PRICE_CANDIDATES)
    log.info("%s price column mapping: %s", asset_class, m)
    if not (m["date"] and m["symbol"] and m["close"]):
        raise ValueError(f"{asset_class} prices: need date/symbol/close columns; got {list(raw.columns)}")
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[m["date"]], errors="coerce").dt.normalize(),
        "symbol": raw[m["symbol"]].astype(str).str.strip().str.upper(),
        "close": pd.to_numeric(raw[m["close"]], errors="coerce"),
    })
    out["volume"] = pd.to_numeric(raw[m["volume"]], errors="coerce") if m["volume"] else np.nan
    if m["turnover"]:
        out["turnover"] = pd.to_numeric(raw[m["turnover"]], errors="coerce")
    else:
        out["turnover"] = out["close"] * out["volume"]
        notes.append(f"{asset_class} turnover not supplied by API; estimated as close x volume.")
    out["mcap"] = pd.to_numeric(raw[m["mcap"]], errors="coerce") if m["mcap"] else np.nan
    out["asset_class"] = asset_class
    out = out.dropna(subset=["date", "close"])
    return out.drop_duplicates(["date", "symbol"], keep="last").sort_values(["symbol", "date"])


def normalise_index(raw: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
    m = map_columns(raw, INDEX_CANDIDATES)
    if not m["date"]:
        raise ValueError(f"Index API: no date column found in {list(raw.columns)}")
    if m["index"] and m["value"]:
        out = pd.DataFrame({"date": pd.to_datetime(raw[m["date"]], errors="coerce").dt.normalize(),
                            "index": raw[m["index"]].astype(str).str.strip().str.upper(),
                            "value": pd.to_numeric(raw[m["value"]], errors="coerce")})
    else:  # wide format: one column per index
        num = raw.drop(columns=[m["date"]]).apply(pd.to_numeric, errors="coerce")
        num = num.loc[:, num.notna().mean() > 0.5]
        num["date"] = pd.to_datetime(raw[m["date"]], errors="coerce").dt.normalize()
        out = num.melt(id_vars="date", var_name="index", value_name="value")
        out["index"] = out["index"].str.upper()
    return out.dropna().drop_duplicates(["date", "index"], keep="last").sort_values(["index", "date"])


# --------------------------------------------------------------------------- #
# Orchestration with a CSV cache (lets you re-render the PDF offline)
# --------------------------------------------------------------------------- #
def load_all(cfg: Settings, use_cache: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    notes: list[str] = []
    cache: Path = cfg.cache_dir
    cache.mkdir(parents=True, exist_ok=True)

    def cached(name: str, fn):
        p = cache / f"{name}.csv"
        if use_cache and p.exists():
            log.info("Using cached %s", p)
            return pd.read_csv(p, low_memory=False)
        df = fn()
        df.to_csv(p, index=False)
        return df

    blocks = normalise_blocks(cached("raw_block_trades", lambda: fetch_block_trades_raw(cfg)), notes)
    eq = normalise_prices(cached("raw_equity_prices", lambda: fetch_json_records(cfg.equity_api)), "Equity", notes)
    try:
        etf = normalise_prices(cached("raw_etf_prices", lambda: fetch_json_records(cfg.etf_api)), "ETF", notes)
    except Exception as exc:  # ETF board is young / may be empty
        notes.append(f"ETF price API unavailable ({exc.__class__.__name__}); ETF context omitted.")
        etf = eq.iloc[0:0]
    try:
        idx = normalise_index(cached("raw_index", lambda: fetch_json_records(cfg.index_api)), notes)
    except Exception as exc:
        notes.append(f"Index API unavailable ({exc.__class__.__name__}); index context omitted.")
        idx = pd.DataFrame(columns=["date", "index", "value"])
    prices = pd.concat([eq, etf], ignore_index=True)
    return blocks, prices, idx, notes
