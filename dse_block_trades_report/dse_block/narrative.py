"""Turns the numbers into an analyst's story: headline, takeaways, section commentary.

Every sentence is driven by the computed metrics, so the text stays truthful when
the data changes. Qualitative context the data cannot know (corporate actions,
foreign-investor flows, policy news) is added via an optional commentary file.
"""
from __future__ import annotations

import numpy as np

from .analytics import ReportData, fmt_pct, fmt_tzs


def _dir(x: float, up="up", down="down", flat="flat") -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return up if x > 0.005 else down if x < -0.005 else flat


def _pp(a: float, b: float) -> str:
    if not (np.isfinite(a) and np.isfinite(b)):
        return "n/a"
    return f"{(a - b) * 100:+.1f} pp"


def headline(rd: ReportData) -> str:
    y, py = rd.scorecard["Y"], rd.prev_scorecard["Y"]
    chg = y["value"] / py["value"] - 1 if py["value"] else np.nan
    verb = "surged" if chg > 0.3 else "rose" if chg > 0.02 else "fell" if chg < -0.02 else "held steady"
    return (f"Negotiated block trades {verb} to {fmt_tzs(y['value'])} year-to-date "
            f"({fmt_pct(chg)} YoY), {fmt_pct(y['block_share'], False, 0)} of all DSE turnover")


def takeaways(rd: ReportData) -> list[str]:
    s, p, ex = rd.scorecard, rd.prev_scorecard, rd.extras
    y, py = s["Y"], p["Y"]
    out = []
    d = ex["decomp_ytd"]
    if np.isfinite(d.get("total", np.nan)):
        lead = "more deals" if abs(d["count_effect"]) >= abs(d["size_effect"]) else "bigger tickets"
        out.append(
            f"<b>Growth is driven by {lead}.</b> Deal count is {fmt_pct(d['deal_growth'])} "
            f"({y['deals']:,} vs {py['deals']:,}) and average ticket {fmt_pct(d['size_growth'])} "
            f"({fmt_tzs(y['avg_deal'])}): they account for {fmt_pct(d['count_effect'])} and "
            f"{fmt_pct(d['size_effect'])} of the {fmt_pct(d['total'])} change.")
    c = ex["contrib_ytd"]
    pos = c[(c.index != "Others") & (c["delta"] > 0)].head(3)
    if len(pos):
        names = ", ".join(f"{i} ({'+' if r.delta >= 0 else ''}{fmt_tzs(r.delta, False)})" for i, r in pos.iterrows())
        share = pos["delta"].sum() / c["delta"].sum() if c["delta"].sum() > 0 else np.nan
        tail = f", together {fmt_pct(share, False, 0)} of the net increase" if np.isfinite(share) and share > 0 else ""
        out.append(f"<b>Leaders of the change:</b> {names}{tail}.")
    out.append(
        f"<b>Concentration:</b> {y['top_symbol']} alone is {fmt_pct(y['top_share'], False, 0)} of YTD block value; "
        f"the HHI of {y['hhi']:,.0f} reads as "
        f"{'highly concentrated' if y['hhi'] > 2500 else 'moderately concentrated' if y['hhi'] > 1500 else 'diversified'}.")
    if np.isfinite(y["wavg_premium"]):
        tone = "discount" if y["wavg_premium"] < 0 else "premium"
        out.append(
            f"<b>Pricing:</b> blocks crossed at a value-weighted {abs(y['wavg_premium']) * 100:.1f}% {tone} "
            f"to the prior close (last year: {fmt_pct(py['wavg_premium'])}). "
            + ("Sellers are paying for liquidity." if y["wavg_premium"] < -0.005 else
               "Buyers are paying up to secure size, a sign of strong demand." if y["wavg_premium"] > 0.005 else
               "Blocks print close to the screen: price discovery is working."))
    imp = ex.get("impact")
    if imp is not None:
        r5 = float(imp.loc[imp.t == 5, "mean"].iloc[0])
        out.append(
            f"<b>Market impact:</b> on average a block security moves {fmt_pct(r5)} "
            f"{'relative to the index ' if rd.headline_index else ''}over the 5 sessions after the trade "
            f"({'information content: follow the block' if r5 > 0.005 else 'temporary price pressure that reverses' if r5 < -0.005 else 'no lasting price effect: blocks are absorbed cleanly'}).")
    return out


def period_commentary(rd: ReportData) -> str:
    s, p = rd.scorecard, rd.prev_scorecard
    parts = []
    for per in rd.periods:
        c, q = s[per.key], p[per.key]
        chg = c["value"] / q["value"] - 1 if q["value"] else np.nan
        parts.append(f"<b>{per.label}</b>: {fmt_tzs(c['value'])} across {c['deals']:,} deal"
                     f"{'s' if c['deals'] != 1 else ''} ({fmt_pct(chg) if np.isfinite(chg) else 'no comparable'} "
                     f"{per.compare_label})")
    return "; ".join(parts) + "."


def trend_commentary(rd: ReportData) -> str:
    m = rd.extras["monthly"]
    last12, prior12 = m["value"].iloc[-12:].sum(), m["value"].iloc[-24:-12].sum()
    peak = m["value"].idxmax()
    share12 = m["share"].iloc[-12:].mean()
    share_prev = m["share"].iloc[-24:-12].mean()
    a = rd.extras["annual"]
    etf = a.get("ETF")
    etf_txt = ""
    if etf is not None and etf.iloc[-1] > 0:
        etf_txt = (f" ETF blocks contribute {fmt_tzs(etf.iloc[-1])} this year "
                   f"({etf.iloc[-1] / a.iloc[-1].sum() * 100:.0f}% of value): institutional money is using the ETF wrapper "
                   f"as a liquidity vehicle.")
    return (f"Over the last twelve months blocks totalled {fmt_tzs(last12)}, {fmt_pct(last12 / prior12 - 1)} on the prior "
            f"twelve. The peak month in the window was {peak:%B %Y} ({fmt_tzs(m.loc[peak, 'value'])}). "
            f"Blocks averaged {share12 * 100:.0f}% of market turnover over the last year versus {share_prev * 100:.0f}% a year "
            f"earlier ({_pp(share12, share_prev)}): the negotiated market is "
            f"{'gaining' if share12 > share_prev else 'losing'} weight against the order book.{etf_txt}")


def drivers_commentary(rd: ReportData) -> str:
    c = rd.extras["contrib_ytd"]
    neg = c[(c.index != "Others") & (c["delta"] < 0)]
    sz, szp = rd.extras["sizes_ytd"], rd.extras["sizes_prev"]
    big = sz.loc[["1-5 bn", "> 5 bn"], "value_share"].sum()
    bigp = szp.loc[["1-5 bn", "> 5 bn"], "value_share"].sum()
    cl = rd.extras.get("cluster_ytd", {})
    txt = (f"Deals above TZS 1 bn now make up {big * 100:.0f}% of block value (last year {bigp * 100:.0f}%), "
           f"so the market is {'more' if big > bigp else 'less'} reliant on a few large institutional tickets. ")
    if len(neg):
        txt += ("Offsetting the gains, " + ", ".join(f"{i} ({fmt_tzs(r.delta, False)})" for i, r in neg.iterrows())
                + " saw lower block activity. ")
    if cl:
        txt += (f"Activity is lumpy: {cl['active_days']} sessions had at least one block this year; the busiest, "
                f"{cl['busiest_day']:%d %b}, printed {fmt_tzs(cl['busiest_day_value'])}. "
                f"{cl['quarter_end_share'] * 100:.0f}% of value traded in the second half of quarter-end months, "
                f"consistent with fund window-dressing and portfolio rebalancing.")
    return txt


def pricing_commentary(rd: ReportData) -> str:
    pm = rd.extras["premium_ytd"]
    if pm is None or pm.empty:
        return "Reference prices were not available to assess block pricing."
    lo, hi = pm.iloc[0], pm.iloc[-1]
    return (f"Discounts are widest in {pm.index[0]} ({fmt_pct(lo.premium)}), where sellers needed to concede to move size. "
            f"The firmest pricing is in {pm.index[-1]} ({fmt_pct(hi.premium)}). Persistent premiums usually signal "
            f"accumulation by strategic or foreign buyers. Persistent discounts point to supply overhang (exits, "
            f"lock-up expiries or forced selling).")


def broker_commentary(rd: ReportData) -> str:
    b = rd.extras.get("brokers_ytd")
    if not b:
        return ""
    t = b["table"]
    nb = t["net"].idxmax()
    ns = t["net"].idxmin()
    return (f"{b['members']} members participated in block trades this year; the top three account for "
            f"{b['top3_share'] * 100:.0f}% of gross flow. {b['cross_share'] * 100:.0f}% of value was crossed in-house "
            f"(same member on both sides). The largest net buyer was {nb} ({fmt_tzs(t.loc[nb, 'net'])}) and the largest "
            f"net seller {ns} ({fmt_tzs(t.loc[ns, 'net'])}), a useful read on which client books are accumulating or "
            f"distributing.")


def outlook(rd: ReportData) -> list[str]:
    y, py = rd.scorecard["Y"], rd.prev_scorecard["Y"]
    ex = rd.extras
    pts = []
    ir = ex.get("index_returns", {})
    if ir:
        pts.append(f"The {rd.headline_index} is {fmt_pct(ir.get('Y', np.nan))} YTD and "
                   f"{fmt_pct(ir.get('Q', np.nan))} QTD. Block appetite has "
                   f"{'tracked' if (ir.get('Y', 0) > 0) == (y['value'] > py['value']) else 'diverged from'} "
                   f"the index, so watch for divergence as an early signal.")
    pts.append("Watch: (1) the quarter-end rebalancing window; (2) dividend record dates and corporate actions in "
               "the most-traded names; (3) concentration risk if the top security keeps dominating flow.")
    pts.append("For investors: block discounts are a liquidity opportunity for patient capital. Bring your "
               "interest to a member broker early, as negotiated size rarely shows on the order book.")
    return pts
