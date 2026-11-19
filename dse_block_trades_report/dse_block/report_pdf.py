"""PDF assembly with ReportLab (A4 portrait, investor-ready)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

from . import charts as C
from . import narrative as N
from .analytics import ReportData, fmt_num, fmt_pct, fmt_tzs

NAVY = colors.HexColor("#0d366b")
BLUE = colors.HexColor(C.BLUE)
INK = colors.HexColor(C.INK)
INK2 = colors.HexColor(C.INK2)
MUTED = colors.HexColor(C.MUTED)
GRID = colors.HexColor(C.GRID)
PANEL = colors.HexColor("#f4f8fd")
GOOD = colors.HexColor("#006300")
BAD = colors.HexColor("#d03b3b")

PAGE_W, PAGE_H = A4

try:  # DejaVu ships with matplotlib and has the ▲ ▼ glyphs Helvetica lacks
    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    pdfmetrics.registerFont(TTFont("DejaVu", str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf")))
    _ARROW_FONT = "DejaVu"
except Exception:  # pragma: no cover
    _ARROW_FONT = None
MARGIN = 0.6 * inch

S = {
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, leading=19, textColor=NAVY, spaceAfter=4),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, leading=15, textColor=NAVY,
                         spaceBefore=8, spaceAfter=4),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=8.6, leading=12.4, textColor=INK, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=7, leading=9.4, textColor=INK2),
    "bullet": ParagraphStyle("bullet", fontName="Helvetica", fontSize=8.6, leading=12.4, textColor=INK,
                             leftIndent=10, bulletIndent=0, spaceAfter=3),
    "kpi_l": ParagraphStyle("kpi_l", fontName="Helvetica", fontSize=7, leading=9, textColor=INK2),
    "kpi_v": ParagraphStyle("kpi_v", fontName="Helvetica-Bold", fontSize=15, leading=18, textColor=INK),
    "kpi_d": ParagraphStyle("kpi_d", fontName="Helvetica", fontSize=7, leading=9, textColor=INK2),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=7, leading=8.6, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=7, leading=8.6, textColor=INK),
    "th": ParagraphStyle("th", fontName="Helvetica-Bold", fontSize=6.8, leading=8.2, textColor=colors.white),
    "cover_t": ParagraphStyle("ct", fontName="Helvetica-Bold", fontSize=24, leading=28, textColor=colors.white),
    "cover_s": ParagraphStyle("cs", fontName="Helvetica", fontSize=11, leading=15, textColor=colors.HexColor("#cde2fb")),
    "headline": ParagraphStyle("hl", fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=NAVY),
}


def _delta(x: float, good_up: bool = True, pp: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "<font color='#898781'>no comparable</font>"
    up = x > 0
    col = "#006300" if up == good_up else "#d03b3b"
    arrow = ("▲" if up else "▼") if _ARROW_FONT else ("+" if up else "-")
    if _ARROW_FONT:
        arrow = f"<font name='{_ARROW_FONT}' size='6'>{arrow}</font>"
    txt = f"{x * 100:+.1f} pp" if pp else fmt_pct(x)
    return f"<font color='{col}'>{arrow} {txt}</font>"


def _chg(a, b):
    return a / b - 1 if b else np.nan


def kpi_row(items: list[tuple[str, str, str]], width: float) -> Table:
    cells = [_stack([Paragraph(l, S["kpi_l"]), Paragraph(v, S["kpi_v"]), Paragraph(d, S["kpi_d"])])
             for l, v, d in items]
    t = Table([cells], colWidths=[width / len(items)] * len(items))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL), ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEABOVE", (0, 0), (-1, 0), 2, BLUE), ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEAFTER", (0, 0), (-2, -1), 3, colors.white),
    ]))
    return t


def _stack(flows):
    t = Table([[f] for f in flows])
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return t


def data_table(header: list[str], rows: list[list], col_w: list[float], bold_first=True, align_right_from=1) -> Table:
    data = [[Paragraph(h, S["th"]) for h in header]]
    for r in rows:
        data.append([Paragraph(str(v), S["cellb"] if (i == 0 and bold_first) else S["cell"]) for i, v in enumerate(r)])
    t = Table(data, colWidths=col_w, repeatRows=1)
    st = [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("LINEBELOW", (0, 1), (-1, -1), 0.4, GRID), ("TOPPADDING", (0, 0), (-1, -1), 3.2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 4),
          ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for i in range(1, len(data)):
        if i % 2 == 0:
            st.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f9f9f7")))
    t.setStyle(TableStyle(st))
    return t


def img(path: Path, width: float) -> Image:
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * ih / iw)


def side_by_side(left, right, width, gap=10):
    t = Table([[left, right]], colWidths=[(width - gap) / 2 + gap / 2, (width - gap) / 2 + gap / 2])
    t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    return t


# --------------------------------------------------------------------------- #
class _Doc(BaseDocTemplate):
    def __init__(self, path, rd: ReportData, demo: bool, **kw):
        super().__init__(str(path), pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=0.85 * inch, bottomMargin=0.7 * inch,
                         title="DSE Block Trades Report", author="Capital Markets Research", **kw)
        self.rd, self.demo = rd, demo
        frame = Frame(MARGIN, 0.7 * inch, PAGE_W - 2 * MARGIN, PAGE_H - 1.55 * inch, id="f", showBoundary=0)
        cover_frame = Frame(MARGIN, 0.7 * inch, PAGE_W - 2 * MARGIN, PAGE_H - 3.6 * inch, id="c")
        self.addPageTemplates([PageTemplate("cover", [cover_frame], onPage=self._cover),
                               PageTemplate("body", [frame], onPage=self._page)])

    def _watermark(self, c):
        if not self.demo:
            return
        c.saveState()
        c.setFont("Helvetica-Bold", 46)
        c.setFillColor(colors.Color(0.82, 0.23, 0.23, alpha=0.10))
        c.translate(PAGE_W / 2, PAGE_H / 2)
        c.rotate(35)
        c.drawCentredString(0, 0, "SYNTHETIC DATA · LAYOUT PREVIEW")
        c.restoreState()

    def _footer(self, c):
        c.setFont("Helvetica", 6.5)
        c.setFillColor(MUTED)
        c.drawString(MARGIN, 0.42 * inch, "Source: DSE ATS deal history (negotiated/block deals), DSE market-price & index "
                                         "APIs. For information only; not investment advice.")
        c.drawRightString(PAGE_W - MARGIN, 0.42 * inch, f"Page {c.getPageNumber()}")

    def _cover(self, c, doc):
        c.saveState()
        c.setFillColor(NAVY)
        c.rect(0, PAGE_H - 2.75 * inch, PAGE_W, 2.75 * inch, stroke=0, fill=1)
        c.setFillColor(BLUE)
        c.rect(0, PAGE_H - 2.75 * inch, PAGE_W, 4, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#86b6ef"))
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(MARGIN, PAGE_H - 0.75 * inch, "DAR ES SALAAM STOCK EXCHANGE · CAPITAL MARKETS RESEARCH")
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 26)
        c.drawString(MARGIN, PAGE_H - 1.35 * inch, "Block Trades Monitor")
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.HexColor("#cde2fb"))
        c.drawString(MARGIN, PAGE_H - 1.68 * inch, "Daily · Weekly · Monthly · Quarterly · Year-to-Date")
        c.setFont("Helvetica-Bold", 10)
        c.setFillColor(colors.white)
        c.drawString(MARGIN, PAGE_H - 2.2 * inch, f"Data as of {self.rd.as_of:%A, %d %B %Y}")
        if self.demo:
            c.setFillColor(colors.HexColor("#fab219"))
            c.setFont("Helvetica-Bold", 8)
            c.drawString(MARGIN, PAGE_H - 2.48 * inch,
                         "PREVIEW BUILT ON SYNTHETIC DATA: re-run on the DSE network for live figures")
        c.restoreState()
        self._watermark(c)
        self._footer(c)

    def _page(self, c, doc):
        c.saveState()
        c.setFillColor(NAVY)
        c.rect(0, PAGE_H - 0.5 * inch, PAGE_W, 0.5 * inch, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(MARGIN, PAGE_H - 0.31 * inch, "DSE Block Trades Monitor")
        c.setFont("Helvetica", 8)
        c.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.31 * inch, f"As of {self.rd.as_of:%d %b %Y}")
        c.restoreState()
        self._watermark(c)
        self._footer(c)


# --------------------------------------------------------------------------- #
def build_pdf(rd: ReportData, out_pdf: Path, work_dir: Path, demo: bool = False,
              commentary: str | None = None) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    W = PAGE_W - 2 * MARGIN
    s, p, ex = rd.scorecard, rd.prev_scorecard, rd.extras
    Y = next(x for x in rd.periods if x.key == "Y")
    ly = Y.prev_start.year
    story = []

    # ---------- charts ----------
    ch = {}
    ch["periods"] = C.period_bars([x.label.replace("-to-date", " TD") for x in rd.periods],
                                  [s[x.key]["value"] for x in rd.periods],
                                  [p[x.key]["value"] for x in rd.periods], work_dir / "periods.png")
    ch["monthly"] = C.monthly_value(ex["monthly"], work_dir / "monthly.png")
    ch["share"] = C.monthly_share(ex["monthly"], work_dir / "share.png")
    ch["annual"] = C.annual_stacked(ex["annual"], rd.as_of, work_dir / "annual.png")
    ch["waterfall"] = C.waterfall(ex["contrib_ytd"], p["Y"]["value"], s["Y"]["value"], f"YTD {ly}",
                                  f"YTD {rd.as_of.year}", work_dir / "waterfall.png",
                                  f"What changed: YTD {rd.as_of.year} vs YTD {ly} by security")
    ch["top"] = C.top_symbols(ex["top_symbols_ytd"], s["Y"]["value"], work_dir / "top.png")
    ch["sizes"] = C.size_mix(ex["sizes_ytd"], ex["sizes_prev"], f"YTD {rd.as_of.year}", f"YTD {ly}",
                             work_dir / "sizes.png")
    ch["heat"] = C.heatmap(ex["heatmap"], work_dir / "heat.png")
    if ex["premium_ytd"] is not None and len(ex["premium_ytd"]):
        ch["prem"] = C.premium_bars(ex["premium_ytd"], work_dir / "prem.png")
    if ex["impact"] is not None:
        ch["impact"] = C.impact_curve(ex["impact"], rd.headline_index is not None, work_dir / "impact.png")
    if ex.get("hi_series") is not None:
        ch["index"] = C.index_context(ex["hi_series"], rd.blocks, rd.headline_index, rd.as_of, work_dir / "index.png")
    if ex.get("brokers_ytd"):
        ch["brokers"] = C.broker_flows(ex["brokers_ytd"]["table"], work_dir / "brokers.png")

    # ---------- page 1: cover + executive summary ----------
    story.append(NextPageTemplate("body"))
    story.append(Paragraph(N.headline(rd), S["headline"]))
    story.append(Spacer(1, 8))
    y, py, d = s["Y"], p["Y"], s["D"]
    story.append(kpi_row([
        ("Block value YTD", fmt_tzs(y["value"]), _delta(_chg(y["value"], py["value"])) + f" vs YTD {ly}"),
        ("Block deals YTD", f"{y['deals']:,}", _delta(_chg(y["deals"], py["deals"])) + f" · avg {fmt_tzs(y['avg_deal'])}"),
        ("Share of turnover", fmt_pct(y["block_share"], False), _delta(y["block_share"] - py["block_share"], pp=True)
         + f" vs {fmt_pct(py['block_share'], False)}"),
        ("Latest session", fmt_tzs(d["value"]), f"{d['deals']} deal{'s' if d['deals'] != 1 else ''} · "
         f"{rd.as_of:%d %b}"),
    ], W))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Key takeaways", S["h2"]))
    for t in N.takeaways(rd):
        story.append(Paragraph(t, S["bullet"], bulletText="•"))
    if commentary:
        story.append(Paragraph("Analyst commentary", S["h2"]))
        for para in [x.strip() for x in commentary.split("\n\n") if x.strip()]:
            story.append(Paragraph(para.replace("\n", " "), S["body"]))
            story.append(Spacer(1, 4))
    story.append(Spacer(1, 8))
    story.append(img(ch["periods"], W))
    story.append(PageBreak())

    # ---------- page 2: period scorecard ----------
    story.append(Paragraph("1. Period scorecard", S["h1"]))
    story.append(Paragraph(N.period_commentary(rd), S["body"]))
    story.append(Spacer(1, 6))
    ir = ex.get("index_returns", {})
    rows = []
    for per in rd.periods:
        c, q = s[per.key], p[per.key]
        win = f"{per.start:%d %b}" if per.start == per.end else f"{per.start:%d %b}–{per.end:%d %b}"
        rows.append([per.label, win, fmt_tzs(c["value"], False), _delta(_chg(c["value"], q["value"])),
                     f"{c['deals']:,}", fmt_tzs(c["avg_deal"], False) if c["deals"] else "-",
                     fmt_pct(c["block_share"], False) if np.isfinite(c["block_share"]) else "-",
                     f"{c['top_symbol']} ({fmt_pct(c['top_share'], False, 0)})" if c["deals"] else "-",
                     fmt_pct(c["wavg_premium"]) if np.isfinite(c["wavg_premium"]) else "-",
                     fmt_pct(ir.get(per.key, np.nan)) if ir else "-"])
    story.append(data_table(["Period", "Window", "Value (TZS)", "Change vs prior", "Deals", "Avg deal",
                             "% turnover", "Top security", "Prem/disc", rd.headline_index or "Index"],
                            rows, [x * inch for x in [0.95, 0.82, 0.72, 0.72, 0.42, 0.62, 0.6, 0.92, 0.6, 0.6]]))
    story.append(Spacer(1, 3))
    story.append(Paragraph("Change vs prior compares each period-to-date window with the same elapsed span of the previous "
                           "period (YTD vs same date last year). Prem/disc = value-weighted block price vs prior close.",
                           S["small"]))
    comp = []
    for per in rd.periods:
        c, q = s[per.key], p[per.key]
        comp.append([per.label, fmt_tzs(q["value"], False), f"{q['deals']:,}", fmt_tzs(q["max_deal"], False)
                     if q["deals"] else "-", fmt_tzs(c["max_deal"], False) if c["deals"] else "-",
                     f"{c['securities']}", fmt_tzs(c["market_turnover"], False)])
    story.append(Paragraph("Comparison detail", S["h2"]))
    story.append(data_table(["Period", "Prior value", "Prior deals", "Prior largest", "Largest deal",
                             "Securities", "Mkt turnover"], comp, [W / 7] * 7))
    story.append(Spacer(1, 10))
    story.append(side_by_side(img(ch["annual"], W / 2 - 6), img(ch["share"], W / 2 - 6), W))
    story.append(PageBreak())

    # ---------- page 3: trend ----------
    story.append(Paragraph("2. The trend: how big is the negotiated market?", S["h1"]))
    story.append(Paragraph(N.trend_commentary(rd), S["body"]))
    story.append(Spacer(1, 6))
    story.append(img(ch["monthly"], W))
    if "index" in ch:
        story.append(Spacer(1, 10))
        story.append(img(ch["index"], W))
    story.append(PageBreak())

    # ---------- page 4: drivers ----------
    story.append(Paragraph("3. The drivers: what moved the numbers", S["h1"]))
    dy = ex["decomp_ytd"]
    if np.isfinite(dy.get("total", np.nan)):
        story.append(Paragraph(
            f"Block value = number of deals × average ticket. YTD value is {fmt_pct(dy['total'])}: "
            f"<b>{fmt_pct(dy['count_effect'])}</b> from deal count ({fmt_pct(dy['deal_growth'])}) and "
            f"<b>{fmt_pct(dy['size_effect'])}</b> from ticket size ({fmt_pct(dy['size_growth'])}). "
            + N.drivers_commentary(rd), S["body"]))
    story.append(Spacer(1, 6))
    story.append(img(ch["waterfall"], W))
    story.append(Spacer(1, 6))
    story.append(side_by_side(img(ch["top"], W / 2 - 6), img(ch["sizes"], W / 2 - 6), W))
    story.append(PageBreak())

    # ---------- page 5: concentration + pricing ----------
    story.append(Paragraph("4. Concentration, pricing and market impact", S["h1"]))
    story.append(Paragraph(
        f"YTD block flow touched {y['securities']} securities. Top security share {fmt_pct(y['top_share'], False, 0)}, "
        f"HHI {y['hhi']:,.0f} (last year {py['hhi']:,.0f}). " + N.pricing_commentary(rd), S["body"]))
    story.append(Spacer(1, 6))
    story.append(img(ch["heat"], W))
    story.append(Spacer(1, 6))
    left = img(ch["prem"], W / 2 - 6) if "prem" in ch else Paragraph("No pricing data.", S["small"])
    right = img(ch["impact"], W / 2 - 6) if "impact" in ch else Paragraph("Too few events for an impact study.", S["small"])
    story.append(side_by_side(left, right, W))
    story.append(Paragraph("Impact study: for each security-day with a block, the security's return from the prior "
                           "close, minus the headline index return over the same window (abnormal return). "
                           "Illiquid names with stale prices bias results toward zero.", S["small"]))
    story.append(PageBreak())

    # ---------- page 6: members + deal log ----------
    story.append(Paragraph("5. Who is trading, and the largest deals", S["h1"]))
    if "brokers" in ch:
        story.append(Paragraph(N.broker_commentary(rd), S["body"]))
        story.append(Spacer(1, 4))
        story.append(img(ch["brokers"], W))
    story.append(Paragraph(f"Largest block trades, YTD {rd.as_of.year}", S["h2"]))
    rows = [[f"{r.date:%d %b %Y}", r.symbol, r.asset_class, fmt_num(r.qty), f"{r.price:,.0f}",
             fmt_tzs(r.value, False), fmt_pct(r.premium) if np.isfinite(r.premium) else "-",
             f"{r.buyer} / {r.seller}" if isinstance(r.buyer, str) else "-"]
            for r in ex["top_deals_ytd"].itertuples()]
    story.append(data_table(["Date", "Security", "Class", "Quantity", "Price", "Value (TZS)", "vs close",
                             "Buyer / Seller member"], rows,
                            [x * inch for x in [0.8, 0.85, 0.5, 0.85, 0.6, 0.75, 0.6, 2.05]], bold_first=False))
    latest = ex["latest_deals"]
    story.append(Paragraph(f"Latest session ({rd.as_of:%d %b %Y}): {len(latest)} block deal(s)", S["h2"]))
    if len(latest):
        rows = [[r.symbol, fmt_num(r.qty), f"{r.price:,.0f}", fmt_tzs(r.value, False),
                 fmt_pct(r.premium) if np.isfinite(r.premium) else "-",
                 f"{r.buyer} / {r.seller}" if isinstance(r.buyer, str) else "-"] for r in latest.head(10).itertuples()]
        story.append(data_table(["Security", "Quantity", "Price", "Value (TZS)", "vs close", "Buyer / Seller member"],
                                rows, [x * inch for x in [1.0, 1.0, 0.8, 1.0, 0.8, 2.4]]))
    else:
        story.append(Paragraph("No negotiated deals were reported in the latest session.", S["body"]))
    story.append(PageBreak())

    # ---------- page 7: outlook + methodology ----------
    story.append(Paragraph("6. What to watch: the investor view", S["h1"]))
    for t in N.outlook(rd):
        story.append(Paragraph(t, S["bullet"], bulletText="•"))
    story.append(Paragraph("Methodology & definitions", S["h2"]))
    meth = [
        "<b>Block trade</b>: a negotiated (pre-arranged) deal in the ATS deal history with Origin = 'P', on "
        "block-eligible instruments, from 2020 onward.",
        "<b>Value</b>: deal consideration in TZS; <b>% of turnover</b>: block value ÷ total market turnover reported "
        "by the DSE price APIs (equity + ETF) for the same window.",
        "<b>Periods</b>: Daily = latest session; Week/Month/Quarter/Year-to-date are calendar windows ending on the "
        "as-of date, each compared with the same elapsed span of the prior period.",
        "<b>Premium/discount</b>: block price ÷ previous session's closing price − 1, value-weighted.",
        "<b>Drivers</b>: change in value split into deal-count and average-ticket effects (log decomposition), "
        "and by security (waterfall).",
        "<b>HHI</b>: Herfindahl-Hirschman Index of security shares of block value (0-10,000; >2,500 = highly "
        "concentrated).",
    ]
    for m_ in meth:
        story.append(Paragraph(m_, S["bullet"], bulletText="–"))
    if rd.notes:
        story.append(Paragraph("Data-quality notes for this run", S["h2"]))
        for n_ in rd.notes:
            story.append(Paragraph(n_, S["bullet"], bulletText="–"))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Disclaimer: this report is prepared for information purposes only and does not constitute an offer, "
        "solicitation or recommendation to buy or sell any security. Past trading activity is not a guide to future "
        "performance. Figures are subject to revision.", S["small"]))

    doc = _Doc(out_pdf, rd, demo)
    doc.build(story)
    return out_pdf
