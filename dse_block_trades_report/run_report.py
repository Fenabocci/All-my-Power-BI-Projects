#!/usr/bin/env python3
"""DSE Block Trades investor report: extract -> analyse -> PDF.

Examples
--------
  python run_report.py                       # live: SQL Server + APIs (run on the DSE network)
  python run_report.py --use-cache           # re-render from data_cache/*.csv without hitting sources
  python run_report.py --as-of 2026-09-30    # historical cut-off (e.g. quarter-end edition)
  python run_report.py --commentary notes.md # add your qualitative market commentary
  python run_report.py --demo                # layout preview on synthetic data (watermarked)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dse_block import analytics, demo_data, extract
from dse_block.config import load_settings
from dse_block.report_pdf import build_pdf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true", help="use synthetic data (layout preview only)")
    ap.add_argument("--use-cache", action="store_true", help="read raw extracts from the cache directory")
    ap.add_argument("--as-of", help="report date (YYYY-MM-DD); default = latest data date")
    ap.add_argument("--commentary", type=Path, help="text/markdown file with analyst commentary (paragraphs)")
    ap.add_argument("--out", type=Path, help="output PDF path")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_settings()
    if a.demo:
        blocks, prices, idx = demo_data.generate(a.as_of or "2026-11-19")
        notes = ["Synthetic data: illustrative only. Member names are anonymised placeholders."]
    else:
        blocks, prices, idx, notes = extract.load_all(cfg, use_cache=a.use_cache)
    if blocks.empty:
        logging.error("No block trades returned; nothing to report.")
        return 1

    rd = analytics.build(blocks, prices, idx, notes, cfg.headline_index, cfg.turnover_includes_blocks, a.as_of)
    out = a.out or Path("output") / f"DSE_Block_Trades_{rd.as_of:%Y%m%d}{'_DEMO' if a.demo else ''}.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    commentary = a.commentary.read_text(encoding="utf-8") if a.commentary else None
    build_pdf(rd, out, out.parent / f"charts_{rd.as_of:%Y%m%d}", demo=a.demo, commentary=commentary)

    # handy Excel/CSV extract of the scorecard for Power BI / email bodies
    import pandas as pd
    sc = pd.DataFrame({p.label: {**rd.scorecard[p.key], **{f"prior_{k}": v for k, v in rd.prev_scorecard[p.key].items()}}
                       for p in rd.periods}).T
    sc.to_csv(out.with_suffix(".scorecard.csv"))
    logging.info("Report written: %s", out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
