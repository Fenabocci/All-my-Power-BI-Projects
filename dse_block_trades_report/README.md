# DSE Block Trades Monitor: Investor PDF report

Pipeline that pulls negotiated/block deals from the DSE ATS (SQL Server), market prices
(Equity + ETF) and index levels from the DSE APIs, analyses them, and produces a
7-page investor PDF covering **Daily, Week-to-date, Month-to-date, Quarter-to-date and YTD**.

`sample_output/DSE_Block_Trades_20261119_DEMO.pdf` is a **layout preview built on synthetic
data** (watermarked). Run the pipeline on the DSE network to get live figures.

## What the report contains
| Page | Content |
|---|---|
| 1 | Cover, headline, KPI tiles, data-driven key takeaways, period comparison chart |
| 2 | Period scorecard: value, change, deals, avg ticket, % of turnover, top security, premium/discount, index return |
| 3 | Trend: 36-month block value, index vs weekly block flow |
| 4 | Drivers: deal-count vs ticket-size decomposition, waterfall by security, top 10, deal-size mix |
| 5 | Concentration heatmap (security × month), block pricing vs prior close, post-trade impact (event study) |
| 6 | Member (broker) flows, largest deals YTD, latest-session deals |
| 7 | Investor outlook, methodology, data-quality notes for the run, disclaimer |

## Setup (Windows or Linux, on the DSE network)
```bash
pip install -r requirements.txt
cp .env.example .env      # then edit: SQL login (read-only account), API URLs
python run_report.py
```
Output: `output/DSE_Block_Trades_YYYYMMDD.pdf` + `.scorecard.csv` (for Power BI or the email body).

Options: `--as-of 2026-09-30` (historical edition), `--use-cache` (re-render offline from
`data_cache/`), `--commentary notes.md` (adds your qualitative "Analyst commentary" section;
paragraphs separated by blank lines), `--demo` (synthetic preview).

## Adapting to the real column names
The ATS and API field names are matched automatically from candidate lists in
`dse_block/extract.py` (`BLOCK_CANDIDATES`, `PRICE_CANDIDATES`, `INDEX_CANDIDATES`).
The first run logs the mapping it chose (`-v` for detail). If a field is not found
(e.g. the buyer/seller member column), add its real name to the list. Missing optional fields
drop the related section and are listed in the report's data-quality notes.

## Security
Credentials are read from environment variables / `.env` (git-ignored), never from code.
Use a dedicated **read-only** SQL login instead of `sa`.

## Scheduling
Windows Task Scheduler / cron, e.g. daily at 17:30 after market close:
`python run_report.py --commentary commentary.md`
