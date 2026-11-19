"""Runtime settings. Everything sensitive comes from environment variables / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no extra dependency). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    sql_server: str
    sql_database: str
    sql_driver: str
    sql_user: str
    sql_password: str
    equity_api: str
    etf_api: str
    index_api: str
    instrument_types: tuple[int, ...]
    block_origin: str
    block_start: str
    headline_index: str
    turnover_includes_blocks: bool
    cache_dir: Path

    @property
    def connection_string(self) -> str:
        base = (
            f"DRIVER={{{self.sql_driver}}};SERVER={self.sql_server};"
            f"DATABASE={self.sql_database};Encrypt=yes;TrustServerCertificate=yes;"
        )
        if self.sql_user:
            return base + f"UID={self.sql_user};PWD={self.sql_password};"
        return base + "Trusted_Connection=yes;"


def load_settings(project_dir: Path | None = None) -> Settings:
    project_dir = project_dir or Path(__file__).resolve().parent.parent
    _load_dotenv(project_dir / ".env")
    env = os.environ.get
    return Settings(
        sql_server=env("DSE_SQL_SERVER", "192.168.2.250"),
        sql_database=env("DSE_SQL_DATABASE", "STT_ATS_MICS"),
        sql_driver=env("DSE_SQL_DRIVER", "ODBC Driver 18 for SQL Server"),
        sql_user=env("DSE_SQL_USER", ""),
        sql_password=env("DSE_SQL_PASSWORD", ""),
        equity_api=env("DSE_EQUITY_API",
                       "https://dse.co.tz/api/get/market/prices/for/range/duration"
                       "?class=EQUITY&security_code=ALL&days=2790"),
        etf_api=env("DSE_ETF_API",
                    "https://dse.co.tz/api/get/market/prices/for/range/duration"
                    "?class=ETF&security_code=ALL&days=1470"),
        index_api=env("DSE_INDEX_API",
                      "http://192.168.2.12:8086/api/index-data?startDate=2016-01-01&endDate=2026-12-31"),
        instrument_types=tuple(int(x) for x in env("DSE_INSTRUMENT_TYPES", "55").split(",") if x.strip()),
        block_origin=env("DSE_BLOCK_ORIGIN", "P"),
        block_start=env("DSE_BLOCK_START", "2020-01-01"),
        headline_index=env("DSE_HEADLINE_INDEX", "DSEI"),
        turnover_includes_blocks=env("DSE_TURNOVER_INCLUDES_BLOCKS", "true").lower() in ("1", "true", "yes"),
        cache_dir=Path(env("DSE_CACHE_DIR", str(project_dir / "data_cache"))),
    )
