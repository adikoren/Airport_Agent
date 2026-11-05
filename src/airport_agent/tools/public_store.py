"""Read the processed public data (BTS + FAA TAF) and the airport reference.

Everything here is read-only. Files are built by the scripts in scripts/.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from airport_agent.settings import get_settings


@lru_cache
def airports() -> pd.DataFrame:
    """One row per airport: airport_id, iata, faa_id, name, city, state, hub_size, lat, lon."""
    return pd.read_parquet(get_settings().reference_dir / "airports.parquet")


@lru_cache
def hourly_cells() -> pd.DataFrame:
    return pd.read_parquet(get_settings().processed_dir / "hourly_cells.parquet")


@lru_cache
def t100_monthly() -> pd.DataFrame:
    return pd.read_parquet(get_settings().processed_dir / "t100_monthly.parquet")


@lru_cache
def ontime_monthly() -> pd.DataFrame:
    return pd.read_parquet(get_settings().processed_dir / "ontime_monthly.parquet")


@lru_cache
def taf() -> pd.DataFrame:
    return pd.read_parquet(get_settings().processed_dir / "taf.parquet")


@dataclass(frozen=True)
class Period:
    """A 12-month window, as month-start timestamps (inclusive)."""
    start: pd.Timestamp
    end: pd.Timestamp

    @property
    def label(self) -> str:
        return f"{self.start:%b %Y} – {self.end:%b %Y}"

    @property
    def fiscal_year(self) -> int:
        """US federal fiscal year containing the end of the window (FY runs Oct–Sep)."""
        return self.end.year + 1 if self.end.month >= 10 else self.end.year


@lru_cache
def default_period() -> Period:
    """The latest 12 months covered by both the on-time data and T-100."""
    end = min(pd.Timestamp(hourly_cells().date.max()).to_period("M").to_timestamp(),
              t100_monthly().month.max())
    return Period(start=end - pd.DateOffset(months=11), end=end)
