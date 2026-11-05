"""Phase 1: FAA Terminal Area Forecast (TAF) -> data/processed/taf.parquet

Usage:
    python scripts/build_taf.py [path/to/taf_folder]   (default: data/raw/taf_2025)

The folder holds the unpacked FAA download (APO100_TAF_Final_<year>.zip from taf.faa.gov):
Enplanements.xlsx and AirportsOperations.xlsx.

Output, one row per airport x fiscal year:
  locid, fiscal_year, is_forecast, enplanements, operations_air_carrier, edition
  enplanements = air carrier + commuter (regional) + air taxi enplanements (the FAA's total)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airport_agent.settings import get_settings  # noqa: E402

warnings.filterwarnings("ignore", message="Workbook contains no default style")


def main() -> None:
    s = get_settings()
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else s.raw_dir / "taf_2025"
    edition = folder.name.replace("taf_", "TAF ")

    e = pd.read_excel(folder / "Enplanements.xlsx")
    e["locid"] = e.locid.astype(str).str.strip()
    e["enplanements"] = e.aac + e.commuter + e.aat
    ops = pd.read_excel(folder / "AirportsOperations.xlsx")
    ops["locid"] = ops.locid.astype(str).str.strip()
    ops["operations_air_carrier"] = ops.itn_Ac

    out = (e[["locid", "ayear", "scenario", "enplanements"]]
           .merge(ops[["locid", "ayear", "operations_air_carrier"]], on=["locid", "ayear"], how="left")
           .rename(columns={"ayear": "fiscal_year"}))
    out["is_forecast"] = out.scenario == 1
    out["edition"] = edition
    out = out.drop(columns="scenario").sort_values(["locid", "fiscal_year"])
    s.processed_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(s.processed_dir / "taf.parquet", index=False)
    print(f"{edition}: {out.locid.nunique():,} airports, fiscal years {out.fiscal_year.min()}-{out.fiscal_year.max()}, "
          f"last actual year {out[~out.is_forecast].fiscal_year.max()}")


if __name__ == "__main__":
    main()
