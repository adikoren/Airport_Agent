"""Phase 1: the airport reference table -> data/reference/airports.parquet

Usage:
    python scripts/build_airport_reference.py [path/to/taf_folder]   (default: data/raw/taf_2025)

Joins:
  - BTS airports (T-100 + on-time): BTS airport id, IATA code, BTS name and city
  - FAA TAF Airports.xlsx: official name, city, state, location and FAA hub size
Join key: the FAA location id. It equals the IATA code for most US airports; the exceptions
are listed in config/faa_ids.yaml and verified here against the FAA's own name and city.
Airports that don't match are kept, with state and hub size empty, and counted in the output.
"""

from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airport_agent.settings import get_settings, load_config  # noqa: E402

warnings.filterwarnings("ignore", message="Workbook contains no default style")
HUB = {3: "large", 2: "medium", 1: "small", 0: "nonhub"}


def dms_to_deg(text: str) -> float | None:
    """'45-28-13.7000N' -> 45.4705"""
    m = re.match(r"(\d+)-(\d+)-([\d.]+)([NSEW])", str(text).strip())
    if not m:
        return None
    d, mi, se, h = m.groups()
    v = int(d) + int(mi) / 60 + float(se) / 3600
    return -v if h in "SW" else v


def main() -> None:
    s = get_settings()
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else s.raw_dir / "taf_2025"

    t = pd.read_parquet(s.processed_dir / "t100_monthly.parquet")
    bts = (t.sort_values("month").drop_duplicates("airport_id", keep="last")
           [["airport_id", "code", "name", "city"]].rename(columns={"code": "iata", "name": "bts_name", "city": "bts_city"}))
    codes = pd.read_parquet(s.processed_dir / "airport_codes.parquet")  # on-time codes override (more recent)
    bts = bts.merge(codes, on="airport_id", how="outer", suffixes=("", "_ontime"))
    bts["iata"] = bts.iata_ontime.fillna(bts.iata)
    bts = bts.drop(columns="iata_ontime")

    faa = pd.read_excel(folder / "Airports.xlsx")
    faa["LOCID"] = faa.LOCID.astype(str).str.strip()
    faa = faa.drop_duplicates("LOCID")
    faa = pd.DataFrame({
        "iata": faa.LOCID,
        "name": faa.APORT_NAME.astype(str).str.strip(),
        "city": faa.CITY.astype(str).str.strip().str.title(),
        "state": faa.STATE.astype(str).str.strip(),
        "hub_size": faa.HUB_SIZE.map(HUB),
        "lat": faa.LATITUDE.map(dms_to_deg),
        "lon": faa.LONGITUDE.map(dms_to_deg),
    })

    exceptions = load_config("faa_ids")["iata_to_faa"]
    bts["faa_id"] = bts.iata.map(lambda c: exceptions.get(c, c))
    faa = faa.rename(columns={"iata": "faa_id"})
    check = bts[bts.iata.isin(exceptions)].merge(faa, on="faa_id", how="left")
    for _, row in check.iterrows():
        bts_words = set(re.findall(r"[a-z]{4,}", f"{row.bts_name} {row.bts_city}".lower()))
        faa_words = set(re.findall(r"[a-z]{4,}", f"{row['name']} {row['city']}".lower()))
        if pd.isna(row["name"]) or not bts_words & faa_words:
            raise SystemExit(f"config/faa_ids.yaml: {row.iata} -> {row.faa_id} doesn't match the FAA list "
                             f"(BTS: {row.bts_name}, {row.bts_city}; FAA: {row['name']}, {row['city']})")
        print(f"verified {row.iata} -> {row.faa_id}: {row['name']}, {row['city']} {row['state']}")

    ref = bts.merge(faa, on="faa_id", how="left")
    ref["name"] = ref.name.fillna(ref.bts_name)
    ref["city"] = ref.city.fillna(ref.bts_city.str.replace(r"\s*\(.*\)$", "", regex=True))
    ref = ref[["airport_id", "iata", "faa_id", "name", "city", "state", "hub_size", "lat", "lon"]].sort_values("airport_id")
    s.reference_dir.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(s.reference_dir / "airports.parquet", index=False)

    with_flights = set(codes.airport_id)
    matched = ref.state.notna()
    print(f"airports: {len(ref):,}; matched to FAA (state + hub size): {matched.sum():,}; "
          f"airports with on-time flights: {len(with_flights)}, of which matched: "
          f"{ref[ref.airport_id.isin(with_flights)].state.notna().sum()}")
    print("hub sizes among matched:", ref.hub_size.value_counts().to_dict())


if __name__ == "__main__":
    main()
