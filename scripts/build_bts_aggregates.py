"""Phase 1: turn the raw BTS files into small Parquet tables the agent reads.

Usage:
    python scripts/build_bts_aggregates.py

Reads data/raw/T_ONTIME_REPORTING_*.csv, and T-100 from the data.bts.gov API (dataset r495-tyji,
"AFF - T100 Segment Summary By Origin Airport"; no key needed). If the API is unreachable, it falls back
to data/raw/AFF-T100_*.csv (the same dataset as a file) and says so.
Writes to data/processed/:
  hourly_cells.parquet    one row per airport x date x hour: departures, arrivals,
                          taxi-out and taxi-in totals and counts (sums, so they can be added up later)
  ontime_monthly.parquet  one row per airport x month: flights, delays, cancellations,
                          long-haul counts (domestic, backup source for long haul)
  t100_monthly.parquet    one row per airport x month: passengers, seats, departures,
                          outbound international passengers (Jan 2014 onwards)
  airport_codes.parquet   BTS airport id -> latest IATA code seen in the on-time data
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airport_agent.settings import get_settings, load_config  # noqa: E402

ONTIME_COLS = ["FL_DATE", "OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM", "ORIGIN_AIRPORT_ID", "ORIGIN",
               "DEST_AIRPORT_ID", "DEST", "CRS_DEP_TIME", "DEP_TIME_BLK", "CRS_ARR_TIME", "TAXI_OUT",
               "TAXI_IN", "DEP_DEL15", "CANCELLED", "CANCELLATION_CODE", "CRS_ELAPSED_TIME", "DISTANCE"]


def load_ontime(raw: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(raw / "T_ONTIME_REPORTING_*.csv")))
    if not files:
        raise FileNotFoundError(f"No on-time files in {raw}")
    df = pd.concat([pd.read_csv(f, usecols=ONTIME_COLS) for f in files], ignore_index=True)
    before = len(df)
    df = df.drop_duplicates()
    print(f"on-time: {before:,} rows from {len(files)} files, {before - len(df):,} exact duplicates removed")
    df["date"] = pd.to_datetime(df.FL_DATE, format="%m/%d/%Y %I:%M:%S %p").dt.date
    df["dep_hour"] = df.DEP_TIME_BLK.str[:2].astype(int)
    df["arr_hour"] = (df.CRS_ARR_TIME // 100) % 24
    return df


def hourly_cells(df: pd.DataFrame) -> pd.DataFrame:
    dep = (df.groupby(["ORIGIN_AIRPORT_ID", "date", "dep_hour"])
           .agg(departures=("CANCELLED", "size"), taxi_out_sum=("TAXI_OUT", "sum"), taxi_out_n=("TAXI_OUT", "count"))
           .rename_axis(["airport_id", "date", "hour"]))
    arr = (df.groupby(["DEST_AIRPORT_ID", "date", "arr_hour"])
           .agg(arrivals=("CANCELLED", "size"), taxi_in_sum=("TAXI_IN", "sum"), taxi_in_n=("TAXI_IN", "count"))
           .rename_axis(["airport_id", "date", "hour"]))
    cells = dep.join(arr, how="outer").fillna(0).reset_index()
    for c in ["departures", "arrivals", "taxi_out_n", "taxi_in_n"]:
        cells[c] = cells[c].astype("int32")
    cells["date"] = pd.to_datetime(cells["date"])
    return cells


def ontime_monthly(df: pd.DataFrame, long_haul: dict) -> pd.DataFrame:
    d = df.assign(month=pd.to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp(),
                  long_haul_time=long_haul["block_time_minutes"] <= df.CRS_ELAPSED_TIME,
                  long_haul_dist=long_haul["distance_miles"] <= df.DISTANCE,
                  block_time_known=df.CRS_ELAPSED_TIME > 0,
                  cancelled_nas=df.CANCELLATION_CODE == "C")
    return (d.groupby(["ORIGIN_AIRPORT_ID", "month"])
            .agg(flights=("CANCELLED", "size"), delayed15=("DEP_DEL15", "sum"), delay_known=("DEP_DEL15", "count"),
                 cancelled=("CANCELLED", "sum"), cancelled_nas=("cancelled_nas", "sum"),
                 long_haul_time=("long_haul_time", "sum"), long_haul_dist=("long_haul_dist", "sum"),
                 block_time_known=("block_time_known", "sum"))
            .rename_axis(["airport_id", "month"]).reset_index())


T100_API = "https://data.bts.gov/resource/r495-tyji.json"
T100_FIELDS = {"origin_airport_id": "airport_id", "origin_airport_code": "code", "origin_airport_name": "name",
               "origin_city_name": "city", "reporting_month": "month", "total_passengers": "passengers",
               "total_seats": "seats", "total_departures": "departures",
               "outbound_international_1": "intl_passengers"}


def t100_from_api(page: int = 50000) -> pd.DataFrame:
    """All rows of the T-100 origin-airport summary from the data.bts.gov (Socrata) API."""
    import httpx
    rows, offset = [], 0
    with httpx.Client(timeout=120, headers={"User-Agent": "airport-agent/0.1"}) as http:
        while True:
            r = http.get(T100_API, params={"$select": ",".join(T100_FIELDS), "$limit": page, "$offset": offset,
                                           "$order": "origin_airport_id,reporting_month"})
            r.raise_for_status()
            batch = r.json()
            rows += batch
            if len(batch) < page:
                break
            offset += page
    out = pd.DataFrame(rows).rename(columns=T100_FIELDS)
    out["month"] = pd.to_datetime(out["month"])
    for c in ["airport_id", "passengers", "seats", "departures", "intl_passengers"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["airport_id"] = out.airport_id.astype("int64")
    out["intl_passengers"] = out["intl_passengers"].fillna(0)
    print(f"T-100 from the data.bts.gov API: {len(out):,} airport-months, {out.airport_id.nunique():,} airports, "
          f"{out.month.min():%Y-%m} to {out.month.max():%Y-%m}")
    return out[["airport_id", "code", "name", "city", "month", "passengers", "seats", "departures", "intl_passengers"]]


def t100_monthly(raw: Path) -> pd.DataFrame:
    """T-100 from the API; the downloaded file is the fallback."""
    try:
        return t100_from_api()
    except Exception as e:  # network or API problem: use the same dataset from the file
        print(f"T-100 API not reachable ({type(e).__name__}); using the file in data/raw instead")
        return t100_from_file(raw)


def t100_from_file(raw: Path) -> pd.DataFrame:
    files = glob.glob(str(raw / "AFF-T100_*.csv"))
    if not files:
        raise FileNotFoundError(f"No T-100 file in {raw}")
    t = pd.read_csv(files[0], skiprows=1, thousands=",")
    t["month"] = pd.to_datetime(t["Date"], format="%Y %b %d %I:%M:%S %p")
    out = t.rename(columns={"ORIGIN_AIRPORT_ID": "airport_id", "Origin Airport Code": "code",
                            "Total Passengers": "passengers", "Total Seats": "seats",
                            "Total Departures": "departures",
                            "Outbound International Passengers": "intl_passengers",
                            "Origin Airport Name": "name", "Origin City Name": "city"})
    out = out[["airport_id", "code", "name", "city", "month", "passengers", "seats", "departures", "intl_passengers"]]
    out["intl_passengers"] = out["intl_passengers"].fillna(0)  # blank international block = no international flights
    print(f"T-100: {len(out):,} airport-months, {out.airport_id.nunique():,} airports, "
          f"{out.month.min():%Y-%m} to {out.month.max():%Y-%m}")
    return out


def main() -> None:
    s = get_settings()
    s.processed_dir.mkdir(parents=True, exist_ok=True)
    long_haul = load_config("scoring")["long_haul"]

    df = load_ontime(s.raw_dir)
    cells = hourly_cells(df)
    cells.to_parquet(s.processed_dir / "hourly_cells.parquet", index=False)
    print(f"hourly_cells: {len(cells):,} rows, {cells.airport_id.nunique()} airports")

    om = ontime_monthly(df, long_haul)
    om.to_parquet(s.processed_dir / "ontime_monthly.parquet", index=False)
    print(f"ontime_monthly: {len(om):,} rows")

    codes = (pd.concat([df[["ORIGIN_AIRPORT_ID", "ORIGIN", "date"]].set_axis(["airport_id", "iata", "date"], axis=1),
                        df[["DEST_AIRPORT_ID", "DEST", "date"]].set_axis(["airport_id", "iata", "date"], axis=1)])
             .sort_values("date").drop_duplicates("airport_id", keep="last")[["airport_id", "iata"]])
    codes.to_parquet(s.processed_dir / "airport_codes.parquet", index=False)
    print(f"airport_codes: {len(codes)} airports")

    t = t100_monthly(s.raw_dir)
    t.to_parquet(s.processed_dir / "t100_monthly.parquet", index=False)


if __name__ == "__main__":
    main()
