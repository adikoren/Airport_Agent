# OAG API — Phase 0 spike findings

Date: 2026-11-05 · Script: `scripts/oag_spike.py` · Fixtures: `tests/fixtures/oag/`

## Connection
- **Endpoint:** `GET https://api.oag.com/flight-instances/`
- **Auth:** header `Subscription-Key: <key>` (works). Key read from `OAG_API_KEY`.
- **Parameters used:** `DepartureDateTime=YYYY-MM-DD`, `DepartureAirport=<IATA>`, `CodeType=IATA`, `version=v2`.
- **Paging:** response `paging = {limit, totalCount, totalPages, next}`. `next` is a full URL on `api.oag.com` with `Limit` and an `After` cursor. Follow `next` until it's absent.
- **No rate-limit or quota headers** were returned.

## Plan limits (free tier)
- `Limit` maximum is **10** per call. Asking for 1,000 returned HTTP 400: *"the maximum allowed for the free tier is 10"*.
- Records per airport-day (including codeshare copies): **BOS 1,769 (177 calls)**, **ANC 488 (49 calls)**.
- **Consequence (decision D2):** OAG is used **only for long-haul share**, for the one airport a question names. All constraint signals come from BTS public data.

## Fields in a flight record
| Need | Field | Notes |
|---|---|---|
| Block time | `elapsedTime` (minutes) | Some records have `0` (seen on cargo flights): impossible, so treated as missing → use distance |
| Distance | `distance.greatCircleMiles` | Present on all sampled records |
| Passenger vs cargo | `serviceType.iata` | `J` = scheduled passenger, `F` = scheduled cargo; sometimes missing |
| Operating vs codeshare copy | `codeshare.operatingFlight` | **Absent on the operating flight**; present on marketing copies (points to the operating carrier and number) |
| Destination | `arrival.airport.iata`, `arrival.country.code` | International = country ≠ US |
| Seats | — | **Not in the records** on this plan |
| Actual times / status | only `statusKey` | **No actual out/off/on/in times** in the records on this plan |

## What the code does with this
- Count only **operating flights** (no `operatingFlight` field), so codeshare copies aren't double-counted.
- Long-haul share is computed for **all flights and for passenger flights** (`serviceType` `J`), because cargo is a large share at some airports.
- **Call budget:** `oag_calls_per_question = 200` (one airport-day up to 2,000 records). If an airport-day needs more, the agent doesn't sample a biased part of the day: it falls back to BTS (domestic only) and says international flights are missing.
