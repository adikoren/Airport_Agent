"""Phase 0: one OAG Flight Info call, to learn the real response shape.

Usage:
    python scripts/oag_spike.py [AIRPORT] [YYYY-MM-DD] [--auth-header NAME]

Reads OAG_API_KEY from the environment (or .env). Sends it only to api.oag.com.
Never prints the key. Saves the raw JSON to tests/fixtures/oag/ and prints its structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from airport_agent.settings import get_settings  # noqa: E402

BASE = "https://api.oag.com/flight-instances/"


def shape(obj, depth=0, max_depth=4):
    """Return a compact description of a JSON structure (keys and types, no values)."""
    pad = "  " * depth
    if isinstance(obj, dict):
        lines = []
        for k, v in obj.items():
            if isinstance(v, (dict, list)) and depth < max_depth:
                lines.append(f"{pad}{k}: {type(v).__name__}")
                lines.append(shape(v, depth + 1, max_depth))
            else:
                sample = v if isinstance(v, (int, float, bool)) or v is None else str(v)[:40]
                lines.append(f"{pad}{k}: {type(v).__name__} = {sample!r}")
        return "\n".join(line for line in lines if line)
    if isinstance(obj, list):
        if not obj:
            return f"{pad}[] (empty)"
        return f"{pad}[{len(obj)} items] first item:\n" + shape(obj[0], depth + 1, max_depth)
    return f"{pad}{obj!r}"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    airport = args[0] if args else "BOS"
    date = args[1] if len(args) > 1 else "2026-10-15"
    header = "Subscription-Key"
    if "--auth-header" in sys.argv:
        header = sys.argv[sys.argv.index("--auth-header") + 1]

    key = get_settings().require_oag_key()
    params = {"DepartureDateTime": date, "DepartureAirport": airport, "CodeType": "IATA", "version": "v2"}
    r = httpx.get(BASE, params=params, headers={header: key}, timeout=60)

    print(f"HTTP {r.status_code}  ({airport}, {date}, auth header '{header}')")
    print("Response headers of interest:")
    for h, v in r.headers.items():
        if any(t in h.lower() for t in ("rate", "limit", "quota", "retry", "remaining", "content-type")):
            print(f"  {h}: {v}")
    try:
        body = r.json()
    except ValueError:
        print("Body is not JSON:", r.text[:500])
        return
    if r.status_code != 200:
        print(json.dumps(body, indent=2)[:1500])
        return

    out = ROOT / "tests" / "fixtures" / "oag" / f"flight_instances_{airport}_{date}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2))
    print(f"Saved raw response to {out.relative_to(ROOT)}\n")
    print("Structure:")
    print(shape(body))


if __name__ == "__main__":
    main()
