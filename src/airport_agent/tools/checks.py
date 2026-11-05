"""Hard checks (ARCHITECTURE §6.4): decide whether an input can be used at all.

Each check returns None if the input passes, or a plain-language reason if it doesn't.
"""

from __future__ import annotations

from airport_agent.settings import load_config


def _cfg() -> dict:
    return load_config("scoring")["checks"]


def enough_flights(n: int, what: str) -> str | None:
    """At least `min_flights_per_mean` observations behind a mean (central-limit rule of thumb)."""
    need = _cfg()["min_flights_per_mean"]
    if n < need:
        return f"only {n} flights for {what} (at least {need} needed for a reliable mean)"
    return None


def load_factor_possible(value: float) -> str | None:
    lo, hi = _cfg()["load_factor_range"]
    if not (lo <= value <= hi):
        return f"load factor {value:.3f} is outside the possible range {lo}–{hi}"
    return None


def complete_months(n_months: int) -> str | None:
    need = _cfg()["months_in_window"]
    if n_months != need:
        return f"only {n_months} of {need} months have data"
    return None


def not_negative(value: float, what: str) -> str | None:
    if value < 0:
        return f"{what} is negative ({value}), which is impossible"
    return None
