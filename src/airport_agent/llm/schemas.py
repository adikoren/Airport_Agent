"""Structured outputs Claude must return (validated by the SDK against these Pydantic models)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["rank", "compare", "metric_lookup", "explain_why", "conceptual", "research", "methodology", "out_of_scope"]
RankBy = Literal["rps", "constraint", "passengers", "growth_share", "load_factor", "peak_saturation",
                 "taxi_out_excess", "taxi_in_excess", "international_share", "delay_rate", "cancellation_rate"]
MetricId = Literal["passengers", "load_factor", "international_share", "growth_share", "peak_saturation",
                   "taxi_out_excess", "taxi_in_excess", "long_haul_share", "delay_rate", "cancellation_rate",
                   "departures"]


class SignalWeights(BaseModel):
    """Only set when the user asks to weight the constraint signals differently."""
    load_factor: float | None = None
    peak_saturation: float | None = None
    taxi_out_excess: float | None = None
    taxi_in_excess: float | None = None


class ParsedRequest(BaseModel):
    intent: Intent = Field(description="What the user wants. Use out_of_scope only if nothing in the request is in scope.")
    airports: list[str] = Field(default_factory=list,
                                description="Every airport mention exactly as the user wrote it (codes, names or cities). "
                                            "For a follow-up that says 'it'/'that one', write the airport it refers to.")
    region: str | None = Field(default=None, description="A region, state or Census division the user named, as written.")
    metrics: list[MetricId] = Field(default_factory=list, description="For metric_lookup: the metrics asked for.")
    horizon_years: Literal[5, 10, 20] | None = Field(default=None, description="Only if the user asks for a horizon.")
    weights: SignalWeights | None = Field(default=None, description="Only if the user asks to weight signals differently.")
    long_haul_definition: Literal["block_time", "distance"] | None = None
    long_haul_threshold: float | None = Field(default=None, description="Hours (block_time) or miles (distance), only if the user gives one.")
    top_n: int | None = Field(description="How many airports the message asks to show: a number of airports "
                              "('top 5', '3 busiest', 'ten largest') or 1 for a single answer ('the biggest', 'which airport'). "
                              "null if the message doesn't say.")
    rank_by: RankBy | None = Field(default=None, description="For rank: the measure to sort airports by. rps = renovation "
                            "candidates (the default); constraint = how constrained/congested; passengers = size/busiest; "
                            "growth_share = forecast growth; or one specific signal or measure.")
    order: Literal["highest", "lowest"] | None = Field(default=None, description="highest for biggest/most/top; lowest for smallest/least.")
    follow_up: bool = Field(description="True if the message builds on the previous answer (same airports or results).")
    out_of_scope_parts: list[str] = Field(default_factory=list, description="Parts of the request the agent can't do.")
    ranked_positions: list[int] = Field(default_factory=list,
                                        description="Positions in the ranking from the earlier answer that the message "
                                                    "refers to: 'the top-ranked airport' or 'the first one' = [1]; 'the "
                                                    "second' = [2]. Empty unless the message refers to a position in an "
                                                    "earlier ranking. Never a position for a new ranking request.")


class AirportChoice(BaseModel):
    mention: str
    choice: str | None = Field(description="The chosen candidate's IATA code, exactly as listed, or null to ask the user.")
    confidence: Literal["high", "medium", "low"]


class Disambiguation(BaseModel):
    choices: list[AirportChoice]
