"""Score aggregation: axes to composite, plus the cost and speed metrics."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class AxisStats:
    mean: float | None
    stdev: float
    n: int


def aggregate_axis(scores: list[int]) -> AxisStats:
    if not scores:
        return AxisStats(mean=None, stdev=0.0, n=0)
    mean = statistics.fmean(scores)
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return AxisStats(mean=mean, stdev=stdev, n=len(scores))


def composite(
    axis_means: dict[str, float | None], weights: dict[str, float]
) -> float | None:
    """Weighted 0-100 quality score.

    A missing axis - every judgment for it errored - is dropped and the
    remaining weights renormalized, so a judge failure lowers confidence
    rather than silently lowering the score.
    """
    present = {
        axis: mean for axis, mean in axis_means.items() if mean is not None
    }
    if not present:
        return None
    total_weight = sum(weights[axis] for axis in present)
    if total_weight <= 0:
        return None
    normalized = sum(
        weights[axis] * (mean - 1.0) / 4.0 for axis, mean in present.items()
    )
    return 100.0 * normalized / total_weight


def refusal_rate(refusals: list[bool]) -> float:
    if not refusals:
        return 0.0
    return 100.0 * sum(1 for r in refusals if r) / len(refusals)


def cost_per_1k_chars(total_cost: float | None, total_chars: int) -> float | None:
    """None means unknown - subscription access - and never zero."""
    if total_cost is None or total_chars <= 0:
        return None
    return total_cost * 1000.0 / total_chars


def speed_per_1k_chars(
    latencies: list[float], chars: list[int]
) -> float | None:
    rates = [
        latency * 1000.0 / char_count
        for latency, char_count in zip(latencies, chars)
        if char_count > 0
    ]
    if not rates:
        return None
    return statistics.median(rates)


def value_score(
    composite_value: float | None, cost: float | None
) -> float | None:
    """Composite points per dollar per 1k source characters."""
    if composite_value is None or cost is None or cost <= 0:
        return None
    return composite_value / cost


def term_stability(term_tables: list[dict[str, str]]) -> float | None:
    """Fraction of recurring source terms rendered identically everywhere.

    Deliberately computed rather than judged: asking a judge to hold ten
    chapters in mind is less reliable than counting the term tables it
    extracted from each chapter separately.

    Terms appearing in only one chapter are excluded - a term used once
    cannot be inconsistent, and counting it inflates the score.
    """
    renderings: dict[str, set[str]] = {}
    for table in term_tables:
        for term, rendering in table.items():
            renderings.setdefault(term, set()).add(rendering.strip().lower())

    counts: dict[str, int] = {}
    for table in term_tables:
        for term in table:
            counts[term] = counts.get(term, 0) + 1

    recurring = [term for term, count in counts.items() if count > 1]
    if not recurring:
        return None
    stable = sum(1 for term in recurring if len(renderings[term]) == 1)
    return stable / len(recurring)


def stability_to_axis_score(stability: float | None) -> float | None:
    """Map 0.0-1.0 stability onto the 1-5 axis scale used by every axis."""
    if stability is None:
        return None
    return 1.0 + 4.0 * stability
