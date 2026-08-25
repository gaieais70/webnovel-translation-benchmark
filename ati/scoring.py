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
