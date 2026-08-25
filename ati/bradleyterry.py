"""Bradley-Terry rating from pairwise comparisons.

Chosen over Elo or Glicko deliberately. Those are sequential algorithms built
for players arriving over time, so ratings depend on the order matches were
played. A benchmark has every comparison in hand at once, and a maximum
likelihood fit is order-independent, reproducible from the same data, and
supports interval estimates.

Ties count as half a win to each side, the same convention used by Chatbot
Arena. Two translations of one sentence are frequently equivalent, and forcing
the judge to name a winner manufactures noise.
"""

from __future__ import annotations

import math
import random

OUTCOMES = ("a", "b", "tie")
ELO_SCALE = 400.0


def _tally(comparisons, prior: float):
    models = sorted({m for c in comparisons for m in (c[0], c[1])})
    index = {m: i for i, m in enumerate(models)}
    wins = [0.0] * len(models)
    pair_counts: dict[tuple[int, int], float] = {}

    for a, b, outcome in comparisons:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}; expected one of {OUTCOMES}")
        ia, ib = index[a], index[b]
        if outcome == "a":
            wins[ia] += 1.0
        elif outcome == "b":
            wins[ib] += 1.0
        else:
            wins[ia] += 0.5
            wins[ib] += 0.5
        key = (min(ia, ib), max(ia, ib))
        pair_counts[key] = pair_counts.get(key, 0.0) + 1.0

    # Regularize: add `prior` worth of tie to every observed pairing. Without
    # this an undefeated model has no finite maximum likelihood estimate and
    # the fit runs away to infinity.
    for (i, j) in list(pair_counts):
        pair_counts[(i, j)] += 2.0 * prior
        wins[i] += prior
        wins[j] += prior

    return models, index, wins, pair_counts


def fit(comparisons, prior: float = 0.5, tol: float = 1e-10, max_iter: int = 10000):
    """Return strength parameters, normalized to geometric mean 1."""
    if not comparisons:
        return {}

    models, index, wins, pair_counts = _tally(comparisons, prior)
    n = len(models)
    strengths = [1.0] * n

    for _ in range(max_iter):
        updated = [0.0] * n
        for i in range(n):
            denominator = 0.0
            for j in range(n):
                if i == j:
                    continue
                count = pair_counts.get((min(i, j), max(i, j)), 0.0)
                if count:
                    denominator += count / (strengths[i] + strengths[j])
            updated[i] = wins[i] / denominator if denominator > 0 else strengths[i]

        mean_log = sum(math.log(v) for v in updated if v > 0) / n
        scale = math.exp(mean_log)
        updated = [v / scale for v in updated]

        shift = max(abs(updated[i] - strengths[i]) for i in range(n))
        strengths = updated
        if shift < tol:
            break

    return {m: strengths[index[m]] for m in models}


def to_elo(strengths, anchor_id: str | None = None, anchor_rating: float = 1000.0):
    """Map strengths onto an Elo-like scale where 400 points is 10:1 odds.

    Only the offset is arbitrary; the spacing carries the meaning. Pinning an
    anchor keeps the numbers stable as models join or leave the pool.
    """
    if not strengths:
        return {}
    raw = {m: ELO_SCALE * math.log10(s) for m, s in strengths.items()}
    if anchor_id and anchor_id in raw:
        offset = anchor_rating - raw[anchor_id]
    else:
        offset = anchor_rating - sum(raw.values()) / len(raw)
    return {m: v + offset for m, v in raw.items()}


def win_probability(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / ELO_SCALE))


def bootstrap_ci(
    comparisons,
    rounds: int = 200,
    seed: int = 0,
    anchor_id: str | None = None,
    anchor_rating: float = 1000.0,
):
    """Percentile bootstrap over comparisons, returning 95% intervals.

    Published so a reader can see when a gap between two models is not
    distinguishable from zero.
    """
    if not comparisons:
        return {}

    rng = random.Random(seed)
    total = len(comparisons)
    samples: dict[str, list[float]] = {}

    for _ in range(rounds):
        resampled = [comparisons[rng.randrange(total)] for _ in range(total)]
        ratings = to_elo(fit(resampled), anchor_id, anchor_rating)
        for model, rating in ratings.items():
            samples.setdefault(model, []).append(rating)

    intervals = {}
    for model, values in samples.items():
        values.sort()
        lo = values[max(0, int(0.025 * len(values)))]
        hi = values[min(len(values) - 1, int(0.975 * len(values)))]
        intervals[model] = (lo, hi)
    return intervals
