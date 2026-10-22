"""`ati cost`: what a run's recorded tokens come to at list price. No calls.

Command Code and opencode Go meter their allowances in dollars, and neither
exposes a balance, so run 2026-09 learned that Command Code's monthly credit
was gone only when every call began failing. This puts a number on the drain
before that happens: recorded tokens times the roster's `price`, and a
projection for what is left from the average completed chapter.

It is a floor. A retried call's record is overwritten by the retry, so the
tokens the first attempt spent are nowhere on disk to count.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ati.fileio import iter_records


@dataclass
class CostRow:
    model_id: str
    priced: bool
    tokens_in: int = 0
    tokens_out: int = 0
    spent: float | None = None
    ok: int = 0
    ok_spent: float = 0.0
    remaining: int = 0
    projected: float | None = None
    unmetered: int = 0


def _dollars(price: dict, tokens_in: int, tokens_out: int, cached: int = 0) -> float:
    cached = min(cached, tokens_in)
    cached_rate = price.get("cached_input", price["input"])
    return (
        (tokens_in - cached) * price["input"]
        + cached * cached_rate
        + tokens_out * price["output"]
    ) / 1_000_000


def record_cost(record: dict, price: dict | None) -> float | None:
    """One call at list price, or None when it cannot be known.

    The route's own list-price figure wins where it reports one: the Claude
    CLI prices its cache reads and writes itself, which tokens times a flat
    rate cannot. Otherwise recorded tokens times the roster's `price`, cached
    input at its `cached_input` rate when the row declares one.
    """
    own = record.get("list_cost_usd")
    if isinstance(own, (int, float)):
        return float(own)
    t_in, t_out = record.get("tokens_in"), record.get("tokens_out")
    if price is None or (t_in is None and t_out is None):
        return None
    return _dollars(price, t_in or 0, t_out or 0, record.get("cached_tokens_in") or 0)


def run_cost(
    run_id: str,
    runs_root: Path,
    prices: dict[str, dict | None],
    instance_ids: list[str],
) -> dict[str, CostRow]:
    """Per model in `prices`: tokens and dollars recorded, and left to spend."""
    rows = {m: CostRow(m, priced=p is not None) for m, p in prices.items()}
    done: dict[str, set[str]] = {m: set() for m in prices}
    for path, record in iter_records(
        sorted((Path(runs_root) / run_id).rglob("*.json")), required=("model_id",)
    ):
        model_id = record["model_id"]
        if model_id not in rows:
            continue
        row, price = rows[model_id], prices[model_id]
        t_in, t_out = record.get("tokens_in"), record.get("tokens_out")
        failed = bool(record.get("error") or record.get("unavailable"))
        if not failed:
            done[model_id].add(path.stem)
        if t_in is None and t_out is None:
            if not failed:
                row.unmetered += 1
            continue
        row.tokens_in += t_in or 0
        row.tokens_out += t_out or 0
        dollars = record_cost(record, price)
        if dollars is not None:
            row.priced = True
            row.spent = (row.spent or 0.0) + dollars
            if not failed:
                row.ok += 1
                row.ok_spent += dollars
    for model_id, row in rows.items():
        row.remaining = sum(1 for iid in instance_ids if iid not in done[model_id])
        if row.priced:
            row.spent = row.spent or 0.0
            if row.remaining == 0:
                row.projected = 0.0
            elif row.ok:
                row.projected = row.ok_spent / row.ok * row.remaining
    return rows


def _money(value: float | None) -> str:
    return "?" if value is None else f"${value:,.2f}"


def render_cost(run_id: str, rows: dict[str, CostRow]) -> str:
    lines = [
        f"run {run_id}: list-price cost of recorded tokens (a floor: retried "
        f"attempts are overwritten and not counted)",
        "",
        f"{'model':30} {'in':>10} {'out':>11} {'spent':>9} {'left':>5} {'to go':>9}",
    ]
    spent = projected = 0.0
    unknown = []
    for model_id in sorted(rows):
        row = rows[model_id]
        lines.append(
            f"{model_id:30} {row.tokens_in:>10,} {row.tokens_out:>11,} "
            f"{_money(row.spent):>9} {row.remaining:>5} {_money(row.projected):>9}"
        )
        if row.spent is None or (row.remaining and row.projected is None):
            unknown.append(model_id)
        spent += row.spent or 0.0
        projected += row.projected or 0.0
    lines += ["", f"total: {_money(spent)} spent, about {_money(projected)} to go"]
    if unknown:
        lines.append(
            f"not counted (no `price` in the roster, or no finished chapter to "
            f"project from): {', '.join(unknown)}"
        )
    return "\n".join(lines)
