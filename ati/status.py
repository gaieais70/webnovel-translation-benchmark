"""`ati status`: where a run stands, without calling anything.

The generic form of scripts/2026-09/status.py, which hardcoded the run id and
eleven chapters. Expected work comes from the roster and the task suite, so a
chapter that was never attempted shows as missing rather than not at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ati.fileio import iter_records


@dataclass
class ModelProgress:
    model_id: str
    ok: int = 0
    refused: int = 0
    failed: int = 0
    missing: int = 0
    latencies: list[float] = field(default_factory=list)
    ratios: list[float] = field(default_factory=list)
    routes: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)


def run_progress(
    run_id: str, runs_root: Path, model_ids: list[str], instance_ids: list[str]
) -> dict[str, ModelProgress]:
    """Per model: translations ok, refused, failed, and never attempted."""
    progress = {m: ModelProgress(m) for m in model_ids}
    seen: set[tuple[str, str]] = set()
    for path, record in iter_records(
        sorted((Path(runs_root) / run_id).rglob("*.json")), required=("model_id",)
    ):
        model_id = record["model_id"]
        row = progress.setdefault(model_id, ModelProgress(model_id))
        seen.add((model_id, path.stem))
        if record.get("error") or record.get("unavailable"):
            row.failed += 1
            label = record.get("passage_id") or path.stem
            row.errors.append(f"{label}: {str(record.get('error'))[:110]}")
        elif record.get("refused"):
            row.refused += 1
        else:
            row.ok += 1
            row.latencies.append(float(record.get("latency_s") or 0.0))
            if record.get("source_chars"):
                row.ratios.append(len(record.get("text") or "") / record["source_chars"])
        if record.get("route"):
            row.routes.add(str(record["route"]))
    for model_id in model_ids:
        progress[model_id].missing = sum(
            1 for iid in instance_ids if (model_id, iid) not in seen
        )
    return progress


def render_status(
    run_id: str,
    progress: dict[str, ModelProgress],
    instances: int,
    active: int,
    judgments: dict | None = None,
    comparisons: dict | None = None,
) -> str:
    lines = [
        f"run {run_id}: {len(progress)} models x {instances} instances",
        "",
        f"{'model':30} {'ok':>4} {'ref':>4} {'fail':>4} {'miss':>4} "
        f"{'avg s':>7} {'ratio':>6}  routes",
    ]
    totals = {"ok": 0, "refused": 0, "failed": 0, "missing": 0}
    for model_id in sorted(progress):
        row = progress[model_id]
        avg = sum(row.latencies) / len(row.latencies) if row.latencies else 0.0
        ratio = sum(row.ratios) / len(row.ratios) if row.ratios else 0.0
        lines.append(
            f"{model_id:30} {row.ok:>4} {row.refused:>4} {row.failed:>4} "
            f"{row.missing:>4} {avg:>7.0f} {'x' + format(ratio, '.2f'):>6}  "
            f"{','.join(sorted(row.routes))}"
        )
        lines += [f"    {e}" for e in row.errors]
        for key in totals:
            totals[key] += getattr(row, key)
    lines += [
        "",
        f"translations: {totals['ok']} ok, {totals['refused']} refused, "
        f"{totals['failed']} failed, {totals['missing']} missing; "
        f"{active} `ati run` process(es) active on this run",
    ]
    if judgments is not None:
        lines.append(
            f"judgments: {judgments['missing']} missing, {judgments['stale']} stale "
            f"(judge {judgments['judge_id']}, fingerprint {judgments['fingerprint']})"
        )
    if comparisons is not None:
        lines.append(
            f"comparisons: {comparisons['count']} on disk covering "
            f"{comparisons['pairs']} model pairs; {comparisons['stale']} stale"
        )
    return "\n".join(lines)
