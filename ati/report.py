"""Aggregation of run and judgment files into results and a Markdown report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ati.config import AXES
from ati.scoring import (
    aggregate_axis,
    composite,
    cost_per_1k_chars,
    refusal_rate,
    speed_per_1k_chars,
    stability_to_axis_score,
    term_stability,
    value_score,
)
from ati.textmetrics import repetition_score, slop_score

FAILURE_THRESHOLD = 0.20
ANCHOR_DRIFT_THRESHOLD = 5.0


@dataclass
class ModelResult:
    model_id: str
    composite: float | None
    axis_means: dict[str, float | None]
    axis_stdevs: dict[str, float]
    refusal_rate: float
    cost_per_1k: float | None
    speed_per_1k: float | None
    value: float | None
    slop: float
    repetition: float
    judge_errors: int
    failed_tasks: int
    total_tasks: int
    anchor: str | None = None
    incomplete: bool = False


def fmt(value: float | None, places: int = 1) -> str:
    """Unknown renders as N/A. Never as zero - that would imply 'free'."""
    if value is None:
        return "N/A"
    return f"{value:.{places}f}"


def _load_all(root: Path) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("*.json"))
    ]


def aggregate(
    run_id: str,
    runs_root: Path,
    judgments_root: Path,
    weights: dict[str, float],
    slop_list: list[str],
    anchors: dict[str, str],
) -> list[ModelResult]:
    runs = _load_all(Path(runs_root) / run_id)
    judgments = _load_all(Path(judgments_root) / run_id)

    by_model_runs: dict[str, list[dict]] = {}
    for r in runs:
        by_model_runs.setdefault(r["model_id"], []).append(r)
    by_model_judgments: dict[str, list[dict]] = {}
    for j in judgments:
        by_model_judgments.setdefault(j["model_id"], []).append(j)

    results: list[ModelResult] = []
    for model_id, model_runs in by_model_runs.items():
        model_judgments = by_model_judgments.get(model_id, [])

        # --- axis scores, averaged across iterations ---
        per_axis: dict[str, list[int]] = {axis: [] for axis in AXES}
        for j in model_judgments:
            for axis, score in (j.get("scores") or {}).items():
                if axis in per_axis:
                    per_axis[axis].append(int(score))
        stats = {axis: aggregate_axis(scores) for axis, scores in per_axis.items()}
        axis_means = {axis: s.mean for axis, s in stats.items()}
        axis_stdevs = {axis: s.stdev for axis, s in stats.items()}

        # --- terminology on consistency tasks is computed, not judged ---
        tables_by_task: dict[str, list[dict]] = {}
        for j in model_judgments:
            if j.get("task_type") == "consistency" and j.get("term_table"):
                tables_by_task.setdefault(j["task_id"], []).append(j["term_table"])
        stabilities = [
            s
            for s in (term_stability(tables) for tables in tables_by_task.values())
            if s is not None
        ]
        if stabilities:
            computed = stability_to_axis_score(sum(stabilities) / len(stabilities))
            axis_means["terminology"] = computed

        # --- diagnostics and operational metrics ---
        scored_runs = [
            r for r in model_runs if not r.get("unavailable") and not r.get("error")
        ]
        texts = [r["text"] for r in scored_runs if r["text"]]
        joined = "\n".join(texts)

        costs = [r.get("cost_usd") for r in scored_runs]
        total_cost = None if any(c is None for c in costs) or not costs else sum(costs)
        total_chars = sum(r.get("source_chars", 0) for r in scored_runs)

        comp = composite(axis_means, weights)
        cost_1k = cost_per_1k_chars(total_cost, total_chars)
        failed = sum(
            1 for r in model_runs if r.get("unavailable") or r.get("error")
        )

        results.append(
            ModelResult(
                model_id=model_id,
                composite=comp,
                axis_means=axis_means,
                axis_stdevs=axis_stdevs,
                refusal_rate=refusal_rate(
                    [bool(r.get("refused")) for r in model_runs]
                ),
                cost_per_1k=cost_1k,
                speed_per_1k=speed_per_1k_chars(
                    [r.get("latency_s", 0.0) for r in scored_runs],
                    [r.get("source_chars", 0) for r in scored_runs],
                ),
                value=value_score(comp, cost_1k),
                slop=slop_score(joined, slop_list),
                repetition=repetition_score(joined),
                judge_errors=sum(
                    1 for j in model_judgments if j.get("judge_error")
                ),
                failed_tasks=failed,
                total_tasks=len(model_runs),
                anchor=anchors.get(model_id),
                incomplete=bool(model_runs)
                and failed / len(model_runs) > FAILURE_THRESHOLD,
            )
        )

    results.sort(key=lambda r: (r.composite is None, -(r.composite or 0.0)))
    return results


def anchor_drift(
    current: list[ModelResult], previous_anchor_composites: dict[str, float]
) -> dict[str, float]:
    drift: dict[str, float] = {}
    for result in current:
        if result.anchor is None or result.composite is None:
            continue
        previous = previous_anchor_composites.get(result.model_id)
        if previous is None:
            continue
        drift[result.model_id] = result.composite - previous
    return drift


def is_comparable(drift: dict[str, float]) -> bool:
    return all(abs(d) <= ANCHOR_DRIFT_THRESHOLD for d in drift.values())


def render_markdown(
    run_id: str,
    results: list[ModelResult],
    drift: dict[str, float],
    comparable: bool,
    metadata: dict,
) -> str:
    ranked = [r for r in results if not r.incomplete]
    incomplete = [r for r in results if r.incomplete]

    lines: list[str] = [
        "# Artificial Translation Index",
        "",
        f"Run `{run_id}` — {metadata.get('date', '')}",
        "",
        f"Judge: `{metadata.get('judge_id', '')}`. "
        "Scores are LLM-judged against a fixed rubric; see "
        "[the rubric](../docs/rubric.md) and "
        "[the methodology](../docs/methodology.md).",
        "",
    ]

    if not comparable:
        lines += [
            "> **Not comparable to previous runs.** A calibration anchor moved "
            f"by more than {ANCHOR_DRIFT_THRESHOLD} composite points "
            f"({', '.join(f'{k}: {v:+.1f}' for k, v in drift.items())}), which "
            "means the judge, the prompt, or a provider changed behavior. "
            "Compare within this run only.",
            "",
        ]

    lines += [
        "## Ranking",
        "",
        "| # | Model | Composite | Refusal % | Cost /1k ch | Sec /1k ch | Value | Slop | Repetition |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(ranked, start=1):
        label = r.model_id + (f" _{r.anchor} anchor_" if r.anchor else "")
        lines.append(
            f"| {i} | {label} | {fmt(r.composite)} | {fmt(r.refusal_rate)} | "
            f"{fmt(r.cost_per_1k, 4)} | {fmt(r.speed_per_1k)} | "
            f"{fmt(r.value, 0)} | {fmt(r.slop, 2)} | {fmt(r.repetition, 3)} |"
        )

    lines += ["", "## Per-axis scores", "", "| Model | " + " | ".join(AXES) + " |"]
    lines.append("|---" * (len(AXES) + 1) + "|")
    for r in ranked:
        cells = " | ".join(
            f"{fmt(r.axis_means.get(axis))} ±{fmt(r.axis_stdevs.get(axis, 0.0))}"
            for axis in AXES
        )
        lines.append(f"| {r.model_id} | {cells} |")

    if incomplete:
        lines += [
            "",
            "## Incomplete",
            "",
            f"These models failed more than {int(FAILURE_THRESHOLD * 100)}% of "
            "their tasks and are excluded from the ranking.",
            "",
            "| Model | Failed | Total |",
            "|---|---|---|",
        ]
        for r in incomplete:
            lines.append(f"| {r.model_id} | {r.failed_tasks} | {r.total_tasks} |")

    lines += [
        "",
        "## Notes",
        "",
        "- **Cost N/A** means access was subscription-backed and per-call cost "
        "is unknown. It does not mean free.",
        "- **Slop** counts LLM-tell and translationese phrases per 1,000 words. "
        "**Repetition** is top-n-gram concentration. Neither enters the "
        "composite; both are diagnostics.",
        "- Model behavior changes under a stable name. These results describe "
        f"the versions tested on {metadata.get('date', 'the run date')}.",
        "",
    ]
    return "\n".join(lines)
