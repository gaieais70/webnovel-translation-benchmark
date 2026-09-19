"""Aggregation of run and judgment files into results and a Markdown report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ati.config import AXES
from ati.fileio import iter_records, read_json
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
    refused_tasks: int = 0


def fmt(value: float | None, places: int = 1) -> str:
    """Unknown renders as N/A. Never as zero - that would imply 'free'."""
    if value is None:
        return "N/A"
    return f"{value:.{places}f}"


def fmt_cost(value: float | None) -> str:
    """Unknown, subscription-covered, and priced are three different things.

    None means the access path reports no cost at all. Zero means the call
    really did bill zero because a subscription covered it. Printing either as
    "0.0000" would read as "free", which misleads the value comparison.
    """
    if value is None:
        return "N/A"
    if value == 0:
        return "sub"
    return f"{value:.4f}"


def _load_all(root: Path) -> list[dict]:
    """Every readable record under `root` that names its model.

    Anything else - a torn file, a stray copy - is reported and left out
    rather than raising, so one bad file cannot stop a report of hundreds.
    """
    return [
        record
        for _, record in iter_records(sorted(root.rglob("*.json")), required=("model_id",))
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
                tables_by_task.setdefault(j.get("task_id"), []).append(j["term_table"])
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
        texts = [r["text"] for r in scored_runs if r.get("text")]
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
                refused_tasks=sum(
                    1
                    for r in model_runs
                    if r.get("refused") and not r.get("error")
                ),
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


def anchor_composites(results: list[ModelResult]) -> dict[str, float]:
    """Composite of every anchor model that has one."""
    return {
        r.model_id: r.composite
        for r in results
        if r.anchor and r.composite is not None
    }


def _numeric(mapping) -> dict[str, float]:
    if not isinstance(mapping, dict):
        return {}
    return {
        k: float(v)
        for k, v in mapping.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def _anchors_of_results_file(data: dict) -> dict[str, float]:
    return {
        r["model_id"]: float(r["composite"])
        for r in data.get("results") or []
        if isinstance(r, dict) and r.get("anchor") and r.get("composite") is not None
    }


def _run_that_wrote(flat: dict[str, float], directory: Path) -> str | None:
    """The run whose results file, beside a legacy anchors.json, holds these
    exact anchor composites - that is, the run that wrote it."""
    for path in sorted(Path(directory).glob("*.json")):
        data = read_json(path)
        if not data or "results" not in data or not data.get("run_id"):
            continue
        if _anchors_of_results_file(data) == flat:
            return str(data["run_id"])
    return None


def read_anchor_baseline(path: Path, run_id: str) -> tuple[str | None, dict[str, float]]:
    """(run id, anchor composites) of the earlier run to measure drift against.

    anchors.json is rewritten by every report. It used to hold only
    {model: composite}, so from the second report of a run onward - after
    the head-to-head pass, say - the "previous run" it offered was the run
    itself, drift was zero, and the not-comparable warning could never fire.
    The file now names its run and carries that run's own baseline forward:

        {"run_id": ..., "anchors": {...}, "baseline": {"run_id", "anchors"}}

    A legacy flat file is still read. Its run is recovered from the results
    file beside it that holds the same composites; if that is this very run,
    it is not a baseline at all. (None, {}) means drift cannot be checked.
    """
    data = read_json(path)
    if data is None:
        return None, {}
    if isinstance(data.get("anchors"), dict):
        if data.get("run_id") != run_id:
            return str(data.get("run_id") or "unknown"), _numeric(data["anchors"])
        baseline = data.get("baseline")
        if isinstance(baseline, dict) and baseline.get("anchors"):
            return str(baseline.get("run_id") or "unknown"), _numeric(
                baseline["anchors"]
            )
        return None, {}
    flat = _numeric(data)
    if not flat or len(flat) != len(data):
        return None, {}
    writer = _run_that_wrote(flat, Path(path).parent)
    if writer == run_id:
        return None, {}
    return writer or "unknown", flat


def anchors_file(
    run_id: str,
    results: list[ModelResult],
    baseline_run: str | None,
    baseline: dict[str, float],
) -> dict:
    """What `ati report` writes to anchors.json for the next run to read."""
    return {
        "run_id": run_id,
        "anchors": anchor_composites(results),
        "baseline": (
            {"run_id": baseline_run, "anchors": baseline} if baseline_run else None
        ),
    }


def pairwise_records(comparisons) -> dict[str, tuple[int, int, int]]:
    """Wins, losses and ties per model from (a, b, outcome) triples."""
    tally: dict[str, list[int]] = {}
    for a, b, outcome in comparisons:
        for model in (a, b):
            tally.setdefault(model, [0, 0, 0])
        if outcome == "a":
            tally[a][0] += 1
            tally[b][1] += 1
        elif outcome == "b":
            tally[b][0] += 1
            tally[a][1] += 1
        else:
            tally[a][2] += 1
            tally[b][2] += 1
    return {m: tuple(v) for m, v in tally.items()}


def render_markdown(
    run_id: str,
    results: list[ModelResult],
    drift: dict[str, float],
    comparable: bool,
    metadata: dict,
    pairwise: dict[str, float] | None = None,
    pairwise_ci: dict[str, tuple[float, float]] | None = None,
    pairwise_anchor: str | None = None,
    pairwise_records: dict[str, tuple[int, int, int]] | None = None,
    docs_prefix: str = "../docs/",
) -> str:
    """The leaderboard.

    `docs_prefix` is the relative path from wherever the file is written to
    the repository's docs/ directory; the default suits results/.
    """
    ranked = [r for r in results if not r.incomplete]
    incomplete = [r for r in results if r.incomplete]

    lines: list[str] = [
        "# Artificial Translation Index",
        "",
        f"Run `{run_id}` — {metadata.get('date', '')}",
        "",
        f"Judge: `{metadata.get('judge_id', '')}`. "
        "Scores are LLM-judged against a fixed rubric; see "
        f"[the rubric]({docs_prefix}rubric.md) and "
        f"[the methodology]({docs_prefix}methodology.md).",
        "",
    ]

    # Said either way, so a reader can tell "checked and fine" from "never
    # checked" - the two used to look identical.
    if "anchor_baseline" in metadata:
        baseline = metadata["anchor_baseline"]
        if baseline is None:
            lines += [
                "Anchor drift: not checked - no earlier run's anchors were found.",
                "",
            ]
        elif drift:
            moved = ", ".join(f"{k} {v:+.1f}" for k, v in drift.items())
            lines += [f"Anchor drift against run `{baseline}`: {moved}.", ""]
        else:
            lines += [
                f"Anchor drift: not checked - no anchor in common with run "
                f"`{baseline}`.",
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
        "| # | Model | Composite | Scored | Refusal % | Cost /1k ch | Sec /1k ch | Value | Slop | Repetition |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(ranked, start=1):
        label = r.model_id + (f" _{r.anchor} anchor_" if r.anchor else "")
        # Shown because a model missing chapters is averaged over fewer, and
        # the chapters that break models are the hard ones - so a composite
        # over 8/11 is not the same measurement as one over 11/11.
        scored = (
            f"{r.total_tasks - r.failed_tasks - r.refused_tasks}/{r.total_tasks}"
        )
        lines.append(
            f"| {i} | {label} | {fmt(r.composite)} | {scored} | {fmt(r.refusal_rate)} | "
            f"{fmt_cost(r.cost_per_1k)} | {fmt(r.speed_per_1k)} | "
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

    if pairwise:
        lines += [
            "",
            "## Head-to-head (Bradley–Terry)",
            "",
            "Ratings from blind pairwise comparison, judged in both orders. "
            "400 points is 10:1 odds; only the offset is arbitrary. Ties are "
            "allowed and a judge that reverses itself when the order flips is "
            "recorded as a tie.",
            "",
            "| Model | Rating | 95% interval | W–L–T |",
            "|---|---|---|---|",
        ]
        ranked_pairwise = sorted(
            pairwise.items(), key=lambda kv: -kv[1]
        )
        for model_id, rating in ranked_pairwise:
            interval = (pairwise_ci or {}).get(model_id)
            if not interval:
                span = "N/A"
            elif abs(interval[1] - interval[0]) < 0.5 and model_id == pairwise_anchor:
                # The scale is pinned to this model, so every bootstrap
                # replicate lands on the same value.
                span = "pinned"
            elif abs(interval[1] - interval[0]) < 0.5:
                # Zero width here is not certainty - it means every resample
                # drew the same handful of comparisons. Printing a tight
                # interval would imply a precision the data does not support.
                span = "too few comparisons"
            else:
                span = f"{interval[0]:.0f} – {interval[1]:.0f}"
            # The record is shown because a rating alone hides how much it
            # rests on: a model compared fewer times, or only on the passages
            # other models survived, can sit high on thin evidence.
            w, l, t = (pairwise_records or {}).get(model_id, (0, 0, 0))
            record = f"{w}–{l}–{t}" if pairwise_records else "N/A"
            lines.append(f"| {model_id} | {rating:.0f} | {span} | {record} |")

        # How much of the field the ratings rest on. A partial schedule was
        # published once already and had to be retracted (run-notes-2026-08).
        coverage = metadata.get("pairwise_coverage")
        if coverage:
            lines += [
                "",
                f"Based on {coverage['comparisons']} comparisons covering "
                f"{coverage['pairs']} of {coverage['possible_pairs']} model pairs.",
            ]

        if pairwise_ci and len(ranked_pairwise) > 1:
            top, second = ranked_pairwise[0][0], ranked_pairwise[1][0]
            first_ci, second_ci = pairwise_ci.get(top), pairwise_ci.get(second)
            if first_ci and second_ci and first_ci[0] <= second_ci[1]:
                lines += [
                    "",
                    f"> The intervals for **{top}** and **{second}** overlap, so "
                    "their ordering is not established by this data. More "
                    "comparisons would be needed to separate them.",
                ]

    lines += [
        "",
        "## Notes",
        "",
        "- **Cost `sub`** means a subscription covered the call, so it billed "
        "zero. That is not the same as the model being free to run. **`N/A`** "
        "means the access path reports no cost at all. Neither is a price.",
        "- **Slop** counts LLM-tell and translationese phrases per 1,000 words. "
        "**Repetition** is top-n-gram concentration. Neither enters the "
        "composite; both are diagnostics.",
        "- Model behavior changes under a stable name. These results describe "
        f"the versions tested on {metadata.get('date', 'the run date')}.",
    ]
    # Only when there is something to say. A composite is averaged over the
    # items that were scored; these are the items that were not.
    errored = [r for r in results if r.judge_errors]
    if errored:
        lines.append(
            "- **Judge errors**: "
            + ", ".join(f"{r.model_id} ({r.judge_errors})" for r in errored)
            + ". The judge gave no usable answer on these items, so those "
            "composites rest on fewer items than the Scored column shows."
        )
    pending = metadata.get("pending_judgments")
    if pending:
        lines.append(
            f"- **Not yet judged**: {pending} outputs have no current judgment "
            "(missing, or made by a different judge or prompt). Run `ati judge` "
            "before reading this ranking as final."
        )
    lines.append("")
    return "\n".join(lines)
