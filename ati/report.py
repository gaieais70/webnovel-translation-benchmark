"""Aggregation of run and judgment files into results and a Markdown report."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ati.config import AXES
from ati.cost import record_cost
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
    # Each panel judge's own composite, when more than one judge scored the
    # run. Shown beside the panel's, so a judge favouring its own family is
    # visible in the table rather than averaged out of sight.
    judge_composites: dict[str, float | None] = field(default_factory=dict)
    # At the roster's list price (or the route's own list figure), averaged
    # over scored chapters. Every route has one, subscription or metered, so
    # unlike the billed cost it compares models whatever they ran on.
    list_cost_per_chapter: float | None = None
    tokens_out_per_chapter: float | None = None


def fmt(value: float | None, places: int = 1) -> str:
    """Unknown renders as N/A. Never as zero - that would imply 'free'."""
    if value is None:
        return "N/A"
    return f"{value:.{places}f}"


def fmt_list_cost(value: float | None) -> str:
    return "N/A" if value is None else f"${value:.4f}"


def fmt_tokens(value: float | None) -> str:
    return "N/A" if value is None else f"{value:,.0f}"


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


def _item_key(judgment: dict) -> tuple:
    """Which translation a judgment scored, so panel judges can be paired."""
    return (
        judgment.get("task_id"),
        judgment.get("passage_id"),
        judgment.get("iteration"),
    )


def _computed_terminology(judgments: list[dict]) -> float | None:
    """Terminology on consistency tasks is computed from term tables, not judged.

    Always from one judge's tables at a time: two judges extract a table
    differently, and pooled, that difference reads as the translator
    rendering one term two ways.
    """
    tables_by_task: dict[str, list[dict]] = {}
    for j in judgments:
        if j.get("task_type") == "consistency" and j.get("term_table"):
            tables_by_task.setdefault(j.get("task_id"), []).append(j["term_table"])
    stabilities = [
        s
        for s in (term_stability(tables) for tables in tables_by_task.values())
        if s is not None
    ]
    if not stabilities:
        return None
    return stability_to_axis_score(sum(stabilities) / len(stabilities))


def _judged_axes(
    judgments_by_judge: dict[str, list[dict]], weights: dict[str, float]
) -> tuple[dict, dict, dict[str, float | None]]:
    """Axis means and spreads over a panel's judgments of one model.

    Each item's axis score is the mean of the judges that scored it, and the
    spread is across items. A panel of one is the single-judge report.
    Returns (axis_means, axis_stdevs, each judge's own composite).
    """
    by_item: dict[tuple, dict[str, list[int]]] = {}
    for judgments in judgments_by_judge.values():
        for j in judgments:
            item = by_item.setdefault(_item_key(j), {axis: [] for axis in AXES})
            for axis, score in (j.get("scores") or {}).items():
                if axis in item:
                    item[axis].append(int(score))
    per_axis: dict[str, list[float]] = {axis: [] for axis in AXES}
    for item in by_item.values():
        for axis, scores in item.items():
            if scores:
                per_axis[axis].append(sum(scores) / len(scores))
    stats = {axis: aggregate_axis(scores) for axis, scores in per_axis.items()}
    axis_means = {axis: s.mean for axis, s in stats.items()}
    axis_stdevs = {axis: s.stdev for axis, s in stats.items()}

    computed = [
        t
        for t in (_computed_terminology(js) for js in judgments_by_judge.values())
        if t is not None
    ]
    if computed:
        axis_means["terminology"] = sum(computed) / len(computed)

    judge_composites: dict[str, float | None] = {}
    if len(judgments_by_judge) > 1:
        for judge, judgments in judgments_by_judge.items():
            own, _, _ = _judged_axes({judge: judgments}, weights)
            judge_composites[judge] = composite(own, weights)
    return axis_means, axis_stdevs, judge_composites


def aggregate(
    run_id: str,
    runs_root: Path,
    judgments_root: Path | None,
    weights: dict[str, float],
    slop_list: list[str],
    anchors: dict[str, str],
    judge_roots: dict[str, Path] | None = None,
    prices: dict[str, dict | None] | None = None,
) -> list[ModelResult]:
    """Results per model. `judge_roots` names a panel, judge id to judgments
    root; without it, `judgments_root` is the one judge."""
    runs = _load_all(Path(runs_root) / run_id)
    roots = judge_roots or {"": Path(judgments_root)}
    by_judge_model: dict[str, dict[str, list[dict]]] = {}
    for judge, root in roots.items():
        for j in _load_all(Path(root) / run_id):
            by_judge_model.setdefault(j["model_id"], {}).setdefault(judge, []).append(j)

    by_model_runs: dict[str, list[dict]] = {}
    for r in runs:
        by_model_runs.setdefault(r["model_id"], []).append(r)

    results: list[ModelResult] = []
    for model_id, model_runs in by_model_runs.items():
        judgments_by_judge = {
            judge: by_judge_model.get(model_id, {}).get(judge, []) for judge in roots
        }
        model_judgments = [j for js in judgments_by_judge.values() for j in js]
        axis_means, axis_stdevs, judge_composites = _judged_axes(
            judgments_by_judge, weights
        )

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

        # --- list-price cost and output tokens, per scored chapter ---
        price = (prices or {}).get(model_id)
        list_costs = [record_cost(r, price) for r in scored_runs]
        list_total = (
            sum(list_costs)
            if list_costs and all(c is not None for c in list_costs)
            else None
        )
        outs = [r["tokens_out"] for r in scored_runs if r.get("tokens_out") is not None]
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
                # Per list dollar: a billed figure is zero or unknown for
                # nearly every route here, so it cannot rank value.
                value=value_score(comp, cost_per_1k_chars(list_total, total_chars)),
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
                judge_composites=judge_composites,
                list_cost_per_chapter=(
                    list_total / len(scored_runs) if list_total is not None else None
                ),
                tokens_out_per_chapter=sum(outs) / len(outs) if outs else None,
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

    # A panel shows each judge's own composite beside the panel's.
    judges = list(next((r.judge_composites for r in ranked if r.judge_composites), {}))
    judge_header = "".join(f" {j} |" for j in judges)
    lines += [
        "## Ranking",
        "",
        "| # | Model | Composite |" + judge_header
        + " Scored | List $ /ch | Out tok /ch | Refusal % | Billed /1k ch | Sec /1k ch"
        " | Value | Slop | Repetition |",
        "|---" * (12 + len(judges)) + "|",
    ]
    for i, r in enumerate(ranked, start=1):
        label = r.model_id + (f" _{r.anchor} anchor_" if r.anchor else "")
        # Shown because a model missing chapters is averaged over fewer, and
        # the chapters that break models are the hard ones - so a composite
        # over 8/11 is not the same measurement as one over 11/11.
        scored = (
            f"{r.total_tasks - r.failed_tasks - r.refused_tasks}/{r.total_tasks}"
        )
        judge_cells = "".join(f" {fmt(r.judge_composites.get(j))} |" for j in judges)
        lines.append(
            f"| {i} | {label} | {fmt(r.composite)} |{judge_cells} {scored} | "
            f"{fmt_list_cost(r.list_cost_per_chapter)} | {fmt_tokens(r.tokens_out_per_chapter)} | "
            f"{fmt(r.refusal_rate)} | "
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
        "- **List $ /ch** is what one chapter costs at the provider's list "
        "price: the tokens each call recorded, priced from the roster (OpenRouter "
        "list), or the route's own list-price figure where it reports one. It "
        "compares every model on one basis, whatever route ran it. Reasoning is "
        "billed as output, so **Out tok /ch** includes it. CLI routes carry "
        "their harness's own input overhead. **Value** is composite points per "
        "list dollar per 1,000 source characters.",
        "- **Billed `sub`** means a subscription covered the call, so it billed "
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
