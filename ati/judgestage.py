"""Judge stage: score every run output, skipping what cannot be scored."""

from __future__ import annotations

import hashlib
from pathlib import Path

from ati.corpus import HashMismatch, Passage, PassageMissing, load_passage
from ati.fileio import iter_records, read_json, write_json_atomic
from ati.judge import (
    JudgeRunnerError,
    JudgeSchemaError,
    build_judge_prompt,
    judge_with_retry,
    rationale_quote_violations,
)
from ati.runstage import is_usage_limit

# Consecutive items the judge could not be reached on before the stage stops.
# One is a blip; two in a row is the judge being down or out of quota, and
# every remaining item would fail the same way.
MAX_CONSECUTIVE_RUNNER_FAILURES = 2
# Consecutive items whose every attempt came back in a shape that could not be
# parsed. Each such item has already cost `attempts` judge calls, so a judge
# that has started answering in the wrong shape would otherwise spend three
# requests on every remaining item - for the Antigravity judge, against the
# operator's own account - while producing nothing that can be scored.
MAX_CONSECUTIVE_SCHEMA_FAILURES = 3


def judgment_output_path(
    judgments_root: Path, run_id: str, model_id: str, instance_id: str
) -> Path:
    safe_model = model_id.replace("/", "__")
    return Path(judgments_root) / run_id / safe_model / f"{instance_id}.json"


def _translation_sha(run_data: dict) -> str:
    """Fingerprint of the translation a judgment was made against."""
    text = run_data.get("text") or ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _judgment_is_current(path: Path, run_data: dict) -> bool:
    """Whether an existing judgment still describes the run beside it.

    Staleness runs in both directions and both matter.

    A judgment recorded as skipped because the translation had failed becomes
    stale the moment that translation succeeds on a later run. Skipping on
    mere file existence would leave the repaired work permanently unjudged,
    scoring the model on fewer tasks than its peers with no indication.

    A scored judgment goes stale the same way when its run is later found
    unusable - reclassified as a runaway or a stub, say. Treating it as
    current would keep rubric scores alive for output that is no longer
    considered a translation, and no amount of re-judging would clear them.
    """
    judgment = read_json(path)
    if judgment is None:
        return False
    # A refusal is recorded as a non-skipped judgment with refused=True, so it
    # is compared on that flag. Folding it into "skipped" made every refusal
    # look stale forever: it was rewritten on every pass, and a driver waiting
    # for the stage to report no work never saw it finish.
    run_failed = bool(run_data.get("error") or run_data.get("unavailable"))
    if bool(judgment.get("skipped")) != run_failed:
        return False
    if bool(judgment.get("refused")) != bool(run_data.get("refused")):
        return False
    # A judge error is a failure to judge, not a judgment. Treating it as
    # current would leave the item unscored for good after one bad response.
    if judgment.get("judge_error"):
        return False
    # A judgment describes one specific translation. Re-running a model - with
    # --force, or through a different provider - replaces the text while
    # leaving the judgment scored and non-skipped, so the check above sees
    # nothing wrong and the old scores are kept against new prose. gpt-5.6-luna
    # and muse-spark were both re-run through a different provider; without
    # this their opencode-route scores would have been reported as Command
    # Code results. A judgment with no recorded hash predates this check and
    # cannot be verified, so it is re-judged once.
    recorded = judgment.get("translation_sha")
    if recorded is None:
        return False
    return recorded == _translation_sha(run_data)


def _write(path: Path, payload: dict) -> None:
    write_json_atomic(path, payload)


# Keys a run record must carry to be judged at all. A file without them is
# not one the run stage wrote - a stray copy, or damage - and is skipped with
# a warning instead of raising out of the stage.
RUN_RECORD_KEYS = (
    "model_id", "task_id", "task_type", "passage_id", "iteration", "text"
)


def judge_run(
    judge_runner,
    judge_id: str,
    run_id: str,
    runs_root: Path,
    judgments_root: Path,
    passages: dict[str, Passage],
    corpus_root: Path,
    template: str,
    force: bool = False,
) -> int:
    run_dir = Path(runs_root) / run_id
    judged = 0
    runner_failures = 0
    schema_failures = 0
    for run_file, run_data in iter_records(
        sorted(run_dir.rglob("*.json")), required=RUN_RECORD_KEYS
    ):
        model_id = run_data["model_id"]
        instance_id = run_file.stem

        out_path = judgment_output_path(
            judgments_root, run_id, model_id, instance_id
        )
        if out_path.exists() and not force and _judgment_is_current(out_path, run_data):
            continue

        base = {
            "model_id": model_id,
            "task_id": run_data["task_id"],
            "task_type": run_data["task_type"],
            "passage_id": run_data["passage_id"],
            "iteration": run_data["iteration"],
            "judge_id": judge_id,
            "scores": {},
            "rationales": {},
            "term_table": None,
            "refused": bool(run_data.get("refused")),
            "skipped": False,
            "judge_error": None,
            "quote_violations": [],
            "translation_sha": _translation_sha(run_data),
        }

        # Nothing to judge: the run failed, the passage was unavailable, or
        # the model refused. Each is recorded, none is scored.
        if run_data.get("unavailable") or run_data.get("error"):
            _write(out_path, {**base, "skipped": True})
            judged += 1
            continue
        if run_data.get("refused"):
            _write(out_path, base)
            judged += 1
            continue

        passage = passages.get(run_data["passage_id"])
        if passage is None:
            _write(out_path, {**base, "skipped": True})
            judged += 1
            continue
        try:
            source_text = load_passage(corpus_root, passage)
        except (HashMismatch, PassageMissing):
            _write(out_path, {**base, "skipped": True})
            judged += 1
            continue

        # A published reference, where one exists, calibrates the judge's sense
        # of a good result. The manifest has always carried has_reference; it
        # was never read, so the calibration described in the methodology was
        # not happening.
        reference = None
        if passage.has_reference:
            reference_file = (
                Path(corpus_root) / "references" / f"{passage.id}.txt"
            )
            if reference_file.exists():
                reference = reference_file.read_text(encoding="utf-8")

        expect_term_table = run_data["task_type"] == "consistency"
        prompt = build_judge_prompt(
            template=template,
            source_text=source_text,
            translation=run_data["text"],
            reference=reference,
            expect_term_table=expect_term_table,
        )

        try:
            judgment = judge_with_retry(
                judge_runner, prompt, expect_term_table=expect_term_table
            )
        except JudgeRunnerError as exc:
            # Nothing is written, so a rerun picks this item up again.
            runner_failures += 1
            if (
                is_usage_limit(str(exc))
                or runner_failures >= MAX_CONSECUTIVE_RUNNER_FAILURES
            ):
                print(
                    f"stopping after {judged} judgments: {exc}. Rerun the same "
                    f"command later; completed judgments are kept."
                )
                return judged
            print(f"  skipping {model_id} {instance_id}: {exc}")
            continue
        except JudgeSchemaError as exc:
            _write(out_path, {**base, "judge_error": str(exc)})
            judged += 1
            runner_failures = 0  # it answered, so it was reachable
            schema_failures += 1
            if schema_failures >= MAX_CONSECUTIVE_SCHEMA_FAILURES:
                print(
                    f"stopping after {judged} judgments: {schema_failures} "
                    f"items in a row got no parseable answer from the judge "
                    f"(last: {exc}). Check the judge before rerunning."
                )
                return judged
            continue

        _write(
            out_path,
            {
                **base,
                "scores": judgment.axes,
                "rationales": judgment.rationales,
                "term_table": judgment.term_table,
                "quote_violations": rationale_quote_violations(judgment),
            },
        )
        judged += 1
        runner_failures = 0
        schema_failures = 0
    return judged
