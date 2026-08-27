"""Judge stage: score every run output, skipping what cannot be scored."""

from __future__ import annotations

import json
from pathlib import Path

from ati.corpus import HashMismatch, Passage, PassageMissing, load_passage
from ati.judge import (
    JudgeSchemaError,
    build_judge_prompt,
    judge_with_retry,
    rationale_quote_violations,
)


def judgment_output_path(
    judgments_root: Path, run_id: str, model_id: str, instance_id: str
) -> Path:
    safe_model = model_id.replace("/", "__")
    return Path(judgments_root) / run_id / safe_model / f"{instance_id}.json"


def _judgment_is_current(path: Path, run_data: dict) -> bool:
    """Whether an existing judgment still describes the run beside it.

    A judgment recorded as skipped because the translation had failed becomes
    stale the moment that translation succeeds on a later run. Skipping on
    mere file existence would leave the repaired work permanently unjudged,
    scoring the model on fewer tasks than its peers with no indication.
    """
    try:
        judgment = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return False
    run_unusable = bool(
        run_data.get("error") or run_data.get("unavailable") or run_data.get("refused")
    )
    if judgment.get("skipped") and not run_unusable:
        return False
    return True


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    for run_file in sorted(run_dir.rglob("*.json")):
        run_data = json.loads(run_file.read_text(encoding="utf-8"))
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
        except JudgeSchemaError as exc:
            _write(out_path, {**base, "judge_error": str(exc)})
            judged += 1
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
    return judged
