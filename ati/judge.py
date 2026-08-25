"""Judge prompt construction and strict parsing of judge responses.

Judge output is validated against a schema rather than trusted: a judge that
returns prose, omits an axis, or invents a score outside 1-5 is a retryable
failure, not a silent zero.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ati.config import AXES

MAX_QUOTED_WORDS = 8

_QUOTE_RE = re.compile(r'"([^"]+)"')


class JudgeSchemaError(Exception):
    """The judge's response did not conform to the required schema."""


@dataclass(frozen=True)
class Judgment:
    axes: dict[str, int]
    rationales: dict[str, str]
    term_table: dict[str, str] | None = None


def _extract_json_object(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise JudgeSchemaError("no JSON object found in judge response")
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeSchemaError(f"no JSON object could be parsed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise JudgeSchemaError("no JSON object found in judge response")
    return parsed


def parse_judgment(raw: str, expect_term_table: bool = False) -> Judgment:
    parsed = _extract_json_object(raw)

    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        raise JudgeSchemaError("judge response has no 'scores' object")

    axes: dict[str, int] = {}
    for axis in AXES:
        if axis not in scores:
            raise JudgeSchemaError(f"missing score for {axis}")
        try:
            value = int(scores[axis])
        except (TypeError, ValueError) as exc:
            raise JudgeSchemaError(f"score for {axis} is not an integer") from exc
        if not 1 <= value <= 5:
            raise JudgeSchemaError(f"score for {axis} is out of range: {value}")
        axes[axis] = value

    raw_rationales = parsed.get("rationales") or {}
    rationales = {axis: str(raw_rationales.get(axis, "")) for axis in AXES}

    term_table = parsed.get("term_table")
    if expect_term_table:
        if not isinstance(term_table, dict) or not term_table:
            raise JudgeSchemaError("expected a non-empty 'term_table' object")
        term_table = {str(k): str(v) for k, v in term_table.items()}
    else:
        term_table = None

    return Judgment(axes=axes, rationales=rationales, term_table=term_table)


def build_judge_prompt(
    template: str,
    source_text: str,
    translation: str,
    reference: str | None,
    expect_term_table: bool,
) -> str:
    parts = [template, "\n\n## SOURCE TEXT\n\n", source_text]
    if reference:
        parts += ["\n\n## REFERENCE TRANSLATION (calibration only, not ground truth)\n\n", reference]
    parts += ["\n\n## TRANSLATION UNDER REVIEW\n\n", translation, "\n\n## REQUIRED OUTPUT\n\n"]

    schema = {
        "scores": {axis: "<integer 1-5>" for axis in AXES},
        "rationales": {axis: "<one short sentence>" for axis in AXES},
    }
    if expect_term_table:
        schema["term_table"] = {
            "<source term>": "<how this translation rendered it>"
        }
        parts.append(
            "Also extract every proper name, title, honorific, and "
            "setting-specific term from the source, mapping it to exactly how "
            "this translation rendered it, as 'term_table'.\n\n"
        )
    parts.append(
        "Reply with a single JSON object and nothing else, matching:\n\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + f"\n\nDo not quote more than {MAX_QUOTED_WORDS} consecutive words "
        "of either text in your rationales."
    )
    return "".join(parts)


def judge_with_retry(
    judge_runner, prompt: str, expect_term_table: bool, attempts: int = 3
) -> Judgment:
    last_error: JudgeSchemaError | None = None
    for _ in range(attempts):
        result = judge_runner.run("", prompt)
        if result.error:
            last_error = JudgeSchemaError(f"judge runner error: {result.error}")
            continue
        try:
            return parse_judgment(result.text, expect_term_table=expect_term_table)
        except JudgeSchemaError as exc:
            last_error = exc
    raise last_error or JudgeSchemaError("judge produced no usable response")


def rationale_quote_violations(judgment: Judgment) -> list[str]:
    """Return axes whose rationale quotes more than MAX_QUOTED_WORDS words.

    Enforces the constraint that published output never reproduces source or
    translation text at length.
    """
    offenders = []
    for axis, text in judgment.rationales.items():
        for quoted in _QUOTE_RE.findall(text):
            if len(quoted.split()) > MAX_QUOTED_WORDS:
                offenders.append(axis)
                break
    return offenders
