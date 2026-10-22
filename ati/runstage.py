"""Run stage: translate every task instance with one model.

Idempotent by construction - an instance whose output file already exists is
skipped, so adding a model to an existing run costs only that model's calls.
"""

from __future__ import annotations

from pathlib import Path

from ati.corpus import HashMismatch, Passage, PassageMissing, load_passage
from ati.fileio import iter_records, read_json, write_json_atomic
from ati.runners.api import is_transient_429
from ati.runners.base import RunResult
from ati.tasks import TaskInstance


def run_output_path(
    runs_root: Path, run_id: str, model_id: str, instance_id: str
) -> Path:
    safe_model = model_id.replace("/", "__")
    return Path(runs_root) / run_id / safe_model / f"{instance_id}.json"


USAGE_LIMIT_MARKERS = (
    "usage limit",
    "quota",
    "429",
    "rate limit",
    # Google's spelling, which is how the Antigravity judge reports a limit.
    "resource_exhausted",
    "resource exhausted",
    "too many requests",
    # OpenRouter's answer (HTTP 402) when prepaid credit runs out. Every
    # remaining call would get it too, so it is a stop, not a task failure.
    "insufficient credit",
    # The Claude Code CLI's spelling: "You've hit your session limit · resets
    # 12am (America/New_York)". Without this it matched no marker, so run
    # 2026-09 spent all three passes retrying it in seconds and recorded five
    # sonnet-5 and ten haiku-4.5 tasks as ordinary failures.
    "session limit",
)

# Wording that names a spent allowance outright. These win over a transient
# marker: `codex exec` reports "You've hit your usage limit ... Try again in 3
# hours", and "try again" alone is how a busy upstream asks for a retry.
# Without this the Codex rows' limit was read as a passing failure and the
# remaining passes spent in seconds.
EXPLICIT_LIMIT_MARKERS = (
    "usage limit",
    "session limit",
    "quota",
    "resource_exhausted",
    "resource exhausted",
    "insufficient credit",
)

# A translation is bounded in length by its source. The api runner already
# refuses to score generation that hit its token ceiling; runners that expose
# no such ceiling - the opencode CLI among them - need the same protection
# measured on the output itself, or a degenerate run is scored as a real
# translation. gpt-5.6-luna returned 141,842 characters for a 6,987-character
# chapter, a 20x expansion, while every honest Korean-to-English chapter in
# this corpus landed between 1.85x and 2.28x. 4.0 sits at nearly double the
# widest legitimate ratio observed, so it catches runaway looping without
# touching a verbose-but-real translation.
MAX_OUTPUT_EXPANSION = 4.0
# A ratio means nothing on a short source: a handful of characters can expand
# many times over for entirely ordinary reasons - a two-character title, a
# single interjection rendered as a phrase. Below this the check is off, which
# also keeps it clear of the 800-character `passage` tasks' noisier end.
MIN_SOURCE_CHARS_FOR_EXPANSION = 200


# The mirror of the expansion cap. Korean-to-English ran 1.85x-2.28x across
# every model and chapter here, so output shorter than half its source is not
# a translation of it. grok-4.6 returned 152 characters for an 8,705-character
# chapter - the agent announcing it would load a skill, then stopping when the
# tool was denied - and it was scored on the rubric as though it were prose.
# A genuine refusal is exempt: on the refusal task type a short answer is the
# measurement, and `refused` already records it.
MIN_OUTPUT_RATIO = 0.5


# Messages a provider's own moderation layer puts in place of - or on the end
# of - a translation. Each was observed on run 2026-09, on chapters with
# bandits and killing in them:
#   Xiaomi (mimo-v2.5, mimo-v2.5-pro): the translation stops mid-sentence and
#     "The request was rejected because it was considered high risk" follows.
#   Alibaba (qwen3.8-flash): HTTP 400 "InternalError.Algo.DataInspectionFailed".
#   Zhipu (glm-5.2): the stream ends with an error object, "Output data may
#     contain inappropriate content" (its input check says "Input data ...").
# For someone translating the work this is a refusal - the route will not
# translate the chapter - so it is recorded as one rather than scored as a
# translation (the fragment before the cut is not one) or retried as though
# the network had failed.
PROVIDER_BLOCK_MARKERS = (
    "the request was rejected because it was considered high risk",
    "datainspectionfailed",
    "data may contain inappropriate content",
)


def provider_blocked(text: str, error: str | None) -> bool:
    """Whether a provider's moderation layer stopped this translation."""
    haystack = f"{text or ''} {error or ''}".lower()
    return any(marker in haystack for marker in PROVIDER_BLOCK_MARKERS)


# Output written mostly in the source's script is not an English translation.
# solar-pro4 returned chapter 4 of 화산귀환 in Korean - the source echoed back
# at a length ratio of exactly 1.00, which clears every length guard.
MAX_SOURCE_SCRIPT_SHARE = 0.5


def _is_source_script(ch: str) -> bool:
    code = ord(ch)
    return (
        0xAC00 <= code <= 0xD7A3  # Hangul syllables
        or 0x1100 <= code <= 0x11FF  # Hangul jamo
        or 0x3130 <= code <= 0x318F  # Hangul compatibility jamo
        or 0x3040 <= code <= 0x30FF  # Hiragana, Katakana
        or 0x4E00 <= code <= 0x9FFF  # CJK unified ideographs
    )


def left_untranslated(text: str) -> bool:
    """Whether most of the letters in the output are in the source's script."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    share = sum(1 for ch in letters if _is_source_script(ch)) / len(letters)
    return share > MAX_SOURCE_SCRIPT_SHARE


def came_up_short(text: str, source_chars: int, refused: bool = False) -> bool:
    """Whether output is too short to be a translation of this source."""
    if refused or source_chars < MIN_SOURCE_CHARS_FOR_EXPANSION:
        return False
    return len(text) / source_chars < MIN_OUTPUT_RATIO


def ran_away(text: str, source_chars: int) -> bool:
    """Whether output is too long to be a translation of this source."""
    if not text or source_chars < MIN_SOURCE_CHARS_FOR_EXPANSION:
        return False
    return len(text) / source_chars > MAX_OUTPUT_EXPANSION


def is_usage_limit(error: str | None) -> bool:
    """Whether a failure means the provider cut us off rather than misbehaved.

    A true usage limit stops the whole run, because every remaining call would
    hit it too. That makes a false positive expensive: it discards every model
    still queued. Command Code returns 429 for "Upstream model provider is
    temporarily unavailable" as well as for quota, and matching the bare string
    "429" cannot tell those apart - hy4-preview alone would have ended a
    twenty-model run at position thirteen. A transient body wins over the
    status code, but not over wording that names a spent allowance.
    """
    if not error:
        return False
    lowered = error.lower()
    if any(marker in lowered for marker in EXPLICIT_LIMIT_MARKERS):
        return True
    if is_transient_429(error):
        return False
    return any(marker in lowered for marker in USAGE_LIMIT_MARKERS)


def _is_complete(path: Path) -> bool:
    """Whether an existing run file represents finished work.

    A recorded error is NOT finished work. Checkpointing failures as if they
    were complete would make a rerun skip them, so a run interrupted by a
    usage limit could never be resumed - it would silently produce nothing.
    A refusal IS finished work: it is the measurement, and is not retried.
    """
    data = read_json(path)
    if data is None:
        return False
    # An unavailable passage is not finished work either. Re-checking costs
    # nothing - no model is called when the hash still mismatches - and
    # treating it as done would permanently exclude the passage from the run
    # once the corpus is repaired, silently scoring that model on fewer tasks
    # than its peers.
    return not data.get("error") and not data.get("unavailable")


def _write(path: Path, payload: dict) -> None:
    write_json_atomic(path, payload)


def _call(runner, system_prompt: str, source_text: str) -> RunResult:
    """One runner call, with any exception recorded as a failed task.

    A runner that raises - an unexpected response shape, a decoding error -
    used to propagate out of run_model and end the run for every model still
    queued behind it. Recorded as an error instead, the task is retried on
    the next pass like any other failure, and the run carries on.
    """
    try:
        return runner.run(system_prompt, source_text)
    except Exception as exc:  # noqa: BLE001 - containment is the point
        return RunResult(
            text="", latency_s=0.0, error=f"runner raised {type(exc).__name__}: {exc}"
        )


def completed_count(runs_root, run_id, model_ids, instance_ids) -> int:
    """How many of these tasks are finished work, by the same rule as a rerun.

    `run_model` returns how many tasks it *attempted*, which is what the pass
    loop needs to decide whether anything is left to retry. `drive` reads that
    number as progress, and an attempt that failed on a usage limit still
    counts as attempted - so a stage whose every attempt fails resets drive's
    give-up counter on each pass and MAX_STOPS never fires. Run 2026-09
    retried an empty credit balance hourly for nine hours that way. Counting
    finished work instead makes "no progress" mean what drive thinks it means.
    """
    return sum(
        1
        for model_id in model_ids
        for instance_id in instance_ids
        if _is_complete(run_output_path(runs_root, run_id, model_id, instance_id))
    )


def run_model(
    runner,
    model_id: str,
    instances: list[TaskInstance],
    passages: dict[str, Passage],
    corpus_root: Path,
    system_prompt: str,
    runs_root: Path,
    run_id: str,
    force: bool = False,
    on_usage_limit=None,
) -> int:
    """Translate every instance not already complete. Returns how many ran.

    `on_usage_limit`, if given, is called with the provider's message when a
    usage limit stops the model, so a caller can hold back other models on
    the same route instead of spending one refused call on each.
    """
    executed = 0
    for instance in instances:
        out_path = run_output_path(
            runs_root, run_id, model_id, instance.instance_id
        )
        if out_path.exists() and not force and _is_complete(out_path):
            continue

        base = {
            "model_id": model_id,
            "task_id": instance.task.id,
            "task_type": instance.task.type,
            "passage_id": instance.passage_id,
            "iteration": instance.iteration,
            "unavailable": False,
        }

        passage = passages.get(instance.passage_id)
        if passage is None:
            _write(
                out_path,
                {
                    **base,
                    "text": "",
                    "source_chars": 0,
                    "latency_s": 0.0,
                    "tokens_in": None,
                    "tokens_out": None,
                    "cost_usd": None,
                    "refused": False,
                    "unavailable": True,
                    "error": f"passage {instance.passage_id} not in manifest",
                },
            )
            executed += 1
            continue

        try:
            source_text = load_passage(corpus_root, passage)
        except (HashMismatch, PassageMissing) as exc:
            _write(
                out_path,
                {
                    **base,
                    "text": "",
                    "source_chars": 0,
                    "latency_s": 0.0,
                    "tokens_in": None,
                    "tokens_out": None,
                    "cost_usd": None,
                    "refused": False,
                    "unavailable": True,
                    "error": str(exc),
                },
            )
            executed += 1
            continue

        # Manual runner needs the instance id to pair prompt and response files.
        set_task_id = getattr(runner, "set_task_id", None)
        if callable(set_task_id):
            set_task_id(instance.instance_id)

        result = _call(runner, system_prompt, source_text)
        text, error, refused, blocked = apply_guards(
            result.text,
            result.error,
            result.refused,
            result.finish_reason,
            len(source_text),
        )
        _write(
            out_path,
            {
                **base,
                "text": text,
                "source_chars": len(source_text),
                "latency_s": result.latency_s,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "reasoning_tokens": result.reasoning_tokens,
                "route": result.route,
                "finish_reason": result.finish_reason,
                "cost_usd": result.cost_usd,
                "refused": refused,
                "blocked_by_provider": blocked,
                "error": error,
            },
        )
        executed += 1

        # A usage limit is not a per-task failure - every remaining call will
        # hit it too. Stop now so the run can be resumed when the quota
        # resets, rather than marking the whole suite failed.
        if is_usage_limit(result.error):
            print(
                f"{model_id}: stopping - provider usage limit reached. "
                f"Rerun the same command once it resets; completed work is kept."
            )
            if on_usage_limit is not None:
                on_usage_limit(result.error)
            return executed
    return executed


def apply_guards(
    text: str,
    error: str | None,
    refused: bool,
    finish_reason: str | None,
    source_chars: int,
) -> tuple[str, str | None, bool, bool]:
    """Classify one output: (text, error, refused, blocked_by_provider).

    The single place the output guards live, used by the run stage on every
    new output and by `recheck` on outputs already on disk, so a record
    written before a guard existed is judged by exactly the rules a new one
    is. The guards only ever take text away; none can restore any.
    """
    blocked = provider_blocked(text, error) or finish_reason == "content_filter"
    if blocked:
        error, text, refused = None, "", True
    if not error and text and left_untranslated(text):
        error = "output is mostly in the source script; not a translation"
        text = ""
    # Recorded as a failure, not scored: judging a degenerate loop would
    # hand the model rubric scores for text that is not a translation.
    # The text is dropped for the same reason the api runner drops
    # truncated output - it cannot be partially credited.
    if not error and came_up_short(text, source_chars, refused):
        error = (
            f"output came up short ({len(text)} characters from "
            f"{source_chars} of source, "
            f"{len(text) / source_chars:.2f}x); not a translation"
        )
        text = ""
    if not error and ran_away(text, source_chars):
        error = (
            f"output ran away ({len(text)} characters from "
            f"{source_chars} of source, "
            f"{len(text) / source_chars:.1f}x); not a translation"
        )
        text = ""
    return text, error, bool(refused), bool(blocked)


def recheck(run_id: str, runs_root: Path, apply: bool = False) -> list[tuple[Path, str]]:
    """Re-apply the current output guards to every run record on disk.

    For records written before a guard existed - on run 2026-09 that took a
    one-off script (scripts/2026-09/reclassify.py) when the provider-block
    and echo guards were added mid-run. Returns (path, what changed) for each
    record the guards now classify differently, and rewrites them only when
    `apply` is set. Makes no model calls; the judge stage then sees the
    changed records as stale and re-judges or skips them.
    """
    changes = []
    for path, record in iter_records(
        sorted((Path(runs_root) / run_id).rglob("*.json")),
        required=("model_id", "passage_id", "text", "source_chars"),
    ):
        # A provider block has already had its evidence - the text or the
        # error that carried the marker - discarded, so the guards could not
        # re-derive it and would un-classify it. A recheck only ever tightens.
        if record.get("unavailable") or record.get("blocked_by_provider"):
            continue
        text, error, refused, blocked = apply_guards(
            record.get("text") or "",
            record.get("error"),
            bool(record.get("refused")),
            record.get("finish_reason"),
            int(record.get("source_chars") or 0),
        )
        refused = refused or bool(record.get("refused"))
        before = (
            record.get("text") or "",
            record.get("error"),
            bool(record.get("refused")),
            bool(record.get("blocked_by_provider")),
        )
        if (text, error, refused, blocked) == before:
            continue
        if blocked and not before[3]:
            what = "refused (provider block)"
        else:
            what = f"failed ({error})" if error else "changed"
        changes.append((path, what))
        if apply:
            _write(
                path,
                {
                    **record,
                    "text": text,
                    "error": error,
                    "refused": refused,
                    "blocked_by_provider": blocked,
                },
            )
    return changes
