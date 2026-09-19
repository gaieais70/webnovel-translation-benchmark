"""Run stage: translate every task instance with one model.

Idempotent by construction - an instance whose output file already exists is
skipped, so adding a model to an existing run costs only that model's calls.
"""

from __future__ import annotations

from pathlib import Path

from ati.corpus import HashMismatch, Passage, PassageMissing, load_passage
from ati.fileio import read_json, write_json_atomic
from ati.runners.api import is_transient_429
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
    status code.
    """
    if not error:
        return False
    if is_transient_429(error):
        return False
    lowered = error.lower()
    return any(marker in lowered for marker in USAGE_LIMIT_MARKERS)


def _is_complete(path: Path) -> bool:
    """Whether an existing run file represents finished work.

    A recorded error is NOT finished work. Checkpointing failures as if they
    were complete would make a rerun skip them, so a run interrupted by a
    usage limit could never be resumed - it would silently produce nothing.
    A passage marked unavailable is different: the corpus itself is wrong, and
    retrying cannot help until that is fixed.
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
) -> int:
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

        result = runner.run(system_prompt, source_text)
        error = result.error
        text = result.text
        refused = result.refused
        blocked = (
            provider_blocked(text, error) or result.finish_reason == "content_filter"
        )
        if blocked:
            error, text, refused = None, "", True
        if not error and text and left_untranslated(text):
            error = "output is mostly in the source script; not a translation"
            text = ""
        # Recorded as a failure, not scored: judging a degenerate loop would
        # hand the model rubric scores for text that is not a translation.
        # The text is dropped for the same reason the api runner drops
        # truncated output - it cannot be partially credited.
        if not error and came_up_short(text, len(source_text), refused):
            error = (
                f"output came up short ({len(text)} characters from "
                f"{len(source_text)} of source, "
                f"{len(text) / len(source_text):.2f}x); not a translation"
            )
            text = ""
        if not error and ran_away(text, len(source_text)):
            error = (
                f"output ran away ({len(text)} characters from "
                f"{len(source_text)} of source, "
                f"{len(text) / len(source_text):.1f}x); not a translation"
            )
            text = ""
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
            return executed
    return executed
