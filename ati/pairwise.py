"""Pairwise comparison stage.

Ratings come from a Bradley-Terry fit rather than Elo or Glicko; see
`ati/bradleyterry.py` for why. Three controls guard the comparison itself:

- Every pair is judged in **both orders**. A judge that flips its answer when
  the order flips has no real preference, so that is recorded as a tie rather
  than averaged into noise.
- Translations are **not truncated**. Truncation is an EQ-Bench control for
  free-form creative writing, where models pad; translations of one source
  passage are naturally similar in length, and cutting them would hide exactly
  the omissions the fidelity axis exists to catch.
- Pairs are chosen **adaptively**, closest-rated and least-compared first, so
  a run interrupted by a usage limit still yields a meaningful ranking.
- Both passage and consistency outputs are comparable. Whole chapters cost the
  judge more per comparison than short excerpts, so budget accordingly.

Every comparison is written to disk the moment it resolves, so an interrupted
run resumes having lost at most the comparison in flight.
"""

from __future__ import annotations

import json
from pathlib import Path

from ati.bradleyterry import OUTCOMES, win_probability
from ati.corpus import HashMismatch, PassageMissing, load_passage, sha256_text
from ati.fileio import iter_records, read_json, write_json_atomic

# One comparison sends the source plus both translations. Whole chapters run
# to ~40,000 characters per prompt, and each pair is judged twice for order,
# so an unguarded run drains a judge subscription far faster than the short
# excerpts this stage was designed around.
MAX_COMPARE_CHARS = 60000

# Comparisons per invocation unless --max-comparisons says otherwise. One
# number, used by both the stage and the CLI; they used to disagree (100 here,
# 50 in the CLI, 200 in the README).
DEFAULT_MAX_COMPARISONS = 50

# Comparisons in a row whose verdict could not be parsed before the stage
# stops. One is a glitch and is skipped. Several in a row is a judge that has
# stopped answering in the expected shape, and a skipped comparison does not
# count against the batch budget - so without this the stage would walk the
# whole schedule, one wasted judge request per pair and passage.
MAX_CONSECUTIVE_BAD_VERDICTS = 3


def comparison_path(root: Path, run_id: str, a: str, b: str, passage_id: str) -> Path:
    first, second = sorted([a, b])
    safe = f"{first}__{second}__{passage_id}".replace("/", "--")
    return Path(root) / run_id / f"{safe}.json"


def parse_verdict(raw: str) -> str:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in verdict")
    winner = str(json.loads(raw[start : end + 1]).get("winner", "")).strip()
    if winner.upper() in ("A", "B"):
        return winner.upper()
    if winner.lower() == "tie":
        return "tie"
    raise ValueError(f"unknown verdict {winner!r}")


def resolve_outcome(forward: str, reverse: str) -> str:
    """Combine both orderings into one outcome.

    In the reverse call the models are swapped, so "B" there means the first
    model won. Agreement yields a winner; anything else is a tie.
    """
    if forward == "A" and reverse == "B":
        return "a"
    if forward == "B" and reverse == "A":
        return "b"
    return "tie"


def build_pair_prompt(template: str, source: str, text_a: str, text_b: str) -> str:
    return (
        f"{template}\n\n## SOURCE\n\n{source}\n\n"
        f"## TRANSLATION A\n\n{text_a}\n\n"
        f"## TRANSLATION B\n\n{text_b}\n\n"
        '## OUTPUT\n\nReply with a single JSON object: '
        '{"winner": "A" | "B" | "tie", "reason": "<one short sentence>"}. '
        'Answer "tie" when neither is clearly better.'
    )


def _ask(judge_runner, prompt: str):
    """One judge call. Any failure to get an answer is a RuntimeError.

    A runner that raises is treated like one that reports an error: the
    stage stops, rather than the exception ending it with a traceback or
    the next comparison repeating the same defect against the judge.
    """
    try:
        result = judge_runner.run("", prompt)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"judge runner raised {type(exc).__name__}: {exc}") from exc
    if result.error:
        raise RuntimeError(result.error)
    return result


def compare(judge_runner, template, source, text_a, text_b):
    """Judge one pair in both orders. Returns (forward, reverse, outcome)."""
    forward_result = _ask(
        judge_runner, build_pair_prompt(template, source, text_a, text_b)
    )
    forward = parse_verdict(forward_result.text)

    reverse_result = _ask(
        judge_runner, build_pair_prompt(template, source, text_b, text_a)
    )
    reverse = parse_verdict(reverse_result.text)

    return forward, reverse, resolve_outcome(forward, reverse)


def select_pairs(models, ratings, counts, k):
    """Rank candidate pairs by expected information per judge call.

    p*(1-p) peaks where the models are evenly matched, which is where a
    comparison tells us most; dividing by prior comparisons spreads coverage.
    """
    candidates = []
    ordered = sorted(models)
    for i in range(len(ordered)):
        for j in range(i + 1, len(ordered)):
            a, b = ordered[i], ordered[j]
            done = counts.get((a, b), 0)
            p = win_probability(ratings.get(a, 1000.0), ratings.get(b, 1000.0))
            candidates.append((p * (1.0 - p) / (1.0 + done), a, b))
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [(a, b) for _, a, b in candidates[:k]]


def _load_outputs(runs_root: Path, run_id: str):
    """model_id -> passage_id -> translation text, for scorable passage tasks."""
    outputs: dict[str, dict[str, str]] = {}
    for _, data in iter_records(
        sorted((Path(runs_root) / run_id).rglob("*.json")),
        required=("model_id", "passage_id"),
    ):
        if data.get("task_type") not in ("passage", "consistency"):
            continue
        if data.get("error") or data.get("unavailable") or data.get("refused"):
            continue
        if not data.get("text"):
            continue
        outputs.setdefault(data["model_id"], {})[data["passage_id"]] = data["text"]
    return outputs


def load_comparisons(pairwise_root: Path, run_id: str):
    """Read completed comparisons back as (a, b, outcome) triples."""
    results = []
    for path, data in iter_records(
        sorted((Path(pairwise_root) / run_id).glob("*.json")),
        required=("a", "b", "outcome"),
    ):
        # An unknown outcome would make the Bradley-Terry fit raise, taking
        # the report down with it over one bad file.
        if data["outcome"] not in OUTCOMES:
            print(f"  skipping {path}: unknown outcome {data['outcome']!r}")
            continue
        results.append((data["a"], data["b"], data["outcome"]))
    return results


def _comparison_is_current(
    path: Path,
    judge_id: str | None,
    judge_fingerprint: str | None,
    sha_a: str,
    sha_b: str,
) -> bool:
    """Whether a recorded comparison still stands for this judge and these texts.

    The pairwise twin of judgestage._judgment_is_current. A field the record
    does not carry is not held against it: every comparison of run 2026-09
    predates these fields, and re-judging them would spend the operator's
    Antigravity quota to reproduce what is already on disk.
    """
    data = read_json(path)
    if data is None:
        return False  # damaged: redo it rather than keep a verdict nobody can read
    checks = (
        ("judge_id", judge_id),
        ("judge_fingerprint", judge_fingerprint),
        ("sha_a", sha_a),
        ("sha_b", sha_b),
    )
    for key, expected in checks:
        recorded = data.get(key)
        if expected and recorded and recorded != expected:
            return False
    return True


def pending_comparisons(pairwise_root: Path, run_id: str, runs_root: Path,
                        judge_id: str | None = None,
                        judge_fingerprint: str | None = None) -> list[Path]:
    """Recorded comparisons the stage would now redo. Makes no calls."""
    outputs = _load_outputs(runs_root, run_id)
    stale = []
    for path, data in iter_records(
        sorted((Path(pairwise_root) / run_id).glob("*.json")),
        required=("a", "b", "passage_id"),
    ):
        a, b, pid = data["a"], data["b"], data["passage_id"]
        text_a = outputs.get(a, {}).get(pid)
        text_b = outputs.get(b, {}).get(pid)
        if text_a is None or text_b is None:
            continue  # no longer comparable; the stage will not revisit it
        if not _comparison_is_current(
            path, judge_id, judge_fingerprint, sha256_text(text_a), sha256_text(text_b)
        ):
            stale.append(path)
    return stale


def run_pairwise(
    judge_runner,
    run_id: str,
    runs_root: Path,
    pairwise_root: Path,
    corpus_root: Path,
    passages: dict,
    template: str,
    models: list[str],
    max_comparisons: int = DEFAULT_MAX_COMPARISONS,
    ratings: dict | None = None,
    max_compare_chars: int = MAX_COMPARE_CHARS,
    judge_id: str | None = None,
    judge_fingerprint: str | None = None,
    on_stop=None,
) -> int:
    """Run up to `max_comparisons` new comparisons. Returns how many ran.

    `on_stop`, if given, is called with the reason when the stage stops
    early, so a caller can wait out a usage limit and resume.
    """
    stop = on_stop or (lambda _reason: None)
    outputs = _load_outputs(runs_root, run_id)
    eligible = [m for m in models if outputs.get(m)]
    if len(eligible) < 2:
        return 0

    existing = load_comparisons(pairwise_root, run_id)
    counts: dict[tuple[str, str], int] = {}
    for a, b, _ in existing:
        key = tuple(sorted([a, b]))
        counts[key] = counts.get(key, 0) + 1

    ratings = ratings or {}
    executed = 0

    # Rank every eligible pair, then take one passage from each in turn.
    # Passing the comparison budget as the pair count would spend it all on
    # the first few pairs, leaving most of the field unjudged and the
    # Bradley-Terry fit resting on a nearly disconnected comparison graph.
    ranked = select_pairs(eligible, ratings, counts, k=len(eligible) ** 2)
    shared_by_pair = {
        (a, b): sorted(set(outputs[a]) & set(outputs[b])) for a, b in ranked
    }
    deepest = max((len(v) for v in shared_by_pair.values()), default=0)

    schedule = []
    for depth in range(deepest):
        for a, b in ranked:
            passages_for_pair = shared_by_pair[(a, b)]
            if depth < len(passages_for_pair):
                schedule.append((a, b, passages_for_pair[depth]))

    oversized = 0
    bad_verdicts = 0
    for a, b, passage_id in schedule:
        if executed >= max_comparisons:
            break

        out_path = comparison_path(pairwise_root, run_id, a, b, passage_id)
        text_a, text_b = outputs[a][passage_id], outputs[b][passage_id]
        sha_a, sha_b = sha256_text(text_a), sha256_text(text_b)
        if out_path.exists() and _comparison_is_current(
            out_path, judge_id, judge_fingerprint, sha_a, sha_b
        ):
            continue

        source = ""
        passage = passages.get(passage_id)
        if passage is not None:
            try:
                source = load_passage(corpus_root, passage)
            except (HashMismatch, PassageMissing):
                continue

        if len(source) + len(text_a) + len(text_b) > max_compare_chars:
            oversized += 1
            continue

        try:
            forward, reverse, outcome = compare(
                judge_runner, template, source, text_a, text_b
            )
        except ValueError as exc:
            # One unparseable verdict is a transient glitch, not a reason
            # to abandon the remaining comparisons.
            print(f"  skipping {a} vs {b} on {passage_id}: {exc}")
            bad_verdicts += 1
            if bad_verdicts >= MAX_CONSECUTIVE_BAD_VERDICTS:
                print(
                    f"stopping after {executed} comparisons: {bad_verdicts} "
                    f"verdicts in a row could not be parsed. Check the judge "
                    f"before rerunning."
                )
                stop(f"{bad_verdicts} unparseable verdicts in a row: {exc}")
                return executed
            continue
        except RuntimeError as exc:
            # The judge itself failed - a usage limit, most likely. Every
            # remaining comparison would fail the same way. Work already
            # written stays on disk; rerunning resumes from here.
            print(f"stopping after {executed} comparisons: {exc}")
            stop(str(exc))
            return executed

        write_json_atomic(
            out_path,
            {
                "a": a,
                "b": b,
                "passage_id": passage_id,
                "forward": forward,
                "reverse": reverse,
                "outcome": outcome,
                "judge_id": judge_id,
                "judge_fingerprint": judge_fingerprint,
                "sha_a": sha_a,
                "sha_b": sha_b,
            },
        )
        executed += 1
        bad_verdicts = 0

    if oversized:
        print(
            f"  {oversized} comparisons skipped: combined text exceeded "
            f"{max_compare_chars} characters"
        )
    return executed
