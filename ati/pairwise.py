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

from ati.bradleyterry import win_probability
from ati.corpus import HashMismatch, PassageMissing, load_passage

VERDICTS = ("A", "B", "tie")


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


def compare(judge_runner, template, source, text_a, text_b):
    """Judge one pair in both orders. Returns (forward, reverse, outcome)."""
    forward_result = judge_runner.run(
        "", build_pair_prompt(template, source, text_a, text_b)
    )
    if forward_result.error:
        raise RuntimeError(forward_result.error)
    forward = parse_verdict(forward_result.text)

    reverse_result = judge_runner.run(
        "", build_pair_prompt(template, source, text_b, text_a)
    )
    if reverse_result.error:
        raise RuntimeError(reverse_result.error)
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
    for path in sorted((Path(runs_root) / run_id).rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
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
    for path in sorted((Path(pairwise_root) / run_id).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        results.append((data["a"], data["b"], data["outcome"]))
    return results


def run_pairwise(
    judge_runner,
    run_id: str,
    runs_root: Path,
    pairwise_root: Path,
    corpus_root: Path,
    passages: dict,
    template: str,
    models: list[str],
    max_comparisons: int = 100,
    ratings: dict | None = None,
) -> int:
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

    for a, b in select_pairs(eligible, ratings, counts, k=max_comparisons):
        shared = sorted(set(outputs[a]) & set(outputs[b]))
        for passage_id in shared:
            if executed >= max_comparisons:
                return executed

            out_path = comparison_path(pairwise_root, run_id, a, b, passage_id)
            if out_path.exists():
                continue

            source = ""
            passage = passages.get(passage_id)
            if passage is not None:
                try:
                    source = load_passage(corpus_root, passage)
                except (HashMismatch, PassageMissing):
                    continue

            try:
                forward, reverse, outcome = compare(
                    judge_runner,
                    template,
                    source,
                    outputs[a][passage_id],
                    outputs[b][passage_id],
                )
            except (RuntimeError, ValueError) as exc:
                # A usage limit or a malformed verdict stops the stage. Work
                # already written stays on disk; rerunning resumes from here.
                print(f"stopping after {executed} comparisons: {exc}")
                return executed

            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(
                json.dumps(
                    {
                        "a": a,
                        "b": b,
                        "passage_id": passage_id,
                        "forward": forward,
                        "reverse": reverse,
                        "outcome": outcome,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            executed += 1

    return executed
