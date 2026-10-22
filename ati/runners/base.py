"""The Runner interface every model-access path implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class RunResult:
    """One model's output for one task.

    cost_usd is None when cost is unknown - subscription-backed access, for
    example. None must propagate to "N/A" in reports and must never be
    coerced to zero, which would misrepresent a paid model as free.
    """

    text: str
    latency_s: float
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    reasoning_tokens: int | None = None
    refused: bool = False
    error: str | None = None
    # Which model slug actually served this call. Set when a runner has more
    # than one route it can use, so a model whose chapters were split between
    # a free route and its paid fallback shows the split in the record instead
    # of averaging two providers into one row silently. Running one model
    # across two providers has already been tried on this project and could not
    # be read as a single result; see the retired hy3 drift probe in
    # docs/run-notes-2026-08.md.
    route: str | None = None
    # Why generation ended, as the provider reported it. Recorded because a
    # provider that ends a generation early does not always say "length":
    # OpenRouter's free routes cut nemotron-3.5-lightning off at 600 seconds
    # mid-sentence, and without this there was no trace of why.
    finish_reason: str | None = None
    # Of tokens_in, how many were read from a prompt cache, where the route
    # says. Priced at the roster's `cached_input` rate when it declares one.
    cached_tokens_in: int | None = None
    # What the call comes to at the provider's list price, as the route itself
    # reports it (the Claude CLI does, cache pricing included). Not what was
    # billed - a subscription paid - so it never fills cost_usd.
    list_cost_usd: float | None = None


class Runner(Protocol):
    def run(self, system_prompt: str, source_text: str) -> RunResult: ...


REFUSAL_MARKERS: tuple[str, ...] = (
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
    "i can't translate",
    "i cannot translate",
    "i'm unable to",
    "i am unable to",
    "i won't be able to",
    "i must decline",
    "against my guidelines",
)


def looks_like_refusal(text: str) -> bool:
    """Heuristic refusal detection over the first part of an output.

    Only the opening is checked: a refusal states itself up front, whereas a
    long translation may legitimately contain similar words in dialogue.
    """
    if not isinstance(text, str):
        return False
    head = text[:400].lower()
    return any(marker in head for marker in REFUSAL_MARKERS)
