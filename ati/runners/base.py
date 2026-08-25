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
    refused: bool = False
    error: str | None = None


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
    head = text[:400].lower()
    return any(marker in head for marker in REFUSAL_MARKERS)
