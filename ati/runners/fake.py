"""In-memory runner used by tests. Never performs I/O."""

from __future__ import annotations

from ati.runners.base import RunResult, looks_like_refusal


class FakeRunner:
    def __init__(
        self,
        responses: list[str] | None = None,
        error: str | None = None,
        cost_usd: float | None = 0.001,
    ) -> None:
        self._responses = list(responses or [])
        self._error = error
        self._cost_usd = cost_usd
        self.calls: list[tuple[str, str]] = []

    def run(self, system_prompt: str, source_text: str) -> RunResult:
        self.calls.append((system_prompt, source_text))
        if self._error is not None:
            return RunResult(text="", latency_s=0.0, error=self._error)
        text = self._responses.pop(0) if self._responses else ""
        return RunResult(
            text=text,
            latency_s=0.0,
            tokens_in=len(source_text),
            tokens_out=len(text),
            cost_usd=self._cost_usd,
            refused=looks_like_refusal(text),
        )
