"""Subprocess runner for subscription-backed CLIs (claude -p, opencode run).

The system prompt and source text are concatenated and delivered on stdin,
because CLI entry points vary too much to assume a system-prompt flag.
"""

from __future__ import annotations

import subprocess
import time

from ati.config import DEFAULT_TIMEOUT_S
from ati.runners.base import RunResult, looks_like_refusal


class CliRunner:
    def __init__(
        self,
        command: list[str],
        timeout_s: float = DEFAULT_TIMEOUT_S,
        runner_fn=None,
    ) -> None:
        self._command = list(command)
        self._timeout_s = timeout_s
        self._runner_fn = runner_fn or subprocess.run

    def run(self, system_prompt: str, source_text: str) -> RunResult:
        stdin_text = f"{system_prompt}\n\n---\n\n{source_text}\n"
        started = time.monotonic()
        try:
            completed = self._runner_fn(
                self._command,
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                encoding="utf-8",
                # A stray non-UTF-8 byte must not raise out of the run stage.
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                text="",
                latency_s=time.monotonic() - started,
                error=f"timed out after {self._timeout_s}s",
            )
        except OSError as exc:
            return RunResult(
                text="", latency_s=time.monotonic() - started, error=str(exc)
            )
        latency = time.monotonic() - started

        if completed.returncode != 0:
            detail = (completed.stderr or "").strip()[:200]
            return RunResult(
                text="",
                latency_s=latency,
                error=f"exit {completed.returncode}: {detail}",
            )

        text = (completed.stdout or "").strip()
        return RunResult(
            text=text,
            latency_s=latency,
            cost_usd=None,  # subscription access: cost is unknown, not zero
            refused=looks_like_refusal(text),
        )
