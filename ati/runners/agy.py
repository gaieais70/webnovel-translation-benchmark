"""Runner for the Antigravity CLI (`agy`), used to reach Gemini by subscription.

Two things differ from the generic CLI runner. The prompt is passed as an
argument to `-p` rather than on stdin, and output is a JSON envelope rather
than bare text. When a JSON schema is supplied, `agy` validates the model's
structured output itself, so the judge receives an already-conforming object
instead of prose that has to be mined for JSON.

`--sandbox` is on by default: an agentic judge with tools could look up the
source work while scoring a translation of it, which would no longer be
measuring the translation.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ati.runners.base import RunResult, looks_like_refusal

DEFAULT_TIMEOUT_S = 600.0


class AgyRunner:
    def __init__(
        self,
        model: str,
        json_schema: Path | str | None = None,
        sandbox: bool = True,
        executable: str = "agy",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        runner_fn=None,
    ) -> None:
        self._model = model
        self._schema = str(json_schema) if json_schema else None
        self._sandbox = sandbox
        self._executable = executable
        self._timeout_s = timeout_s
        self._runner_fn = runner_fn or subprocess.run

    def _command(self, prompt: str) -> list[str]:
        cmd = [
            self._executable,
            "-p",
            prompt,
            "--model",
            self._model,
            "--output-format",
            "json",
            "--dangerously-skip-permissions",
        ]
        if self._schema:
            # --json-schema is only accepted alongside --output-format json
            cmd += ["--json-schema", self._schema]
        if self._sandbox:
            cmd.append("--sandbox")
        return cmd

    def run(self, system_prompt: str, source_text: str) -> RunResult:
        prompt = f"{system_prompt}\n\n{source_text}".strip()
        try:
            completed = self._runner_fn(
                self._command(prompt),
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                encoding="utf-8",
            )
        except subprocess.TimeoutExpired:
            return RunResult(
                text="", latency_s=0.0, error=f"timed out after {self._timeout_s}s"
            )
        except OSError as exc:
            return RunResult(text="", latency_s=0.0, error=str(exc))

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[:200]
            return RunResult(
                text="", latency_s=0.0, error=f"exit {completed.returncode}: {detail}"
            )

        try:
            envelope = json.loads(completed.stdout)
        except (ValueError, TypeError) as exc:
            return RunResult(
                text="", latency_s=0.0, error=f"unparseable agy output: {exc}"
            )

        usage = envelope.get("usage") or {}
        latency = float(envelope.get("duration_seconds") or 0.0)

        if self._schema:
            structured = envelope.get("structured_output")
            if structured is None:
                return RunResult(
                    text="",
                    latency_s=latency,
                    error="agy returned no structured_output despite a schema",
                )
            text = json.dumps(structured, ensure_ascii=False)
        else:
            text = (envelope.get("response") or "").strip()

        return RunResult(
            text=text,
            latency_s=latency,
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
            reasoning_tokens=usage.get("thinking_tokens"),
            cost_usd=None,  # subscription access: unknown, not zero
            refused=looks_like_refusal(text) if not self._schema else False,
        )
