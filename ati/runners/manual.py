"""File-drop runner for models reachable only through a web UI.

First pass writes a prompt file and reports the task as pending. After the
human pastes the model's output into the matching .response file, rerunning
picks it up.
"""

from __future__ import annotations

from pathlib import Path

from ati.runners.base import RunResult, looks_like_refusal


class ResponsePending(Exception):
    """A manual response file has not been provided yet."""


class ManualRunner:
    def __init__(self, root: Path, model_id: str) -> None:
        self._dir = Path(root) / model_id
        self._task_id = "pending"

    def set_task_id(self, task_id: str) -> None:
        """Set by the run stage so prompt and response files pair up."""
        self._task_id = task_id

    def run(self, system_prompt: str, source_text: str) -> RunResult:
        self._dir.mkdir(parents=True, exist_ok=True)
        response_file = self._dir / f"{self._task_id}.response"
        if response_file.exists():
            text = response_file.read_text(encoding="utf-8").strip()
            return RunResult(
                text=text,
                latency_s=0.0,
                cost_usd=None,
                refused=looks_like_refusal(text),
            )

        prompt_file = self._dir / f"{self._task_id}.prompt"
        prompt_file.write_text(
            f"{system_prompt}\n\n---\n\n{source_text}\n", encoding="utf-8"
        )
        return RunResult(
            text="", latency_s=0.0, cost_usd=None, error="awaiting manual response"
        )
