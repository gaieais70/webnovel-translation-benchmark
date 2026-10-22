"""Subprocess runner for subscription-backed CLIs (claude -p, opencode run).

The system prompt and source text are concatenated and delivered on stdin,
because CLI entry points vary too much to assume a system-prompt flag. A row
can instead name a flag (`system_prompt_flag`), or put `{system_prompt_file}`
in its command for a CLI that reads the prompt only from a file.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import subprocess
import time

from ati.config import DEFAULT_TIMEOUT_S
from ati.fileio import write_text_atomic
from ati.runners.base import RunResult, looks_like_refusal

SYSTEM_PROMPT_FILE = "{system_prompt_file}"

# Structured output a row can ask its CLI for, so the record carries token
# counts. Cost drives translation choices as much as quality does, and plain
# text output left every CLI row of run 2026-09 with no tokens to price.
OUTPUT_FORMATS = ("text", "claude-json", "codex-jsonl")


class UnparsedOutput(ValueError):
    """The CLI's structured output was not what its format promised."""


def _int(value) -> int | None:
    return value if isinstance(value, int) else None


def parse_claude_json(stdout: str) -> dict:
    """`claude -p --output-format json`: one result object."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise UnparsedOutput(f"claude JSON output could not be parsed: {exc}") from exc
    if not isinstance(data, dict):
        raise UnparsedOutput("claude JSON output could not be parsed: not an object")
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    parts = [
        _int(usage.get(k))
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    ]
    details = usage.get("output_tokens_details") or {}
    cost = data.get("total_cost_usd")
    return {
        "text": str(data.get("result") or ""),
        "error": str(data.get("result") or "error result") if data.get("is_error") else None,
        "tokens_in": sum(p for p in parts if p is not None) if any(p is not None for p in parts) else None,
        "cached_tokens_in": _int(usage.get("cache_read_input_tokens")),
        "tokens_out": _int(usage.get("output_tokens")),
        "reasoning_tokens": _int(details.get("thinking_tokens")),
        "list_cost_usd": float(cost) if isinstance(cost, (int, float)) else None,
    }


def parse_codex_jsonl(stdout: str) -> dict:
    """`codex exec --json`: one event per line; the last agent message is the
    answer, and turn.completed carries usage."""
    text, usage, failure, seen = None, {}, None, False
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen = True
        kind = event.get("type")
        item = event.get("item") or {}
        if kind == "item.completed" and item.get("type") == "agent_message":
            text = item.get("text") or ""
        elif kind == "turn.completed":
            usage = event.get("usage") or {}
        elif kind == "turn.failed":
            failure = (event.get("error") or {}).get("message") or "turn failed"
        elif kind == "error":
            failure = event.get("message") or "error"
    if not seen:
        raise UnparsedOutput("codex JSONL output could not be parsed: no events")
    return {
        "text": text or "",
        "error": failure if failure else (None if text else "no agent message in codex output"),
        "tokens_in": _int(usage.get("input_tokens")),
        "cached_tokens_in": _int(usage.get("cached_input_tokens")),
        "tokens_out": _int(usage.get("output_tokens")),
        "reasoning_tokens": _int(usage.get("reasoning_output_tokens")),
        "list_cost_usd": None,
    }


PARSERS = {"claude-json": parse_claude_json, "codex-jsonl": parse_codex_jsonl}


def resolve_executable(command: list[str], which=shutil.which) -> list[str]:
    """The command with its program resolved on PATH, extension included.

    On Windows, npm installs a CLI as an extensionless shell script plus a
    `.cmd` shim, and subprocess without a shell can only run the shim: a bare
    `opencode` failed WinError 2 on every call of run 2026-08, recorded as the
    model failing (run-notes-2026-08 §2). `shutil.which` applies PATHEXT, so
    it finds `opencode.cmd`. A program it cannot find is left as written, so
    the failure still surfaces, as an error on the task.
    """
    if not command:
        return list(command)
    resolved = which(command[0])
    return [resolved or command[0], *command[1:]]


class CliRunner:
    def __init__(
        self,
        command: list[str],
        timeout_s: float = DEFAULT_TIMEOUT_S,
        runner_fn=None,
        system_prompt_flag: str | None = None,
        workspace: Path | None = None,
        prompt_dir: Path | None = None,
        output_format: str | None = None,
    ) -> None:
        self._command = resolve_executable(list(command))
        self._timeout_s = timeout_s
        self._runner_fn = runner_fn or subprocess.run
        self._system_prompt_flag = system_prompt_flag
        self._workspace = workspace
        self._parse = PARSERS.get(output_format or "text")
        # Outside the workspace, which stays empty: the CLI under test is run
        # from a directory with nothing in it to read.
        self._prompt_dir = prompt_dir or (
            Path(tempfile.gettempdir()) / "ati-cli" / "_prompts"
        )

    def _system_prompt_path(self, system_prompt: str) -> Path:
        """A file holding the prompt, named by its content.

        Content-addressed so tasks with different prompts running at once
        never overwrite each other's file while a call is reading it.
        """
        digest = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
        path = self._prompt_dir / f"{digest}.txt"
        if not path.exists():
            self._prompt_dir.mkdir(parents=True, exist_ok=True)
            write_text_atomic(path, system_prompt)
        return path

    def run(self, system_prompt: str, source_text: str) -> RunResult:
        command = list(self._command)
        if self._system_prompt_flag:
            # `claude --system-prompt` REPLACES the agent's own system prompt.
            # Appending would leave the coding-agent framing in place, and the
            # row would measure that agent rather than the model.
            command += [self._system_prompt_flag, system_prompt]
            stdin_text = f"{source_text}\n"
        elif any(SYSTEM_PROMPT_FILE in arg for arg in command):
            # `codex exec` has no system-prompt argument; its base
            # instructions come from a file named in a config override. Same
            # separation as the flag: prompt replaced, source alone on stdin.
            path = self._system_prompt_path(system_prompt).as_posix()
            command = [arg.replace(SYSTEM_PROMPT_FILE, path) for arg in command]
            stdin_text = f"{source_text}\n"
        else:
            stdin_text = f"{system_prompt}\n\n---\n\n{source_text}\n"
        extra = {}
        if self._workspace is not None:
            self._workspace.mkdir(parents=True, exist_ok=True)
            extra["cwd"] = str(self._workspace)
        started = time.monotonic()
        try:
            completed = self._runner_fn(
                command,
                input=stdin_text,
                **extra,
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

        if self._parse is not None:
            return self._structured(completed, latency)

        if completed.returncode != 0:
            # stdout as well as stderr: `claude -p` prints a usage-limit
            # message on stdout and exits non-zero, so reading stderr alone
            # discarded the one thing that identified the cause. Run 2026-09
            # recorded nine sonnet-5 and eleven haiku-4.5 failures as a bare
            # "exit 1: " - is_usage_limit had nothing to match, so the stage
            # treated a rate limit as ordinary failure and spent passes 2 and 3
            # on it in seconds instead of waiting.
            #
            # Both streams, and the END of stderr: `codex exec` opens stderr
            # with a banner (version, workdir, model) and states its limit
            # last, and the Claude CLI now prints a notice on stderr while its
            # limit message stays on stdout. Keeping the first 200 characters
            # of stderr alone dropped the limit message in both.
            detail = " | ".join(
                part
                for part in (
                    (completed.stderr or "").strip()[-300:],
                    (completed.stdout or "").strip()[:300],
                )
                if part
            )
            return RunResult(
                text="",
                latency_s=latency,
                error=(
                    f"exit {completed.returncode}: {detail}"
                    if detail
                    else f"exit {completed.returncode} with no output on either stream"
                ),
            )

        text = (completed.stdout or "").strip()
        return RunResult(
            text=text,
            latency_s=latency,
            cost_usd=None,  # subscription access: cost is unknown, not zero
            refused=looks_like_refusal(text),
        )

    def _structured(self, completed, latency: float) -> RunResult:
        """A result from structured output, on success and on failure alike:
        both CLIs report their limit inside it, not only on stderr."""
        try:
            parsed = self._parse(completed.stdout or "")
        except UnparsedOutput as exc:
            stderr = (completed.stderr or "").strip()[-300:]
            return RunResult(
                text="",
                latency_s=latency,
                error=f"exit {completed.returncode}: {exc}"
                + (f" | {stderr}" if stderr else ""),
            )
        error = parsed["error"]
        if completed.returncode != 0 and not error:
            error = (completed.stderr or "").strip()[-300:] or "no error message"
        if error:
            return RunResult(
                text="",
                latency_s=latency,
                error=f"exit {completed.returncode}: {error}",
                tokens_in=parsed["tokens_in"],
                tokens_out=parsed["tokens_out"],
            )
        text = parsed["text"].strip()
        return RunResult(
            text=text,
            latency_s=latency,
            tokens_in=parsed["tokens_in"],
            tokens_out=parsed["tokens_out"],
            reasoning_tokens=parsed["reasoning_tokens"],
            cached_tokens_in=parsed["cached_tokens_in"],
            list_cost_usd=parsed["list_cost_usd"],
            cost_usd=None,  # subscription access: billed nothing per call
            refused=looks_like_refusal(text),
        )
