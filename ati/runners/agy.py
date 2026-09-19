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
import os
import subprocess
import tempfile
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ati.fileio import write_text_atomic
from ati.runners.base import RunResult, looks_like_refusal

DEFAULT_TIMEOUT_S = 600.0

# agy runs on the operator's personal Google account, which is the one thing
# in this project that cannot be re-provisioned if it is lost. The gate below
# keeps usage looking like what it is - one person's tool making one request at
# a time - rather than a batch job:
#
# - one call at a time across every process on the machine, not just this one,
#   so a rubric stage and a pairwise stage started together still queue;
# - a minimum gap between the end of one call and the start of the next;
# - an empty working directory, so an agentic judge running with
#   --dangerously-skip-permissions has no .env, corpus or run output in reach.
#
# Stopping at the first sign of a usage limit, rather than retrying into it,
# is the stages' job: see judgestage.py and pairwise.py.
DEFAULT_MIN_GAP_S = 10.0
DEFAULT_COURTESY_S = 2.0
GATE_ROOT = Path(tempfile.gettempdir()) / "ati-agy"


def _acquire(handle, sleep_fn) -> None:
    if os.name == "nt":
        import msvcrt

        while True:
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError:
                sleep_fn(0.2)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _release(handle) -> None:
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class CallGate:
    """Serializes agy calls machine-wide and spaces them out.

    Fair between processes: whoever made the last call waits `courtesy_s`
    longer than anyone else before trying again, so a second process that is
    waiting always gets the next turn instead of racing the first for it.
    """

    def __init__(
        self,
        root: Path | str = GATE_ROOT,
        min_gap_s: float = DEFAULT_MIN_GAP_S,
        sleep_fn=time.sleep,
        clock_fn=time.time,
        courtesy_s: float = DEFAULT_COURTESY_S,
    ) -> None:
        self._root = Path(root)
        self._min_gap_s = min_gap_s
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._courtesy_s = courtesy_s
        # Per gate rather than per process id, so two gates in one process -
        # as in the tests - are still told apart.
        self._token = uuid.uuid4().hex

    @property
    def workspace(self) -> Path:
        path = self._root / "workspace"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _last(self) -> tuple[float, str]:
        try:
            stamp, _, owner = (
                (self._root / "last").read_text(encoding="utf-8").partition(" ")
            )
            return float(stamp), owner.strip()
        except FileNotFoundError:
            return 0.0, ""  # no call has ever been made from this machine
        except (OSError, ValueError):
            # The stamp exists but cannot be read. Assume a call just ended:
            # waiting one gap too many costs seconds, while reading damage as
            # "long ago" would let the next call through with no gap at all.
            return self._clock(), ""

    def _remaining(self, courtesy: bool) -> float:
        stamp, owner = self._last()
        extra = self._courtesy_s if courtesy and owner == self._token else 0.0
        return stamp + self._min_gap_s + extra - self._clock()

    @contextmanager
    def slot(self):
        self._root.mkdir(parents=True, exist_ok=True)
        with open(self._root / "lock", "a+b") as handle:
            while True:
                # The gap is waited out WITHOUT the lock. Waiting while holding
                # it let the process that just finished re-take the lock within
                # milliseconds of releasing it, so a second process polling for
                # its turn was starved for twenty minutes.
                wait = self._remaining(courtesy=True)
                if wait > 0:
                    self._sleep(wait)
                _acquire(handle, self._sleep)
                if self._remaining(courtesy=False) <= 0:
                    break
                _release(handle)  # another process called during the wait
            try:
                yield
            finally:
                # Recorded on failure too: a call that errored still reached
                # the account, and the gap is measured from the account's view.
                # Atomic, so a process killed mid-write cannot leave a stamp
                # that the next reader has to guess about.
                try:
                    write_text_atomic(
                        self._root / "last", f"{self._clock()!r} {self._token}"
                    )
                finally:
                    _release(handle)


# Windows caps a whole command line at 32,767 characters, and this runner
# passes the prompt as an argument to --print. A rubric prompt carries one
# translation and fits; a pairwise prompt carries the source plus TWO whole
# translations - 45,000+ characters - and does not. All 300 comparisons of the
# first pairwise run failed with WinError 206 before a single judge call was
# made. Above this length the prompt goes over stdin as stream-json instead,
# which has no such limit and still honours --json-schema.
SAFE_ARG_PROMPT_CHARS = 30000

# How much of agy's own error text is kept. Its quota refusal ends with the
# time until the quota returns ("... Resets in 1h17m35s.") and runs to ~170
# characters; the waiting loop in ati/waiting.py reads that figure, so the
# cut must not fall before it.
ERROR_DETAIL_CHARS = 400

# agy is an agent, not a completion endpoint: it has a shell and file tools, and
# --dangerously-skip-permissions approves whatever it reaches for. Measured on
# run 2026-09: 3 of the first 30 judge calls started a shell command
# mid-judgment, and every one of them stalled until the timeout - all of the
# hangs, none of the successful calls. A judge that runs commands is also not
# judging only the text in front of it. agy's own PreToolUse deny hook would
# block this outright, but it loads only in a workspace the operator has
# trusted in their Antigravity settings, so the instruction is carried in the
# prompt instead.
NO_TOOLS_PREAMBLE = (
    "You are being used as a text-only evaluator. Do not call any tools: do "
    "not run commands, open, read or write files, or browse. Everything you "
    "need is in this message. Respond directly."
)


class AgyRunner:
    def __init__(
        self,
        model: str,
        json_schema: Path | str | None = None,
        sandbox: bool = True,
        executable: str = "agy",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        runner_fn=None,
        gate: CallGate | None = None,
    ) -> None:
        self._model = model
        # Absolute, because a gated call runs from the empty workspace and a
        # path relative to the project root would no longer resolve there.
        self._schema = str(Path(json_schema).resolve()) if json_schema else None
        self._sandbox = sandbox
        self._executable = executable
        self._timeout_s = timeout_s
        self._runner_fn = runner_fn or subprocess.run
        self._gate = gate

    def _print_timeout(self) -> list[str]:
        # agy has its own wait limit in print mode, 5 minutes by default, which
        # would end a long judge call before the subprocess timeout does and
        # report it as an agy error rather than a timeout.
        return ["--print-timeout", f"{int(self._timeout_s)}s"]

    def _stream_command(self) -> list[str]:
        """Long-prompt form: the prompt arrives on stdin, not the argv."""
        cmd = [
            self._executable,
            "--print=",
            "--model",
            self._model,
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--dangerously-skip-permissions",
            *self._print_timeout(),
        ]
        if self._schema:
            cmd += ["--json-schema", self._schema]
        if self._sandbox:
            cmd.append("--sandbox")
        return cmd

    @staticmethod
    def _stream_input(prompt: str) -> str:
        payload = {"event": "user", "message": {"role": "user", "content": prompt}}
        return json.dumps(payload, ensure_ascii=False) + chr(10)

    @staticmethod
    def _result_from_stream(stdout: str):
        """The final `result` event; earlier events are progress, not output."""
        found = None
        for line in (stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("event") == "result":
                found = event.get("result")
        return found

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
            *self._print_timeout(),
        ]
        if self._schema:
            # --json-schema is only accepted alongside --output-format json
            cmd += ["--json-schema", self._schema]
        if self._sandbox:
            cmd.append("--sandbox")
        return cmd

    def run(self, system_prompt: str, source_text: str) -> RunResult:
        prompt = f"{NO_TOOLS_PREAMBLE}\n\n{system_prompt}\n\n{source_text}".strip()
        streaming = len(prompt) > SAFE_ARG_PROMPT_CHARS
        kwargs = dict(
            capture_output=True,
            text=True,
            timeout=self._timeout_s,
            encoding="utf-8",
            # One stray byte that is not UTF-8 - in a progress line, say -
            # would otherwise raise UnicodeDecodeError after the call had
            # already reached the account, and take the stage down with it.
            errors="replace",
        )
        if streaming:
            cmd = self._stream_command()
            kwargs["input"] = self._stream_input(prompt)
        else:
            cmd = self._command(prompt)
        try:
            if self._gate is None:
                completed = self._runner_fn(cmd, **kwargs)
            else:
                with self._gate.slot():
                    completed = self._runner_fn(
                        cmd, cwd=str(self._gate.workspace), **kwargs
                    )
        except subprocess.TimeoutExpired:
            return RunResult(
                text="", latency_s=0.0, error=f"timed out after {self._timeout_s}s"
            )
        except OSError as exc:
            return RunResult(text="", latency_s=0.0, error=str(exc))

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            return RunResult(
                text="",
                latency_s=0.0,
                error=f"exit {completed.returncode}: {detail[:ERROR_DETAIL_CHARS]}",
            )

        if streaming:
            envelope = self._result_from_stream(completed.stdout)
            if envelope is None:
                return RunResult(
                    text="", latency_s=0.0,
                    error="agy stream produced no result event",
                )
        else:
            try:
                envelope = json.loads(completed.stdout)
            except (ValueError, TypeError) as exc:
                return RunResult(
                    text="", latency_s=0.0, error=f"unparseable agy output: {exc}"
                )
        # Valid JSON is not necessarily an envelope: a list, a string or a
        # number would otherwise raise AttributeError below, uncaught, and end
        # the stage on a call that had already cost the account one request.
        if not isinstance(envelope, dict):
            return RunResult(
                text="",
                latency_s=0.0,
                error=f"unexpected agy output: a JSON {type(envelope).__name__}, "
                "not an object",
            )
        if envelope.get("status") == "ERROR":
            return RunResult(
                text="",
                latency_s=_seconds(envelope.get("duration_seconds")),
                error=f"agy error: {str(envelope.get('error'))[:ERROR_DETAIL_CHARS]}",
            )

        usage = envelope.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        latency = _seconds(envelope.get("duration_seconds"))

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
            response = envelope.get("response")
            text = response.strip() if isinstance(response, str) else ""

        return RunResult(
            text=text,
            latency_s=latency,
            tokens_in=usage.get("input_tokens"),
            tokens_out=usage.get("output_tokens"),
            reasoning_tokens=usage.get("thinking_tokens"),
            cost_usd=None,  # subscription access: unknown, not zero
            refused=looks_like_refusal(text) if not self._schema else False,
        )


def _seconds(value) -> float:
    """agy's duration_seconds, or 0.0 when it is absent or not a number."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
