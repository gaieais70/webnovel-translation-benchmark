# Pairwise Ranking and Gemini Judging — Implementation Plan

> **For agentic workers:** implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Rank opencode-go models head-to-head using Gemini 3.7 Flash (via the `agy` CLI) as judge, with Bradley–Terry ratings, ties, and bootstrap confidence intervals — resumable across subscription usage limits.

**Architecture:** Translations run through the existing `ApiRunner` against opencode-go's OpenAI-compatible endpoint. Judging runs through a new `AgyRunner` that drives `agy -p` in sandboxed, schema-enforced JSON mode. A new pairwise stage generates comparisons adaptively, checkpoints each one to disk, and fits a Bradley–Terry model.

**Tech Stack:** Python 3.11+, httpx, PyYAML, pytest. Standard library `math`/`random`/`statistics` for the fitting — no numpy.

**Base commit:** `2d378bb` (141 tests passing)

## Verified environment facts

These were confirmed by direct testing; do not re-derive them.

- **opencode-go endpoint:** `https://opencode.ai/zen/go/v1/chat/completions`, OpenAI-compatible, key in `OPENCODE_API_KEY`. Returns `cost` as a **top-level** key (not inside `usage`), and `usage.completion_tokens_details.reasoning_tokens`. On a subscription, `cost` is `0`.
- **agy CLI:** `agy -p "<prompt>" --model gemini-3.7-flash-high --output-format json --json-schema <file> --sandbox --dangerously-skip-permissions`. Returns one JSON object with `response`, `structured_output`, `duration_seconds`, and `usage` (`input_tokens`, `output_tokens`, `thinking_tokens`). `--json-schema` requires `--output-format json`.
- **UTF-8:** CJK passes through httpx correctly. Only console printing needs `PYTHONIOENCODING=utf-8`; all file writes already specify UTF-8.

## Global Constraints

- **Do not modify** `ati/corpus.py`, `ati/tasks.py`, `ati/runstage.py`, `ati/judge.py`, `ati/judgestage.py`, `ati/scoring.py`, `ati/textmetrics.py`, `ati/intake.py`, or `ati/runners/cli.py` / `manual.py` / `fake.py`.
- **No live network calls in tests, and no real `agy` invocations in tests.** Every runner takes an injectable `runner_fn`.
- **Cost `None` means unknown; cost `0` means subscription-covered.** They are different and must stay different all the way to the report.
- **Every pairwise comparison checkpoints to disk the moment it completes.** Hitting a usage limit must lose at most one comparison.
- Python 3.11+ syntax. Only httpx, PyYAML, pytest. No numpy, no pydantic, no jinja2.

---

### Task 16: Capture top-level cost and reasoning tokens

**Files:**
- Modify: `ati/runners/base.py`, `ati/runners/api.py`, `tests/test_runners_api.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_runners_api.py`:

```python
def test_api_runner_reads_top_level_cost():
    """opencode-go returns cost outside usage, unlike OpenRouter."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "t"}}],
                "cost": 0.0031,
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    assert runner.run("s", "u").cost_usd == 0.0031


def test_api_runner_prefers_usage_cost_over_top_level():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "t"}}],
                "cost": 9.99,
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.5},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    assert runner.run("s", "u").cost_usd == 0.5


def test_api_runner_keeps_zero_cost_distinct_from_unknown():
    """Subscription access bills zero. That is not the same as unknown."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "t"}}], "cost": 0},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    assert runner.run("s", "u").cost_usd == 0.0


def test_api_runner_captures_reasoning_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "t"}}],
                "usage": {
                    "prompt_tokens": 43,
                    "completion_tokens": 178,
                    "completion_tokens_details": {"reasoning_tokens": 153},
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    result = runner.run("s", "u")
    assert result.reasoning_tokens == 153
    assert result.tokens_out == 178


def test_api_runner_reasoning_tokens_none_when_absent():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "t"}}], "usage": {}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    assert runner.run("s", "u").reasoning_tokens is None
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_runners_api.py -v`
Expected: FAIL — `RunResult` has no `reasoning_tokens`, top-level cost ignored.

- [ ] **Step 3: Add the field**

In `ati/runners/base.py`, add one field to `RunResult`, after `cost_usd`:

```python
    reasoning_tokens: int | None = None
```

- [ ] **Step 4: Read both cost locations**

In `ati/runners/api.py`, replace the block that builds the final `RunResult` with:

```python
        usage = body.get("usage") or {}
        # OpenRouter reports cost inside usage; opencode-go reports it at the
        # top level. Zero is a real value on subscription access and must not
        # collapse into None, which means "unknown".
        cost = usage.get("cost")
        if cost is None:
            cost = body.get("cost")
        details = usage.get("completion_tokens_details") or {}
        return RunResult(
            text=text,
            latency_s=latency,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            cost_usd=float(cost) if cost is not None else None,
            refused=looks_like_refusal(text),
        )
```

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest -q` — expect 146 passed.

```bash
git add ati/runners/base.py ati/runners/api.py tests/test_runners_api.py
git commit -m "feat: read top-level cost and reasoning tokens"
```

---

### Task 17: `AgyRunner`

**Files:**
- Create: `ati/runners/agy.py`, `tests/test_runners_agy.py`

**Interfaces:**
- Produces: `ati.runners.agy.AgyRunner(model, json_schema=None, sandbox=True, executable="agy", timeout_s=600.0, runner_fn=None)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_runners_agy.py`:

```python
import json
import subprocess

from ati.runners.agy import AgyRunner


def envelope(**overrides):
    payload = {
        "conversation_id": "x",
        "status": "SUCCESS",
        "response": "Yun Che raised his head.",
        "duration_seconds": 5.9,
        "num_turns": 2,
        "usage": {"input_tokens": 28754, "output_tokens": 758, "thinking_tokens": 403},
    }
    payload.update(overrides)
    return json.dumps(payload)


def fake_run_factory(stdout="", stderr="", returncode=0, capture=None):
    def fake_run(cmd, **kwargs):
        if capture is not None:
            capture.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    return fake_run


def test_agy_runner_builds_a_sandboxed_print_mode_command():
    capture = []
    runner = AgyRunner(
        model="gemini-3.7-flash-high",
        runner_fn=fake_run_factory(stdout=envelope(), capture=capture),
    )
    runner.run("SYSTEM", "SOURCE")
    cmd, _ = capture[0]
    assert cmd[0] == "agy"
    assert "-p" in cmd
    assert "--model" in cmd and "gemini-3.7-flash-high" in cmd
    assert "--output-format" in cmd and "json" in cmd
    assert "--sandbox" in cmd
    assert "--dangerously-skip-permissions" in cmd


def test_agy_runner_passes_prompt_as_argument_not_stdin():
    capture = []
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=envelope(), capture=capture))
    runner.run("SYSTEM", "SOURCE")
    cmd, kwargs = capture[0]
    joined = " ".join(cmd)
    assert "SYSTEM" in joined and "SOURCE" in joined
    assert not kwargs.get("input")


def test_agy_runner_returns_response_text():
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=envelope()))
    result = runner.run("s", "u")
    assert result.text == "Yun Che raised his head."
    assert result.error is None


def test_agy_runner_maps_usage_and_duration():
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=envelope()))
    result = runner.run("s", "u")
    assert result.tokens_in == 28754
    assert result.tokens_out == 758
    assert result.reasoning_tokens == 403
    assert result.latency_s == 5.9


def test_agy_runner_reports_cost_none_for_subscription():
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=envelope()))
    assert runner.run("s", "u").cost_usd is None


def test_agy_runner_returns_structured_output_as_json_text_when_schema_given(tmp_path):
    schema = tmp_path / "s.json"
    schema.write_text("{}", encoding="utf-8")
    out = envelope(structured_output={"scores": {"accuracy": 4}})
    capture = []
    runner = AgyRunner(
        "m", json_schema=schema, runner_fn=fake_run_factory(stdout=out, capture=capture)
    )
    result = runner.run("s", "u")
    assert json.loads(result.text) == {"scores": {"accuracy": 4}}
    cmd, _ = capture[0]
    assert "--json-schema" in cmd


def test_agy_runner_errors_when_schema_requested_but_absent_from_envelope(tmp_path):
    schema = tmp_path / "s.json"
    schema.write_text("{}", encoding="utf-8")
    runner = AgyRunner("m", json_schema=schema, runner_fn=fake_run_factory(stdout=envelope()))
    result = runner.run("s", "u")
    assert result.error is not None
    assert "structured_output" in result.error


def test_agy_runner_records_nonzero_exit():
    runner = AgyRunner(
        "m", runner_fn=fake_run_factory(stdout="", stderr="quota exceeded", returncode=1)
    )
    result = runner.run("s", "u")
    assert "quota exceeded" in result.error
    assert result.text == ""


def test_agy_runner_records_unparseable_output():
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout="not json at all"))
    result = runner.run("s", "u")
    assert result.error is not None
    assert result.text == ""


def test_agy_runner_records_timeout():
    def timeout_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    runner = AgyRunner("m", timeout_s=1, runner_fn=timeout_run)
    assert "timed out" in runner.run("s", "u").error


def test_agy_runner_can_disable_sandbox():
    capture = []
    runner = AgyRunner(
        "m", sandbox=False, runner_fn=fake_run_factory(stdout=envelope(), capture=capture)
    )
    runner.run("s", "u")
    assert "--sandbox" not in capture[0][0]
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_runners_agy.py -v`
Expected: FAIL — no module `ati.runners.agy`.

- [ ] **Step 3: Write the implementation**

Create `ati/runners/agy.py`:

```python
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
```

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest -q` — expect 157 passed.

```bash
git add ati/runners/agy.py tests/test_runners_agy.py
git commit -m "feat: agy runner for Gemini via Antigravity CLI"
```

---

### Task 18: Route the judge through `models.yaml`

**Files:**
- Modify: `ati/config.py`, `ati/cli.py`, `tests/test_cli.py`, `config/models.yaml`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_build_runner_returns_agy_runner_for_agy_models(tmp_path):
    from ati.runners.agy import AgyRunner

    m = ModelConfig(id="judge", runner="agy", model="gemini-3.7-flash-high")
    assert isinstance(build_runner(m, None, tmp_path), AgyRunner)


def test_load_models_accepts_agy_runner(tmp_path):
    from ati.config import load_models

    p = tmp_path / "models.yaml"
    p.write_text(
        "- id: judge\n  runner: agy\n  model: gemini-3.7-flash-high\n"
        "  json_schema: config/schemas/judgment.json\n  sandbox: true\n",
        encoding="utf-8",
    )
    m = load_models(p)[0]
    assert m.runner == "agy"
    assert m.json_schema == "config/schemas/judgment.json"
    assert m.sandbox is True


def test_load_models_rejects_agy_entry_without_model(tmp_path):
    from ati.config import ConfigError, load_models

    p = tmp_path / "models.yaml"
    p.write_text("- id: judge\n  runner: agy\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="requires 'model'"):
        load_models(p)


def test_build_judge_runner_uses_models_yaml_entry(tmp_path):
    from ati.cli import build_judge_runner
    from ati.runners.agy import AgyRunner

    p = tmp_path / "models.yaml"
    p.write_text(
        "- id: gem-judge\n  runner: agy\n  model: gemini-3.7-flash-high\n",
        encoding="utf-8",
    )
    runner = build_judge_runner("gem-judge", p, None, tmp_path, None)
    assert isinstance(runner, AgyRunner)


def test_build_judge_runner_falls_back_to_api_for_unknown_id(tmp_path):
    from ati.cli import build_judge_runner
    from ati.runners.api import ApiRunner

    p = tmp_path / "models.yaml"
    p.write_text("- id: other\n  runner: cli\n  command: [\"x\"]\n", encoding="utf-8")
    runner = build_judge_runner(
        "anthropic/claude-sonnet-5", p, "http://x", tmp_path, "key"
    )
    assert isinstance(runner, ApiRunner)
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL — `agy` is not a valid runner, `build_judge_runner` does not exist.

- [ ] **Step 3: Extend `ati/config.py`**

Change `VALID_RUNNERS` to include `agy`:

```python
VALID_RUNNERS = ("api", "cli", "manual", "agy")
```

Add two fields to `ModelConfig`, after `anchor`:

```python
    json_schema: str | None = None
    sandbox: bool = True
```

In `load_models`, add the `agy` validation next to the existing `api` check:

```python
        if runner == "agy" and not entry.get("model"):
            raise ConfigError(f"model {model_id}: runner 'agy' requires 'model'")
```

and pass the two new fields when constructing `ModelConfig`:

```python
                json_schema=entry.get("json_schema"),
                sandbox=bool(entry.get("sandbox", True)),
```

- [ ] **Step 4: Extend `ati/cli.py`**

Add the import:

```python
from ati.runners.agy import AgyRunner
```

In `build_runner`, add the `agy` branch before the final `return ManualRunner(...)`:

```python
    if model.runner == "agy":
        return AgyRunner(
            model=model.model or model.id,
            json_schema=model.json_schema,
            sandbox=model.sandbox,
        )
```

Add a new function above `cmd_judge`:

```python
def build_judge_runner(judge_id, models_path, endpoint, manual_root, api_key):
    """Resolve the judge against models.yaml, falling back to a plain API call.

    The judge was previously hardcoded to ApiRunner, which made a
    subscription-backed CLI judge impossible. Declaring it in models.yaml like
    any other model also gets the secondary judge for free.
    """
    try:
        declared = {m.id: m for m in load_models(Path(models_path))}
    except (OSError, ConfigError):
        declared = {}
    if judge_id in declared:
        return build_runner(declared[judge_id], api_key, Path(manual_root))
    return ApiRunner(
        endpoint=endpoint or DEFAULT_ENDPOINT,
        model=judge_id,
        api_key=api_key or "",
        temperature=0.0,
    )
```

Add `ConfigError` to the `ati.config` import line.

Replace the first four lines of `cmd_judge` with:

```python
def cmd_judge(args) -> int:
    judge_runner = build_judge_runner(
        args.judge_model,
        args.models,
        args.judge_endpoint,
        Path(args.manual),
        os.environ.get("OPENROUTER_API_KEY"),
    )
```

- [ ] **Step 5: Add the judge entry to `config/models.yaml`**

Append:

```yaml
# Judge. Runs through the Antigravity CLI on a Gemini subscription.
# --sandbox is deliberate: a judge with tools could look up the source work
# while scoring a translation of it.
- id: gemini-judge
  runner: agy
  model: gemini-3.7-flash-high
  sandbox: true
```

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest -q` — expect 162 passed.

```bash
git add ati/config.py ati/cli.py config/models.yaml tests/test_cli.py
git commit -m "feat: judge resolves through models.yaml, enabling CLI judges"
```

---

### Task 19: Bradley–Terry fitting with ties and bootstrap intervals

**Files:**
- Create: `ati/bradleyterry.py`, `tests/test_bradleyterry.py`

**Interfaces:**
- Produces:
  - `ati.bradleyterry.fit(comparisons, prior=0.5, tol=1e-10, max_iter=10000) -> dict[str, float]` — comparisons are `(a, b, outcome)` with outcome in `{"a", "b", "tie"}`; returns strength parameters normalized to geometric mean 1.
  - `ati.bradleyterry.to_elo(strengths, anchor_id=None, anchor_rating=1000.0) -> dict[str, float]`
  - `ati.bradleyterry.bootstrap_ci(comparisons, rounds=200, seed=0, anchor_id=None) -> dict[str, tuple[float, float]]`
  - `ati.bradleyterry.win_probability(rating_a, rating_b) -> float`

- [ ] **Step 1: Write the failing test**

Create `tests/test_bradleyterry.py`:

```python
import pytest

from ati.bradleyterry import bootstrap_ci, fit, to_elo, win_probability


def test_fit_returns_empty_for_no_comparisons():
    assert fit([]) == {}


def test_stronger_model_gets_higher_strength():
    comparisons = [("a", "b", "a")] * 8 + [("a", "b", "b")] * 2
    s = fit(comparisons)
    assert s["a"] > s["b"]


def test_equal_records_give_equal_strength():
    comparisons = [("a", "b", "a")] * 5 + [("a", "b", "b")] * 5
    s = fit(comparisons)
    assert s["a"] == pytest.approx(s["b"], rel=1e-6)


def test_all_ties_give_equal_strength():
    s = fit([("a", "b", "tie")] * 6)
    assert s["a"] == pytest.approx(s["b"], rel=1e-6)


def test_ties_count_as_half_a_win_each():
    """A tie must sit strictly between a win and a loss."""
    tie_only = fit([("a", "b", "tie")] * 10)
    a_wins = fit([("a", "b", "a")] * 10)
    assert a_wins["a"] > tie_only["a"]


def test_transitivity_is_recovered():
    comparisons = (
        [("a", "b", "a")] * 9
        + [("b", "c", "b")] * 9
        + [("a", "c", "a")] * 9
    )
    s = fit(comparisons)
    assert s["a"] > s["b"] > s["c"]


def test_undefeated_model_does_not_diverge():
    """Without regularization BT has no finite MLE for an undefeated model."""
    s = fit([("a", "b", "a")] * 20)
    assert s["a"] < float("inf")
    assert s["a"] > s["b"] > 0


def test_fit_is_order_independent():
    forward = [("a", "b", "a"), ("b", "c", "b"), ("a", "c", "a")] * 3
    backward = list(reversed(forward))
    assert fit(forward) == pytest.approx(fit(backward), rel=1e-9)


def test_strengths_normalize_to_geometric_mean_one():
    import math

    s = fit([("a", "b", "a")] * 4 + [("b", "c", "tie")] * 4)
    logs = [math.log(v) for v in s.values()]
    assert sum(logs) / len(logs) == pytest.approx(0.0, abs=1e-9)


def test_to_elo_centres_on_mean_without_anchor():
    elo = to_elo({"a": 2.0, "b": 0.5})
    assert sum(elo.values()) / 2 == pytest.approx(1000.0)


def test_to_elo_pins_the_anchor():
    elo = to_elo({"a": 2.0, "weak": 0.5}, anchor_id="weak", anchor_rating=1000.0)
    assert elo["weak"] == pytest.approx(1000.0)
    assert elo["a"] > elo["weak"]


def test_elo_gap_of_400_is_ten_to_one_odds():
    elo = to_elo({"a": 10.0, "b": 1.0})
    assert elo["a"] - elo["b"] == pytest.approx(400.0)


def test_win_probability_is_symmetric_and_calibrated():
    assert win_probability(1000, 1000) == pytest.approx(0.5)
    assert win_probability(1400, 1000) == pytest.approx(10 / 11, rel=1e-6)


def test_bootstrap_ci_brackets_the_point_estimate():
    comparisons = [("a", "b", "a")] * 15 + [("a", "b", "b")] * 5
    point = to_elo(fit(comparisons))
    ci = bootstrap_ci(comparisons, rounds=80, seed=1)
    lo, hi = ci["a"]
    assert lo <= point["a"] <= hi


def test_bootstrap_ci_is_deterministic_for_a_fixed_seed():
    comparisons = [("a", "b", "a")] * 10 + [("a", "b", "b")] * 4
    assert bootstrap_ci(comparisons, rounds=50, seed=7) == bootstrap_ci(
        comparisons, rounds=50, seed=7
    )


def test_bootstrap_interval_is_wider_with_less_data():
    few = [("a", "b", "a")] * 3 + [("a", "b", "b")] * 1
    many = [("a", "b", "a")] * 30 + [("a", "b", "b")] * 10
    w_few = bootstrap_ci(few, rounds=80, seed=3)["a"]
    w_many = bootstrap_ci(many, rounds=80, seed=3)["a"]
    assert (w_few[1] - w_few[0]) > (w_many[1] - w_many[0])


def test_fit_rejects_unknown_outcome():
    with pytest.raises(ValueError, match="unknown outcome"):
        fit([("a", "b", "maybe")])
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest tests/test_bradleyterry.py -v`
Expected: FAIL — no module `ati.bradleyterry`.

- [ ] **Step 3: Write the implementation**

Create `ati/bradleyterry.py`:

```python
"""Bradley-Terry rating from pairwise comparisons.

Chosen over Elo or Glicko deliberately. Those are sequential algorithms built
for players arriving over time, so ratings depend on the order matches were
played. A benchmark has every comparison in hand at once, and a maximum
likelihood fit is order-independent, reproducible from the same data, and
supports interval estimates.

Ties count as half a win to each side, the same convention used by Chatbot
Arena. Two translations of one sentence are frequently equivalent, and forcing
the judge to name a winner manufactures noise.
"""

from __future__ import annotations

import math
import random

OUTCOMES = ("a", "b", "tie")
ELO_SCALE = 400.0


def _tally(comparisons, prior: float):
    models = sorted({m for c in comparisons for m in (c[0], c[1])})
    index = {m: i for i, m in enumerate(models)}
    wins = [0.0] * len(models)
    pair_counts: dict[tuple[int, int], float] = {}

    for a, b, outcome in comparisons:
        if outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {outcome!r}; expected one of {OUTCOMES}")
        ia, ib = index[a], index[b]
        if outcome == "a":
            wins[ia] += 1.0
        elif outcome == "b":
            wins[ib] += 1.0
        else:
            wins[ia] += 0.5
            wins[ib] += 0.5
        key = (min(ia, ib), max(ia, ib))
        pair_counts[key] = pair_counts.get(key, 0.0) + 1.0

    # Regularize: add `prior` worth of tie to every observed pairing. Without
    # this an undefeated model has no finite maximum likelihood estimate and
    # the fit runs away to infinity.
    for (i, j) in list(pair_counts):
        pair_counts[(i, j)] += 2.0 * prior
        wins[i] += prior
        wins[j] += prior

    return models, index, wins, pair_counts


def fit(comparisons, prior: float = 0.5, tol: float = 1e-10, max_iter: int = 10000):
    """Return strength parameters, normalized to geometric mean 1."""
    if not comparisons:
        return {}

    models, index, wins, pair_counts = _tally(comparisons, prior)
    n = len(models)
    strengths = [1.0] * n

    for _ in range(max_iter):
        updated = [0.0] * n
        for i in range(n):
            denominator = 0.0
            for j in range(n):
                if i == j:
                    continue
                count = pair_counts.get((min(i, j), max(i, j)), 0.0)
                if count:
                    denominator += count / (strengths[i] + strengths[j])
            updated[i] = wins[i] / denominator if denominator > 0 else strengths[i]

        mean_log = sum(math.log(v) for v in updated if v > 0) / n
        scale = math.exp(mean_log)
        updated = [v / scale for v in updated]

        shift = max(abs(updated[i] - strengths[i]) for i in range(n))
        strengths = updated
        if shift < tol:
            break

    return {m: strengths[index[m]] for m in models}


def to_elo(strengths, anchor_id: str | None = None, anchor_rating: float = 1000.0):
    """Map strengths onto an Elo-like scale where 400 points is 10:1 odds.

    Only the offset is arbitrary; the spacing carries the meaning. Pinning an
    anchor keeps the numbers stable as models join or leave the pool.
    """
    if not strengths:
        return {}
    raw = {m: ELO_SCALE * math.log10(s) for m, s in strengths.items()}
    if anchor_id and anchor_id in raw:
        offset = anchor_rating - raw[anchor_id]
    else:
        offset = anchor_rating - sum(raw.values()) / len(raw)
    return {m: v + offset for m, v in raw.items()}


def win_probability(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / ELO_SCALE))


def bootstrap_ci(
    comparisons,
    rounds: int = 200,
    seed: int = 0,
    anchor_id: str | None = None,
    anchor_rating: float = 1000.0,
):
    """Percentile bootstrap over comparisons, returning 95% intervals.

    Published so a reader can see when a gap between two models is not
    distinguishable from zero.
    """
    if not comparisons:
        return {}

    rng = random.Random(seed)
    total = len(comparisons)
    samples: dict[str, list[float]] = {}

    for _ in range(rounds):
        resampled = [comparisons[rng.randrange(total)] for _ in range(total)]
        ratings = to_elo(fit(resampled), anchor_id, anchor_rating)
        for model, rating in ratings.items():
            samples.setdefault(model, []).append(rating)

    intervals = {}
    for model, values in samples.items():
        values.sort()
        lo = values[max(0, int(0.025 * len(values)))]
        hi = values[min(len(values) - 1, int(0.975 * len(values)))]
        intervals[model] = (lo, hi)
    return intervals
```

- [ ] **Step 4: Verify and commit**

Run: `python -m pytest -q` — expect 179 passed.

```bash
git add ati/bradleyterry.py tests/test_bradleyterry.py
git commit -m "feat: Bradley-Terry fitting with ties and bootstrap intervals"
```

---

### Task 20: Pairwise comparison stage

**Files:**
- Replace: `ati/pairwise.py` (currently a stub)
- Create: `tests/test_pairwise_stage.py`
- Modify: `tests/test_pairwise.py` (the stub tests must go)
- Create: `config/prompts/pairwise.md`, `config/schemas/pairwise.json`

**Interfaces:**
- Produces:
  - `ati.pairwise.comparison_path(root, run_id, a, b, passage_id) -> Path`
  - `ati.pairwise.parse_verdict(raw: str) -> str` — returns `"A"`, `"B"`, or `"tie"`
  - `ati.pairwise.resolve_outcome(forward: str, reverse: str) -> str` — returns `"a"`, `"b"`, or `"tie"`
  - `ati.pairwise.select_pairs(models, ratings, counts, k) -> list[tuple[str, str]]`
  - `ati.pairwise.build_pair_prompt(template, source, text_a, text_b) -> str`
  - `ati.pairwise.compare(judge_runner, template, source, text_a, text_b) -> tuple[str, str, str]`
  - `ati.pairwise.load_comparisons(root, run_id) -> list[tuple[str, str, str]]`
  - `ati.pairwise.run_pairwise(...) -> int` — number of comparisons executed

- [ ] **Step 1: Delete the stub tests**

Delete `tests/test_pairwise.py` entirely — it asserts `NotImplementedError` and the docstring wording, both of which this task replaces.

```bash
git rm tests/test_pairwise.py
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_pairwise_stage.py`:

```python
import json

import pytest

from ati.pairwise import (
    build_pair_prompt,
    comparison_path,
    compare,
    load_comparisons,
    parse_verdict,
    resolve_outcome,
    run_pairwise,
    select_pairs,
)
from ati.runners.fake import FakeRunner
from ati.runstage import run_output_path


def verdict(winner):
    return json.dumps({"winner": winner, "reason": "clearer phrasing"})


def test_parse_verdict_reads_winner():
    assert parse_verdict(verdict("A")) == "A"
    assert parse_verdict(verdict("B")) == "B"
    assert parse_verdict(verdict("tie")) == "tie"


def test_parse_verdict_is_case_insensitive():
    assert parse_verdict(json.dumps({"winner": "a"})) == "A"


def test_parse_verdict_rejects_garbage():
    with pytest.raises(ValueError):
        parse_verdict("the first one I suppose")


def test_resolve_outcome_agrees():
    # forward: A=a wins. reverse: A=b, and B (which is a) wins.
    assert resolve_outcome("A", "B") == "a"
    assert resolve_outcome("B", "A") == "b"


def test_resolve_outcome_treats_a_flip_as_a_tie():
    """If swapping the order swaps the verdict, the judge has no real preference."""
    assert resolve_outcome("A", "A") == "tie"
    assert resolve_outcome("B", "B") == "tie"


def test_resolve_outcome_propagates_explicit_ties():
    assert resolve_outcome("tie", "tie") == "tie"
    assert resolve_outcome("A", "tie") == "tie"


def test_build_pair_prompt_contains_both_translations_and_source():
    p = build_pair_prompt("TEMPLATE", "SOURCE", "ALPHA", "BETA")
    assert "TEMPLATE" in p and "SOURCE" in p
    assert "ALPHA" in p and "BETA" in p


def test_build_pair_prompt_does_not_name_the_models():
    p = build_pair_prompt("T", "S", "ALPHA", "BETA")
    assert "gpt" not in p.lower() and "claude" not in p.lower()


def test_compare_runs_both_directions():
    runner = FakeRunner(responses=[verdict("A"), verdict("B")])
    forward, reverse, outcome = compare(runner, "T", "SRC", "TEXT_A", "TEXT_B")
    assert forward == "A"
    assert reverse == "B"
    assert outcome == "a"
    assert len(runner.calls) == 2


def test_compare_swaps_order_on_the_second_call():
    runner = FakeRunner(responses=[verdict("A"), verdict("B")])
    compare(runner, "T", "SRC", "AAA", "BBB")
    first_prompt = runner.calls[0][1]
    second_prompt = runner.calls[1][1]
    assert first_prompt.index("AAA") < first_prompt.index("BBB")
    assert second_prompt.index("BBB") < second_prompt.index("AAA")


def test_select_pairs_prefers_close_and_rarely_compared():
    models = ["a", "b", "c"]
    ratings = {"a": 1000.0, "b": 1005.0, "c": 1600.0}
    counts = {}
    pairs = select_pairs(models, ratings, counts, k=1)
    assert pairs == [("a", "b")]


def test_select_pairs_deprioritizes_already_compared_pairs():
    models = ["a", "b", "c"]
    ratings = {"a": 1000.0, "b": 1005.0, "c": 1010.0}
    counts = {("a", "b"): 20}
    pairs = select_pairs(models, ratings, counts, k=1)
    assert pairs != [("a", "b")]


def test_select_pairs_is_deterministic():
    models = ["a", "b", "c", "d"]
    ratings = {m: 1000.0 for m in models}
    assert select_pairs(models, ratings, {}, k=3) == select_pairs(
        models, ratings, {}, k=3
    )


def seed_run(tmp_path, model_id, text, passage_id="cn-01"):
    payload = {
        "model_id": model_id,
        "task_id": "t",
        "task_type": "passage",
        "passage_id": passage_id,
        "iteration": 0,
        "text": text,
        "source_chars": 10,
        "latency_s": 1.0,
        "tokens_in": 1,
        "tokens_out": 1,
        "cost_usd": 0.0,
        "refused": False,
        "unavailable": False,
        "error": None,
    }
    p = run_output_path(tmp_path / "runs", "r1", model_id, f"t.{passage_id}.0")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload), encoding="utf-8")


def pairwise_args(tmp_path, runner, **kw):
    args = dict(
        judge_runner=runner,
        run_id="r1",
        runs_root=tmp_path / "runs",
        pairwise_root=tmp_path / "pairwise",
        corpus_root=tmp_path / "corpus",
        passages={},
        template="TEMPLATE",
        models=["m1", "m2"],
        max_comparisons=10,
    )
    args.update(kw)
    return args


def test_run_pairwise_writes_one_file_per_comparison(tmp_path):
    seed_run(tmp_path, "m1", "alpha")
    seed_run(tmp_path, "m2", "beta")
    runner = FakeRunner(responses=[verdict("A"), verdict("B")])
    count = run_pairwise(**pairwise_args(tmp_path, runner))
    assert count == 1
    path = comparison_path(tmp_path / "pairwise", "r1", "m1", "m2", "cn-01")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["outcome"] == "a"
    assert data["forward"] == "A"
    assert data["reverse"] == "B"


def test_run_pairwise_skips_completed_comparisons(tmp_path):
    seed_run(tmp_path, "m1", "alpha")
    seed_run(tmp_path, "m2", "beta")
    run_pairwise(**pairwise_args(tmp_path, FakeRunner(responses=[verdict("A"), verdict("B")])))
    second = FakeRunner(responses=[verdict("B"), verdict("A")])
    count = run_pairwise(**pairwise_args(tmp_path, second))
    assert count == 0
    assert second.calls == []


def test_run_pairwise_stops_cleanly_on_judge_error(tmp_path):
    """A usage limit must lose at most the comparison in flight."""
    seed_run(tmp_path, "m1", "alpha")
    seed_run(tmp_path, "m2", "beta")
    runner = FakeRunner(error="quota exceeded")
    count = run_pairwise(**pairwise_args(tmp_path, runner))
    assert count == 0
    assert not comparison_path(
        tmp_path / "pairwise", "r1", "m1", "m2", "cn-01"
    ).exists()


def test_run_pairwise_respects_max_comparisons(tmp_path):
    for model in ("m1", "m2", "m3"):
        seed_run(tmp_path, model, f"text-{model}")
    runner = FakeRunner(responses=[verdict("A"), verdict("B")] * 10)
    count = run_pairwise(
        **pairwise_args(tmp_path, runner, models=["m1", "m2", "m3"], max_comparisons=2)
    )
    assert count == 2


def test_run_pairwise_ignores_models_with_no_output(tmp_path):
    seed_run(tmp_path, "m1", "alpha")
    runner = FakeRunner(responses=[verdict("A"), verdict("B")])
    count = run_pairwise(**pairwise_args(tmp_path, runner))
    assert count == 0


def test_load_comparisons_reads_back_outcomes(tmp_path):
    seed_run(tmp_path, "m1", "alpha")
    seed_run(tmp_path, "m2", "beta")
    run_pairwise(**pairwise_args(tmp_path, FakeRunner(responses=[verdict("A"), verdict("B")])))
    comparisons = load_comparisons(tmp_path / "pairwise", "r1")
    assert comparisons == [("m1", "m2", "a")]
```

- [ ] **Step 3: Run to confirm failure**

Run: `python -m pytest tests/test_pairwise_stage.py -v`
Expected: FAIL — the stub raises `NotImplementedError` and none of these functions exist.

- [ ] **Step 4: Replace `ati/pairwise.py`**

Overwrite the file completely:

```python
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
        if data.get("task_type") != "passage":
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
```

- [ ] **Step 5: Create the pairwise prompt and schema**

Create `config/prompts/pairwise.md`:

```markdown
You are comparing two English translations of the same passage of web serial
fiction.

Judge which translation a reader of this genre would rather read, weighing:
accuracy to the source, consistency of names and setting terms, preservation
of character voice and register, natural English prose, completeness, and
whether violent or sexual content in the source survives intact.

Ignore length differences that follow from the source. Do not reward padding
or added explanation.

Answer "tie" whenever neither translation is clearly better. Ties are
expected and are not a failure to decide — two renderings of one sentence are
often equally good.
```

Create `config/schemas/pairwise.json`:

```json
{
  "type": "object",
  "properties": {
    "winner": {"type": "string", "enum": ["A", "B", "tie"]},
    "reason": {"type": "string"}
  },
  "required": ["winner", "reason"]
}
```

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest -q` — expect 195 passed (179 + 18 new − 2 deleted stub tests).

```bash
git add ati/pairwise.py config/prompts/pairwise.md config/schemas/pairwise.json tests/test_pairwise_stage.py
git commit -m "feat: pairwise comparison stage with bidirectional judging"
```

---

### Task 21: `ati pairwise` command and report integration

**Files:**
- Modify: `ati/cli.py`, `ati/report.py`, `tests/test_report.py`, `tests/test_cli.py`
- Create: `config/schemas/judgment.json`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_report.py`:

```python
def test_fmt_cost_labels_subscription_zero():
    from ati.report import fmt_cost

    assert fmt_cost(None) == "N/A"
    assert fmt_cost(0.0) == "sub"
    assert fmt_cost(0.0031) == "0.0031"


def test_render_markdown_includes_pairwise_ratings_when_present(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    md = render_markdown(
        run_id="r1",
        results=results,
        drift={},
        comparable=True,
        metadata={"judge_id": "gemini-judge", "date": "2026-08-25"},
        pairwise={"m1": 1240.0},
        pairwise_ci={"m1": (1180.0, 1300.0)},
    )
    assert "Head-to-head" in md
    assert "1240" in md
    assert "1180" in md


def test_render_markdown_omits_pairwise_section_when_absent(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    md = render_markdown(
        "r1", results, {}, True, {"judge_id": "j", "date": "d"}
    )
    assert "Head-to-head" not in md


def test_render_markdown_warns_when_intervals_overlap(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    seed(tmp_path, "m2", "t.cn-01.0", scores={axis: 3 for axis in AXES})
    results = run_aggregate(tmp_path)
    md = render_markdown(
        "r1",
        results,
        {},
        True,
        {"judge_id": "j", "date": "d"},
        pairwise={"m1": 1200.0, "m2": 1190.0},
        pairwise_ci={"m1": (1100.0, 1300.0), "m2": (1090.0, 1290.0)},
    )
    assert "overlap" in md.lower()
```

Append to `tests/test_cli.py`:

```python
def test_pairwise_command_reports_when_nothing_to_compare(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "models.yaml").write_text(
        "- id: gemini-judge\n  runner: agy\n  model: gemini-3.7-flash-high\n",
        encoding="utf-8",
    )
    (tmp_path / "config" / "prompts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "prompts" / "pairwise.md").write_text("T", encoding="utf-8")
    (tmp_path / "corpus").mkdir(exist_ok=True)
    (tmp_path / "corpus" / "manifest.json").write_text(
        json.dumps({"passages": []}), encoding="utf-8"
    )
    rc = main(["pairwise", "--run", "empty", "--judge-model", "gemini-judge"])
    assert rc == 0
    assert "0 comparisons" in capsys.readouterr().out
```

- [ ] **Step 2: Run to confirm failure**

Run: `python -m pytest -q`
Expected: FAIL — `fmt_cost` missing, `render_markdown` takes no `pairwise`, no `pairwise` command.

- [ ] **Step 3: Extend `ati/report.py`**

Add after `fmt`:

```python
def fmt_cost(value: float | None) -> str:
    """Unknown, subscription-covered, and priced are three different things.

    None means the access path reports no cost at all. Zero means the call
    really did bill zero because a subscription covered it. Printing either as
    "0.0000" would read as "free", which misleads the value comparison.
    """
    if value is None:
        return "N/A"
    if value == 0:
        return "sub"
    return f"{value:.4f}"
```

Change `render_markdown`'s signature to accept two optional arguments:

```python
def render_markdown(
    run_id: str,
    results: list[ModelResult],
    drift: dict[str, float],
    comparable: bool,
    metadata: dict,
    pairwise: dict[str, float] | None = None,
    pairwise_ci: dict[str, tuple[float, float]] | None = None,
) -> str:
```

In the ranking table row builder, replace `{fmt(r.cost_per_1k, 4)}` with `{fmt_cost(r.cost_per_1k)}`.

Immediately before the `## Notes` section, insert the head-to-head block:

```python
    if pairwise:
        lines += [
            "",
            "## Head-to-head (Bradley–Terry)",
            "",
            "Ratings from blind pairwise comparison, judged in both orders. "
            "400 points is 10:1 odds; only the offset is arbitrary. Ties are "
            "allowed and a judge that reverses itself when the order flips is "
            "recorded as a tie.",
            "",
            "| Model | Rating | 95% interval |",
            "|---|---|---|",
        ]
        ranked_pairwise = sorted(
            pairwise.items(), key=lambda kv: -kv[1]
        )
        for model_id, rating in ranked_pairwise:
            interval = (pairwise_ci or {}).get(model_id)
            span = (
                f"{interval[0]:.0f} – {interval[1]:.0f}" if interval else "N/A"
            )
            lines.append(f"| {model_id} | {rating:.0f} | {span} |")

        if pairwise_ci and len(ranked_pairwise) > 1:
            top, second = ranked_pairwise[0][0], ranked_pairwise[1][0]
            first_ci, second_ci = pairwise_ci.get(top), pairwise_ci.get(second)
            if first_ci and second_ci and first_ci[0] <= second_ci[1]:
                lines += [
                    "",
                    f"> The intervals for **{top}** and **{second}** overlap, so "
                    "their ordering is not established by this data. More "
                    "comparisons would be needed to separate them.",
                ]
```

- [ ] **Step 4: Extend `ati/cli.py`**

Add imports:

```python
from ati.bradleyterry import bootstrap_ci, fit, to_elo
from ati.pairwise import load_comparisons, run_pairwise
```

Add the command handler:

```python
def cmd_pairwise(args) -> int:
    judge_runner = build_judge_runner(
        args.judge_model,
        args.models,
        args.judge_endpoint,
        Path(args.manual),
        os.environ.get("OPENROUTER_API_KEY"),
    )
    declared = load_models(Path(args.models))
    pool = [m.id for m in declared if m.runner != "agy"]
    if args.model:
        pool = [m for m in pool if m in args.model]

    passages = {p.id: p for p in load_manifest(Path(args.manifest))}
    template = Path(args.pairwise_prompt).read_text(encoding="utf-8")

    existing = load_comparisons(Path(args.pairwise), args.run)
    ratings = to_elo(fit(existing)) if existing else {}

    executed = run_pairwise(
        judge_runner=judge_runner,
        run_id=args.run,
        runs_root=Path(args.runs),
        pairwise_root=Path(args.pairwise),
        corpus_root=Path(args.corpus),
        passages=passages,
        template=template,
        models=pool,
        max_comparisons=args.max_comparisons,
        ratings=ratings,
    )
    print(f"{executed} comparisons executed")

    comparisons = load_comparisons(Path(args.pairwise), args.run)
    if comparisons:
        anchor = next((m.id for m in declared if m.anchor == "weak"), None)
        final = to_elo(fit(comparisons), anchor_id=anchor)
        for model_id, rating in sorted(final.items(), key=lambda kv: -kv[1]):
            print(f"  {rating:7.0f}  {model_id}")
    return 0
```

Add these to the `common` parser:

```python
    common.add_argument("--pairwise", default="pairwise")
    common.add_argument("--pairwise-prompt", default="config/prompts/pairwise.md")
    common.add_argument("--max-comparisons", type=int, default=50)
```

Add the subparser and handler entry:

```python
    sub.add_parser(
        "pairwise", help="blind head-to-head comparison", parents=[common]
    )
```

```python
        "pairwise": cmd_pairwise,
```

In `cmd_report`, load pairwise data and pass it through. Replace the `render_markdown` call with:

```python
    comparisons = load_comparisons(Path(args.pairwise), args.run)
    pairwise_ratings, pairwise_intervals = {}, {}
    if comparisons:
        anchor = next(
            (m.id for m in load_models(Path(args.models)) if m.anchor == "weak"), None
        )
        pairwise_ratings = to_elo(fit(comparisons), anchor_id=anchor)
        pairwise_intervals = bootstrap_ci(comparisons, anchor_id=anchor)

    (results_dir / "leaderboard.md").write_text(
        render_markdown(
            args.run,
            results,
            drift,
            comparable,
            metadata,
            pairwise_ratings,
            pairwise_intervals,
        ),
        encoding="utf-8",
    )
```

- [ ] **Step 5: Create the judgment schema**

Create `config/schemas/judgment.json`:

```json
{
  "type": "object",
  "properties": {
    "scores": {
      "type": "object",
      "properties": {
        "accuracy": {"type": "integer"},
        "terminology": {"type": "integer"},
        "register": {"type": "integer"},
        "fluency": {"type": "integer"},
        "fidelity": {"type": "integer"},
        "sanitization": {"type": "integer"}
      },
      "required": [
        "accuracy", "terminology", "register",
        "fluency", "fidelity", "sanitization"
      ]
    },
    "rationales": {"type": "object"},
    "term_table": {"type": "object"}
  },
  "required": ["scores", "rationales"]
}
```

- [ ] **Step 6: Verify and commit**

Run: `python -m pytest -q` — expect 200 passed.

```bash
git add ati/cli.py ati/report.py config/schemas/judgment.json tests/test_report.py tests/test_cli.py
git commit -m "feat: ati pairwise command and head-to-head report section"
```

---

### Task 22: Configure the opencode-go pool and document the workflow

**Files:**
- Modify: `config/models.yaml`, `config/ati.yaml`, `README.md`, `docs/methodology.md`

- [ ] **Step 1: Write the model pool**

Replace `config/models.yaml` with the opencode-go roster plus the judge. Every entry uses the OpenAI-compatible endpoint verified against the live API.

```yaml
# Models under test: the opencode-go subscription roster.
# Endpoint is OpenAI-compatible; key comes from OPENCODE_API_KEY.
#
# Anchors: qwen3.8-max as the strong reference, deepseek-v4-flash as the weak
# one. They run every time and exist to detect drift, not to be ranked.
- id: glm-5.3
  runner: api
  endpoint: https://opencode.ai/zen/go/v1/chat/completions
  model: glm-5.3
- id: kimi-k3
  runner: api
  endpoint: https://opencode.ai/zen/go/v1/chat/completions
  model: kimi-k3
- id: minimax-m3
  runner: api
  endpoint: https://opencode.ai/zen/go/v1/chat/completions
  model: minimax-m3
- id: grok-4.5
  runner: api
  endpoint: https://opencode.ai/zen/go/v1/chat/completions
  model: grok-4.5
- id: gpt-5.6-luna
  runner: api
  endpoint: https://opencode.ai/zen/go/v1/chat/completions
  model: gpt-5.6-luna
- id: deepseek-v4-pro
  runner: api
  endpoint: https://opencode.ai/zen/go/v1/chat/completions
  model: deepseek-v4-pro
- id: qwen3.7-max
  runner: api
  endpoint: https://opencode.ai/zen/go/v1/chat/completions
  model: qwen3.7-max
- id: qwen3.8-max
  runner: api
  endpoint: https://opencode.ai/zen/go/v1/chat/completions
  model: qwen3.8-max
  anchor: strong
- id: deepseek-v4-flash
  runner: api
  endpoint: https://opencode.ai/zen/go/v1/chat/completions
  model: deepseek-v4-flash
  anchor: weak

# Judge. Gemini through the Antigravity CLI on a subscription.
# --sandbox is deliberate: a judge with tools could look up the source work
# while scoring a translation of it, which stops it measuring the translation.
- id: gemini-judge
  runner: agy
  model: gemini-3.7-flash-high
  json_schema: config/schemas/judgment.json
  sandbox: true
```

- [ ] **Step 2: Point the defaults at the Gemini judge**

Replace `config/ati.yaml`:

```yaml
# Operator defaults. CLI flags override these; these override built-ins.
judge_model: gemini-judge
iterations: 1
```

- [ ] **Step 3: Document the pairwise stage in `docs/methodology.md`**

Add a new section immediately before `## Judge bias`:

```markdown
## Head-to-head ratings

Rubric scores compress: judges cluster absolute ratings in a narrow band, and
near-equal models become indistinguishable. A second pass compares
translations directly and fits a Bradley–Terry model to the results.

Bradley–Terry rather than Elo or Glicko, because those are sequential
algorithms whose output depends on the order matches were played, and a
benchmark holds every comparison at once. The fit is order-independent and
reproducible from the same data.

Controls on each comparison:

- **Both orders.** Every pair is judged twice with the positions swapped. A
  judge that reverses itself is recorded as a tie, not averaged into noise.
- **Ties are allowed.** Two renderings of one sentence are often equally good;
  forcing a winner manufactures noise.
- **No truncation.** EQ-Bench truncates to control length bias in free-form
  creative writing. Translations of one source passage are naturally similar
  in length, and truncating would conceal the omissions the fidelity axis
  exists to catch.
- **Adaptive pairing.** Closest-rated, least-compared pairs are judged first,
  so an interrupted run still produces a meaningful ranking.
- **95% bootstrap intervals** are published with every rating. Where the top
  two intervals overlap, the report says so: the ordering is not established.

Ratings use a scale on which 400 points is 10:1 odds. Only the offset is
arbitrary — it is pinned to the weak calibration anchor so numbers stay
comparable as models join or leave the pool.
```

- [ ] **Step 4: Document the commands in `README.md`**

After the `ati bench` block in `## Usage`, add:

```markdown
Then run the head-to-head pass, which is separate because it is the expensive
part:

```bash
ati pairwise --run 2026-08 --max-comparisons 200
```

Every comparison is checkpointed as it completes. If the judge subscription
hits a usage limit the stage stops cleanly and prints how far it got; rerun
the same command later and it resumes without repeating work. Rerun
`ati report` afterwards to fold the ratings into the leaderboard.
```

- [ ] **Step 5: Verify and commit**

Run: `python -m pytest -q` — expect 200 passed.

```bash
git add config/models.yaml config/ati.yaml README.md docs/methodology.md
git commit -m "feat: configure opencode-go pool and document head-to-head stage"
```
