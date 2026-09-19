# Artificial Translation Index Implementation Plan

> **Superseded design history (August 2026).** This document describes an
> earlier revision and embeds code that has since changed. The code in `ati/`
> and `docs/methodology.md` are authoritative; `docs/runbook.md` is how a run
> is operated today. Kept for the record, not as guidance.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI that runs a fixed translation task suite against configured AI models, scores the outputs with a judge model, and publishes a Markdown leaderboard ranking translation quality for CJK→EN web serial fiction.

**Architecture:** Four disk-backed stages — `fetch` (rebuild local corpus from a pointer manifest), `run` (translate via pluggable runners), `judge` (LLM rubric scoring to structured JSON), `report` (aggregate to scores + Markdown). Each stage is resumable and idempotent: work whose output file already exists is skipped, so adding one model is an O(1) operation. Model access is abstracted behind a `Runner` protocol so API, subscription-CLI, and manual-paste access all work identically.

**Tech Stack:** Python 3.11+, `httpx` (HTTP), `PyYAML` (config), `pytest` (tests). Standard library for everything else.

**Spec:** `docs/superpowers/specs/2026-08-23-translation-index-design.md`

## Global Constraints

- **Python 3.11+.** Development machine has 3.12.10.
- **Dependencies are limited to `httpx`, `PyYAML`, and `pytest`.** No ML libraries. No `pydantic` — validation is hand-written against dataclasses. No `jinja2` — the Markdown report is built with f-strings. (These two are deliberate simplifications of the spec's stated stack, made to keep the dependency surface minimal; the spec's behavior is unchanged.)
- **No source text or translation text is ever committed.** `corpus/passages/`, `runs/`, `judgments/`, and `manual/` are gitignored (already configured). Only `corpus/manifest.json` and `results/` are committed.
- **`cost_usd = None` means unknown, never zero.** It must propagate to "N/A" in every output. Never coerce to `0.0`.
- **No live network calls in tests.** Every test uses fakes or fixtures.
- **Rubric axes are exactly these six, in this order:** `accuracy`, `terminology`, `register`, `fluency`, `fidelity`, `sanitization`.
- **Default weights:** accuracy 0.25, terminology 0.20, fluency 0.15, register 0.15, fidelity 0.15, sanitization 0.10. They must sum to 1.0.
- **Commit after every task.** Conventional commit prefixes (`feat:`, `test:`, `chore:`).

---

## File Structure

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, pytest config |
| `ati/config.py` | Load and validate `models.yaml`, `weights.yaml` |
| `ati/corpus.py` | Manifest load, passage hashing, fetch verification |
| `ati/tasks.py` | Task definitions, loading, iteration expansion |
| `ati/runners/base.py` | `Runner` protocol, `RunResult` dataclass |
| `ati/runners/fake.py` | `FakeRunner` for tests |
| `ati/runners/api.py` | OpenAI-compatible HTTP runner |
| `ati/runners/cli.py` | Subprocess runner for subscription CLIs |
| `ati/runners/manual.py` | File-drop runner for web-only models |
| `ati/runstage.py` | Run-stage orchestration and idempotency |
| `ati/judge.py` | Judge prompt construction, response parsing, retry |
| `ati/judgestage.py` | Judge-stage orchestration |
| `ati/scoring.py` | Axis aggregation, composite, term stability, cost/value |
| `ati/textmetrics.py` | Slop and repetition metrics |
| `ati/report.py` | Anchor drift detection, Markdown rendering |
| `ati/pairwise.py` | Deferred-stage interface stub |
| `ati/cli.py` | Argument parsing, stage dispatch |

---

### Task 1: Project scaffolding and config loading

**Files:**
- Create: `pyproject.toml`, `ati/__init__.py`, `ati/config.py`, `config/models.yaml`, `config/weights.yaml`, `tests/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `ati.config.ModelConfig` dataclass with fields: `id: str`, `runner: str`, `model: str | None`, `endpoint: str | None`, `command: list[str] | None`, `temperature: float`, `anchor: str | None`
  - `ati.config.load_models(path: Path) -> list[ModelConfig]`
  - `ati.config.load_weights(path: Path) -> dict[str, float]`
  - `ati.config.ConfigError(Exception)`
  - `ati.config.AXES: tuple[str, ...]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from ati.config import AXES, ConfigError, load_models, load_weights


def write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_axes_are_the_six_rubric_axes():
    assert AXES == (
        "accuracy",
        "terminology",
        "register",
        "fluency",
        "fidelity",
        "sanitization",
    )


def test_load_models_reads_api_and_cli_entries(tmp_path):
    p = write(
        tmp_path,
        "models.yaml",
        """
- id: openrouter/some-model
  runner: api
  endpoint: https://openrouter.ai/api/v1/chat/completions
  model: vendor/some-model
- id: claude-code
  runner: cli
  command: ["claude", "-p"]
  temperature: 0.0
""",
    )
    models = load_models(p)
    assert [m.id for m in models] == ["openrouter/some-model", "claude-code"]
    assert models[0].runner == "api"
    assert models[0].temperature == 0.3  # default
    assert models[1].command == ["claude", "-p"]
    assert models[1].temperature == 0.0
    assert models[0].anchor is None


def test_load_models_rejects_duplicate_ids(tmp_path):
    p = write(
        tmp_path,
        "models.yaml",
        """
- id: dupe
  runner: cli
  command: ["x"]
- id: dupe
  runner: cli
  command: ["y"]
""",
    )
    with pytest.raises(ConfigError, match="duplicate model id"):
        load_models(p)


def test_load_models_rejects_unknown_runner(tmp_path):
    p = write(tmp_path, "models.yaml", "- id: m\n  runner: telepathy\n")
    with pytest.raises(ConfigError, match="unknown runner"):
        load_models(p)


def test_load_models_rejects_api_entry_without_model(tmp_path):
    p = write(tmp_path, "models.yaml", "- id: m\n  runner: api\n  endpoint: http://x\n")
    with pytest.raises(ConfigError, match="requires 'model'"):
        load_models(p)


def test_load_models_rejects_cli_entry_without_command(tmp_path):
    p = write(tmp_path, "models.yaml", "- id: m\n  runner: cli\n")
    with pytest.raises(ConfigError, match="requires 'command'"):
        load_models(p)


def test_load_models_accepts_anchor_labels(tmp_path):
    p = write(
        tmp_path,
        "models.yaml",
        """
- id: strong
  runner: cli
  command: ["x"]
  anchor: strong
- id: weak
  runner: cli
  command: ["y"]
  anchor: weak
""",
    )
    models = load_models(p)
    assert [m.anchor for m in models] == ["strong", "weak"]


def test_load_models_rejects_bad_anchor_label(tmp_path):
    p = write(
        tmp_path,
        "models.yaml",
        "- id: m\n  runner: cli\n  command: [\"x\"]\n  anchor: medium\n",
    )
    with pytest.raises(ConfigError, match="anchor must be"):
        load_models(p)


def test_load_weights_returns_all_axes(tmp_path):
    p = write(
        tmp_path,
        "weights.yaml",
        """
accuracy: 0.25
terminology: 0.20
register: 0.15
fluency: 0.15
fidelity: 0.15
sanitization: 0.10
""",
    )
    w = load_weights(p)
    assert set(w) == set(AXES)
    assert w["accuracy"] == 0.25


def test_load_weights_rejects_sum_not_one(tmp_path):
    p = write(
        tmp_path,
        "weights.yaml",
        """
accuracy: 0.5
terminology: 0.5
register: 0.5
fluency: 0.15
fidelity: 0.15
sanitization: 0.10
""",
    )
    with pytest.raises(ConfigError, match="must sum to 1.0"):
        load_weights(p)


def test_load_weights_rejects_missing_axis(tmp_path):
    p = write(tmp_path, "weights.yaml", "accuracy: 1.0\n")
    with pytest.raises(ConfigError, match="missing axis"):
        load_weights(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ati'`

- [ ] **Step 3: Create the package scaffolding**

Create `pyproject.toml`:

```toml
[project]
name = "ati"
version = "0.1.0"
description = "Artificial Translation Index - benchmark AI translation of web serial fiction"
requires-python = ">=3.11"
dependencies = ["httpx>=0.27", "PyYAML>=6.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
ati = "ati.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["ati*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Create empty `ati/__init__.py` and `tests/__init__.py`.

- [ ] **Step 4: Write the implementation**

Create `ati/config.py`:

```python
"""Loading and validation of models.yaml and weights.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

AXES: tuple[str, ...] = (
    "accuracy",
    "terminology",
    "register",
    "fluency",
    "fidelity",
    "sanitization",
)

VALID_RUNNERS = ("api", "cli", "manual")
VALID_ANCHORS = ("strong", "weak")
DEFAULT_TEMPERATURE = 0.3


class ConfigError(Exception):
    """Raised when a config file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class ModelConfig:
    id: str
    runner: str
    model: str | None = None
    endpoint: str | None = None
    command: list[str] | None = None
    temperature: float = DEFAULT_TEMPERATURE
    anchor: str | None = None


def load_models(path: Path) -> list[ModelConfig]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ConfigError("models.yaml must contain a list of model entries")

    models: list[ModelConfig] = []
    seen: set[str] = set()
    for entry in raw:
        model_id = entry.get("id")
        if not model_id:
            raise ConfigError("every model entry requires 'id'")
        if model_id in seen:
            raise ConfigError(f"duplicate model id: {model_id}")
        seen.add(model_id)

        runner = entry.get("runner")
        if runner not in VALID_RUNNERS:
            raise ConfigError(f"unknown runner {runner!r} for model {model_id}")
        if runner == "api" and not entry.get("model"):
            raise ConfigError(f"model {model_id}: runner 'api' requires 'model'")
        if runner == "cli" and not entry.get("command"):
            raise ConfigError(f"model {model_id}: runner 'cli' requires 'command'")

        anchor = entry.get("anchor")
        if anchor is not None and anchor not in VALID_ANCHORS:
            raise ConfigError(
                f"model {model_id}: anchor must be 'strong' or 'weak', got {anchor!r}"
            )

        models.append(
            ModelConfig(
                id=model_id,
                runner=runner,
                model=entry.get("model"),
                endpoint=entry.get("endpoint"),
                command=entry.get("command"),
                temperature=float(entry.get("temperature", DEFAULT_TEMPERATURE)),
                anchor=anchor,
            )
        )
    return models


def load_weights(path: Path) -> dict[str, float]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("weights.yaml must contain a mapping of axis to weight")

    for axis in AXES:
        if axis not in raw:
            raise ConfigError(f"weights.yaml missing axis: {axis}")
    extra = set(raw) - set(AXES)
    if extra:
        raise ConfigError(f"weights.yaml has unknown axes: {sorted(extra)}")

    weights = {axis: float(raw[axis]) for axis in AXES}
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"weights must sum to 1.0, got {total}")
    return weights
```

- [ ] **Step 5: Create the real config files**

Create `config/weights.yaml`:

```yaml
# Axis weights for the quality composite. Must sum to 1.0.
# Reweight and rerun `ati report` to recompute rankings without rerunning models.
accuracy: 0.25
terminology: 0.20
register: 0.15
fluency: 0.15
fidelity: 0.15
sanitization: 0.10
```

Create `config/models.yaml`:

```yaml
# Models under test. `anchor: strong` / `anchor: weak` marks the two
# calibration models that run every time to detect drift.
#
# runner: api     -> OpenAI-compatible HTTP endpoint, reports real cost
# runner: cli     -> subprocess; source on stdin, translation on stdout
# runner: manual  -> prompts written to manual/, responses pasted back
- id: claude-code-opus
  runner: cli
  command: ["claude", "-p", "--model", "opus"]
  temperature: 0.3
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pip install -e ".[dev]"` then `python -m pytest tests/test_config.py -v`
Expected: all 11 tests PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml ati/ config/ tests/
git commit -m "feat: project scaffolding and config loading"
```

---

### Task 2: Corpus manifest and hash verification

**Files:**
- Create: `ati/corpus.py`, `corpus/manifest.json`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `ati.config.ConfigError` (not used; corpus has its own error type).
- Produces:
  - `ati.corpus.Passage` dataclass: `id: str`, `lang: str`, `genre: str`, `source_url: str`, `chapter: str`, `start: int`, `end: int`, `sha256: str`, `has_reference: bool`, `series: str | None`, `series_index: int | None`
  - `ati.corpus.load_manifest(path: Path) -> list[Passage]`
  - `ati.corpus.sha256_text(text: str) -> str`
  - `ati.corpus.passage_path(root: Path, passage_id: str) -> Path`
  - `ati.corpus.load_passage(root: Path, passage: Passage) -> str` — raises `HashMismatch` if content does not match
  - `ati.corpus.HashMismatch(Exception)`, `ati.corpus.PassageMissing(Exception)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_corpus.py`:

```python
import json

import pytest

from ati.corpus import (
    HashMismatch,
    PassageMissing,
    load_manifest,
    load_passage,
    passage_path,
    sha256_text,
)


def test_sha256_is_stable_and_utf8_based():
    assert sha256_text("霜降") == sha256_text("霜降")
    assert sha256_text("a") != sha256_text("b")
    assert len(sha256_text("a")) == 64


def test_load_manifest_parses_entries(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "passages": [
                    {
                        "id": "cn-01",
                        "lang": "zh",
                        "genre": "xianxia",
                        "source_url": "https://example.invalid/book/1",
                        "chapter": "1",
                        "start": 0,
                        "end": 800,
                        "sha256": "0" * 64,
                        "has_reference": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    passages = load_manifest(manifest)
    assert len(passages) == 1
    assert passages[0].id == "cn-01"
    assert passages[0].lang == "zh"
    assert passages[0].has_reference is False
    assert passages[0].series is None


def test_load_manifest_rejects_duplicate_passage_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    entry = {
        "id": "dupe",
        "lang": "zh",
        "genre": "x",
        "source_url": "u",
        "chapter": "1",
        "start": 0,
        "end": 1,
        "sha256": "0" * 64,
        "has_reference": False,
    }
    manifest.write_text(json.dumps({"passages": [entry, entry]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate passage id"):
        load_manifest(manifest)


def test_load_passage_returns_text_when_hash_matches(tmp_path):
    from ati.corpus import Passage

    text = "the quick brown fox"
    p = Passage(
        id="ctrl-01",
        lang="en",
        genre="control",
        source_url="u",
        chapter="1",
        start=0,
        end=len(text),
        sha256=sha256_text(text),
        has_reference=True,
    )
    passage_path(tmp_path, p.id).parent.mkdir(parents=True, exist_ok=True)
    passage_path(tmp_path, p.id).write_text(text, encoding="utf-8")
    assert load_passage(tmp_path, p) == text


def test_load_passage_raises_on_hash_mismatch(tmp_path):
    from ati.corpus import Passage

    p = Passage(
        id="cn-01",
        lang="zh",
        genre="x",
        source_url="u",
        chapter="1",
        start=0,
        end=3,
        sha256=sha256_text("original"),
        has_reference=False,
    )
    passage_path(tmp_path, p.id).parent.mkdir(parents=True, exist_ok=True)
    passage_path(tmp_path, p.id).write_text("changed", encoding="utf-8")
    with pytest.raises(HashMismatch, match="cn-01"):
        load_passage(tmp_path, p)


def test_load_passage_raises_when_file_absent(tmp_path):
    from ati.corpus import Passage

    p = Passage(
        id="jp-01",
        lang="ja",
        genre="x",
        source_url="u",
        chapter="1",
        start=0,
        end=3,
        sha256="0" * 64,
        has_reference=False,
    )
    with pytest.raises(PassageMissing, match="jp-01"):
        load_passage(tmp_path, p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ati.corpus'`

- [ ] **Step 3: Write the implementation**

Create `ati/corpus.py`:

```python
"""Corpus manifest handling.

The manifest stores pointers and hashes only - never source text. Passage
text lives in a gitignored local directory rebuilt by `ati fetch`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class HashMismatch(Exception):
    """Local passage text does not match the hash recorded in the manifest."""


class PassageMissing(Exception):
    """Local passage text has not been fetched."""


@dataclass(frozen=True)
class Passage:
    id: str
    lang: str
    genre: str
    source_url: str
    chapter: str
    start: int
    end: int
    sha256: str
    has_reference: bool
    series: str | None = None
    series_index: int | None = None


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> list[Passage]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    passages: list[Passage] = []
    seen: set[str] = set()
    for entry in data.get("passages", []):
        pid = entry["id"]
        if pid in seen:
            raise ValueError(f"duplicate passage id: {pid}")
        seen.add(pid)
        passages.append(
            Passage(
                id=pid,
                lang=entry["lang"],
                genre=entry["genre"],
                source_url=entry["source_url"],
                chapter=str(entry["chapter"]),
                start=int(entry["start"]),
                end=int(entry["end"]),
                sha256=entry["sha256"],
                has_reference=bool(entry["has_reference"]),
                series=entry.get("series"),
                series_index=entry.get("series_index"),
            )
        )
    return passages


def passage_path(root: Path, passage_id: str) -> Path:
    return Path(root) / "passages" / f"{passage_id}.txt"


def load_passage(root: Path, passage: Passage) -> str:
    path = passage_path(root, passage.id)
    if not path.exists():
        raise PassageMissing(
            f"{passage.id}: not fetched. Run `ati fetch` to rebuild the corpus."
        )
    text = path.read_text(encoding="utf-8")
    actual = sha256_text(text)
    if actual != passage.sha256:
        raise HashMismatch(
            f"{passage.id}: local text hash {actual[:12]} does not match "
            f"manifest hash {passage.sha256[:12]}. The source may have changed; "
            f"this passage is unavailable for scoring."
        )
    return text
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_corpus.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Create the empty manifest**

Create `corpus/manifest.json`:

```json
{
  "methodology_version": 1,
  "passages": []
}
```

- [ ] **Step 6: Commit**

```bash
git add ati/corpus.py corpus/manifest.json tests/test_corpus.py
git commit -m "feat: corpus manifest loading and hash verification"
```

---

### Task 3: Runner protocol, fake runner, and API runner

**Files:**
- Create: `ati/runners/__init__.py`, `ati/runners/base.py`, `ati/runners/fake.py`, `ati/runners/api.py`
- Test: `tests/test_runners_api.py`

**Interfaces:**
- Consumes: `ati.config.ModelConfig`
- Produces:
  - `ati.runners.base.RunResult` dataclass: `text: str`, `latency_s: float`, `tokens_in: int | None`, `tokens_out: int | None`, `cost_usd: float | None`, `refused: bool`, `error: str | None`
  - `ati.runners.base.Runner` Protocol with `run(self, system_prompt: str, source_text: str) -> RunResult`
  - `ati.runners.base.REFUSAL_MARKERS: tuple[str, ...]`
  - `ati.runners.base.looks_like_refusal(text: str) -> bool`
  - `ati.runners.fake.FakeRunner(responses: list[str] | None = None, error: str | None = None)`
  - `ati.runners.api.ApiRunner(endpoint: str, model: str, api_key: str, temperature: float, client=None)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_runners_api.py`:

```python
import httpx
import pytest

from ati.runners.api import ApiRunner
from ati.runners.base import RunResult, looks_like_refusal
from ati.runners.fake import FakeRunner


def test_fake_runner_returns_queued_responses():
    r = FakeRunner(responses=["first", "second"])
    assert r.run("sys", "src").text == "first"
    assert r.run("sys", "src").text == "second"


def test_fake_runner_records_calls():
    r = FakeRunner(responses=["x"])
    r.run("SYSTEM", "SOURCE")
    assert r.calls == [("SYSTEM", "SOURCE")]


def test_fake_runner_can_simulate_error():
    r = FakeRunner(error="boom")
    result = r.run("sys", "src")
    assert result.error == "boom"
    assert result.text == ""


def test_looks_like_refusal_detects_common_phrasings():
    assert looks_like_refusal("I can't help with that request.")
    assert looks_like_refusal("I'm unable to translate this content.")
    assert not looks_like_refusal("He drew his sword and cut the man down.")


def test_api_runner_parses_a_successful_response():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "translated text"}}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cost": 0.0012,
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner(
        endpoint="https://example.invalid/v1/chat/completions",
        model="vendor/model",
        api_key="k",
        temperature=0.3,
        client=client,
    )
    result = runner.run("sys", "src")
    assert isinstance(result, RunResult)
    assert result.text == "translated text"
    assert result.tokens_in == 100
    assert result.tokens_out == 50
    assert result.cost_usd == 0.0012
    assert result.error is None
    assert result.latency_s >= 0


def test_api_runner_reports_cost_none_when_usage_omits_it():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "t"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    result = runner.run("sys", "src")
    assert result.cost_usd is None


def test_api_runner_records_http_error_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limited")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    result = runner.run("sys", "src")
    assert result.error is not None
    assert "429" in result.error
    assert result.text == ""


def test_api_runner_flags_refusals():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "I can't help with that."}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    runner = ApiRunner("http://x", "m", "k", 0.3, client=client)
    assert runner.run("sys", "src").refused is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runners_api.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ati.runners'`

- [ ] **Step 3: Write the base module**

Create empty `ati/runners/__init__.py`, then `ati/runners/base.py`:

```python
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
```

- [ ] **Step 4: Write the fake runner**

Create `ati/runners/fake.py`:

```python
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
```

- [ ] **Step 5: Write the API runner**

Create `ati/runners/api.py`:

```python
"""OpenAI-compatible HTTP runner. Used for OpenRouter and direct providers."""

from __future__ import annotations

import time

import httpx

from ati.runners.base import RunResult, looks_like_refusal

TIMEOUT_S = 300.0


class ApiRunner:
    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key: str,
        temperature: float,
        client: httpx.Client | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client = client or httpx.Client(timeout=TIMEOUT_S)

    def run(self, system_prompt: str, source_text: str) -> RunResult:
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": source_text},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}"}
        started = time.monotonic()
        try:
            response = self._client.post(
                self._endpoint, json=payload, headers=headers
            )
        except httpx.HTTPError as exc:
            return RunResult(
                text="", latency_s=time.monotonic() - started, error=str(exc)
            )
        latency = time.monotonic() - started

        if response.status_code != 200:
            return RunResult(
                text="",
                latency_s=latency,
                error=f"HTTP {response.status_code}: {response.text[:200]}",
            )

        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError) as exc:
            return RunResult(
                text="", latency_s=latency, error=f"malformed response: {exc}"
            )

        usage = body.get("usage") or {}
        cost = usage.get("cost")
        return RunResult(
            text=text,
            latency_s=latency,
            tokens_in=usage.get("prompt_tokens"),
            tokens_out=usage.get("completion_tokens"),
            cost_usd=float(cost) if cost is not None else None,
            refused=looks_like_refusal(text),
        )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_runners_api.py -v`
Expected: all 8 tests PASS

- [ ] **Step 7: Commit**

```bash
git add ati/runners/ tests/test_runners_api.py
git commit -m "feat: runner protocol with fake and API implementations"
```

---

### Task 4: CLI and manual runners

**Files:**
- Create: `ati/runners/cli.py`, `ati/runners/manual.py`
- Test: `tests/test_runners_cli.py`, `tests/test_runners_manual.py`

**Interfaces:**
- Consumes: `ati.runners.base.RunResult`, `ati.runners.base.looks_like_refusal`
- Produces:
  - `ati.runners.cli.CliRunner(command: list[str], timeout_s: float = 600.0, runner_fn=None)` — `runner_fn` defaults to `subprocess.run` and exists so tests can inject a fake
  - `ati.runners.manual.ManualRunner(root: Path, model_id: str)`
  - `ati.runners.manual.ResponsePending(Exception)`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_runners_cli.py`:

```python
import subprocess

from ati.runners.cli import CliRunner


def fake_run_factory(stdout="", stderr="", returncode=0, capture=None):
    def fake_run(cmd, **kwargs):
        if capture is not None:
            capture.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            args=cmd, returncode=returncode, stdout=stdout, stderr=stderr
        )

    return fake_run


def test_cli_runner_sends_source_on_stdin_and_reads_stdout():
    capture = []
    runner = CliRunner(
        command=["claude", "-p"],
        runner_fn=fake_run_factory(stdout="translated\n", capture=capture),
    )
    result = runner.run("SYSTEM PROMPT", "SOURCE TEXT")
    assert result.text == "translated"
    assert result.error is None
    cmd, kwargs = capture[0]
    assert cmd == ["claude", "-p"]
    assert "SYSTEM PROMPT" in kwargs["input"]
    assert "SOURCE TEXT" in kwargs["input"]


def test_cli_runner_reports_cost_none():
    runner = CliRunner(command=["x"], runner_fn=fake_run_factory(stdout="t"))
    assert runner.run("s", "t").cost_usd is None


def test_cli_runner_records_nonzero_exit_as_error():
    runner = CliRunner(
        command=["x"],
        runner_fn=fake_run_factory(stdout="", stderr="not logged in", returncode=1),
    )
    result = runner.run("s", "t")
    assert result.error is not None
    assert "not logged in" in result.error
    assert result.text == ""


def test_cli_runner_records_timeout_as_error():
    def timeout_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    runner = CliRunner(command=["x"], timeout_s=1, runner_fn=timeout_run)
    result = runner.run("s", "t")
    assert "timed out" in result.error


def test_cli_runner_flags_refusals():
    runner = CliRunner(
        command=["x"],
        runner_fn=fake_run_factory(stdout="I cannot translate this content."),
    )
    assert runner.run("s", "t").refused is True
```

Create `tests/test_runners_manual.py`:

```python
from ati.runners.manual import ManualRunner


def test_manual_runner_writes_prompt_when_no_response_exists(tmp_path):
    runner = ManualRunner(root=tmp_path, model_id="web-model")
    result = runner.run("SYSTEM", "SOURCE")
    prompt_file = tmp_path / "web-model" / "pending.prompt"
    assert prompt_file.exists()
    assert "SOURCE" in prompt_file.read_text(encoding="utf-8")
    assert result.error == "awaiting manual response"
    assert result.text == ""


def test_manual_runner_reads_pasted_response(tmp_path):
    runner = ManualRunner(root=tmp_path, model_id="web-model")
    runner.set_task_id("cn-01.0")
    (tmp_path / "web-model").mkdir(parents=True, exist_ok=True)
    (tmp_path / "web-model" / "cn-01.0.response").write_text(
        "pasted translation", encoding="utf-8"
    )
    result = runner.run("SYSTEM", "SOURCE")
    assert result.text == "pasted translation"
    assert result.error is None
    assert result.cost_usd is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_runners_cli.py tests/test_runners_manual.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ati.runners.cli'`

- [ ] **Step 3: Write the CLI runner**

Create `ati/runners/cli.py`:

```python
"""Subprocess runner for subscription-backed CLIs (claude -p, opencode run).

The system prompt and source text are concatenated and delivered on stdin,
because CLI entry points vary too much to assume a system-prompt flag.
"""

from __future__ import annotations

import subprocess
import time

from ati.runners.base import RunResult, looks_like_refusal

DEFAULT_TIMEOUT_S = 600.0


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
```

- [ ] **Step 4: Write the manual runner**

Create `ati/runners/manual.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_runners_cli.py tests/test_runners_manual.py -v`
Expected: all 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add ati/runners/cli.py ati/runners/manual.py tests/test_runners_cli.py tests/test_runners_manual.py
git commit -m "feat: CLI and manual runners for subscription and web-only access"
```

---

### Task 5: Task definitions and iteration expansion

**Files:**
- Create: `ati/tasks.py`, `config/tasks.yaml`, `config/prompts/translate.md`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Consumes: `ati.corpus.Passage`
- Produces:
  - `ati.tasks.Task` dataclass: `id: str`, `type: str`, `passage_ids: list[str]`
  - `ati.tasks.TaskInstance` dataclass: `task: Task`, `passage_id: str`, `iteration: int`, `instance_id: str`
  - `ati.tasks.load_tasks(path: Path) -> list[Task]`
  - `ati.tasks.expand(tasks: list[Task], iterations: int) -> list[TaskInstance]`
  - `ati.tasks.TASK_TYPES: tuple[str, ...]` == `("passage", "consistency", "refusal")`
  - `ati.tasks.TaskError(Exception)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tasks.py`:

```python
import pytest

from ati.tasks import TASK_TYPES, Task, TaskError, expand, load_tasks


def test_task_types_are_the_three_from_the_spec():
    assert TASK_TYPES == ("passage", "consistency", "refusal")


def test_load_tasks_parses_entries(tmp_path):
    p = tmp_path / "tasks.yaml"
    p.write_text(
        """
- id: cn-passages
  type: passage
  passages: [cn-01, cn-02]
- id: cn-series
  type: consistency
  passages: [cn-ch01, cn-ch02]
""",
        encoding="utf-8",
    )
    tasks = load_tasks(p)
    assert [t.id for t in tasks] == ["cn-passages", "cn-series"]
    assert tasks[0].type == "passage"
    assert tasks[1].passage_ids == ["cn-ch01", "cn-ch02"]


def test_load_tasks_rejects_unknown_type(tmp_path):
    p = tmp_path / "tasks.yaml"
    p.write_text("- id: x\n  type: poetry\n  passages: [a]\n", encoding="utf-8")
    with pytest.raises(TaskError, match="unknown task type"):
        load_tasks(p)


def test_load_tasks_rejects_empty_passage_list(tmp_path):
    p = tmp_path / "tasks.yaml"
    p.write_text("- id: x\n  type: passage\n  passages: []\n", encoding="utf-8")
    with pytest.raises(TaskError, match="at least one passage"):
        load_tasks(p)


def test_expand_produces_one_instance_per_passage_per_iteration():
    tasks = [Task(id="t", type="passage", passage_ids=["a", "b"])]
    instances = expand(tasks, iterations=3)
    assert len(instances) == 6
    assert instances[0].instance_id == "t.a.0"
    assert instances[-1].instance_id == "t.b.2"
    assert {i.iteration for i in instances} == {0, 1, 2}


def test_expand_rejects_zero_iterations():
    tasks = [Task(id="t", type="passage", passage_ids=["a"])]
    with pytest.raises(TaskError, match="iterations must be at least 1"):
        expand(tasks, iterations=0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tasks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ati.tasks'`

- [ ] **Step 3: Write the implementation**

Create `ati/tasks.py`:

```python
"""Task suite definitions and expansion into per-iteration work items."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

TASK_TYPES: tuple[str, ...] = ("passage", "consistency", "refusal")


class TaskError(Exception):
    """Raised when the task suite is malformed."""


@dataclass(frozen=True)
class Task:
    id: str
    type: str
    passage_ids: list[str]


@dataclass(frozen=True)
class TaskInstance:
    """One passage, translated once. The unit of work for the run stage."""

    task: Task
    passage_id: str
    iteration: int

    @property
    def instance_id(self) -> str:
        return f"{self.task.id}.{self.passage_id}.{self.iteration}"


def load_tasks(path: Path) -> list[Task]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    tasks: list[Task] = []
    seen: set[str] = set()
    for entry in raw:
        task_id = entry.get("id")
        if not task_id:
            raise TaskError("every task entry requires 'id'")
        if task_id in seen:
            raise TaskError(f"duplicate task id: {task_id}")
        seen.add(task_id)

        task_type = entry.get("type")
        if task_type not in TASK_TYPES:
            raise TaskError(f"unknown task type {task_type!r} in task {task_id}")

        passages = entry.get("passages") or []
        if not passages:
            raise TaskError(f"task {task_id} needs at least one passage")

        tasks.append(Task(id=task_id, type=task_type, passage_ids=list(passages)))
    return tasks


def expand(tasks: list[Task], iterations: int) -> list[TaskInstance]:
    if iterations < 1:
        raise TaskError("iterations must be at least 1")
    instances: list[TaskInstance] = []
    for task in tasks:
        for passage_id in task.passage_ids:
            for i in range(iterations):
                instances.append(
                    TaskInstance(task=task, passage_id=passage_id, iteration=i)
                )
    return instances
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tasks.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Create the task suite and translation prompt**

Create `config/tasks.yaml`:

```yaml
# Task suite. Passage ids reference corpus/manifest.json.
# Populate once the corpus is selected - see the spec's deferred decisions.
[]
```

Create `config/prompts/translate.md`:

```markdown
You are translating a chapter of serialized web fiction into English.

Translate the text that follows. Requirements:

- Translate the entire passage. Do not summarize, abridge, or skip content.
- Preserve the author's register and each character's voice, including crude,
  violent, or sexual content where the source contains it.
- Render names, titles, honorifics, and setting-specific terms consistently.
- Produce natural English prose, not a literal gloss.
- Output only the translation. No notes, no preamble, no commentary.
```

- [ ] **Step 6: Commit**

```bash
git add ati/tasks.py config/tasks.yaml config/prompts/translate.md tests/test_tasks.py
git commit -m "feat: task suite definitions and iteration expansion"
```

---

### Task 6: Run stage orchestration and idempotency

**Files:**
- Create: `ati/runstage.py`
- Test: `tests/test_runstage.py`

**Interfaces:**
- Consumes: `ati.config.ModelConfig`, `ati.corpus.Passage`, `ati.corpus.load_passage`, `ati.tasks.TaskInstance`, `ati.runners.base.RunResult`
- Produces:
  - `ati.runstage.run_output_path(runs_root: Path, run_id: str, model_id: str, instance_id: str) -> Path`
  - `ati.runstage.run_model(runner, model_id, instances, passages, corpus_root, system_prompt, runs_root, run_id, force=False) -> int` — returns count of instances executed (skipped ones not counted)

- [ ] **Step 1: Write the failing test**

Create `tests/test_runstage.py`:

```python
import json

from ati.corpus import Passage, passage_path, sha256_text
from ati.runners.fake import FakeRunner
from ati.runstage import run_model, run_output_path
from ati.tasks import Task, TaskInstance


def make_passage(tmp_path, pid, text):
    p = Passage(
        id=pid,
        lang="zh",
        genre="xianxia",
        source_url="u",
        chapter="1",
        start=0,
        end=len(text),
        sha256=sha256_text(text),
        has_reference=False,
    )
    path = passage_path(tmp_path / "corpus", pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return p


def test_run_model_writes_one_file_per_instance(tmp_path):
    passage = make_passage(tmp_path, "cn-01", "源文")
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]
    runner = FakeRunner(responses=["English output"])

    count = run_model(
        runner=runner,
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )

    assert count == 1
    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["text"] == "English output"
    assert data["model_id"] == "m1"
    assert data["task_id"] == "t"
    assert data["task_type"] == "passage"
    assert data["passage_id"] == "cn-01"
    assert data["iteration"] == 0
    assert data["source_chars"] == 2
    assert data["error"] is None


def test_run_model_skips_completed_instances(tmp_path):
    passage = make_passage(tmp_path, "cn-01", "源文")
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]

    first = FakeRunner(responses=["first"])
    run_model(
        runner=first,
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )

    second = FakeRunner(responses=["second"])
    count = run_model(
        runner=second,
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )

    assert count == 0
    assert second.calls == []
    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    assert json.loads(out.read_text(encoding="utf-8"))["text"] == "first"


def test_run_model_force_overwrites(tmp_path):
    passage = make_passage(tmp_path, "cn-01", "源文")
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]
    common = dict(
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )
    run_model(runner=FakeRunner(responses=["first"]), **common)
    count = run_model(runner=FakeRunner(responses=["second"]), force=True, **common)
    assert count == 1
    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    assert json.loads(out.read_text(encoding="utf-8"))["text"] == "second"


def test_run_model_records_runner_error_and_continues(tmp_path):
    p1 = make_passage(tmp_path, "cn-01", "一")
    p2 = make_passage(tmp_path, "cn-02", "二")
    task = Task(id="t", type="passage", passage_ids=["cn-01", "cn-02"])
    instances = [
        TaskInstance(task=task, passage_id="cn-01", iteration=0),
        TaskInstance(task=task, passage_id="cn-02", iteration=0),
    ]
    runner = FakeRunner(error="provider exploded")

    count = run_model(
        runner=runner,
        model_id="m1",
        instances=instances,
        passages={p1.id: p1, p2.id: p2},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )

    assert count == 2  # both attempted, neither aborted the run
    for pid in ("cn-01", "cn-02"):
        out = run_output_path(tmp_path / "runs", "r1", "m1", f"t.{pid}.0")
        assert json.loads(out.read_text(encoding="utf-8"))["error"] == "provider exploded"


def test_run_model_marks_unavailable_when_passage_hash_mismatches(tmp_path):
    passage = make_passage(tmp_path, "cn-01", "原文")
    passage_path(tmp_path / "corpus", "cn-01").write_text("changed", encoding="utf-8")
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]
    runner = FakeRunner(responses=["should not be called"])

    run_model(
        runner=runner,
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )

    assert runner.calls == []
    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["unavailable"] is True
    assert "hash" in data["error"].lower()


def test_run_model_sets_task_id_on_manual_runner(tmp_path):
    passage = make_passage(tmp_path, "cn-01", "源")
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]

    class SpyManual(FakeRunner):
        def __init__(self):
            super().__init__(responses=["x"])
            self.task_ids = []

        def set_task_id(self, task_id):
            self.task_ids.append(task_id)

    runner = SpyManual()
    run_model(
        runner=runner,
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )
    assert runner.task_ids == ["t.cn-01.0"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_runstage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ati.runstage'`

- [ ] **Step 3: Write the implementation**

Create `ati/runstage.py`:

```python
"""Run stage: translate every task instance with one model.

Idempotent by construction - an instance whose output file already exists is
skipped, so adding a model to an existing run costs only that model's calls.
"""

from __future__ import annotations

import json
from pathlib import Path

from ati.corpus import HashMismatch, Passage, PassageMissing, load_passage
from ati.tasks import TaskInstance


def run_output_path(
    runs_root: Path, run_id: str, model_id: str, instance_id: str
) -> Path:
    safe_model = model_id.replace("/", "__")
    return Path(runs_root) / run_id / safe_model / f"{instance_id}.json"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_model(
    runner,
    model_id: str,
    instances: list[TaskInstance],
    passages: dict[str, Passage],
    corpus_root: Path,
    system_prompt: str,
    runs_root: Path,
    run_id: str,
    force: bool = False,
) -> int:
    executed = 0
    for instance in instances:
        out_path = run_output_path(
            runs_root, run_id, model_id, instance.instance_id
        )
        if out_path.exists() and not force:
            continue

        base = {
            "model_id": model_id,
            "task_id": instance.task.id,
            "task_type": instance.task.type,
            "passage_id": instance.passage_id,
            "iteration": instance.iteration,
            "unavailable": False,
        }

        passage = passages.get(instance.passage_id)
        if passage is None:
            _write(
                out_path,
                {
                    **base,
                    "text": "",
                    "source_chars": 0,
                    "latency_s": 0.0,
                    "tokens_in": None,
                    "tokens_out": None,
                    "cost_usd": None,
                    "refused": False,
                    "unavailable": True,
                    "error": f"passage {instance.passage_id} not in manifest",
                },
            )
            executed += 1
            continue

        try:
            source_text = load_passage(corpus_root, passage)
        except (HashMismatch, PassageMissing) as exc:
            _write(
                out_path,
                {
                    **base,
                    "text": "",
                    "source_chars": 0,
                    "latency_s": 0.0,
                    "tokens_in": None,
                    "tokens_out": None,
                    "cost_usd": None,
                    "refused": False,
                    "unavailable": True,
                    "error": str(exc),
                },
            )
            executed += 1
            continue

        # Manual runner needs the instance id to pair prompt and response files.
        set_task_id = getattr(runner, "set_task_id", None)
        if callable(set_task_id):
            set_task_id(instance.instance_id)

        result = runner.run(system_prompt, source_text)
        _write(
            out_path,
            {
                **base,
                "text": result.text,
                "source_chars": len(source_text),
                "latency_s": result.latency_s,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "cost_usd": result.cost_usd,
                "refused": result.refused,
                "error": result.error,
            },
        )
        executed += 1
    return executed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_runstage.py -v`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ati/runstage.py tests/test_runstage.py
git commit -m "feat: idempotent run stage orchestration"
```

---

### Task 7: Judge prompt construction and response parsing

**Files:**
- Create: `ati/judge.py`, `config/prompts/judge.md`, `docs/rubric.md`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `ati.config.AXES`, `ati.runners.base.RunResult`
- Produces:
  - `ati.judge.Judgment` dataclass: `axes: dict[str, int]`, `rationales: dict[str, str]`, `term_table: dict[str, str] | None`
  - `ati.judge.JudgeSchemaError(Exception)`
  - `ati.judge.parse_judgment(raw: str, expect_term_table: bool = False) -> Judgment`
  - `ati.judge.build_judge_prompt(template: str, source_text: str, translation: str, reference: str | None, expect_term_table: bool) -> str`
  - `ati.judge.judge_with_retry(judge_runner, prompt: str, expect_term_table: bool, attempts: int = 3) -> Judgment`
  - `ati.judge.MAX_QUOTED_WORDS: int` == 8
  - `ati.judge.rationale_quote_violations(judgment: Judgment) -> list[str]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge.py`:

```python
import json

import pytest

from ati.config import AXES
from ati.judge import (
    Judgment,
    JudgeSchemaError,
    build_judge_prompt,
    judge_with_retry,
    parse_judgment,
    rationale_quote_violations,
)
from ati.runners.fake import FakeRunner


def valid_payload(**overrides):
    payload = {
        "scores": {axis: 4 for axis in AXES},
        "rationales": {axis: "reads well" for axis in AXES},
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_parse_judgment_reads_scores_and_rationales():
    j = parse_judgment(valid_payload())
    assert j.axes["accuracy"] == 4
    assert j.rationales["fluency"] == "reads well"
    assert j.term_table is None


def test_parse_judgment_tolerates_markdown_fenced_json():
    raw = "Here you go:\n```json\n" + valid_payload() + "\n```\n"
    assert parse_judgment(raw).axes["register"] == 4


def test_parse_judgment_rejects_missing_axis():
    payload = json.loads(valid_payload())
    del payload["scores"]["fidelity"]
    with pytest.raises(JudgeSchemaError, match="missing score for fidelity"):
        parse_judgment(json.dumps(payload))


def test_parse_judgment_rejects_out_of_range_score():
    payload = json.loads(valid_payload())
    payload["scores"]["accuracy"] = 9
    with pytest.raises(JudgeSchemaError, match="out of range"):
        parse_judgment(json.dumps(payload))


def test_parse_judgment_rejects_non_json():
    with pytest.raises(JudgeSchemaError, match="no JSON object"):
        parse_judgment("I'd rather not score this.")


def test_parse_judgment_requires_term_table_when_expected():
    with pytest.raises(JudgeSchemaError, match="term_table"):
        parse_judgment(valid_payload(), expect_term_table=True)


def test_parse_judgment_reads_term_table():
    raw = valid_payload(term_table={"云澈": "Yun Che", "元气": "Profound Energy"})
    j = parse_judgment(raw, expect_term_table=True)
    assert j.term_table == {"云澈": "Yun Che", "元气": "Profound Energy"}


def test_build_judge_prompt_includes_source_translation_and_reference():
    prompt = build_judge_prompt(
        template="TEMPLATE",
        source_text="SOURCE",
        translation="TRANSLATION",
        reference="REFERENCE",
        expect_term_table=False,
    )
    assert "TEMPLATE" in prompt
    assert "SOURCE" in prompt
    assert "TRANSLATION" in prompt
    assert "REFERENCE" in prompt


def test_build_judge_prompt_omits_reference_section_when_absent():
    prompt = build_judge_prompt("T", "S", "X", None, False)
    assert "REFERENCE TRANSLATION" not in prompt


def test_build_judge_prompt_requests_term_table_for_consistency_tasks():
    prompt = build_judge_prompt("T", "S", "X", None, expect_term_table=True)
    assert "term_table" in prompt


def test_judge_with_retry_succeeds_after_one_bad_response():
    runner = FakeRunner(responses=["garbage", valid_payload()])
    j = judge_with_retry(runner, "prompt", expect_term_table=False)
    assert j.axes["accuracy"] == 4
    assert len(runner.calls) == 2


def test_judge_with_retry_raises_after_three_failures():
    runner = FakeRunner(responses=["a", "b", "c"])
    with pytest.raises(JudgeSchemaError):
        judge_with_retry(runner, "prompt", expect_term_table=False)
    assert len(runner.calls) == 3


def test_rationale_quote_violations_flags_long_quoted_spans():
    j = Judgment(
        axes={axis: 3 for axis in AXES},
        rationales={
            **{axis: "fine" for axis in AXES},
            "accuracy": 'renders it as "one two three four five six seven eight nine"',
        },
        term_table=None,
    )
    assert rationale_quote_violations(j) == ["accuracy"]


def test_rationale_quote_violations_allows_short_quotes():
    j = Judgment(
        axes={axis: 3 for axis in AXES},
        rationales={**{axis: "fine" for axis in AXES}, "accuracy": 'uses "sect elder"'},
        term_table=None,
    )
    assert rationale_quote_violations(j) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_judge.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ati.judge'`

- [ ] **Step 3: Write the implementation**

Create `ati/judge.py`:

```python
"""Judge prompt construction and strict parsing of judge responses.

Judge output is validated against a schema rather than trusted: a judge that
returns prose, omits an axis, or invents a score outside 1-5 is a retryable
failure, not a silent zero.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ati.config import AXES

MAX_QUOTED_WORDS = 8

_QUOTE_RE = re.compile(r'"([^"]+)"')


class JudgeSchemaError(Exception):
    """The judge's response did not conform to the required schema."""


@dataclass(frozen=True)
class Judgment:
    axes: dict[str, int]
    rationales: dict[str, str]
    term_table: dict[str, str] | None = None


def _extract_json_object(raw: str) -> dict:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise JudgeSchemaError("no JSON object found in judge response")
    try:
        parsed = json.loads(raw[start : end + 1])
    except json.JSONDecodeError as exc:
        raise JudgeSchemaError(f"no JSON object could be parsed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise JudgeSchemaError("no JSON object found in judge response")
    return parsed


def parse_judgment(raw: str, expect_term_table: bool = False) -> Judgment:
    parsed = _extract_json_object(raw)

    scores = parsed.get("scores")
    if not isinstance(scores, dict):
        raise JudgeSchemaError("judge response has no 'scores' object")

    axes: dict[str, int] = {}
    for axis in AXES:
        if axis not in scores:
            raise JudgeSchemaError(f"missing score for {axis}")
        try:
            value = int(scores[axis])
        except (TypeError, ValueError) as exc:
            raise JudgeSchemaError(f"score for {axis} is not an integer") from exc
        if not 1 <= value <= 5:
            raise JudgeSchemaError(f"score for {axis} is out of range: {value}")
        axes[axis] = value

    raw_rationales = parsed.get("rationales") or {}
    rationales = {axis: str(raw_rationales.get(axis, "")) for axis in AXES}

    term_table = parsed.get("term_table")
    if expect_term_table:
        if not isinstance(term_table, dict) or not term_table:
            raise JudgeSchemaError("expected a non-empty 'term_table' object")
        term_table = {str(k): str(v) for k, v in term_table.items()}
    else:
        term_table = None

    return Judgment(axes=axes, rationales=rationales, term_table=term_table)


def build_judge_prompt(
    template: str,
    source_text: str,
    translation: str,
    reference: str | None,
    expect_term_table: bool,
) -> str:
    parts = [template, "\n\n## SOURCE TEXT\n\n", source_text]
    if reference:
        parts += ["\n\n## REFERENCE TRANSLATION (calibration only, not ground truth)\n\n", reference]
    parts += ["\n\n## TRANSLATION UNDER REVIEW\n\n", translation, "\n\n## REQUIRED OUTPUT\n\n"]

    schema = {
        "scores": {axis: "<integer 1-5>" for axis in AXES},
        "rationales": {axis: "<one short sentence>" for axis in AXES},
    }
    if expect_term_table:
        schema["term_table"] = {
            "<source term>": "<how this translation rendered it>"
        }
        parts.append(
            "Also extract every proper name, title, honorific, and "
            "setting-specific term from the source, mapping it to exactly how "
            "this translation rendered it, as 'term_table'.\n\n"
        )
    parts.append(
        "Reply with a single JSON object and nothing else, matching:\n\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
        + f"\n\nDo not quote more than {MAX_QUOTED_WORDS} consecutive words "
        "of either text in your rationales."
    )
    return "".join(parts)


def judge_with_retry(
    judge_runner, prompt: str, expect_term_table: bool, attempts: int = 3
) -> Judgment:
    last_error: JudgeSchemaError | None = None
    for _ in range(attempts):
        result = judge_runner.run("", prompt)
        if result.error:
            last_error = JudgeSchemaError(f"judge runner error: {result.error}")
            continue
        try:
            return parse_judgment(result.text, expect_term_table=expect_term_table)
        except JudgeSchemaError as exc:
            last_error = exc
    raise last_error or JudgeSchemaError("judge produced no usable response")


def rationale_quote_violations(judgment: Judgment) -> list[str]:
    """Return axes whose rationale quotes more than MAX_QUOTED_WORDS words.

    Enforces the constraint that published output never reproduces source or
    translation text at length.
    """
    offenders = []
    for axis, text in judgment.rationales.items():
        for quoted in _QUOTE_RE.findall(text):
            if len(quoted.split()) > MAX_QUOTED_WORDS:
                offenders.append(axis)
                break
    return offenders
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_judge.py -v`
Expected: all 14 tests PASS

- [ ] **Step 5: Write the rubric and judge template**

Create `docs/rubric.md`:

```markdown
# Rubric

Six axes, each scored 1-5. Anchors below are included verbatim in the judge
prompt so scores mean the same thing across runs and across models.

## accuracy — does the English convey what the source says?

- **1** — Substantial mistranslation. A reader would come away with a wrong
  understanding of events.
- **3** — Broadly correct, with errors in detail: a misread modifier, a
  confused subject, an idiom taken literally.
- **5** — Faithful throughout, including idioms, implication, and subtext.

## terminology — are names and setting terms rendered consistently?

- **1** — The same name or term appears in two or more forms with no reason.
- **3** — Mostly consistent; minor drift in secondary terms.
- **5** — Every name, title, honorific, and setting term is stable.

## register — is voice, formality, and tone preserved?

- **1** — Flattened to one uniform voice; formal and crude speech read alike.
- **3** — Distinguishes characters, but tone slips in places.
- **5** — Each character's voice and the narration's tone survive intact.

## fluency — does it read as natural English prose?

- **1** — Word-order artifacts and calques throughout; reads as raw MTL.
- **3** — Understandable but stilted; a reader notices the translation.
- **5** — Reads as English prose written by a competent author.

## fidelity — absence of omission, addition, and hallucination

- **1** — Whole sentences dropped, or content invented that is not in the source.
- **3** — Minor omissions or small unsupported embellishments.
- **5** — Complete, with nothing added and nothing dropped.

## sanitization — absence of softening or censoring

- **1** — Violent, sexual, or dark content removed or rewritten as inoffensive.
- **3** — Content present but noticeably toned down.
- **5** — Rendered at the source's full intensity.

A hard refusal is not scored on this scale. It is recorded separately in the
refusal rate.
```

Create `config/prompts/judge.md`:

```markdown
You are evaluating a translation of serialized web fiction into English.

Score the translation on six axes, 1 to 5, using these anchors:

**accuracy** — 1: substantial mistranslation. 3: broadly correct with errors
of detail. 5: faithful including idiom, implication, and subtext.

**terminology** — 1: the same name or term appears in conflicting forms.
3: mostly consistent, minor drift. 5: every name and term stable.

**register** — 1: flattened to one uniform voice. 3: distinguishes characters
but tone slips. 5: voice and tone survive intact.

**fluency** — 1: calques and word-order artifacts throughout. 3: understandable
but stilted. 5: reads as competent English prose.

**fidelity** — 1: whole sentences dropped or content invented. 3: minor
omissions or embellishments. 5: complete, nothing added or dropped.

**sanitization** — 1: violent, sexual, or dark content removed or rewritten.
3: present but toned down. 5: rendered at full intensity.

Judge what is in front of you. Where a reference translation is supplied, use
it to calibrate your sense of what a good result looks like — do not penalize
differences in wording that are equally valid.
```

- [ ] **Step 6: Commit**

```bash
git add ati/judge.py config/prompts/judge.md docs/rubric.md tests/test_judge.py
git commit -m "feat: judge prompt construction and strict response parsing"
```

---

### Task 8: Judge stage orchestration

**Files:**
- Create: `ati/judgestage.py`
- Test: `tests/test_judgestage.py`

**Interfaces:**
- Consumes: `ati.judge.judge_with_retry`, `ati.judge.build_judge_prompt`, `ati.judge.JudgeSchemaError`, `ati.judge.rationale_quote_violations`, `ati.runstage.run_output_path`, `ati.corpus.load_passage`
- Produces:
  - `ati.judgestage.judgment_output_path(judgments_root: Path, run_id: str, model_id: str, instance_id: str) -> Path`
  - `ati.judgestage.judge_run(judge_runner, judge_id, run_id, runs_root, judgments_root, passages, corpus_root, template, force=False) -> int`

- [ ] **Step 1: Write the failing test**

Create `tests/test_judgestage.py`:

```python
import json

from ati.config import AXES
from ati.corpus import Passage, passage_path, sha256_text
from ati.judgestage import judge_run, judgment_output_path
from ati.runners.fake import FakeRunner
from ati.runstage import run_output_path


def valid_judge_reply(score=4, term_table=None):
    payload = {
        "scores": {axis: score for axis in AXES},
        "rationales": {axis: "ok" for axis in AXES},
    }
    if term_table is not None:
        payload["term_table"] = term_table
    return json.dumps(payload)


def seed_run(tmp_path, instance_id="t.cn-01.0", task_type="passage", **overrides):
    text = "源文"
    p = Passage(
        id="cn-01",
        lang="zh",
        genre="x",
        source_url="u",
        chapter="1",
        start=0,
        end=len(text),
        sha256=sha256_text(text),
        has_reference=False,
    )
    path = passage_path(tmp_path / "corpus", p.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

    payload = {
        "model_id": "m1",
        "task_id": "t",
        "task_type": task_type,
        "passage_id": "cn-01",
        "iteration": 0,
        "text": "English output",
        "source_chars": len(text),
        "latency_s": 1.0,
        "tokens_in": 10,
        "tokens_out": 5,
        "cost_usd": 0.001,
        "refused": False,
        "unavailable": False,
        "error": None,
    }
    payload.update(overrides)
    out = run_output_path(tmp_path / "runs", "r1", "m1", instance_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return p


def common_args(tmp_path, passage, judge_runner):
    return dict(
        judge_runner=judge_runner,
        judge_id="judge-model",
        run_id="r1",
        runs_root=tmp_path / "runs",
        judgments_root=tmp_path / "judgments",
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        template="TEMPLATE",
    )


def test_judge_run_writes_a_judgment_per_run_file(tmp_path):
    p = seed_run(tmp_path)
    runner = FakeRunner(responses=[valid_judge_reply()])
    count = judge_run(**common_args(tmp_path, p, runner))

    assert count == 1
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["scores"]["accuracy"] == 4
    assert data["judge_id"] == "judge-model"
    assert data["judge_error"] is None


def test_judge_run_skips_already_judged_instances(tmp_path):
    p = seed_run(tmp_path)
    judge_run(**common_args(tmp_path, p, FakeRunner(responses=[valid_judge_reply()])))
    second = FakeRunner(responses=[valid_judge_reply(score=1)])
    count = judge_run(**common_args(tmp_path, p, second))
    assert count == 0
    assert second.calls == []


def test_judge_run_records_judge_error_after_retries(tmp_path):
    p = seed_run(tmp_path)
    runner = FakeRunner(responses=["bad", "bad", "bad"])
    judge_run(**common_args(tmp_path, p, runner))
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["judge_error"] is not None
    assert data["scores"] == {}
    assert len(runner.calls) == 3


def test_judge_run_skips_unavailable_and_errored_runs(tmp_path):
    p = seed_run(tmp_path, unavailable=True, error="hash mismatch", text="")
    runner = FakeRunner(responses=[valid_judge_reply()])
    judge_run(**common_args(tmp_path, p, runner))
    assert runner.calls == []
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["skipped"] is True


def test_judge_run_records_refusal_without_judging(tmp_path):
    p = seed_run(tmp_path, refused=True, text="I can't help with that.")
    runner = FakeRunner(responses=[valid_judge_reply()])
    judge_run(**common_args(tmp_path, p, runner))
    assert runner.calls == []
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["refused"] is True
    assert data["scores"] == {}


def test_judge_run_requests_term_table_for_consistency_tasks(tmp_path):
    p = seed_run(tmp_path, task_type="consistency")
    runner = FakeRunner(
        responses=[valid_judge_reply(term_table={"云澈": "Yun Che"})]
    )
    judge_run(**common_args(tmp_path, p, runner))
    _, prompt = runner.calls[0]
    assert "term_table" in prompt
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["term_table"] == {"云澈": "Yun Che"}


def test_judge_run_flags_overlong_quotes_in_rationales(tmp_path):
    p = seed_run(tmp_path)
    payload = json.loads(valid_judge_reply())
    payload["rationales"]["accuracy"] = 'says "one two three four five six seven eight nine ten"'
    runner = FakeRunner(responses=[json.dumps(payload)])
    judge_run(**common_args(tmp_path, p, runner))
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["quote_violations"] == ["accuracy"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_judgestage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ati.judgestage'`

- [ ] **Step 3: Write the implementation**

Create `ati/judgestage.py`:

```python
"""Judge stage: score every run output, skipping what cannot be scored."""

from __future__ import annotations

import json
from pathlib import Path

from ati.corpus import HashMismatch, Passage, PassageMissing, load_passage
from ati.judge import (
    JudgeSchemaError,
    build_judge_prompt,
    judge_with_retry,
    rationale_quote_violations,
)


def judgment_output_path(
    judgments_root: Path, run_id: str, model_id: str, instance_id: str
) -> Path:
    safe_model = model_id.replace("/", "__")
    return Path(judgments_root) / run_id / safe_model / f"{instance_id}.json"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def judge_run(
    judge_runner,
    judge_id: str,
    run_id: str,
    runs_root: Path,
    judgments_root: Path,
    passages: dict[str, Passage],
    corpus_root: Path,
    template: str,
    force: bool = False,
) -> int:
    run_dir = Path(runs_root) / run_id
    judged = 0
    for run_file in sorted(run_dir.rglob("*.json")):
        run_data = json.loads(run_file.read_text(encoding="utf-8"))
        model_id = run_data["model_id"]
        instance_id = run_file.stem

        out_path = judgment_output_path(
            judgments_root, run_id, model_id, instance_id
        )
        if out_path.exists() and not force:
            continue

        base = {
            "model_id": model_id,
            "task_id": run_data["task_id"],
            "task_type": run_data["task_type"],
            "passage_id": run_data["passage_id"],
            "iteration": run_data["iteration"],
            "judge_id": judge_id,
            "scores": {},
            "rationales": {},
            "term_table": None,
            "refused": bool(run_data.get("refused")),
            "skipped": False,
            "judge_error": None,
            "quote_violations": [],
        }

        # Nothing to judge: the run failed, the passage was unavailable, or
        # the model refused. Each is recorded, none is scored.
        if run_data.get("unavailable") or run_data.get("error"):
            _write(out_path, {**base, "skipped": True})
            judged += 1
            continue
        if run_data.get("refused"):
            _write(out_path, base)
            judged += 1
            continue

        passage = passages.get(run_data["passage_id"])
        if passage is None:
            _write(out_path, {**base, "skipped": True})
            judged += 1
            continue
        try:
            source_text = load_passage(corpus_root, passage)
        except (HashMismatch, PassageMissing):
            _write(out_path, {**base, "skipped": True})
            judged += 1
            continue

        expect_term_table = run_data["task_type"] == "consistency"
        prompt = build_judge_prompt(
            template=template,
            source_text=source_text,
            translation=run_data["text"],
            reference=None,
            expect_term_table=expect_term_table,
        )

        try:
            judgment = judge_with_retry(
                judge_runner, prompt, expect_term_table=expect_term_table
            )
        except JudgeSchemaError as exc:
            _write(out_path, {**base, "judge_error": str(exc)})
            judged += 1
            continue

        _write(
            out_path,
            {
                **base,
                "scores": judgment.axes,
                "rationales": judgment.rationales,
                "term_table": judgment.term_table,
                "quote_violations": rationale_quote_violations(judgment),
            },
        )
        judged += 1
    return judged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_judgestage.py -v`
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ati/judgestage.py tests/test_judgestage.py
git commit -m "feat: judge stage orchestration with skip and error recording"
```

---

### Task 9: Scoring — aggregation, composite, cost, and value

**Files:**
- Create: `ati/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Consumes: `ati.config.AXES`
- Produces:
  - `ati.scoring.AxisStats` dataclass: `mean: float | None`, `stdev: float`, `n: int`
  - `ati.scoring.aggregate_axis(scores: list[int]) -> AxisStats`
  - `ati.scoring.composite(axis_means: dict[str, float | None], weights: dict[str, float]) -> float | None`
  - `ati.scoring.refusal_rate(refusals: list[bool]) -> float`
  - `ati.scoring.cost_per_1k_chars(total_cost: float | None, total_chars: int) -> float | None`
  - `ati.scoring.speed_per_1k_chars(latencies: list[float], chars: list[int]) -> float | None`
  - `ati.scoring.value_score(composite_value: float | None, cost: float | None) -> float | None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scoring.py`:

```python
import pytest

from ati.config import AXES
from ati.scoring import (
    aggregate_axis,
    composite,
    cost_per_1k_chars,
    refusal_rate,
    speed_per_1k_chars,
    value_score,
)


def test_aggregate_axis_computes_mean_and_stdev():
    stats = aggregate_axis([4, 4, 5, 3])
    assert stats.mean == 4.0
    assert stats.n == 4
    # population stdev, matching aggregate_axis: with 2-3 iterations the
    # sample form inflates spread (by ~41% at n=2) and this number describes
    # the runs actually performed, not an inference about a wider population
    assert stats.stdev == pytest.approx(0.7071067812, rel=1e-6)


def test_aggregate_axis_of_single_score_has_zero_stdev():
    stats = aggregate_axis([5])
    assert stats.mean == 5.0
    assert stats.stdev == 0.0


def test_aggregate_axis_with_no_scores_returns_none_mean():
    stats = aggregate_axis([])
    assert stats.mean is None
    assert stats.n == 0
    assert stats.stdev == 0.0


def test_composite_maps_all_fives_to_one_hundred():
    weights = {axis: 1 / len(AXES) for axis in AXES}
    assert composite({axis: 5.0 for axis in AXES}, weights) == pytest.approx(100.0)


def test_composite_maps_all_ones_to_zero():
    weights = {axis: 1 / len(AXES) for axis in AXES}
    assert composite({axis: 1.0 for axis in AXES}, weights) == pytest.approx(0.0)


def test_composite_respects_weights():
    weights = {
        "accuracy": 0.25,
        "terminology": 0.20,
        "register": 0.15,
        "fluency": 0.15,
        "fidelity": 0.15,
        "sanitization": 0.10,
    }
    means = {axis: 3.0 for axis in AXES}
    means["accuracy"] = 5.0
    # 0.25 axis at 1.0 normalized, rest at 0.5 -> 0.25 + 0.75*0.5 = 0.625
    assert composite(means, weights) == pytest.approx(62.5)


def test_composite_renormalizes_when_an_axis_is_excluded():
    weights = {
        "accuracy": 0.25,
        "terminology": 0.20,
        "register": 0.15,
        "fluency": 0.15,
        "fidelity": 0.15,
        "sanitization": 0.10,
    }
    means = {axis: 5.0 for axis in AXES}
    means["sanitization"] = None  # judge errored on this axis
    # Remaining axes all perfect -> still 100, not 90
    assert composite(means, weights) == pytest.approx(100.0)


def test_composite_returns_none_when_every_axis_is_missing():
    weights = {axis: 1 / len(AXES) for axis in AXES}
    assert composite({axis: None for axis in AXES}, weights) is None


def test_refusal_rate_is_a_percentage():
    assert refusal_rate([True, False, False, False]) == 25.0
    assert refusal_rate([]) == 0.0


def test_cost_per_1k_chars():
    assert cost_per_1k_chars(0.02, 4000) == pytest.approx(0.005)


def test_cost_per_1k_chars_is_none_when_cost_unknown():
    assert cost_per_1k_chars(None, 4000) is None


def test_cost_per_1k_chars_is_none_when_no_chars():
    assert cost_per_1k_chars(0.02, 0) is None


def test_speed_per_1k_chars_uses_median():
    assert speed_per_1k_chars([10.0, 20.0, 30.0], [1000, 1000, 1000]) == 20.0


def test_speed_per_1k_chars_is_none_without_data():
    assert speed_per_1k_chars([], []) is None


def test_value_score_is_composite_per_dollar():
    assert value_score(80.0, 0.004) == pytest.approx(20000.0)


def test_value_score_is_none_when_cost_unknown():
    assert value_score(80.0, None) is None


def test_value_score_is_none_when_composite_unknown():
    assert value_score(None, 0.004) is None


def test_value_score_is_none_when_cost_is_zero():
    assert value_score(80.0, 0.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ati.scoring'`

- [ ] **Step 3: Write the implementation**

Create `ati/scoring.py`:

```python
"""Score aggregation: axes to composite, plus the cost and speed metrics."""

from __future__ import annotations

import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class AxisStats:
    mean: float | None
    stdev: float
    n: int


def aggregate_axis(scores: list[int]) -> AxisStats:
    if not scores:
        return AxisStats(mean=None, stdev=0.0, n=0)
    mean = statistics.fmean(scores)
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    return AxisStats(mean=mean, stdev=stdev, n=len(scores))


def composite(
    axis_means: dict[str, float | None], weights: dict[str, float]
) -> float | None:
    """Weighted 0-100 quality score.

    A missing axis - every judgment for it errored - is dropped and the
    remaining weights renormalized, so a judge failure lowers confidence
    rather than silently lowering the score.
    """
    present = {
        axis: mean for axis, mean in axis_means.items() if mean is not None
    }
    if not present:
        return None
    total_weight = sum(weights[axis] for axis in present)
    if total_weight <= 0:
        return None
    normalized = sum(
        weights[axis] * (mean - 1.0) / 4.0 for axis, mean in present.items()
    )
    return 100.0 * normalized / total_weight


def refusal_rate(refusals: list[bool]) -> float:
    if not refusals:
        return 0.0
    return 100.0 * sum(1 for r in refusals if r) / len(refusals)


def cost_per_1k_chars(total_cost: float | None, total_chars: int) -> float | None:
    """None means unknown - subscription access - and never zero."""
    if total_cost is None or total_chars <= 0:
        return None
    return total_cost * 1000.0 / total_chars


def speed_per_1k_chars(
    latencies: list[float], chars: list[int]
) -> float | None:
    rates = [
        latency * 1000.0 / char_count
        for latency, char_count in zip(latencies, chars)
        if char_count > 0
    ]
    if not rates:
        return None
    return statistics.median(rates)


def value_score(
    composite_value: float | None, cost: float | None
) -> float | None:
    """Composite points per dollar per 1k source characters."""
    if composite_value is None or cost is None or cost <= 0:
        return None
    return composite_value / cost
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_scoring.py -v`
Expected: all 18 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ati/scoring.py tests/test_scoring.py
git commit -m "feat: score aggregation, composite, and cost/value metrics"
```

---

### Task 10: Term stability and deterministic text metrics

**Files:**
- Modify: `ati/scoring.py` (append `term_stability`)
- Create: `ati/textmetrics.py`, `config/slop_list.txt`
- Test: `tests/test_term_stability.py`, `tests/test_textmetrics.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `ati.scoring.term_stability(term_tables: list[dict[str, str]]) -> float | None` — returns 0.0–1.0
  - `ati.scoring.stability_to_axis_score(stability: float | None) -> float | None` — maps 0.0–1.0 onto the 1–5 axis scale
  - `ati.textmetrics.load_slop_list(path: Path) -> list[str]`
  - `ati.textmetrics.slop_score(text: str, slop_list: list[str]) -> float` — hits per 1,000 words
  - `ati.textmetrics.repetition_score(text: str) -> float` — 0.0–1.0

- [ ] **Step 1: Write the failing tests**

Create `tests/test_term_stability.py`:

```python
import pytest

from ati.scoring import stability_to_axis_score, term_stability


def test_term_stability_is_one_when_every_term_is_stable():
    tables = [
        {"云澈": "Yun Che", "元气": "Profound Energy"},
        {"云澈": "Yun Che", "元气": "Profound Energy"},
    ]
    assert term_stability(tables) == 1.0


def test_term_stability_counts_only_terms_seen_more_than_once():
    tables = [
        {"云澈": "Yun Che", "只出现一次": "Once Only"},
        {"云澈": "Yun Che"},
    ]
    # the singleton term cannot be inconsistent, so it is excluded
    assert term_stability(tables) == 1.0


def test_term_stability_penalizes_drift():
    tables = [
        {"云澈": "Yun Che", "元气": "Profound Energy"},
        {"云澈": "Cloud Clear", "元气": "Profound Energy"},
    ]
    assert term_stability(tables) == 0.5


def test_term_stability_ignores_case_and_surrounding_space():
    tables = [{"元气": "Profound Energy"}, {"元气": " profound energy "}]
    assert term_stability(tables) == 1.0


def test_term_stability_is_none_without_shared_terms():
    assert term_stability([{"a": "x"}]) is None
    assert term_stability([]) is None


def test_stability_maps_onto_the_five_point_axis_scale():
    assert stability_to_axis_score(1.0) == 5.0
    assert stability_to_axis_score(0.0) == 1.0
    assert stability_to_axis_score(0.5) == 3.0
    assert stability_to_axis_score(None) is None
```

Create `tests/test_textmetrics.py`:

```python
import pytest

from ati.textmetrics import load_slop_list, repetition_score, slop_score


def test_load_slop_list_strips_comments_and_blanks(tmp_path):
    p = tmp_path / "slop.txt"
    p.write_text(
        "# comment\n\na testament to\ncouldn't help but\n", encoding="utf-8"
    )
    assert load_slop_list(p) == ["a testament to", "couldn't help but"]


def test_slop_score_counts_hits_per_1000_words():
    text = " ".join(["word"] * 998) + " a testament to"
    assert slop_score(text, ["a testament to"]) == pytest.approx(1.0, rel=1e-3)


def test_slop_score_is_case_insensitive():
    assert slop_score("A Testament To greatness", ["a testament to"]) > 0


def test_slop_score_counts_repeats():
    text = "a testament to a testament to " + " ".join(["x"] * 994)
    assert slop_score(text, ["a testament to"]) == pytest.approx(2.0, rel=1e-2)


def test_slop_score_is_zero_for_clean_text():
    assert slop_score("he drew his sword", ["a testament to"]) == 0.0


def test_slop_score_is_zero_for_empty_text():
    assert slop_score("", ["a testament to"]) == 0.0


def test_repetition_score_is_higher_for_repetitive_text():
    repetitive = "the sect elder said " * 50
    varied = " ".join(f"word{i}" for i in range(200))
    assert repetition_score(repetitive) > repetition_score(varied)


def test_repetition_score_is_bounded():
    assert 0.0 <= repetition_score("the the the the the") <= 1.0


def test_repetition_score_is_zero_for_empty_text():
    assert repetition_score("") == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_term_stability.py tests/test_textmetrics.py -v`
Expected: FAIL — `ImportError: cannot import name 'term_stability'` and `ModuleNotFoundError: No module named 'ati.textmetrics'`

- [ ] **Step 3: Append term stability to scoring**

Append to `ati/scoring.py`:

```python
def term_stability(term_tables: list[dict[str, str]]) -> float | None:
    """Fraction of recurring source terms rendered identically everywhere.

    Deliberately computed rather than judged: asking a judge to hold ten
    chapters in mind is less reliable than counting the term tables it
    extracted from each chapter separately.

    Terms appearing in only one chapter are excluded - a term used once
    cannot be inconsistent, and counting it inflates the score.
    """
    renderings: dict[str, set[str]] = {}
    for table in term_tables:
        for term, rendering in table.items():
            renderings.setdefault(term, set()).add(rendering.strip().lower())

    counts: dict[str, int] = {}
    for table in term_tables:
        for term in table:
            counts[term] = counts.get(term, 0) + 1

    recurring = [term for term, count in counts.items() if count > 1]
    if not recurring:
        return None
    stable = sum(1 for term in recurring if len(renderings[term]) == 1)
    return stable / len(recurring)


def stability_to_axis_score(stability: float | None) -> float | None:
    """Map 0.0-1.0 stability onto the 1-5 axis scale used by every axis."""
    if stability is None:
        return None
    return 1.0 + 4.0 * stability
```

- [ ] **Step 4: Write the text metrics module**

Create `ati/textmetrics.py`:

```python
"""Deterministic diagnostics computed from output text alone.

No judge is involved, so these cost nothing and are perfectly reproducible.
They are the cheapest defense against a judge that rewards fluent-sounding
filler: a model can score well on the rubric and still read like machine
translation, and these two numbers show it.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

# Digits are part of a word: web serials are full of numerals ("3,000 years",
# "Level 7"). Dropping them also undercounts the denominator in slop_score's
# per-1,000-words normalization, which would inflate every slop rate.
_WORD_RE = re.compile(r"[a-z0-9']+")
_TOP_N = 20


def load_slop_list(path: Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [
        line.strip().lower()
        for line in lines
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def slop_score(text: str, slop_list: list[str]) -> float:
    """Slop-phrase occurrences per 1,000 words."""
    words = _words(text)
    if not words:
        return 0.0
    lowered = text.lower()
    hits = sum(lowered.count(phrase) for phrase in slop_list)
    return hits * 1000.0 / len(words)


def repetition_score(text: str) -> float:
    """Share of output accounted for by its most frequent n-grams.

    Averages the top-20 concentration of unigrams, bigrams, and trigrams.
    Higher means more repetitive. Bounded to 0.0-1.0.
    """
    words = _words(text)
    if not words:
        return 0.0

    shares: list[float] = []
    for n in (1, 2, 3):
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if not grams:
            continue
        top = Counter(grams).most_common(_TOP_N)
        shares.append(sum(count for _, count in top) / len(grams))
    if not shares:
        return 0.0
    return min(1.0, sum(shares) / len(shares))
```

- [ ] **Step 5: Create the slop list**

Create `config/slop_list.txt`:

```text
# Phrases over-represented in LLM output, plus translationese markers common
# in machine-translated web fiction. One phrase per line, lowercase.
# Lines starting with # are comments.
#
# Seeded from the approach used by EQ-Bench's slop score:
# https://eqbench.com/about.html

# --- general LLM tells ---
a testament to
couldn't help but
could not help but
a mix of
a mixture of
sent shivers down
sent a shiver down
barely above a whisper
a mixture of fear and
little did he know
little did she know
little did they know
in that moment
at that moment
it was clear that
a wave of
washed over
eyes narrowed
lips curled
let out a breath
he didn't know
the air was thick with

# --- translationese markers ---
as expected of
is it not
i see, so that's how it is
in other words
that being said
so to speak
at this time
at this moment in time
the current situation
it seems that
it appears that
this humble one
this king
this young master
the so-called
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_term_stability.py tests/test_textmetrics.py -v`
Expected: all 15 tests PASS

- [ ] **Step 7: Commit**

```bash
git add ati/scoring.py ati/textmetrics.py config/slop_list.txt tests/test_term_stability.py tests/test_textmetrics.py
git commit -m "feat: term stability and deterministic slop/repetition metrics"
```

---

### Task 11: Aggregation, anchor drift, and the Markdown report

**Files:**
- Create: `ati/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `ati.config.AXES`, `ati.scoring.*`, `ati.textmetrics.*`
- Produces:
  - `ati.report.ModelResult` dataclass: `model_id`, `composite`, `axis_means`, `axis_stdevs`, `refusal_rate`, `cost_per_1k`, `speed_per_1k`, `value`, `slop`, `repetition`, `judge_errors`, `failed_tasks`, `total_tasks`, `anchor`, `incomplete`
  - `ati.report.FAILURE_THRESHOLD: float` == 0.20
  - `ati.report.ANCHOR_DRIFT_THRESHOLD: float` == 5.0
  - `ati.report.aggregate(run_id, runs_root, judgments_root, weights, slop_list, anchors) -> list[ModelResult]`
  - `ati.report.anchor_drift(current: list[ModelResult], previous_anchor_composites: dict[str, float]) -> dict[str, float]`
  - `ati.report.is_comparable(drift: dict[str, float]) -> bool`
  - `ati.report.render_markdown(run_id, results, drift, comparable, metadata) -> str`
  - `ati.report.fmt(value: float | None, places: int = 1) -> str` — returns `"N/A"` for `None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
import json

import pytest

from ati.config import AXES
from ati.report import (
    ANCHOR_DRIFT_THRESHOLD,
    ModelResult,
    aggregate,
    anchor_drift,
    fmt,
    is_comparable,
    render_markdown,
)
from ati.judgestage import judgment_output_path
from ati.runstage import run_output_path

WEIGHTS = {
    "accuracy": 0.25,
    "terminology": 0.20,
    "register": 0.15,
    "fluency": 0.15,
    "fidelity": 0.15,
    "sanitization": 0.10,
}


def seed(tmp_path, model_id, instance_id, *, scores=None, run=None, judgment=None):
    run_payload = {
        "model_id": model_id,
        "task_id": "t",
        "task_type": "passage",
        "passage_id": "cn-01",
        "iteration": 0,
        "text": "he drew his sword and cut the man down",
        "source_chars": 800,
        "latency_s": 8.0,
        "tokens_in": 500,
        "tokens_out": 400,
        "cost_usd": 0.004,
        "refused": False,
        "unavailable": False,
        "error": None,
    }
    run_payload.update(run or {})
    rp = run_output_path(tmp_path / "runs", "r1", model_id, instance_id)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(run_payload), encoding="utf-8")

    judgment_payload = {
        "model_id": model_id,
        "task_id": run_payload["task_id"],
        "task_type": run_payload["task_type"],
        "passage_id": run_payload["passage_id"],
        "iteration": run_payload["iteration"],
        "judge_id": "judge",
        "scores": scores if scores is not None else {axis: 4 for axis in AXES},
        "rationales": {axis: "ok" for axis in AXES},
        "term_table": None,
        "refused": run_payload["refused"],
        "skipped": False,
        "judge_error": None,
        "quote_violations": [],
    }
    judgment_payload.update(judgment or {})
    jp = judgment_output_path(tmp_path / "judgments", "r1", model_id, instance_id)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(judgment_payload), encoding="utf-8")


def run_aggregate(tmp_path, anchors=None):
    return aggregate(
        run_id="r1",
        runs_root=tmp_path / "runs",
        judgments_root=tmp_path / "judgments",
        weights=WEIGHTS,
        slop_list=["a testament to"],
        anchors=anchors or {},
    )


def test_fmt_renders_none_as_na():
    assert fmt(None) == "N/A"
    assert fmt(3.14159, 2) == "3.14"


def test_aggregate_computes_composite_from_judgments(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    assert len(results) == 1
    assert results[0].model_id == "m1"
    assert results[0].composite == pytest.approx(75.0)


def test_aggregate_averages_iterations_and_reports_stdev(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", scores={axis: 5 for axis in AXES})
    seed(
        tmp_path,
        "m1",
        "t.cn-01.1",
        scores={axis: 3 for axis in AXES},
        run={"iteration": 1},
        judgment={"iteration": 1},
    )
    results = run_aggregate(tmp_path)
    assert results[0].axis_means["accuracy"] == pytest.approx(4.0)
    assert results[0].axis_stdevs["accuracy"] == pytest.approx(1.0)


def test_aggregate_excludes_judge_errored_axes_and_counts_them(tmp_path):
    seed(
        tmp_path,
        "m1",
        "t.cn-01.0",
        judgment={"scores": {}, "judge_error": "schema failure"},
    )
    seed(tmp_path, "m1", "t.cn-01.1", run={"iteration": 1}, judgment={"iteration": 1})
    results = run_aggregate(tmp_path)
    assert results[0].judge_errors == 1
    assert results[0].composite == pytest.approx(75.0)


def test_aggregate_reports_cost_none_for_subscription_models(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", run={"cost_usd": None})
    results = run_aggregate(tmp_path)
    assert results[0].cost_per_1k is None
    assert results[0].value is None


def test_aggregate_never_treats_unknown_cost_as_free(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", run={"cost_usd": None})
    results = run_aggregate(tmp_path)
    assert results[0].cost_per_1k != 0.0


def test_aggregate_computes_refusal_rate(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", run={"refused": True}, judgment={"refused": True, "scores": {}})
    seed(tmp_path, "m1", "t.cn-01.1", run={"iteration": 1}, judgment={"iteration": 1})
    results = run_aggregate(tmp_path)
    assert results[0].refusal_rate == pytest.approx(50.0)


def test_aggregate_marks_model_incomplete_above_failure_threshold(tmp_path):
    for i in range(4):
        seed(
            tmp_path,
            "m1",
            f"t.cn-01.{i}",
            run={"iteration": i, "error": "provider down", "text": ""},
            judgment={"iteration": i, "skipped": True, "scores": {}},
        )
    seed(tmp_path, "m1", "t.cn-01.9", run={"iteration": 9}, judgment={"iteration": 9})
    results = run_aggregate(tmp_path)
    assert results[0].failed_tasks == 4
    assert results[0].incomplete is True


def test_aggregate_leaves_healthy_model_complete(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    assert run_aggregate(tmp_path)[0].incomplete is False


def test_aggregate_sorts_by_composite_descending(tmp_path):
    seed(tmp_path, "low", "t.cn-01.0", scores={axis: 2 for axis in AXES})
    seed(tmp_path, "high", "t.cn-01.0", scores={axis: 5 for axis in AXES})
    assert [r.model_id for r in run_aggregate(tmp_path)] == ["high", "low"]


def test_aggregate_labels_anchor_models(tmp_path):
    seed(tmp_path, "anchor-weak", "t.cn-01.0")
    results = run_aggregate(tmp_path, anchors={"anchor-weak": "weak"})
    assert results[0].anchor == "weak"


def test_anchor_drift_reports_difference_from_previous_run():
    results = [
        ModelResult(
            model_id="a",
            composite=70.0,
            axis_means={},
            axis_stdevs={},
            refusal_rate=0.0,
            cost_per_1k=None,
            speed_per_1k=None,
            value=None,
            slop=0.0,
            repetition=0.0,
            judge_errors=0,
            failed_tasks=0,
            total_tasks=1,
            anchor="strong",
            incomplete=False,
        )
    ]
    assert anchor_drift(results, {"a": 62.0}) == {"a": 8.0}


def test_is_comparable_is_false_above_threshold():
    assert is_comparable({"a": ANCHOR_DRIFT_THRESHOLD + 0.1}) is False
    assert is_comparable({"a": 1.0}) is True
    assert is_comparable({}) is True


def test_render_markdown_contains_ranking_and_na_for_unknown_cost(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0", run={"cost_usd": None})
    results = run_aggregate(tmp_path)
    md = render_markdown(
        run_id="r1",
        results=results,
        drift={},
        comparable=True,
        metadata={"judge_id": "judge", "date": "2026-08-23"},
    )
    assert "m1" in md
    assert "N/A" in md
    assert "Artificial Translation Index" in md
    assert "judge" in md


def test_render_markdown_warns_when_not_comparable(tmp_path):
    seed(tmp_path, "m1", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    md = render_markdown(
        run_id="r1",
        results=results,
        drift={"anchor-strong": 9.0},
        comparable=False,
        metadata={"judge_id": "judge", "date": "2026-08-23"},
    )
    assert "not comparable" in md.lower()


def test_render_markdown_lists_incomplete_models_separately(tmp_path):
    for i in range(4):
        seed(
            tmp_path,
            "broken",
            f"t.cn-01.{i}",
            run={"iteration": i, "error": "down", "text": ""},
            judgment={"iteration": i, "skipped": True, "scores": {}},
        )
    seed(tmp_path, "broken", "t.cn-01.9", run={"iteration": 9}, judgment={"iteration": 9})
    seed(tmp_path, "healthy", "t.cn-01.0")
    results = run_aggregate(tmp_path)
    md = render_markdown("r1", results, {}, True, {"judge_id": "j", "date": "d"})
    assert "Incomplete" in md
    assert "broken" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ati.report'`

- [ ] **Step 3: Write the implementation**

Create `ati/report.py`:

```python
"""Aggregation of run and judgment files into results and a Markdown report."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ati.config import AXES
from ati.scoring import (
    aggregate_axis,
    composite,
    cost_per_1k_chars,
    refusal_rate,
    speed_per_1k_chars,
    stability_to_axis_score,
    term_stability,
    value_score,
)
from ati.textmetrics import repetition_score, slop_score

FAILURE_THRESHOLD = 0.20
ANCHOR_DRIFT_THRESHOLD = 5.0


@dataclass
class ModelResult:
    model_id: str
    composite: float | None
    axis_means: dict[str, float | None]
    axis_stdevs: dict[str, float]
    refusal_rate: float
    cost_per_1k: float | None
    speed_per_1k: float | None
    value: float | None
    slop: float
    repetition: float
    judge_errors: int
    failed_tasks: int
    total_tasks: int
    anchor: str | None = None
    incomplete: bool = False


def fmt(value: float | None, places: int = 1) -> str:
    """Unknown renders as N/A. Never as zero - that would imply 'free'."""
    if value is None:
        return "N/A"
    return f"{value:.{places}f}"


def _load_all(root: Path) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.rglob("*.json"))
    ]


def aggregate(
    run_id: str,
    runs_root: Path,
    judgments_root: Path,
    weights: dict[str, float],
    slop_list: list[str],
    anchors: dict[str, str],
) -> list[ModelResult]:
    runs = _load_all(Path(runs_root) / run_id)
    judgments = _load_all(Path(judgments_root) / run_id)

    by_model_runs: dict[str, list[dict]] = {}
    for r in runs:
        by_model_runs.setdefault(r["model_id"], []).append(r)
    by_model_judgments: dict[str, list[dict]] = {}
    for j in judgments:
        by_model_judgments.setdefault(j["model_id"], []).append(j)

    results: list[ModelResult] = []
    for model_id, model_runs in by_model_runs.items():
        model_judgments = by_model_judgments.get(model_id, [])

        # --- axis scores, averaged across iterations ---
        per_axis: dict[str, list[int]] = {axis: [] for axis in AXES}
        for j in model_judgments:
            for axis, score in (j.get("scores") or {}).items():
                if axis in per_axis:
                    per_axis[axis].append(int(score))
        stats = {axis: aggregate_axis(scores) for axis, scores in per_axis.items()}
        axis_means = {axis: s.mean for axis, s in stats.items()}
        axis_stdevs = {axis: s.stdev for axis, s in stats.items()}

        # --- terminology on consistency tasks is computed, not judged ---
        tables_by_task: dict[str, list[dict]] = {}
        for j in model_judgments:
            if j.get("task_type") == "consistency" and j.get("term_table"):
                tables_by_task.setdefault(j["task_id"], []).append(j["term_table"])
        stabilities = [
            s
            for s in (term_stability(tables) for tables in tables_by_task.values())
            if s is not None
        ]
        if stabilities:
            computed = stability_to_axis_score(sum(stabilities) / len(stabilities))
            axis_means["terminology"] = computed

        # --- diagnostics and operational metrics ---
        scored_runs = [
            r for r in model_runs if not r.get("unavailable") and not r.get("error")
        ]
        texts = [r["text"] for r in scored_runs if r["text"]]
        joined = "\n".join(texts)

        costs = [r.get("cost_usd") for r in scored_runs]
        total_cost = None if any(c is None for c in costs) or not costs else sum(costs)
        total_chars = sum(r.get("source_chars", 0) for r in scored_runs)

        comp = composite(axis_means, weights)
        cost_1k = cost_per_1k_chars(total_cost, total_chars)
        failed = sum(
            1 for r in model_runs if r.get("unavailable") or r.get("error")
        )

        results.append(
            ModelResult(
                model_id=model_id,
                composite=comp,
                axis_means=axis_means,
                axis_stdevs=axis_stdevs,
                refusal_rate=refusal_rate(
                    [bool(r.get("refused")) for r in model_runs]
                ),
                cost_per_1k=cost_1k,
                speed_per_1k=speed_per_1k_chars(
                    [r.get("latency_s", 0.0) for r in scored_runs],
                    [r.get("source_chars", 0) for r in scored_runs],
                ),
                value=value_score(comp, cost_1k),
                slop=slop_score(joined, slop_list),
                repetition=repetition_score(joined),
                judge_errors=sum(
                    1 for j in model_judgments if j.get("judge_error")
                ),
                failed_tasks=failed,
                total_tasks=len(model_runs),
                anchor=anchors.get(model_id),
                incomplete=bool(model_runs)
                and failed / len(model_runs) > FAILURE_THRESHOLD,
            )
        )

    results.sort(key=lambda r: (r.composite is None, -(r.composite or 0.0)))
    return results


def anchor_drift(
    current: list[ModelResult], previous_anchor_composites: dict[str, float]
) -> dict[str, float]:
    drift: dict[str, float] = {}
    for result in current:
        if result.anchor is None or result.composite is None:
            continue
        previous = previous_anchor_composites.get(result.model_id)
        if previous is None:
            continue
        drift[result.model_id] = result.composite - previous
    return drift


def is_comparable(drift: dict[str, float]) -> bool:
    return all(abs(d) <= ANCHOR_DRIFT_THRESHOLD for d in drift.values())


def render_markdown(
    run_id: str,
    results: list[ModelResult],
    drift: dict[str, float],
    comparable: bool,
    metadata: dict,
) -> str:
    ranked = [r for r in results if not r.incomplete]
    incomplete = [r for r in results if r.incomplete]

    lines: list[str] = [
        "# Artificial Translation Index",
        "",
        f"Run `{run_id}` — {metadata.get('date', '')}",
        "",
        f"Judge: `{metadata.get('judge_id', '')}`. "
        "Scores are LLM-judged against a fixed rubric; see "
        "[the rubric](../docs/rubric.md) and "
        "[the methodology](../docs/methodology.md).",
        "",
    ]

    if not comparable:
        lines += [
            "> **Not comparable to previous runs.** A calibration anchor moved "
            f"by more than {ANCHOR_DRIFT_THRESHOLD} composite points "
            f"({', '.join(f'{k}: {v:+.1f}' for k, v in drift.items())}), which "
            "means the judge, the prompt, or a provider changed behavior. "
            "Compare within this run only.",
            "",
        ]

    lines += [
        "## Ranking",
        "",
        "| # | Model | Composite | Refusal % | Cost /1k ch | Sec /1k ch | Value | Slop | Repetition |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(ranked, start=1):
        label = r.model_id + (f" _{r.anchor} anchor_" if r.anchor else "")
        lines.append(
            f"| {i} | {label} | {fmt(r.composite)} | {fmt(r.refusal_rate)} | "
            f"{fmt(r.cost_per_1k, 4)} | {fmt(r.speed_per_1k)} | "
            f"{fmt(r.value, 0)} | {fmt(r.slop, 2)} | {fmt(r.repetition, 3)} |"
        )

    lines += ["", "## Per-axis scores", "", "| Model | " + " | ".join(AXES) + " |"]
    lines.append("|---" * (len(AXES) + 1) + "|")
    for r in ranked:
        cells = " | ".join(
            f"{fmt(r.axis_means.get(axis))} ±{fmt(r.axis_stdevs.get(axis, 0.0))}"
            for axis in AXES
        )
        lines.append(f"| {r.model_id} | {cells} |")

    if incomplete:
        lines += [
            "",
            "## Incomplete",
            "",
            f"These models failed more than {int(FAILURE_THRESHOLD * 100)}% of "
            "their tasks and are excluded from the ranking.",
            "",
            "| Model | Failed | Total |",
            "|---|---|---|",
        ]
        for r in incomplete:
            lines.append(f"| {r.model_id} | {r.failed_tasks} | {r.total_tasks} |")

    lines += [
        "",
        "## Notes",
        "",
        "- **Cost N/A** means access was subscription-backed and per-call cost "
        "is unknown. It does not mean free.",
        "- **Slop** counts LLM-tell and translationese phrases per 1,000 words. "
        "**Repetition** is top-n-gram concentration. Neither enters the "
        "composite; both are diagnostics.",
        "- Model behavior changes under a stable name. These results describe "
        f"the versions tested on {metadata.get('date', 'the run date')}.",
        "",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py -v`
Expected: all 16 tests PASS

- [ ] **Step 5: Commit**

```bash
git add ati/report.py tests/test_report.py
git commit -m "feat: aggregation, anchor drift detection, and Markdown report"
```

---

### Task 12: CLI wiring, fetch, pairwise stub, and methodology doc

**Files:**
- Create: `ati/cli.py`, `ati/pairwise.py`, `docs/methodology.md`, `docs/spot-checks.md`, `README.md`
- Test: `tests/test_cli.py`, `tests/test_pairwise.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `ati.cli.main(argv: list[str] | None = None) -> int`
  - `ati.cli.build_runner(model: ModelConfig, api_key: str | None, manual_root: Path) -> Runner`
  - `ati.cli.cmd_fetch`, `cmd_run`, `cmd_judge`, `cmd_report`
  - `ati.pairwise.PairwiseNotImplemented(NotImplementedError)`
  - `ati.pairwise.rank_pairwise(...) -> dict[str, float]` — raises

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pairwise.py`:

```python
import pytest

from ati.pairwise import PairwiseNotImplemented, rank_pairwise


def test_rank_pairwise_is_a_documented_stub():
    with pytest.raises(PairwiseNotImplemented, match="deferred"):
        rank_pairwise(judge_runner=None, run_id="r1", runs_root=None, top_n=5)


def test_stub_documents_the_intended_design():
    doc = rank_pairwise.__doc__ or ""
    for expected in ("bidirectional", "truncat", "Glicko", "anchor"):
        assert expected.lower() in doc.lower()
```

Create `tests/test_cli.py`:

```python
import json
from pathlib import Path

import pytest

from ati.cli import build_runner, main
from ati.config import ModelConfig
from ati.runners.api import ApiRunner
from ati.runners.cli import CliRunner
from ati.runners.manual import ManualRunner


def test_build_runner_returns_api_runner_for_api_models(tmp_path):
    m = ModelConfig(
        id="m", runner="api", model="vendor/m", endpoint="http://x", temperature=0.3
    )
    assert isinstance(build_runner(m, "key", tmp_path), ApiRunner)


def test_build_runner_returns_cli_runner_for_cli_models(tmp_path):
    m = ModelConfig(id="m", runner="cli", command=["echo"])
    assert isinstance(build_runner(m, None, tmp_path), CliRunner)


def test_build_runner_returns_manual_runner_for_manual_models(tmp_path):
    m = ModelConfig(id="m", runner="manual")
    assert isinstance(build_runner(m, None, tmp_path), ManualRunner)


def test_main_with_no_args_prints_usage_and_returns_nonzero(capsys):
    assert main([]) != 0
    assert "usage" in capsys.readouterr().out.lower()


def test_main_unknown_command_returns_nonzero(capsys):
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_report_command_writes_leaderboard(tmp_path, monkeypatch):
    # minimal run + judgment so report has something to aggregate
    from ati.config import AXES
    from ati.judgestage import judgment_output_path
    from ati.runstage import run_output_path

    run_payload = {
        "model_id": "m1",
        "task_id": "t",
        "task_type": "passage",
        "passage_id": "cn-01",
        "iteration": 0,
        "text": "output text",
        "source_chars": 800,
        "latency_s": 5.0,
        "tokens_in": 1,
        "tokens_out": 1,
        "cost_usd": 0.004,
        "refused": False,
        "unavailable": False,
        "error": None,
    }
    rp = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(run_payload), encoding="utf-8")

    jp = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(
        json.dumps(
            {
                "model_id": "m1",
                "task_id": "t",
                "task_type": "passage",
                "passage_id": "cn-01",
                "iteration": 0,
                "judge_id": "judge",
                "scores": {axis: 4 for axis in AXES},
                "rationales": {axis: "ok" for axis in AXES},
                "term_table": None,
                "refused": False,
                "skipped": False,
                "judge_error": None,
                "quote_violations": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    Path(tmp_path / "config").mkdir(exist_ok=True)
    Path(tmp_path / "config" / "weights.yaml").write_text(
        "accuracy: 0.25\nterminology: 0.20\nregister: 0.15\n"
        "fluency: 0.15\nfidelity: 0.15\nsanitization: 0.10\n",
        encoding="utf-8",
    )
    Path(tmp_path / "config" / "models.yaml").write_text(
        '- id: m1\n  runner: cli\n  command: ["x"]\n', encoding="utf-8"
    )
    Path(tmp_path / "config" / "slop_list.txt").write_text("a testament to\n", encoding="utf-8")

    assert main(["report", "--run", "r1"]) == 0
    leaderboard = tmp_path / "results" / "leaderboard.md"
    assert leaderboard.exists()
    assert "m1" in leaderboard.read_text(encoding="utf-8")
    assert (tmp_path / "results" / "r1.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py tests/test_pairwise.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ati.cli'`

- [ ] **Step 3: Write the pairwise stub**

Create `ati/pairwise.py`:

```python
"""Pairwise tie-breaking stage. Deferred - interface only.

Recorded here so v1's data format does not need to change when this ships.

Intended design:

- Only the top N models by composite (default 5) are paired, on passage
  tasks only.
- The judge sees two anonymized translations of the same source passage and
  returns a winner plus a margin of 1-5.
- Every pairing runs **bidirectionally** (A vs B, then B vs A) and the
  results are averaged, cancelling position bias.
- Outputs are **truncated** to a fixed character count before comparison,
  cancelling length bias.
- Ratings use **Glicko weighted by win margin**, with the two calibration
  **anchor** models pinned to fixed ratings so that adding a model does not
  shift the existing field.

Pairwise output is a separate rating column in the report. It is never
merged into the rubric composite.
"""

from __future__ import annotations


class PairwiseNotImplemented(NotImplementedError):
    """Raised because the pairwise stage is deferred past v1."""


def rank_pairwise(judge_runner, run_id: str, runs_root, top_n: int = 5) -> dict:
    """Return model_id -> Glicko rating. See module docstring for the design.

    Bidirectional comparison, truncated outputs, Glicko with margin
    weighting, anchor models pinned.
    """
    raise PairwiseNotImplemented(
        "The pairwise stage is deferred past v1. See ati/pairwise.py for the "
        "intended design."
    )
```

- [ ] **Step 4: Write the CLI**

Create `ati/cli.py`:

```python
"""Command line entry point: fetch, run, judge, report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import httpx

from ati.config import ModelConfig, load_models, load_weights
from ati.corpus import (
    HashMismatch,
    PassageMissing,
    load_manifest,
    load_passage,
    passage_path,
    sha256_text,
)
from ati.judgestage import judge_run
from ati.report import aggregate, anchor_drift, is_comparable, render_markdown
from ati.runners.api import ApiRunner
from ati.runners.cli import CliRunner
from ati.runners.manual import ManualRunner
from ati.runstage import run_model
from ati.tasks import expand, load_tasks
from ati.textmetrics import load_slop_list

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def build_runner(model: ModelConfig, api_key: str | None, manual_root: Path):
    if model.runner == "api":
        return ApiRunner(
            endpoint=model.endpoint or DEFAULT_ENDPOINT,
            model=model.model or model.id,
            api_key=api_key or "",
            temperature=model.temperature,
        )
    if model.runner == "cli":
        return CliRunner(command=model.command or [])
    return ManualRunner(root=manual_root, model_id=model.id)


def cmd_fetch(args) -> int:
    """Verify local corpus against the manifest.

    Automatic downloading is deliberately not implemented: sources vary and
    scraping them is the operator's decision, not the tool's. This reports
    what is missing or changed so the operator can act.
    """
    passages = load_manifest(Path(args.manifest))
    missing, changed, ok = [], [], 0
    for p in passages:
        try:
            load_passage(Path(args.corpus), p)
            ok += 1
        except PassageMissing:
            missing.append(p)
        except HashMismatch:
            changed.append(p)

    print(f"verified: {ok}/{len(passages)}")
    for p in missing:
        print(f"  MISSING  {p.id}  {p.source_url} chapter {p.chapter} "
              f"chars {p.start}-{p.end} -> {passage_path(Path(args.corpus), p.id)}")
    for p in changed:
        print(f"  CHANGED  {p.id}  source no longer matches manifest hash; "
              "this passage is unavailable for scoring")
    return 0 if not missing and not changed else 1


def cmd_run(args) -> int:
    models = load_models(Path(args.models))
    if args.model:
        models = [m for m in models if m.id in args.model]
    tasks = load_tasks(Path(args.tasks))
    if args.task:
        tasks = [t for t in tasks if t.id in args.task]
    instances = expand(tasks, iterations=args.iterations)
    passages = {p.id: p for p in load_manifest(Path(args.manifest))}
    system_prompt = Path(args.prompt).read_text(encoding="utf-8")
    api_key = os.environ.get("OPENROUTER_API_KEY")

    total = 0
    for model in models:
        runner = build_runner(model, api_key, Path(args.manual))
        executed = run_model(
            runner=runner,
            model_id=model.id,
            instances=instances,
            passages=passages,
            corpus_root=Path(args.corpus),
            system_prompt=system_prompt,
            runs_root=Path(args.runs),
            run_id=args.run,
            force=args.force,
        )
        print(f"{model.id}: {executed} executed, {len(instances) - executed} skipped")
        total += executed
    print(f"total executed: {total}")
    return 0


def cmd_judge(args) -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    judge_runner = ApiRunner(
        endpoint=args.judge_endpoint,
        model=args.judge_model,
        api_key=api_key or "",
        temperature=0.0,
    )
    passages = {p.id: p for p in load_manifest(Path(args.manifest))}
    template = Path(args.judge_prompt).read_text(encoding="utf-8")
    judged = judge_run(
        judge_runner=judge_runner,
        judge_id=args.judge_model,
        run_id=args.run,
        runs_root=Path(args.runs),
        judgments_root=Path(args.judgments),
        passages=passages,
        corpus_root=Path(args.corpus),
        template=template,
        force=args.force,
    )
    print(f"judged: {judged}")
    return 0


def cmd_report(args) -> int:
    weights = load_weights(Path(args.weights))
    slop_list = load_slop_list(Path(args.slop))
    anchors = {
        m.id: m.anchor for m in load_models(Path(args.models)) if m.anchor
    }

    results = aggregate(
        run_id=args.run,
        runs_root=Path(args.runs),
        judgments_root=Path(args.judgments),
        weights=weights,
        slop_list=slop_list,
        anchors=anchors,
    )

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    previous: dict[str, float] = {}
    prev_path = results_dir / "anchors.json"
    if prev_path.exists():
        previous = json.loads(prev_path.read_text(encoding="utf-8"))

    drift = anchor_drift(results, previous)
    comparable = is_comparable(drift)
    metadata = {"judge_id": args.judge_model, "date": date.today().isoformat()}

    (results_dir / f"{args.run}.json").write_text(
        json.dumps(
            {
                "run_id": args.run,
                "metadata": metadata,
                "weights": weights,
                "anchor_drift": drift,
                "comparable": comparable,
                "results": [asdict(r) for r in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (results_dir / "leaderboard.md").write_text(
        render_markdown(args.run, results, drift, comparable, metadata),
        encoding="utf-8",
    )
    prev_path.write_text(
        json.dumps(
            {
                r.model_id: r.composite
                for r in results
                if r.anchor and r.composite is not None
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {results_dir / f'{args.run}.json'}")
    print(f"wrote {results_dir / 'leaderboard.md'}")
    if not comparable:
        print("WARNING: anchor drift exceeds threshold; run is not comparable "
              "to previous runs")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Shared options live on a parent parser so they are accepted both before
    # and after the subcommand: `ati --run r1 report` and `ati report --run r1`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", default="corpus/manifest.json")
    common.add_argument("--corpus", default="corpus")
    common.add_argument("--models", default="config/models.yaml")
    common.add_argument("--tasks", default="config/tasks.yaml")
    common.add_argument("--weights", default="config/weights.yaml")
    common.add_argument("--slop", default="config/slop_list.txt")
    common.add_argument("--prompt", default="config/prompts/translate.md")
    common.add_argument("--judge-prompt", default="config/prompts/judge.md")
    common.add_argument("--judge-model", default="anthropic/claude-sonnet-5")
    common.add_argument("--judge-endpoint", default=DEFAULT_ENDPOINT)
    common.add_argument("--runs", default="runs")
    common.add_argument("--judgments", default="judgments")
    common.add_argument("--results", default="results")
    common.add_argument("--manual", default="manual")
    common.add_argument("--run", default="latest", help="run id")
    common.add_argument("--force", action="store_true")

    parser = argparse.ArgumentParser(
        prog="ati",
        description="Artificial Translation Index",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "fetch",
        help="verify the local corpus against the manifest",
        parents=[common],
    )

    run_parser = sub.add_parser(
        "run", help="translate every task instance", parents=[common]
    )
    run_parser.add_argument("--model", action="append", help="limit to model id")
    run_parser.add_argument("--task", action="append", help="limit to task id")
    run_parser.add_argument("--iterations", type=int, default=1)

    sub.add_parser("judge", help="score run outputs", parents=[common])
    sub.add_parser(
        "report", help="aggregate into results and leaderboard", parents=[common]
    )

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_usage()
        return 2

    handlers = {
        "fetch": cmd_fetch,
        "run": cmd_run,
        "judge": cmd_judge,
        "report": cmd_report,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_cli.py tests/test_pairwise.py -v`
Expected: all 8 tests PASS

- [ ] **Step 6: Write the methodology docs and README**

Create `docs/methodology.md`:

```markdown
# Methodology

## What this measures

How well AI models translate serialized web fiction from Chinese, Japanese,
and Korean into English. It is aimed at people translating works that have no
official translation.

## What this publishes

Rankings and methodology. No source text and no translation text is
published or committed — the repository stores only pointers and hashes, and
`ati fetch` rebuilds a corpus locally for anyone reproducing a run.

## Pipeline

1. `ati fetch` — verify the local corpus against `corpus/manifest.json`.
2. `ati run` — every model translates every passage, once per iteration.
3. `ati judge` — a pinned judge model scores each output 1–5 on six axes.
4. `ati report` — aggregate into `results/` and `results/leaderboard.md`.

Stages are idempotent. Adding one model costs only that model's calls.

## Axes and weights

Six axes, defined with anchors in [the rubric](rubric.md): accuracy,
terminology, register, fluency, fidelity, sanitization. Weights live in
`config/weights.yaml` and are published with every run — reweight and rerun
`ati report` to recompute the ranking without rerunning any model.

Terminology on consistency tasks is **computed, not judged**: the judge
extracts a term table per chapter and the harness measures how many recurring
terms were rendered identically across all of them.

## What is reported outside the composite

Refusal rate, cost per 1,000 source characters, seconds per 1,000 source
characters, value, slop, and repetition. These sit beside the composite
rather than inside it, because their importance depends entirely on the
reader — censorship is disqualifying for some and irrelevant to others.

**Cost N/A means unknown, not free.** Subscription-backed access reports no
per-call cost, so those models rank on quality but not on value.

## Calibration anchors

One known-strong and one known-weak model run in every evaluation. They are
not there to be ranked — they detect drift. If an anchor's composite moves
more than 5 points between runs, the judge, the prompt, or a provider has
changed behavior, and the report marks the run **not comparable** to earlier
runs rather than charting them together.

## Judge bias

A single pinned judge, temperature 0, structured JSON output validated
against a schema; malformed responses are retried twice and then excluded
rather than scored as zero. The judge never evaluates its own outputs.

Self-preference is **disclosed, not corrected**: the judge's model family is
named in every report so readers can discount accordingly. The top 5 models
are re-judged by a secondary judge and the agreement rate is published.
Human spot-checks are recorded in [spot-checks.md](spot-checks.md).

## Known limitations

- LLM judges compress absolute scores toward the middle. Anchors detect
  drift but do not fix compression; pairwise tie-breaking, which would, is
  deferred (see `ati/pairwise.py`).
- Refusal detection is a keyword heuristic over the start of an output.
- Model behavior changes under a stable name. Every run records its date.

## Prior art

The design of the anchors, the deterministic slop and repetition metrics, and
the bias controls planned for the pairwise stage follow
[EQ-Bench Creative Writing v3](https://eqbench.com/creative_writing.html),
which does the same job for English creative writing.
```

Create `docs/spot-checks.md`:

```markdown
# Human spot-checks

Judge scores are validated by human review of a sampled subset. Each entry
records what was checked and whether the judge's score held up.

| Date | Run | Model | Task | Judge score | Human assessment | Agreed? |
|---|---|---|---|---|---|---|

No spot-checks recorded yet.
```

Create `README.md`:

```markdown
# Artificial Translation Index

A benchmark ranking AI models on translating serialized web fiction
(Chinese, Japanese, Korean → English), for people translating works that
have no official translation.

**This project publishes rankings, not translations.** No source text or
translation text is committed or published.

- [Leaderboard](results/leaderboard.md)
- [Methodology](docs/methodology.md)
- [Rubric](docs/rubric.md)

## Usage

```bash
pip install -e ".[dev]"
```

```bash
ati fetch
```

```bash
ati run --run 2026-08 --iterations 1
```

```bash
ati judge --run 2026-08
```

```bash
ati report --run 2026-08
```

Set `OPENROUTER_API_KEY` for API-backed models. Subscription-backed models
are configured as `runner: cli` in `config/models.yaml` and report cost as
N/A.

## Reweighting

Edit `config/weights.yaml` and rerun `ati report`. No model calls are made.
```

- [ ] **Step 7: Run the full test suite**

Run: `python -m pytest -v`
Expected: all tests PASS (roughly 100 across 12 files)

- [ ] **Step 8: Commit**

```bash
git add ati/cli.py ati/pairwise.py docs/ README.md tests/test_cli.py tests/test_pairwise.py
git commit -m "feat: CLI wiring, pairwise stub, and methodology documentation"
```

---

## Post-implementation: what the operator must supply

The harness is complete at Task 12, but produces no ranking until these are
filled in. They are deliberately left to the operator because they are
editorial decisions, not code:

1. **Corpus selection.** Populate `corpus/manifest.json` with passage
   pointers and hashes. Criteria: publicly readable, representative of a
   common genre, at least 10 consecutive chapters available for each
   consistency series. Target: 4 passages per language + 3 public-domain
   control passages + one 10-chapter series per language + 6 refusal
   passages.
2. **Task suite.** Populate `config/tasks.yaml` referencing those passage ids.
3. **Judge selection.** Pick primary and secondary judge models; pass via
   `--judge-model`.
4. **Anchor selection.** Mark one strong and one weak model with
   `anchor: strong` / `anchor: weak` in `config/models.yaml`.

## Known gap against the spec

The spec calls for the **top 5 models to be re-judged by a secondary judge,
with the inter-judge agreement rate published**. Tasks 1–12 make the second
pass possible but do not automate the comparison:

```bash
ati judge --run 2026-08 --judge-model <secondary> --judgments judgments-secondary
```

That writes a parallel judgment tree, but nothing computes or publishes the
agreement rate between the two trees. Closing this needs a follow-up task
adding `ati.report.judge_agreement(primary_root, secondary_root, run_id,
top_n) -> float` and a row in the report. It is deliberately out of v1
scope: the second pass costs a full judge run, and it is only worth
automating once there are real results worth validating.
