# Streamlined Intake and Pipeline — Implementation Plan

> **For agentic workers:** implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Remove the two worst frictions in operating the index: hand-writing corpus manifest JSON, and chaining four commands by hand.

**Architecture:** Two additive CLI commands. `ati add` ingests local chapter text files — hashing, slicing, id generation, manifest entry, and task-suite entry — so the operator never writes JSON. `ati bench` chains fetch → run → judge → report, with defaults loaded from `config/ati.yaml`. Nothing in the scoring core changes.

**Tech Stack:** Python 3.11+, PyYAML, pytest. Standard library otherwise.

**Base commit:** `54d5e8f` (122 tests passing)

## Global Constraints

- **Do not modify** `ati/runners/`, `ati/judge.py`, `ati/judgestage.py`, `ati/runstage.py`, `ati/scoring.py`, `ati/textmetrics.py`, `ati/report.py`, `ati/corpus.py`, or `ati/tasks.py`. The scoring core is tested and correct.
- **No source text is ever committed.** `ati add` writes text into `corpus/passages/`, which is gitignored. Only hashes and pointers go into `corpus/manifest.json`.
- **`ati add` never overwrites silently.** Re-adding identical text is a no-op; re-adding *changed* text under an existing id raises `IntakeError`.
- **`ati add` appends to `config/tasks.yaml`, never clobbers it.** Existing unrelated task entries must survive.
- Python 3.11+ syntax. Only httpx, PyYAML, pytest. No pydantic, no jinja2, no ML libraries.
- No live network calls in tests.

---

### Task 13: Corpus intake (`ati/intake.py`)

**Files:**
- Create: `ati/intake.py`, `tests/test_intake.py`

**Interfaces:**
- Produces:
  - `ati.intake.IntakeError(Exception)`
  - `ati.intake.KINDS: tuple[str, ...]` == `("passage", "consistency", "refusal")`
  - `ati.intake.DEFAULT_CHARS: int` == 800
  - `ati.intake.normalize(raw: str) -> str`
  - `ati.intake.make_id(work: str, kind: str, index: int) -> str`
  - `ati.intake.IntakeResult` dataclass: `added: list[str]`, `unchanged: list[str]`, `task_id: str`
  - `ati.intake.add_files(files, work, lang, genre, kind, url, corpus_root, manifest_path, tasks_path, chars=DEFAULT_CHARS, has_reference=False) -> IntakeResult`

- [ ] **Step 1: Write the failing test**

Create `tests/test_intake.py`:

```python
import json

import pytest
import yaml

from ati.corpus import passage_path, sha256_text
from ati.intake import (
    DEFAULT_CHARS,
    IntakeError,
    add_files,
    make_id,
    normalize,
)


def write_raw(tmp_path, name, text):
    d = tmp_path / "raw"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p


def call(tmp_path, files, kind="passage", work="skyfall", chars=DEFAULT_CHARS):
    return add_files(
        files=files,
        work=work,
        lang="zh",
        genre="xianxia",
        kind=kind,
        url="https://example.invalid/skyfall",
        corpus_root=tmp_path / "corpus",
        manifest_path=tmp_path / "corpus" / "manifest.json",
        tasks_path=tmp_path / "config" / "tasks.yaml",
        chars=chars,
    )


def read_manifest(tmp_path):
    return json.loads(
        (tmp_path / "corpus" / "manifest.json").read_text(encoding="utf-8")
    )


def read_tasks(tmp_path):
    return yaml.safe_load(
        (tmp_path / "config" / "tasks.yaml").read_text(encoding="utf-8")
    )


def test_normalize_strips_bom_and_crlf():
    assert normalize("﻿line one\r\nline two") == "line one\nline two"


def test_make_id_numbers_passages_and_chapters_differently():
    assert make_id("skyfall", "passage", 1) == "skyfall-01"
    assert make_id("skyfall", "refusal", 12) == "skyfall-12"
    assert make_id("skyfall", "consistency", 1) == "skyfall-ch01"


def test_add_files_writes_passage_text_and_manifest_entry(tmp_path):
    f = write_raw(tmp_path, "a.txt", "云澈抬起头")
    result = call(tmp_path, [f])

    assert result.added == ["skyfall-01"]
    assert result.unchanged == []

    stored = passage_path(tmp_path / "corpus", "skyfall-01")
    assert stored.read_text(encoding="utf-8") == "云澈抬起头"

    entry = read_manifest(tmp_path)["passages"][0]
    assert entry["id"] == "skyfall-01"
    assert entry["lang"] == "zh"
    assert entry["genre"] == "xianxia"
    assert entry["sha256"] == sha256_text("云澈抬起头")
    assert entry["start"] == 0
    assert entry["end"] == 5
    assert entry["has_reference"] is False
    assert entry["source_file"] == "a.txt"


def test_add_files_slices_passages_to_chars(tmp_path):
    f = write_raw(tmp_path, "a.txt", "x" * 2000)
    call(tmp_path, [f], chars=800)
    entry = read_manifest(tmp_path)["passages"][0]
    assert entry["end"] == 800
    stored = passage_path(tmp_path / "corpus", "skyfall-01")
    assert len(stored.read_text(encoding="utf-8")) == 800
    assert entry["sha256"] == sha256_text("x" * 800)


def test_add_files_keeps_whole_chapter_for_consistency(tmp_path):
    f = write_raw(tmp_path, "ch1.txt", "y" * 2000)
    call(tmp_path, [f], kind="consistency")
    entry = read_manifest(tmp_path)["passages"][0]
    assert entry["id"] == "skyfall-ch01"
    assert entry["end"] == 2000
    assert entry["series"] == "skyfall"
    assert entry["series_index"] == 1
    assert entry["chapter"] == "1"


def test_add_files_is_idempotent_for_identical_text(tmp_path):
    f = write_raw(tmp_path, "a.txt", "same text")
    call(tmp_path, [f])
    result = call(tmp_path, [f])
    assert result.added == []
    assert result.unchanged == ["skyfall-01"]
    assert len(read_manifest(tmp_path)["passages"]) == 1


def test_add_files_raises_when_text_changed_under_existing_id(tmp_path):
    f = write_raw(tmp_path, "a.txt", "original")
    call(tmp_path, [f])
    f.write_text("different", encoding="utf-8")
    with pytest.raises(IntakeError, match="skyfall-01"):
        call(tmp_path, [f])


def test_add_files_continues_numbering_on_later_calls(tmp_path):
    a = write_raw(tmp_path, "a.txt", "first")
    call(tmp_path, [a])
    b = write_raw(tmp_path, "b.txt", "second")
    result = call(tmp_path, [b])
    assert result.added == ["skyfall-02"]
    assert [e["id"] for e in read_manifest(tmp_path)["passages"]] == [
        "skyfall-01",
        "skyfall-02",
    ]


def test_add_files_numbers_multiple_files_in_sorted_order(tmp_path):
    c = write_raw(tmp_path, "ch03.txt", "three")
    a = write_raw(tmp_path, "ch01.txt", "one")
    b = write_raw(tmp_path, "ch02.txt", "two")
    result = call(tmp_path, [c, a, b], kind="consistency")
    assert result.added == ["skyfall-ch01", "skyfall-ch02", "skyfall-ch03"]
    stored = passage_path(tmp_path / "corpus", "skyfall-ch01")
    assert stored.read_text(encoding="utf-8") == "one"


def test_add_files_writes_task_entry(tmp_path):
    f = write_raw(tmp_path, "a.txt", "text")
    result = call(tmp_path, [f])
    assert result.task_id == "skyfall-passage"
    tasks = read_tasks(tmp_path)
    assert tasks == [
        {"id": "skyfall-passage", "type": "passage", "passages": ["skyfall-01"]}
    ]


def test_add_files_extends_existing_task_without_duplicates(tmp_path):
    a = write_raw(tmp_path, "a.txt", "one")
    call(tmp_path, [a])
    b = write_raw(tmp_path, "b.txt", "two")
    call(tmp_path, [b])
    call(tmp_path, [b])
    tasks = read_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0]["passages"] == ["skyfall-01", "skyfall-02"]


def test_add_files_preserves_unrelated_task_entries(tmp_path):
    tasks_path = tmp_path / "config" / "tasks.yaml"
    tasks_path.parent.mkdir(parents=True, exist_ok=True)
    tasks_path.write_text(
        yaml.safe_dump(
            [{"id": "other-passage", "type": "passage", "passages": ["other-01"]}]
        ),
        encoding="utf-8",
    )
    f = write_raw(tmp_path, "a.txt", "text")
    call(tmp_path, [f])
    tasks = read_tasks(tmp_path)
    assert {t["id"] for t in tasks} == {"other-passage", "skyfall-passage"}


def test_add_files_rejects_unknown_kind(tmp_path):
    f = write_raw(tmp_path, "a.txt", "text")
    with pytest.raises(IntakeError, match="unknown kind"):
        call(tmp_path, [f], kind="poetry")


def test_add_files_rejects_empty_file(tmp_path):
    f = write_raw(tmp_path, "a.txt", "   \n  ")
    with pytest.raises(IntakeError, match="empty"):
        call(tmp_path, [f])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_intake.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'ati.intake'`

- [ ] **Step 3: Write the implementation**

Create `ati/intake.py`:

```python
"""Corpus intake: turn local chapter text files into manifest entries.

The operator saves chapter text however they like; this module handles the
mechanical part - slicing, hashing, id generation, and registering both the
manifest entry and the task-suite entry. No source text reaches the
repository: text is written into the gitignored corpus directory, and only
hashes and pointers are recorded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ati.corpus import passage_path, sha256_text

KINDS: tuple[str, ...] = ("passage", "consistency", "refusal")
DEFAULT_CHARS = 800


class IntakeError(Exception):
    """Raised when intake cannot proceed without destroying or corrupting data."""


@dataclass
class IntakeResult:
    added: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    task_id: str = ""


def normalize(raw: str) -> str:
    """Strip a UTF-8 BOM and normalize line endings.

    Without this, the same chapter saved on Windows and on Linux hashes
    differently and looks like a changed source.
    """
    return raw.lstrip("﻿").replace("\r\n", "\n")


def make_id(work: str, kind: str, index: int) -> str:
    if kind == "consistency":
        return f"{work}-ch{index:02d}"
    return f"{work}-{index:02d}"


def _load_manifest(path: Path) -> dict:
    if Path(path).exists():
        return json.loads(Path(path).read_text(encoding="utf-8"))
    return {"methodology_version": 1, "passages": []}


def _load_tasks(path: Path) -> list[dict]:
    if Path(path).exists():
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    return []


def _next_index(manifest: dict, work: str, kind: str) -> int:
    """Highest existing index for this work and kind, plus one.

    Uses the maximum rather than the count so that a manually removed entry
    does not cause a later add to reuse a retired id.
    """
    prefix = make_id(work, kind, 0)[:-2]
    highest = 0
    for entry in manifest["passages"]:
        pid = entry["id"]
        if not pid.startswith(prefix):
            continue
        suffix = pid[len(prefix) :]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1


def add_files(
    files,
    work: str,
    lang: str,
    genre: str,
    kind: str,
    url: str,
    corpus_root: Path,
    manifest_path: Path,
    tasks_path: Path,
    chars: int = DEFAULT_CHARS,
    has_reference: bool = False,
) -> IntakeResult:
    if kind not in KINDS:
        raise IntakeError(f"unknown kind {kind!r}; expected one of {KINDS}")

    manifest = _load_manifest(manifest_path)
    result = IntakeResult(task_id=f"{work}-{kind}")

    # Identity is the source file name, not a positional counter: re-running
    # add on the same files must recognize them rather than allocate new ids.
    # Two raw files sharing a base name are treated as the same passage.
    by_source = {
        entry["source_file"]: entry
        for entry in manifest["passages"]
        if entry.get("source_file")
    }

    index = _next_index(manifest, work, kind)
    for source in sorted(Path(f) for f in files):
        raw = normalize(source.read_text(encoding="utf-8"))
        if not raw.strip():
            raise IntakeError(f"{source}: file is empty")

        end = len(raw) if kind == "consistency" else min(chars, len(raw))
        text = raw[:end]
        digest = sha256_text(text)

        # A file already registered is either unchanged - a no-op - or
        # changed, which would silently invalidate every score recorded
        # against its id. The second case must stop the run.
        existing = by_source.get(source.name)
        if existing is not None:
            if existing["sha256"] == digest:
                result.unchanged.append(existing["id"])
                continue
            raise IntakeError(
                f"{existing['id']}: already registered from {source.name} with "
                f"different text. Rescoring against altered text would corrupt "
                f"results already recorded against this id; remove the manifest "
                f"entry deliberately if the source really changed."
            )

        # Capture the position before advancing: chapter and series_index
        # below must record THIS passage's number, not the next one's.
        position = index
        candidate_id = make_id(work, kind, position)
        index += 1

        target = passage_path(corpus_root, candidate_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

        entry = {
            "id": candidate_id,
            "lang": lang,
            "genre": genre,
            "source_url": url,
            "chapter": str(position),
            "start": 0,
            "end": end,
            "sha256": digest,
            "has_reference": has_reference,
            "source_file": source.name,
        }
        if kind == "consistency":
            entry["series"] = work
            entry["series_index"] = position

        manifest["passages"].append(entry)
        by_source[source.name] = entry
        result.added.append(candidate_id)

    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Append to the task suite rather than rewriting it: the operator may
    # have entries here that this call knows nothing about.
    tasks = _load_tasks(tasks_path)
    task = next((t for t in tasks if t.get("id") == result.task_id), None)
    ids = result.added + result.unchanged
    if task is None:
        tasks.append({"id": result.task_id, "type": kind, "passages": list(ids)})
    else:
        for pid in ids:
            if pid not in task["passages"]:
                task["passages"].append(pid)

    Path(tasks_path).parent.mkdir(parents=True, exist_ok=True)
    Path(tasks_path).write_text(
        yaml.safe_dump(tasks, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_intake.py -v`
Expected: all 14 tests PASS

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: 136 passed

- [ ] **Step 6: Commit**

```bash
git add ati/intake.py tests/test_intake.py
git commit -m "feat: corpus intake from local chapter files"
```

---

### Task 14: `ati add`, `ati bench`, and config defaults

**Files:**
- Modify: `ati/cli.py`, `ati/config.py`, `tests/test_cli.py`
- Create: `config/ati.yaml`

**Interfaces:**
- Consumes: `ati.intake.add_files`, `ati.intake.DEFAULT_CHARS`, `ati.intake.KINDS`
- Produces:
  - `ati.config.load_defaults(path: Path) -> dict`
  - `ati.cli.cmd_add(args) -> int`
  - `ati.cli.cmd_bench(args) -> int`
  - `ati.cli.DEFAULT_JUDGE_MODEL: str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_cli.py`:

```python
def test_load_defaults_returns_empty_dict_when_absent(tmp_path):
    from ati.config import load_defaults

    assert load_defaults(tmp_path / "nope.yaml") == {}


def test_load_defaults_reads_yaml(tmp_path):
    from ati.config import load_defaults

    p = tmp_path / "ati.yaml"
    p.write_text("judge_model: some/judge\niterations: 3\n", encoding="utf-8")
    d = load_defaults(p)
    assert d["judge_model"] == "some/judge"
    assert d["iterations"] == 3


def test_add_command_registers_passage_and_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "ch1.txt"
    raw.write_text("云澈抬起头", encoding="utf-8")

    rc = main(
        [
            "add",
            str(raw),
            "--work",
            "skyfall",
            "--lang",
            "zh",
            "--genre",
            "xianxia",
            "--type",
            "passage",
            "--url",
            "https://example.invalid/skyfall",
        ]
    )
    assert rc == 0

    manifest = json.loads(
        (tmp_path / "corpus" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["passages"][0]["id"] == "skyfall-01"
    assert (tmp_path / "corpus" / "passages" / "skyfall-01.txt").exists()

    import yaml

    tasks = yaml.safe_load(
        (tmp_path / "config" / "tasks.yaml").read_text(encoding="utf-8")
    )
    assert tasks[0]["id"] == "skyfall-passage"


def test_add_command_reports_changed_text_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "ch1.txt"
    raw.write_text("original", encoding="utf-8")
    args = [
        "add",
        str(raw),
        "--work",
        "skyfall",
        "--lang",
        "zh",
        "--genre",
        "x",
        "--type",
        "passage",
        "--url",
        "u",
    ]
    assert main(args) == 0
    raw.write_text("changed", encoding="utf-8")
    assert main(args) != 0
    assert "already registered" in capsys.readouterr().out


def test_bench_stops_when_fetch_reports_missing_passages(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "manifest.json").write_text(
        json.dumps(
            {
                "passages": [
                    {
                        "id": "gone-01",
                        "lang": "zh",
                        "genre": "x",
                        "source_url": "u",
                        "chapter": "1",
                        "start": 0,
                        "end": 3,
                        "sha256": "0" * 64,
                        "has_reference": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rc = main(["bench", "--run", "r1"])
    assert rc != 0
    out = capsys.readouterr().out
    assert "MISSING" in out
    assert "stopping" in out.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_defaults'` and unknown command `add`

- [ ] **Step 3: Add `load_defaults` to `ati/config.py`**

Append to `ati/config.py`:

```python
def load_defaults(path: Path) -> dict:
    """Operator defaults from config/ati.yaml. Absent file means no defaults.

    Precedence is: explicit CLI flag, then this file, then the built-in
    default baked into the CLI.
    """
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("ati.yaml must contain a mapping")
    return raw
```

- [ ] **Step 4: Wire the CLI**

In `ati/cli.py`, make these changes.

Add to the imports:

```python
from ati.config import ModelConfig, load_defaults, load_models, load_weights
from ati.intake import DEFAULT_CHARS, KINDS, IntakeError, add_files
```

Add near `DEFAULT_ENDPOINT`:

```python
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_ITERATIONS = 1
```

Add these two command handlers:

```python
def cmd_add(args) -> int:
    """Ingest local chapter files into the corpus and the task suite."""
    try:
        result = add_files(
            files=args.files,
            work=args.work,
            lang=args.lang,
            genre=args.genre,
            kind=args.type,
            url=args.url,
            corpus_root=Path(args.corpus),
            manifest_path=Path(args.manifest),
            tasks_path=Path(args.tasks),
            chars=args.chars,
            has_reference=args.reference,
        )
    except IntakeError as exc:
        print(f"error: {exc}")
        return 1

    for pid in result.added:
        print(f"  added      {pid}")
    for pid in result.unchanged:
        print(f"  unchanged  {pid}")
    print(
        f"{len(result.added)} added, {len(result.unchanged)} unchanged; "
        f"task '{result.task_id}' updated in {args.tasks}"
    )
    return 0


def cmd_bench(args) -> int:
    """Run the whole pipeline: fetch, run, judge, report."""
    print("== fetch ==")
    if cmd_fetch(args) != 0:
        print(
            "stopping: corpus is incomplete or has changed. Scoring against a "
            "broken corpus would produce meaningless results."
        )
        return 1

    if not args.skip_run:
        print("== run ==")
        cmd_run(args)

    if not args.skip_judge:
        print("== judge ==")
        cmd_judge(args)

    print("== report ==")
    return cmd_report(args)
```

Replace the three flag definitions on `common` so they default to `None` and can be resolved from `config/ati.yaml`:

```python
    common.add_argument("--judge-model", default=None)
    common.add_argument("--judge-endpoint", default=None)
    common.add_argument("--iterations", type=int, default=None)
    common.add_argument("--model", action="append", help="limit to model id")
    common.add_argument("--task", action="append", help="limit to task id")
    common.add_argument("--defaults", default="config/ati.yaml")
```

Remove the now-duplicated `--model`, `--task`, and `--iterations` arguments from the `run` subparser — argparse raises on a conflicting option string if they are defined in both places. The `run` subparser becomes:

```python
    sub.add_parser("run", help="translate every task instance", parents=[common])
```

Add the two new subparsers:

```python
    add_parser = sub.add_parser(
        "add", help="ingest local chapter files into the corpus", parents=[common]
    )
    add_parser.add_argument("files", nargs="+", help="chapter text files")
    add_parser.add_argument("--work", required=True, help="short work slug, e.g. skyfall")
    add_parser.add_argument("--lang", required=True, help="source language, e.g. zh")
    add_parser.add_argument("--genre", required=True, help="e.g. xianxia")
    add_parser.add_argument("--type", required=True, choices=list(KINDS))
    add_parser.add_argument("--url", required=True, help="source URL")
    add_parser.add_argument("--chars", type=int, default=DEFAULT_CHARS)
    add_parser.add_argument("--reference", action="store_true")

    bench_parser = sub.add_parser(
        "bench", help="fetch, run, judge, and report in one command", parents=[common]
    )
    bench_parser.add_argument("--skip-run", action="store_true")
    bench_parser.add_argument("--skip-judge", action="store_true")
```

After `args = parser.parse_args(argv)` and the `if not args.command` check, resolve defaults:

```python
    defaults = load_defaults(Path(args.defaults))
    if args.judge_model is None:
        args.judge_model = defaults.get("judge_model", DEFAULT_JUDGE_MODEL)
    if args.judge_endpoint is None:
        args.judge_endpoint = defaults.get("judge_endpoint", DEFAULT_ENDPOINT)
    if args.iterations is None:
        args.iterations = int(defaults.get("iterations", DEFAULT_ITERATIONS))
```

Add both commands to the `handlers` dict:

```python
    handlers = {
        "add": cmd_add,
        "fetch": cmd_fetch,
        "run": cmd_run,
        "judge": cmd_judge,
        "report": cmd_report,
        "bench": cmd_bench,
    }
```

- [ ] **Step 5: Create `config/ati.yaml`**

```yaml
# Operator defaults. CLI flags override these; these override built-ins.
judge_model: anthropic/claude-sonnet-5
judge_endpoint: https://openrouter.ai/api/v1/chat/completions
iterations: 1
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest -q`
Expected: 141 passed

- [ ] **Step 7: Commit**

```bash
git add ati/cli.py ati/config.py config/ati.yaml tests/test_cli.py
git commit -m "feat: ati add and ati bench with config defaults"
```

---

### Task 15: Update the operator documentation

**Files:**
- Modify: `README.md`, `docs/methodology.md`

- [ ] **Step 1: Rewrite the README usage section**

Replace the `## Usage` section of `README.md` with:

````markdown
## Usage

Install:

```bash
pip install -e ".[dev]"
```

Add chapter text you have saved locally. This hashes each file, slices it,
writes the manifest entry, and registers the task — no JSON by hand:

```bash
ati add ./raw/ch*.txt --work skyfall --lang zh --genre xianxia --type consistency --url https://example.com/skyfall
```

`--type passage` takes the first 800 characters of each file; `--type
consistency` keeps whole chapters for the multi-chapter terminology test;
`--type refusal` is for passages with violent or sexual content.

Then run everything:

```bash
ati bench --run 2026-08
```

That chains fetch → run → judge → report and writes
`results/leaderboard.md`. The individual stages remain available
(`ati fetch`, `ati run`, `ati judge`, `ati report`) and every stage is
resumable — work already done is skipped.

Set `OPENROUTER_API_KEY` for API-backed models. Subscription-backed models
are configured as `runner: cli` in `config/models.yaml` and report cost as
N/A. Defaults live in `config/ati.yaml`.

## Reweighting

Edit `config/weights.yaml` and rerun `ati report`. No model calls are made.
````

- [ ] **Step 2: Add an intake note to `docs/methodology.md`**

In `docs/methodology.md`, replace the `## Pipeline` section's numbered list with:

```markdown
0. `ati add` — register locally saved chapter files: hash, slice, and record
   pointers. Source text stays in the gitignored corpus directory.
1. `ati fetch` — verify the local corpus against `corpus/manifest.json`.
2. `ati run` — every model translates every passage, once per iteration.
3. `ati judge` — a pinned judge model scores each output 1–5 on six axes.
4. `ati report` — aggregate into `results/` and `results/leaderboard.md`.

`ati bench` runs steps 1–4 in one command.
```

- [ ] **Step 3: Commit**

```bash
git add README.md docs/methodology.md
git commit -m "docs: document ati add and ati bench"
```
