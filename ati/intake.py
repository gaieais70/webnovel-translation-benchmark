"""Corpus intake: turn local chapter text files into manifest entries.

The operator saves chapter text however they like; this module handles the
mechanical part - slicing, hashing, id generation, and registering both the
manifest entry and the task-suite entry. No source text reaches the
repository: text is written into the gitignored corpus directory, and only
hashes and pointers are recorded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ati.corpus import passage_path, sha256_text
from ati.fileio import write_text_atomic

KINDS: tuple[str, ...] = ("passage", "consistency", "refusal")
DEFAULT_CHARS = 800


class IntakeError(Exception):
    """Raised when intake cannot proceed without destroying or corrupting data."""


@dataclass
class IntakeResult:
    added: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    restored: list[str] = field(default_factory=list)
    task_id: str = ""


def normalize(raw: str) -> str:
    """Strip a UTF-8 BOM and normalize line endings.

    Without this, the same chapter saved on Windows and on Linux hashes
    differently and looks like a changed source.
    """
    return raw.lstrip("﻿").replace("\r\n", "\n")


HEADER_SCAN_LINES = 25
# Reader-UI furniture that scrapers pick up along with the prose. These are
# the font-size control on Korean web-novel readers.
CHROME_TOKENS = {"글자", "기본", "-", "+", "가", "원래대로"}
_FONT_SIZE_RE = re.compile(r"^\d+\s*px$", re.IGNORECASE)
# " - 284화", " 284화", " chapter 284", " 第284章"
_CHAPTER_SUFFIX_RE = re.compile(
    r"\s*[-–—]?\s*(?:"
    r"(?:제\s*)?\d+\s*(?:화|장|회|章|話)"  # 284화, 제3장
    r"|(?:chapter|chap\.?|ch\.?|episode|ep\.?)\s*\d+"  # chapter 12
    r")\s*$",
    re.IGNORECASE,
)


def title_stem(line: str) -> str:
    """The work title with any trailing chapter marker removed."""
    return _CHAPTER_SUFFIX_RE.sub("", line.strip()).strip()


def strip_boilerplate(text: str) -> str:
    """Remove scraped reader chrome from the top of a chapter file.

    Scraped chapters arrive with the site's font-size widget and the title
    repeated two or three times before the prose starts. Left in place the
    models dutifully translate "Text / - / 16px / + / Default", and the judge
    then scores them on how well they rendered a font-size control.

    Only the first few lines are examined, and only lines that repeat the
    title or match known chrome are dropped, so a chapter that happens to open
    with short dialogue is left alone.
    """
    lines = text.split("\n")
    if not lines:
        return text

    stem = title_stem(lines[0])
    last_header = -1
    chrome_hits = 0
    stem_hits = 0
    for i, line in enumerate(lines[:HEADER_SCAN_LINES]):
        stripped = line.strip()
        if not stripped:
            continue
        is_chrome = stripped in CHROME_TOKENS or bool(_FONT_SIZE_RE.match(stripped))
        is_title = len(stem) > 3 and stem in stripped
        if is_chrome:
            chrome_hits += 1
        if is_title:
            stem_hits += 1
        if is_chrome or is_title:
            last_header = i

    # Evidence is required before cutting anything. A first line that merely
    # looks like a title is not enough - in a clean file that first line is
    # the opening sentence, and stripping it would silently delete prose.
    # Real scraped chrome shows up either as widget tokens or as the title
    # repeated more than once.
    if last_header < 0 or (chrome_hits == 0 and stem_hits < 2):
        return text
    return "\n".join(lines[last_header + 1 :]).lstrip("\n")


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
    strip_headers: bool = True,
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
        if strip_headers:
            raw = strip_boilerplate(raw)
        if not raw.strip():
            raise IntakeError(f"{source}: file is empty")

        # chars <= 0 means take the whole file, whatever its length.
        if kind == "consistency" or chars <= 0:
            end = len(raw)
        else:
            end = min(chars, len(raw))
        text = raw[:end]
        digest = sha256_text(text)

        # A file already registered is either unchanged - a no-op - or
        # changed, which would silently invalidate every score recorded
        # against its id. The second case must stop the run.
        existing = by_source.get(source.name)
        if existing is not None:
            if existing["sha256"] == digest:
                # The manifest already has it. If the local passage text is
                # missing or damaged, this source reproduces it exactly - the
                # hash just proved so - and this is the only way a fresh
                # clone or a wiped corpus directory gets its text back.
                target = passage_path(corpus_root, existing["id"])
                try:
                    intact = sha256_text(target.read_text(encoding="utf-8")) == digest
                except OSError:
                    intact = False
                if intact:
                    result.unchanged.append(existing["id"])
                else:
                    write_text_atomic(target, text)
                    result.restored.append(existing["id"])
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

        write_text_atomic(passage_path(corpus_root, candidate_id), text)

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

    write_text_atomic(
        manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2)
    )

    # Append to the task suite rather than rewriting it: the operator may
    # have entries here that this call knows nothing about.
    tasks = _load_tasks(tasks_path)
    task = next((t for t in tasks if t.get("id") == result.task_id), None)
    ids = result.added + result.unchanged + result.restored
    if task is None:
        tasks.append({"id": result.task_id, "type": kind, "passages": list(ids)})
    else:
        for pid in ids:
            if pid not in task["passages"]:
                task["passages"].append(pid)

    write_text_atomic(
        tasks_path, yaml.safe_dump(tasks, allow_unicode=True, sort_keys=False)
    )
    return result
