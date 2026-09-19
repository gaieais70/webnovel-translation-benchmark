"""Corpus manifest handling.

The manifest stores pointers and hashes only - never source text. Passage
text lives in a gitignored local directory, corpus/passages/, written by
`ati add`. Nothing downloads it: to restore it, run `ati add` again on the
original source files with the same options, and every file that reproduces
its manifest hash is written back. `ati fetch` only verifies.
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
    # Chapter order within a work. Recorded for consistency tasks; nothing
    # reads it for scoring, it is the machine-readable record of the order.
    series: str | None = None
    series_index: int | None = None
    # The file `ati add` read, which is what restoring the text needs.
    source_file: str | None = None


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
                source_file=entry.get("source_file"),
            )
        )
    return passages


def passage_path(root: Path, passage_id: str) -> Path:
    return Path(root) / "passages" / f"{passage_id}.txt"


def load_passage(root: Path, passage: Passage) -> str:
    path = passage_path(root, passage.id)
    if not path.exists():
        raise PassageMissing(
            f"{passage.id}: no local text at {path}. Restore it by running "
            f"`ati add` on the original source file with the same options; "
            f"`ati fetch` lists what is missing."
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
