"""Writes that are all-or-nothing, and reads that survive a bad file.

Run, judgment and comparison records are written by one process while
another reads them: a judge pass runs while translation lanes are still
writing, a pairwise batch reads every translation, and `ati report` can be
run at any moment. `Path.write_text` truncates the file and then fills it, so
a reader arriving in between saw an empty or half-written file, `json.loads`
raised, and the whole stage died with it. Every record is now written to a
temporary file beside its target and moved into place with `os.replace`,
which a reader sees as either the old file or the new one, never a mix.

The temporary name ends in `.tmp`, so no `*.json` glob ever picks one up.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Iterator

# Windows refuses to replace a file that another process has open, which is
# exactly what a concurrent reader does for the few milliseconds it takes to
# read one record. The handle is released almost at once, so a short retry
# is enough; this is not a lock and waits for nothing else.
REPLACE_ATTEMPTS = 20
REPLACE_BACKOFF_S = 0.05


def write_text_atomic(path: Path | str, text: str, sleep_fn=time.sleep) -> None:
    """Write `text` to `path` so that no reader ever sees a partial file.

    Same encoding and newline handling as `Path.write_text(text,
    encoding="utf-8")`, so files written here are byte-identical to the ones
    written before this existed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        for attempt in range(REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt + 1 == REPLACE_ATTEMPTS:
                    raise
                sleep_fn(REPLACE_BACKOFF_S * (attempt + 1))
    finally:
        # Only still present if the write or the move failed.
        try:
            tmp.unlink()
        except OSError:
            pass


def write_json_atomic(path: Path | str, payload, sleep_fn=time.sleep) -> None:
    """The project's record format: UTF-8 JSON, non-ASCII kept, indented."""
    write_text_atomic(
        path, json.dumps(payload, ensure_ascii=False, indent=2), sleep_fn=sleep_fn
    )


def read_json(path: Path | str) -> dict | None:
    """The JSON object in `path`, or None if it cannot be used.

    None covers a missing or unreadable file, a torn or corrupt one, and a
    file whose top level is not an object. Callers decide what None means for
    them; none of them should crash a stage over one bad file.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def iter_records(
    paths, required: tuple[str, ...] = (), warn=print
) -> Iterator[tuple[Path, dict]]:
    """(path, record) for every readable record carrying `required` keys.

    A file that cannot be read, or lacks a key the caller depends on, is
    reported and skipped rather than raised: one stray or damaged file in a
    run directory must not take down a stage that has hundreds of good ones.
    """
    for path in paths:
        record = read_json(path)
        if record is None:
            warn(f"  skipping unreadable record {path}")
            continue
        missing = [key for key in required if key not in record]
        if missing:
            warn(f"  skipping {path}: missing {', '.join(missing)}")
            continue
        yield Path(path), record
