"""Markers saying an `ati run` process is working on a run right now.

Run 2026-09's judge driver decided translations were finished by looking for
the line "=== <lane> done" in five named log files that only the lane scripts
knew about: rename a lane or lose a log and the judge waited forever. The run
stage now says so itself. While `ati run` works on a run it keeps a marker in
runs/<run>/.active/ and touches it every minute; the marker is removed when
the process ends, and one left behind by a killed process goes stale after
ten minutes. `ati judge --follow` keeps waiting while any marker is fresh.

The marker is not a .json file, so no stage ever reads it as a record.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

ACTIVE_DIR = ".active"
HEARTBEAT_S = 60.0
STALE_AFTER_S = 600.0


def _dir(runs_root: Path | str, run_id: str) -> Path:
    return Path(runs_root) / run_id / ACTIVE_DIR


@contextmanager
def run_activity(runs_root: Path | str, run_id: str, label: str, heartbeat_s=HEARTBEAT_S):
    """Hold a fresh marker for the duration of the block."""
    directory = _dir(runs_root, run_id)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / f"{os.getpid()}-{uuid.uuid4().hex[:8]}.active"
    marker.write_text(
        json.dumps({"pid": os.getpid(), "label": label, "started": time.time()}),
        encoding="utf-8",
    )
    stop = threading.Event()

    def beat():
        while not stop.wait(heartbeat_s):
            try:
                os.utime(marker, None)
            except OSError:
                pass

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield marker
    finally:
        stop.set()
        thread.join(timeout=5)
        try:
            marker.unlink()
        except OSError:
            pass


def active_markers(
    runs_root: Path | str, run_id: str, now: float | None = None,
    stale_after_s: float = STALE_AFTER_S,
) -> list[Path]:
    """Markers touched within `stale_after_s`: processes still at work."""
    directory = _dir(runs_root, run_id)
    if not directory.is_dir():
        return []
    now = time.time() if now is None else now
    fresh = []
    for marker in directory.glob("*.active"):
        try:
            if now - marker.stat().st_mtime <= stale_after_s:
                fresh.append(marker)
        except OSError:
            continue  # removed between listing and stat: that process is done
    return sorted(fresh)
