"""Keep a judge stage going for run 2026-09 until it has nothing left to do.

    python runs/logs-2026-09/judge_driver.py rubric
    python runs/logs-2026-09/judge_driver.py pairwise TARGET
    python runs/logs-2026-09/judge_driver.py all TARGET [START_DELAY_S]
    python runs/logs-2026-09/judge_driver.py finish TARGET   (pairwise, then report)

Every judge call goes through the Antigravity CallGate, so this adds no
concurrency; it only decides when to invoke the stage again. It backs off on
any stop, and backs off far longer on anything that looks like a usage limit:
the account behind the judge is the operator's own Google account, and a
refused request is never retried quickly.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "runs" / "logs-2026-09"
LANES = ("ccA", "ccB", "ccC", "orD", "orE")
# Antigravity says when the quota comes back: "Resets in 59m36s". Waiting for
# exactly that, plus a margin, means one refused call per quota window at most.
RESET_RE = re.compile(r"resets in (?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", re.IGNORECASE)
RESET_MARGIN_S = 3 * 60
LIMIT_MARKERS = ("usage limit", "quota", "429", "rate limit", "resource_exhausted",
                 "resource exhausted", "too many requests")
LIMIT_BACKOFF_S = 60 * 60
ERROR_BACKOFF_S = 10 * 60
IDLE_WAIT_S = 5 * 60
# Stops in a row before the driver gives up and leaves it to the operator.
MAX_STOPS = 6


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def translations_finished() -> bool:
    for lane in LANES:
        path = LOGS / f"{lane}.log"
        if not path.exists() or f"=== {lane} done" not in path.read_text(
            encoding="utf-8", errors="replace"
        ):
            return False
    return True


def stage(args: list[str]) -> str:
    cmd = [sys.executable, "-u", "-m", "ati.cli", *args, "--run", "2026-09"]
    done = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace",
    )
    out = (done.stdout or "") + (done.stderr or "")
    for line in out.splitlines():
        if "loaded from .env" not in line:
            print("    " + line, flush=True)
    return out


def backoff(out: str) -> int:
    match = RESET_RE.search(out)
    if match and any(match.groups()):
        h, m, sec = (int(g or 0) for g in match.groups())
        return h * 3600 + m * 60 + sec + RESET_MARGIN_S
    lowered = out.lower()
    return LIMIT_BACKOFF_S if any(m in lowered for m in LIMIT_MARKERS) else ERROR_BACKOFF_S


def progressed(out: str) -> bool:
    """Whether a stage that stopped had done any work first."""
    match = re.search(r"stopping after (\d+)", out)
    return bool(match and int(match.group(1)) > 0)


def rubric() -> None:
    stops = 0
    while True:
        finished_before = translations_finished()
        out = stage(["judge"])
        if "stopping after" in out or "Traceback" in out:
            stops = 1 if progressed(out) else stops + 1
            if stops >= MAX_STOPS:
                log("too many stops in a row; leaving it for the operator")
                return
            wait = backoff(out)
            log(f"judge stopped ({stops} in a row); waiting {wait // 60} min")
            time.sleep(wait)
            continue
        stops = 0
        judged = next(
            (int(l.split(":")[1]) for l in out.splitlines() if l.startswith("judged:")),
            0,
        )
        if judged == 0 and finished_before:
            log("rubric complete")
            return
        if judged == 0:
            time.sleep(IDLE_WAIT_S)


def pairwise(target: int) -> None:
    stops = 0
    while True:
        existing = len(list((ROOT / "pairwise" / "2026-09").glob("*.json")))
        if existing >= target:
            log(f"pairwise reached {existing} comparisons")
            return
        batch = min(25, target - existing)
        out = stage(["pairwise", "--max-comparisons", str(batch)])
        if "stopping after" in out or "Traceback" in out:
            stops = 1 if progressed(out) else stops + 1
            if stops >= MAX_STOPS:
                log("too many stops in a row; leaving it for the operator")
                return
            wait = backoff(out)
            log(f"pairwise stopped ({stops} in a row); waiting {wait // 60} min")
            time.sleep(wait)
            continue
        stops = 0
        if "0 comparisons executed" in out:
            log("pairwise has nothing left to compare")
            return


if __name__ == "__main__":
    # Optional: seconds to wait before the first call, e.g. until a quota reset.
    if len(sys.argv) > 3:
        log(f"waiting {int(sys.argv[3]) // 60} min before starting")
        time.sleep(int(sys.argv[3]))
    if sys.argv[1] == "rubric":
        rubric()
    elif sys.argv[1] == "all":
        # One process, one stage at a time, so the rubric and pairwise stages
        # never split a quota window between them.
        rubric()
        pairwise(int(sys.argv[2]))
    elif sys.argv[1] == "finish":
        # Pairwise to the target, then write the leaderboard. Used when the
        # driver runs detached, so the run completes without a session open.
        pairwise(int(sys.argv[2]))
        stage(["report"])
        log("report written")
    else:
        pairwise(int(sys.argv[2]))
