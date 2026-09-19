"""Keep a stage going through usage limits: `--wait` on run, judge, pairwise.

This is the loop run 2026-09 ran by hand in scripts/2026-09/judge_driver.py,
moved into the package so any run can use it and the tests cover it. It adds
no concurrency and no calls of its own: it only decides when to run the stage
again. Every Antigravity call still goes through CallGate, and each stage
still stops at the first sign of a usage limit - this loop then waits the
limit out instead of leaving it to the operator.

How long it waits:

- the provider said when the quota returns ("Resets in 1h17m35s", as
  Antigravity does): exactly that, plus a margin. At most one refused call
  reaches the account per quota window;
- a usage limit with no time given: an hour;
- any other stop (the judge unreachable, unparseable answers): ten minutes.

After MAX_STOPS stops in a row that made no progress it gives up and exits
non-zero, leaving the rest to the operator rather than knocking on a door
that is not opening.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

from ati.runstage import is_usage_limit

RESET_RE = re.compile(
    r"resets?\s+in\s+(?:(\d+)\s*h)?\s*(?:(\d+)\s*m(?!s))?\s*(?:(\d+)\s*s)?",
    re.IGNORECASE,
)
RESET_MARGIN_S = 3 * 60
LIMIT_BACKOFF_S = 60 * 60
ERROR_BACKOFF_S = 10 * 60
IDLE_WAIT_S = 5 * 60
MAX_STOPS = 6


@dataclass(frozen=True)
class Pass:
    """What one invocation of a stage did: work done, and why it stopped early."""

    done: int
    stopped: str | None = None


def reset_wait_s(message: str | None) -> int | None:
    """Seconds until the quota returns, when the message says so."""
    match = RESET_RE.search(message or "")
    if not match or not any(match.groups()):
        return None
    hours, minutes, seconds = (int(g or 0) for g in match.groups())
    return hours * 3600 + minutes * 60 + seconds


def backoff_s(message: str | None) -> int:
    reset = reset_wait_s(message)
    if reset is not None:
        return reset + RESET_MARGIN_S
    return LIMIT_BACKOFF_S if is_usage_limit(message) else ERROR_BACKOFF_S


def drive(
    pass_fn,
    *,
    wait: bool = True,
    finished=lambda result: result.done == 0,
    active=lambda: False,
    sleep_fn=None,
    log=print,
    max_stops: int = MAX_STOPS,
    idle_wait_s: int = IDLE_WAIT_S,
) -> bool:
    """Run `pass_fn` until `finished` says so. True on completion.

    `pass_fn` returns a Pass. A pass that stopped early is waited out (when
    `wait` is set) and repeated. A pass that finished is repeated only while
    `active()` was true before it began - another process is still producing
    work for this stage, as translation lanes do for the judge.
    """
    sleep_fn = sleep_fn or time.sleep  # looked up per call, so tests can patch it
    stops = 0
    while True:
        was_active = active()
        try:
            result = pass_fn()
        except Exception as exc:  # noqa: BLE001 - one bad pass must not end the loop
            result = Pass(0, f"{type(exc).__name__}: {exc}")
        if result.stopped:
            if not wait:
                return False
            # Progress before the stop means the window was used well; only
            # stops that achieved nothing count toward giving up.
            stops = 1 if result.done else stops + 1
            if stops >= max_stops:
                log(f"{stops} stops in a row without progress; leaving it for the operator")
                return False
            delay = backoff_s(result.stopped)
            until = (datetime.now() + timedelta(seconds=delay)).strftime("%H:%M")
            log(f"stopped ({stops} in a row): waiting {delay // 60} min, until about {until}")
            sleep_fn(delay)
            continue
        stops = 0
        if finished(result):
            if was_active:
                log(f"another process is still working on this run; checking again "
                    f"in {idle_wait_s // 60} min")
                sleep_fn(idle_wait_s)
                continue
            return True
