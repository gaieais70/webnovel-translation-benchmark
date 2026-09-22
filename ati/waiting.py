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
from datetime import datetime, timedelta, timezone

from ati.runstage import is_usage_limit

RESET_RE = re.compile(
    r"resets?\s+in\s+(?:(\d+)\s*h)?\s*(?:(\d+)\s*m(?!s))?\s*(?:(\d+)\s*s)?",
    re.IGNORECASE,
)

# Command Code states the other kind of reset: an absolute instant, not a
# duration ("Your limit resets at 2026-09-22T01:40:52.743Z"). Its weekly limit
# can be a day or more away, and without this the message parsed as "no time
# given", fell back to the one-hour backoff, and `--wait` gave up after
# MAX_STOPS hours of knocking - six hours into a thirty-three hour wait.
ABSOLUTE_RESET_RE = re.compile(
    r"resets?\s+at\s+"
    r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)",
    re.IGNORECASE,
)

# The third spelling, the Claude Code CLI's: a wall-clock time in a named zone
# ("You've hit your session limit - resets 12am (America/New_York)"). Unparsed
# it fell back to the hourly backoff, and MAX_STOPS gave up six hours in - so a
# limit hit in the afternoon ended the run hours before the midnight reset.
CLOCK_RESET_RE = re.compile(
    r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*"
    r"\(([A-Za-z_]+(?:/[A-Za-z0-9_+\-]+)+)\)",
    re.IGNORECASE,
)
# A provider saying the balance is empty is not saying a window is closed.
# Command Code answers HTTP 400 "You have insufficient credits to make this
# request. Please purchase more credits"; OpenRouter answers 402. Both stop the
# stage, like a quota, but no amount of waiting refills a balance, so waiting
# only hides the failure behind hours of retries.
CREDIT_MARKERS = (
    "insufficient credit",
    "purchase more credits",
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


def reset_wait_s(message: str | None, now: datetime | None = None) -> int | None:
    """Seconds until the quota returns, when the message says so.

    Three spellings: a duration ("Resets in 1h17m35s", Antigravity), an
    instant ("resets at 2026-09-22T01:40:52.743Z", Command Code), or a wall
    clock time in a named zone ("resets 12am (America/New_York)", the Claude
    Code CLI). `now` is injectable so the last two are testable.
    """
    match = RESET_RE.search(message or "")
    if match and any(match.groups()):
        hours, minutes, seconds = (int(g or 0) for g in match.groups())
        return hours * 3600 + minutes * 60 + seconds
    absolute = _absolute_reset_wait_s(message, now)
    if absolute is not None:
        return absolute
    return _clock_reset_wait_s(message, now)


def _absolute_reset_wait_s(message: str | None, now: datetime | None) -> int | None:
    match = ABSOLUTE_RESET_RE.search(message or "")
    if not match:
        return None
    text = match.group(1).replace(" ", "T")
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        when = datetime.fromisoformat(text)
    except ValueError:
        return None
    # A bare timestamp from an API is UTC; reading it as local time would wait
    # hours too few or too many depending on the operator's offset.
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    # A reset already in the past means the window is open: retry at once
    # rather than returning a negative sleep.
    return max(0, int((when - now).total_seconds()))


def out_of_credit(message: str | None) -> bool:
    """Whether the provider says the account has no balance left to spend."""
    lowered = (message or "").lower()
    return any(marker in lowered for marker in CREDIT_MARKERS)


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
            if out_of_credit(result.stopped):
                log(
                    "the provider says the account is out of credit; waiting does "
                    "not refill a balance, so this is where it stops"
                )
                return False
            # Progress before the stop means the window was used well; only
            # stops that achieved nothing count toward giving up.
            stops = 1 if result.done else stops + 1
            if stops >= max_stops:
                log(f"{stops} stops in a row without progress; leaving it for the operator")
                return False
            delay = backoff_s(result.stopped)
            until = datetime.now() + timedelta(seconds=delay)
            # A weekly limit is waited out in days, so minutes stop being a
            # readable unit and the date matters, not just the clock time.
            if delay >= 12 * 3600:
                span = f"{delay / 3600:.1f} h"
                stamp = until.strftime("%a %d %b %H:%M")
            else:
                span = f"{delay // 60} min"
                stamp = until.strftime("%H:%M")
            log(f"stopped ({stops} in a row): waiting {span}, until about {stamp}")
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


def _clock_reset_wait_s(message: str | None, now: datetime | None) -> int | None:
    match = CLOCK_RESET_RE.search(message or "")
    if not match:
        return None
    hour, minute, meridiem, zone_name = match.groups()
    hour, minute = int(hour) % 12, int(minute or 0)
    if meridiem.lower() == "pm":
        hour += 12
    try:
        from zoneinfo import ZoneInfo

        zone = ZoneInfo(zone_name)
    except Exception:  # noqa: BLE001 - no tz database: fall back to hourly
        return None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    local = now.astimezone(zone)
    when = local.replace(hour=hour, minute=minute, second=0, microsecond=0)
    # "resets 12am" names the next midnight, never the one already past.
    if when <= local:
        when += timedelta(days=1)
    return int((when - local).total_seconds())
