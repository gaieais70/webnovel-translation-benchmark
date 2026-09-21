import os
import time
from datetime import datetime, timezone

import pytest

from ati.activity import active_markers, run_activity
from ati.waiting import (
    ERROR_BACKOFF_S,
    LIMIT_BACKOFF_S,
    RESET_MARGIN_S,
    Pass,
    backoff_s,
    drive,
    reset_wait_s,
)

QUOTA = (
    "agy error: API error (attempt 7): RESOURCE_EXHAUSTED (code 429): Individual "
    "quota reached. Please upgrade your subscription to increase your limits. "
    "Resets in 1h17m35s."
)


@pytest.mark.parametrize(
    "message, seconds",
    [
        (QUOTA, 4655),
        ("Resets in 59m36s.", 3576),
        ("resets in 45s", 45),
        ("Resets in 2h", 7200),
        ("usage limit reached", None),
        (None, None),
    ],
)
def test_the_reset_time_is_read_from_the_message(message, seconds):
    assert reset_wait_s(message) == seconds


def test_a_stated_reset_is_waited_out_exactly_plus_a_margin():
    assert backoff_s(QUOTA) == 4655 + RESET_MARGIN_S


def test_an_unstated_usage_limit_waits_an_hour_anything_else_ten_minutes():
    assert backoff_s("HTTP 429: You've reached your 5-hour usage limit") == LIMIT_BACKOFF_S
    assert backoff_s("judge runner error: exit 1: connection reset") == ERROR_BACKOFF_S


class Script:
    """A stage whose passes return the scripted results in turn."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_a_quota_stop_is_waited_out_and_the_stage_resumed():
    slept, logs = [], []
    stage = Script(Pass(40, QUOTA), Pass(12), Pass(0))
    assert drive(stage, sleep_fn=slept.append, log=logs.append) is True
    assert slept == [4655 + RESET_MARGIN_S]
    assert stage.calls == 3


def test_without_wait_the_first_stop_ends_it():
    slept = []
    stage = Script(Pass(3, QUOTA))
    assert drive(stage, wait=False, sleep_fn=slept.append, log=lambda m: None) is False
    assert slept == []


def test_stops_without_progress_give_up_after_the_limit():
    """Knocking on a door that is not opening is itself a risk to the account."""
    slept = []
    stage = Script(*[Pass(0, "quota exceeded")] * 10)
    assert drive(stage, sleep_fn=slept.append, log=lambda m: None, max_stops=3) is False
    assert stage.calls == 3
    assert len(slept) == 2


def test_progress_before_a_stop_resets_the_count():
    stage = Script(
        Pass(0, "quota"), Pass(0, "quota"), Pass(5, "quota"), Pass(0, "quota"), Pass(0)
    )
    assert drive(stage, sleep_fn=lambda s: None, log=lambda m: None, max_stops=3) is True


def test_an_exception_in_a_pass_is_a_stop_not_a_crash():
    stage = Script(OSError("disk busy"), Pass(0))
    slept = []
    assert drive(stage, sleep_fn=slept.append, log=lambda m: None) is True
    assert slept == [ERROR_BACKOFF_S]


def test_it_keeps_following_while_another_process_is_active():
    active = iter([True, True, False])
    slept = []
    stage = Script(Pass(0), Pass(2), Pass(0))
    assert drive(
        stage, active=lambda: next(active), sleep_fn=slept.append, log=lambda m: None,
        idle_wait_s=300,
    ) is True
    assert stage.calls == 3
    assert slept == [300]


def test_a_custom_finish_condition_is_honoured():
    stage = Script(Pass(25), Pass(25))
    assert drive(stage, finished=lambda p: True, sleep_fn=lambda s: None) is True
    assert stage.calls == 1


# --- activity markers ---------------------------------------------------------


def test_a_marker_is_held_for_the_block_and_removed_after(tmp_path):
    with run_activity(tmp_path, "r1", "m1,m2") as marker:
        assert marker.exists()
        assert active_markers(tmp_path, "r1") == [marker]
        assert marker.suffix == ".active"  # never read as a *.json record
    assert not marker.exists()
    assert active_markers(tmp_path, "r1") == []


def test_a_marker_left_by_a_killed_process_goes_stale(tmp_path):
    directory = tmp_path / "r1" / ".active"
    directory.mkdir(parents=True)
    marker = directory / "123-abc.active"
    marker.write_text("{}", encoding="utf-8")
    old = time.time() - 3600
    os.utime(marker, (old, old))
    assert active_markers(tmp_path, "r1") == []
    assert active_markers(tmp_path, "r1", stale_after_s=7200) == [marker]


def test_the_heartbeat_keeps_a_long_run_fresh(tmp_path):
    with run_activity(tmp_path, "r1", "m", heartbeat_s=0.05) as marker:
        old = time.time() - 3600
        os.utime(marker, (old, old))
        deadline = time.time() + 5
        while time.time() < deadline and not active_markers(tmp_path, "r1"):
            time.sleep(0.05)
        assert active_markers(tmp_path, "r1") == [marker]


CC_WEEKLY = (
    "HTTP 429: {\"error\":{\"message\":\"You've reached your weekly usage limit "
    "for your plan. Your limit resets at 2026-09-22T01:40:52.743Z. Please wait "
    "for the window to reset or upgrade your plan to continue.\"}}"
)


def test_reset_wait_s_reads_an_absolute_reset_instant():
    """Command Code gives an instant, not a duration."""
    now = datetime(2026, 9, 20, 17, 40, 52, tzinfo=timezone.utc)
    assert reset_wait_s(CC_WEEKLY, now=now) == 32 * 3600


def test_reset_wait_s_treats_a_past_reset_as_open():
    now = datetime(2026, 9, 23, 0, 0, 0, tzinfo=timezone.utc)
    assert reset_wait_s(CC_WEEKLY, now=now) == 0


def test_backoff_waits_out_a_weekly_limit_instead_of_an_hour():
    """The bug this covers: a weekly limit backed off one hour, so `--wait`
    hit MAX_STOPS six hours into a wait of more than a day and gave up.

    backoff_s reads the real clock, so the reset instant is built from it: a
    fixed date made this test fail once that date came within twelve hours.
    """
    from datetime import timedelta

    later = (datetime.now(timezone.utc) + timedelta(hours=30)).strftime(
        "%Y-%m-%dT%H:%M:%S.000Z"
    )
    message = f"HTTP 429: weekly usage limit. Your limit resets at {later}."
    assert backoff_s(message) > 29 * 3600
    assert backoff_s(message) != LIMIT_BACKOFF_S


def test_relative_resets_still_parse():
    assert reset_wait_s(QUOTA) == 4655


CLAUDE_SESSION = "exit 1: You've hit your session limit \u00b7 resets 12am (America/New_York)"


def test_claude_cli_session_limit_is_a_usage_limit():
    """The CLI's own spelling. Unmatched, it read as an ordinary task failure
    and the run stage retried it instead of waiting."""
    from ati.runstage import is_usage_limit

    assert is_usage_limit(CLAUDE_SESSION)


def test_clock_time_reset_waits_until_that_time_in_that_zone():
    """13:00 in New York; the limit resets at midnight there - eleven hours.

    Before this the message fell back to the hourly backoff, and MAX_STOPS
    ended the run six hours in, well short of a midnight reset.
    """
    now = datetime(2026, 9, 21, 17, 0, tzinfo=timezone.utc)  # 13:00 EDT
    assert reset_wait_s(CLAUDE_SESSION, now=now) == 11 * 3600


def test_clock_time_reset_is_the_next_occurrence_not_the_last():
    now = datetime(2026, 9, 21, 22, 30, tzinfo=timezone.utc)  # 18:30 EDT
    msg = "exit 1: You've hit your session limit · resets 5pm (America/New_York)"
    assert reset_wait_s(msg, now=now) == 22 * 3600 + 30 * 60


def test_clock_time_reset_with_unknown_zone_falls_back_to_hourly():
    msg = "exit 1: You've hit your session limit · resets 12am (Nowhere/Atlantis)"
    assert reset_wait_s(msg) is None
    assert backoff_s(msg) == LIMIT_BACKOFF_S


def test_iso_instant_is_not_misread_as_a_clock_time():
    now = datetime(2026, 9, 20, 17, 40, 52, tzinfo=timezone.utc)
    assert reset_wait_s(CC_WEEKLY, now=now) == 32 * 3600
