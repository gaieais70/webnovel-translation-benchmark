import os
import time

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
