import json
import subprocess

import pytest

from ati.runners.agy import AgyRunner


def envelope(**overrides):
    payload = {
        "conversation_id": "x",
        "status": "SUCCESS",
        "response": "Yun Che raised his head.",
        "duration_seconds": 5.9,
        "num_turns": 2,
        "usage": {"input_tokens": 28754, "output_tokens": 758, "thinking_tokens": 403},
    }
    payload.update(overrides)
    return json.dumps(payload)


def fake_run_factory(stdout="", stderr="", returncode=0, capture=None):
    def fake_run(cmd, **kwargs):
        if capture is not None:
            capture.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)

    return fake_run


def test_agy_runner_builds_a_sandboxed_print_mode_command():
    capture = []
    runner = AgyRunner(
        model="gemini-3.7-flash-high",
        runner_fn=fake_run_factory(stdout=envelope(), capture=capture),
    )
    runner.run("SYSTEM", "SOURCE")
    cmd, _ = capture[0]
    assert cmd[0] == "agy"
    assert "-p" in cmd
    assert "--model" in cmd and "gemini-3.7-flash-high" in cmd
    assert "--output-format" in cmd and "json" in cmd
    assert "--sandbox" in cmd
    assert "--dangerously-skip-permissions" in cmd


def test_agy_runner_passes_prompt_as_argument_not_stdin():
    capture = []
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=envelope(), capture=capture))
    runner.run("SYSTEM", "SOURCE")
    cmd, kwargs = capture[0]
    joined = " ".join(cmd)
    assert "SYSTEM" in joined and "SOURCE" in joined
    assert not kwargs.get("input")


def test_agy_runner_returns_response_text():
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=envelope()))
    result = runner.run("s", "u")
    assert result.text == "Yun Che raised his head."
    assert result.error is None


def test_agy_runner_maps_usage_and_duration():
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=envelope()))
    result = runner.run("s", "u")
    assert result.tokens_in == 28754
    assert result.tokens_out == 758
    assert result.reasoning_tokens == 403
    assert result.latency_s == 5.9


def test_agy_runner_reports_cost_none_for_subscription():
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=envelope()))
    assert runner.run("s", "u").cost_usd is None


def test_agy_runner_returns_structured_output_as_json_text_when_schema_given(tmp_path):
    schema = tmp_path / "s.json"
    schema.write_text("{}", encoding="utf-8")
    out = envelope(structured_output={"scores": {"accuracy": 4}})
    capture = []
    runner = AgyRunner(
        "m", json_schema=schema, runner_fn=fake_run_factory(stdout=out, capture=capture)
    )
    result = runner.run("s", "u")
    assert json.loads(result.text) == {"scores": {"accuracy": 4}}
    cmd, _ = capture[0]
    assert "--json-schema" in cmd


def test_agy_runner_errors_when_schema_requested_but_absent_from_envelope(tmp_path):
    schema = tmp_path / "s.json"
    schema.write_text("{}", encoding="utf-8")
    runner = AgyRunner("m", json_schema=schema, runner_fn=fake_run_factory(stdout=envelope()))
    result = runner.run("s", "u")
    assert result.error is not None
    assert "structured_output" in result.error


def test_agy_runner_records_nonzero_exit():
    runner = AgyRunner(
        "m", runner_fn=fake_run_factory(stdout="", stderr="quota exceeded", returncode=1)
    )
    result = runner.run("s", "u")
    assert "quota exceeded" in result.error
    assert result.text == ""


def test_agy_runner_records_unparseable_output():
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout="not json at all"))
    result = runner.run("s", "u")
    assert result.error is not None
    assert result.text == ""


def test_agy_runner_records_timeout():
    def timeout_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    runner = AgyRunner("m", timeout_s=1, runner_fn=timeout_run)
    assert "timed out" in runner.run("s", "u").error


def test_agy_runner_can_disable_sandbox():
    capture = []
    runner = AgyRunner(
        "m", sandbox=False, runner_fn=fake_run_factory(stdout=envelope(), capture=capture)
    )
    runner.run("s", "u")
    assert "--sandbox" not in capture[0][0]


def stream_lines(**overrides):
    """agy stream-json output: progress events, then one final result event."""
    result = {
        "conversation_id": "x",
        "status": "SUCCESS",
        "response": "Yun Che raised his head.",
        "duration_seconds": 5.9,
        "num_turns": 1,
        "usage": {"input_tokens": 28754, "output_tokens": 758, "thinking_tokens": 403},
    }
    result.update(overrides)
    return (
        json.dumps({"event": "init", "init": {"model": "m"}})
        + chr(10)
        + json.dumps({"event": "result", "result": result})
        + chr(10)
    )


def test_a_long_prompt_goes_over_stdin_instead_of_the_command_line():
    """Windows caps argv at 32,767 chars; a pairwise prompt carries two whole
    translations and exceeded it, failing all 300 comparisons with WinError 206
    before any judge call was made."""
    capture = []
    runner = AgyRunner(
        model="gemini-3.7-flash-high",
        runner_fn=fake_run_factory(stdout=stream_lines(), capture=capture),
    )
    runner.run("SYSTEM", "x" * 40000)
    cmd, kwargs = capture[0]
    assert "--input-format" in cmd and "stream-json" in cmd
    assert "-p" not in cmd
    # the prompt is in stdin, not in any argument
    assert not any(len(part) > 30000 for part in cmd)
    sent = json.loads(kwargs["input"].strip())
    assert sent["event"] == "user"
    assert sent["message"]["role"] == "user"
    assert len(sent["message"]["content"]) > 40000


def test_a_short_prompt_still_uses_the_argument_form():
    capture = []
    runner = AgyRunner(
        model="gemini-3.7-flash-high",
        runner_fn=fake_run_factory(stdout=envelope(), capture=capture),
    )
    runner.run("SYSTEM", "SOURCE")
    cmd, kwargs = capture[0]
    assert "-p" in cmd
    assert "--input-format" not in cmd
    assert "input" not in kwargs


def test_the_final_result_event_is_read_out_of_the_stream():
    runner = AgyRunner(
        model="m", runner_fn=fake_run_factory(stdout=stream_lines())
    )
    result = runner.run("SYSTEM", "x" * 40000)
    assert result.error is None
    assert result.text == "Yun Che raised his head."
    assert result.tokens_in == 28754


def test_a_stream_error_status_is_reported_not_silently_empty():
    runner = AgyRunner(
        model="m",
        runner_fn=fake_run_factory(
            stdout=stream_lines(status="ERROR", response="", error="empty prompt")
        ),
    )
    result = runner.run("SYSTEM", "x" * 40000)
    assert result.error is not None and "empty prompt" in result.error
    assert result.text == ""


def test_a_stream_with_no_result_event_is_an_error():
    runner = AgyRunner(
        model="m",
        runner_fn=fake_run_factory(
            stdout=json.dumps({"event": "init", "init": {}}) + chr(10)
        ),
    )
    result = runner.run("SYSTEM", "x" * 40000)
    assert result.error is not None and "no result event" in result.error


# --- account safety gate ----------------------------------------------------


class FakeClock:
    def __init__(self, now=1000.0):
        self.now = now
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_the_gate_waits_out_the_remainder_of_the_minimum_gap(tmp_path):
    from ati.runners.agy import CallGate

    clock = FakeClock()
    gate = CallGate(tmp_path, min_gap_s=10.0, sleep_fn=clock.sleep, clock_fn=clock, courtesy_s=0.0)
    with gate.slot():
        pass
    clock.now += 4.0
    with gate.slot():
        pass
    assert clock.slept == [6.0]


def test_the_gate_does_not_wait_once_the_gap_has_passed(tmp_path):
    from ati.runners.agy import CallGate

    clock = FakeClock()
    gate = CallGate(tmp_path, min_gap_s=10.0, sleep_fn=clock.sleep, clock_fn=clock, courtesy_s=0.0)
    with gate.slot():
        pass
    clock.now += 30.0
    with gate.slot():
        pass
    assert clock.slept == []


def test_the_gap_is_shared_by_every_gate_on_the_same_root(tmp_path):
    """A second process builds its own gate; the spacing must still hold."""
    from ati.runners.agy import CallGate

    clock = FakeClock()
    first = CallGate(tmp_path, min_gap_s=10.0, sleep_fn=clock.sleep, clock_fn=clock, courtesy_s=0.0)
    second = CallGate(tmp_path, min_gap_s=10.0, sleep_fn=clock.sleep, clock_fn=clock, courtesy_s=0.0)
    with first.slot():
        pass
    with second.slot():
        pass
    assert clock.slept == [10.0]


def test_a_failed_call_still_counts_toward_the_gap(tmp_path):
    from ati.runners.agy import CallGate

    clock = FakeClock()
    gate = CallGate(tmp_path, min_gap_s=10.0, sleep_fn=clock.sleep, clock_fn=clock, courtesy_s=0.0)
    try:
        with gate.slot():
            raise RuntimeError("agy failed")
    except RuntimeError:
        pass
    with gate.slot():
        pass
    assert clock.slept == [10.0]


def test_the_gate_admits_one_holder_at_a_time(tmp_path):
    import threading
    import time

    from ati.runners.agy import CallGate

    events = []
    gate_a = CallGate(tmp_path, min_gap_s=0.0, courtesy_s=0.0)
    gate_b = CallGate(tmp_path, min_gap_s=0.0, courtesy_s=0.0)
    entered = threading.Event()

    def hold():
        with gate_a.slot():
            events.append("a-in")
            entered.set()
            time.sleep(0.5)
            events.append("a-out")

    worker = threading.Thread(target=hold)
    worker.start()
    entered.wait(5)
    with gate_b.slot():
        events.append("b-in")
    worker.join()
    assert events == ["a-in", "a-out", "b-in"]


def test_a_gated_call_runs_from_the_empty_workspace(tmp_path):
    from ati.runners.agy import CallGate

    capture = []
    gate = CallGate(tmp_path / "gate", min_gap_s=0.0)
    runner = AgyRunner(
        "m", gate=gate, runner_fn=fake_run_factory(stdout=envelope(), capture=capture)
    )
    runner.run("s", "u")
    _, kwargs = capture[0]
    assert kwargs["cwd"] == str(tmp_path / "gate" / "workspace")
    assert list((tmp_path / "gate" / "workspace").iterdir()) == []


def test_the_schema_path_is_absolute_so_it_resolves_from_the_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "s.json").write_text("{}", encoding="utf-8")
    out = envelope(structured_output={"winner": "A"})
    capture = []
    runner = AgyRunner(
        "m", json_schema="s.json", runner_fn=fake_run_factory(stdout=out, capture=capture)
    )
    runner.run("s", "u")
    cmd, _ = capture[0]
    schema_arg = cmd[cmd.index("--json-schema") + 1]
    assert schema_arg == str(tmp_path / "s.json")


def test_agy_gets_a_print_timeout_matching_the_subprocess_timeout():
    capture = []
    runner = AgyRunner(
        "m", timeout_s=900, runner_fn=fake_run_factory(stdout=envelope(), capture=capture)
    )
    runner.run("s", "u")
    cmd, _ = capture[0]
    assert cmd[cmd.index("--print-timeout") + 1] == "900s"


def test_every_prompt_tells_the_judge_not_to_use_tools():
    """Every judge call that reached for a shell stalled until the timeout."""
    from ati.runners.agy import NO_TOOLS_PREAMBLE

    capture = []
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=envelope(), capture=capture))
    runner.run("SYSTEM", "SOURCE")
    cmd, _ = capture[0]
    prompt = cmd[cmd.index("-p") + 1]
    assert prompt.startswith(NO_TOOLS_PREAMBLE)
    assert prompt.index("SYSTEM") < prompt.index("SOURCE")


def test_a_busy_process_does_not_starve_a_second_one(tmp_path):
    """The rubric stage calling back-to-back once locked a pairwise process
    out for twenty minutes: it re-took the lock the instant it released it."""
    import threading
    import time

    from ati.runners.agy import CallGate

    order = []
    busy = CallGate(tmp_path, min_gap_s=0.3, courtesy_s=0.3)
    other = CallGate(tmp_path, min_gap_s=0.3, courtesy_s=0.3)

    def hammer():
        for i in range(6):
            with busy.slot():
                order.append(f"busy{i}")
                time.sleep(0.05)

    worker = threading.Thread(target=hammer)
    worker.start()
    time.sleep(0.1)  # the busy process is already mid-call
    with other.slot():
        order.append("other")
    worker.join()
    assert order.index("other") < order.index("busy5")


def test_the_last_caller_yields_to_anyone_else_waiting(tmp_path):
    from ati.runners.agy import CallGate

    clock = FakeClock()
    first = CallGate(tmp_path, min_gap_s=10.0, sleep_fn=clock.sleep, clock_fn=clock, courtesy_s=2.0)
    second = CallGate(tmp_path, min_gap_s=10.0, sleep_fn=clock.sleep, clock_fn=clock, courtesy_s=2.0)
    with first.slot():
        pass
    assert first._remaining(courtesy=True) == 12.0
    assert second._remaining(courtesy=True) == 10.0


@pytest.mark.parametrize("stdout", ["[1, 2]", '"just a string"', "42", "null"])
def test_json_that_is_not_an_envelope_is_an_error_not_a_crash(stdout):
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=stdout))
    result = runner.run("s", "u")
    assert result.text == ""
    assert "unexpected agy output" in result.error


def test_a_stream_result_that_is_not_an_object_is_an_error():
    lines = json.dumps({"event": "result", "result": "plain text"}) + chr(10)
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=lines))
    result = runner.run("SYSTEM", "x" * 40000)
    assert "unexpected agy output" in result.error


def test_stream_lines_that_are_not_objects_are_skipped():
    lines = "[1, 2]" + chr(10) + '"progress"' + chr(10) + stream_lines()
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=lines))
    result = runner.run("SYSTEM", "x" * 40000)
    assert result.error is None
    assert result.text == "Yun Che raised his head."


def test_odd_usage_and_duration_fields_do_not_crash():
    runner = AgyRunner(
        "m",
        runner_fn=fake_run_factory(
            stdout=envelope(usage="n/a", duration_seconds="fast", response=["x"])
        ),
    )
    result = runner.run("s", "u")
    assert result.error is None
    assert result.latency_s == 0.0
    assert result.tokens_in is None
    assert result.text == ""


def test_output_is_decoded_leniently():
    """A stray non-UTF-8 byte must not raise after the call reached the account."""
    capture = []
    runner = AgyRunner("m", runner_fn=fake_run_factory(stdout=envelope(), capture=capture))
    runner.run("s", "u")
    _, kwargs = capture[0]
    assert kwargs["encoding"] == "utf-8"
    assert kwargs["errors"] == "replace"


def test_the_quota_reset_time_survives_the_error_cut():
    """The waiting loop reads 'Resets in ...' from this text."""
    message = (
        "API error (attempt 7): RESOURCE_EXHAUSTED (code 429): Individual quota "
        "reached. Please upgrade your subscription to increase your limits. "
        + "x" * 60
        + " Resets in 1h17m35s."
    )
    runner = AgyRunner(
        "m", runner_fn=fake_run_factory(stdout=envelope(status="ERROR", error=message))
    )
    assert "Resets in 1h17m35s" in runner.run("s", "u").error
    runner = AgyRunner(
        "m", runner_fn=fake_run_factory(stderr=message, returncode=1)
    )
    assert "Resets in 1h17m35s" in runner.run("s", "u").error


def test_a_damaged_stamp_enforces_the_full_gap(tmp_path):
    """Reading damage as 'long ago' would let the next call through with no gap."""
    from ati.runners.agy import CallGate

    clock = FakeClock()
    (tmp_path / "last").write_text("garbage", encoding="utf-8")
    gate = CallGate(
        tmp_path, min_gap_s=10.0, sleep_fn=clock.sleep, clock_fn=clock, courtesy_s=0.0
    )
    assert gate._remaining(courtesy=False) == 10.0


def test_the_stamp_is_written_beside_the_workspace_not_inside_it(tmp_path):
    from ati.runners.agy import CallGate

    gate = CallGate(tmp_path, min_gap_s=0.0, courtesy_s=0.0)
    _ = gate.workspace
    with gate.slot():
        pass
    assert sorted(p.name for p in tmp_path.iterdir()) == ["last", "lock", "workspace"]
    assert list((tmp_path / "workspace").iterdir()) == []
