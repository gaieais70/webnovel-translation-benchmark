import json
import subprocess

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
