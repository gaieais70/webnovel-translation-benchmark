import subprocess

from ati.runners.cli import CliRunner


def fake_run_factory(stdout="", stderr="", returncode=0, capture=None):
    def fake_run(cmd, **kwargs):
        if capture is not None:
            capture.append((cmd, kwargs))
        return subprocess.CompletedProcess(
            args=cmd, returncode=returncode, stdout=stdout, stderr=stderr
        )

    return fake_run


def test_cli_runner_sends_source_on_stdin_and_reads_stdout():
    capture = []
    runner = CliRunner(
        command=["no-such-cli-for-tests", "-p"],
        runner_fn=fake_run_factory(stdout="translated\n", capture=capture),
    )
    result = runner.run("SYSTEM PROMPT", "SOURCE TEXT")
    assert result.text == "translated"
    assert result.error is None
    cmd, kwargs = capture[0]
    assert cmd == ["no-such-cli-for-tests", "-p"]  # not on PATH: left as written
    assert "SYSTEM PROMPT" in kwargs["input"]
    assert "SOURCE TEXT" in kwargs["input"]


def test_cli_runner_reports_cost_none():
    runner = CliRunner(command=["x"], runner_fn=fake_run_factory(stdout="t"))
    assert runner.run("s", "t").cost_usd is None


def test_cli_runner_records_nonzero_exit_as_error():
    runner = CliRunner(
        command=["x"],
        runner_fn=fake_run_factory(stdout="", stderr="not logged in", returncode=1),
    )
    result = runner.run("s", "t")
    assert result.error is not None
    assert "not logged in" in result.error
    assert result.text == ""


def test_cli_runner_records_timeout_as_error():
    def timeout_run(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    runner = CliRunner(command=["x"], timeout_s=1, runner_fn=timeout_run)
    result = runner.run("s", "t")
    assert "timed out" in result.error


def test_cli_runner_flags_refusals():
    runner = CliRunner(
        command=["x"],
        runner_fn=fake_run_factory(stdout="I cannot translate this content."),
    )
    assert runner.run("s", "t").refused is True


def test_the_program_is_resolved_to_the_shim_windows_can_run():
    """A bare `opencode` failed WinError 2 on every call of run 2026-08."""
    from ati.runners.cli import resolve_executable

    shims = {"opencode": r"C:\npm\opencode.CMD"}
    assert resolve_executable(["opencode", "run"], which=shims.get) == [
        r"C:\npm\opencode.CMD", "run",
    ]
    assert resolve_executable(["missing", "x"], which=shims.get) == ["missing", "x"]
    assert resolve_executable([], which=shims.get) == []


def test_cli_runner_decodes_output_leniently():
    """A stray non-UTF-8 byte must not raise UnicodeDecodeError out of the run."""
    capture = []
    runner = CliRunner(command=["x"], runner_fn=fake_run_factory(stdout="ok", capture=capture))
    runner.run("s", "t")
    _, kwargs = capture[0]
    assert kwargs["errors"] == "replace"
