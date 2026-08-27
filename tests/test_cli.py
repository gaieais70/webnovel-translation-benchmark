import json
from pathlib import Path

import pytest

from ati.cli import build_runner, main
from ati.config import ModelConfig
from ati.runners.api import ApiRunner
from ati.runners.cli import CliRunner
from ati.runners.manual import ManualRunner


def test_build_runner_returns_api_runner_for_api_models(tmp_path):
    m = ModelConfig(
        id="m", runner="api", model="vendor/m", endpoint="http://x", temperature=0.3
    )
    assert isinstance(build_runner(m, "key", tmp_path), ApiRunner)


def test_build_runner_returns_cli_runner_for_cli_models(tmp_path):
    m = ModelConfig(id="m", runner="cli", command=["echo"])
    assert isinstance(build_runner(m, None, tmp_path), CliRunner)


def test_build_runner_returns_manual_runner_for_manual_models(tmp_path):
    m = ModelConfig(id="m", runner="manual")
    assert isinstance(build_runner(m, None, tmp_path), ManualRunner)


def test_main_with_no_args_prints_usage_and_returns_nonzero(capsys):
    assert main([]) != 0
    assert "usage" in capsys.readouterr().out.lower()


def test_main_unknown_command_returns_nonzero(capsys):
    with pytest.raises(SystemExit):
        main(["frobnicate"])


def test_report_command_writes_leaderboard(tmp_path, monkeypatch):
    # minimal run + judgment so report has something to aggregate
    from ati.config import AXES
    from ati.judgestage import judgment_output_path
    from ati.runstage import run_output_path

    run_payload = {
        "model_id": "m1",
        "task_id": "t",
        "task_type": "passage",
        "passage_id": "cn-01",
        "iteration": 0,
        "text": "output text",
        "source_chars": 800,
        "latency_s": 5.0,
        "tokens_in": 1,
        "tokens_out": 1,
        "cost_usd": 0.004,
        "refused": False,
        "unavailable": False,
        "error": None,
    }
    rp = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(run_payload), encoding="utf-8")

    jp = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(
        json.dumps(
            {
                "model_id": "m1",
                "task_id": "t",
                "task_type": "passage",
                "passage_id": "cn-01",
                "iteration": 0,
                "judge_id": "judge",
                "scores": {axis: 4 for axis in AXES},
                "rationales": {axis: "ok" for axis in AXES},
                "term_table": None,
                "refused": False,
                "skipped": False,
                "judge_error": None,
                "quote_violations": [],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    Path(tmp_path / "config").mkdir(exist_ok=True)
    Path(tmp_path / "config" / "weights.yaml").write_text(
        "accuracy: 0.25\nterminology: 0.20\nregister: 0.15\n"
        "fluency: 0.15\nfidelity: 0.15\nsanitization: 0.10\n",
        encoding="utf-8",
    )
    Path(tmp_path / "config" / "models.yaml").write_text(
        '- id: m1\n  runner: cli\n  command: ["x"]\n', encoding="utf-8"
    )
    Path(tmp_path / "config" / "slop_list.txt").write_text("a testament to\n", encoding="utf-8")

    assert main(["report", "--run", "r1"]) == 0
    leaderboard = tmp_path / "results" / "leaderboard.md"
    assert leaderboard.exists()
    assert "m1" in leaderboard.read_text(encoding="utf-8")
    assert (tmp_path / "results" / "r1.json").exists()


def test_load_defaults_returns_empty_dict_when_absent(tmp_path):
    from ati.config import load_defaults

    assert load_defaults(tmp_path / "nope.yaml") == {}


def test_load_defaults_reads_yaml(tmp_path):
    from ati.config import load_defaults

    p = tmp_path / "ati.yaml"
    p.write_text("judge_model: some/judge\niterations: 3\n", encoding="utf-8")
    d = load_defaults(p)
    assert d["judge_model"] == "some/judge"
    assert d["iterations"] == 3


def test_add_command_registers_passage_and_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "ch1.txt"
    raw.write_text("云澈抬起头", encoding="utf-8")

    rc = main(
        [
            "add",
            str(raw),
            "--work",
            "skyfall",
            "--lang",
            "zh",
            "--genre",
            "xianxia",
            "--type",
            "passage",
            "--url",
            "https://example.invalid/skyfall",
        ]
    )
    assert rc == 0

    manifest = json.loads(
        (tmp_path / "corpus" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["passages"][0]["id"] == "skyfall-01"
    assert (tmp_path / "corpus" / "passages" / "skyfall-01.txt").exists()

    import yaml

    tasks = yaml.safe_load(
        (tmp_path / "config" / "tasks.yaml").read_text(encoding="utf-8")
    )
    assert tasks[0]["id"] == "skyfall-passage"


def test_add_command_reports_changed_text_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "ch1.txt"
    raw.write_text("original", encoding="utf-8")
    args = [
        "add",
        str(raw),
        "--work",
        "skyfall",
        "--lang",
        "zh",
        "--genre",
        "x",
        "--type",
        "passage",
        "--url",
        "u",
    ]
    assert main(args) == 0
    raw.write_text("changed", encoding="utf-8")
    assert main(args) != 0
    assert "already registered" in capsys.readouterr().out


def test_bench_stops_when_fetch_reports_missing_passages(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "manifest.json").write_text(
        json.dumps(
            {
                "passages": [
                    {
                        "id": "gone-01",
                        "lang": "zh",
                        "genre": "x",
                        "source_url": "u",
                        "chapter": "1",
                        "start": 0,
                        "end": 3,
                        "sha256": "0" * 64,
                        "has_reference": False,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    rc = main(["bench", "--run", "r1"])
    assert rc != 0
    out = capsys.readouterr().out
    assert "MISSING" in out
    assert "stopping" in out.lower()


def test_build_runner_returns_agy_runner_for_agy_models(tmp_path):
    from ati.runners.agy import AgyRunner

    m = ModelConfig(id="judge", runner="agy", model="gemini-3.7-flash-high")
    assert isinstance(build_runner(m, None, tmp_path), AgyRunner)


def test_load_models_accepts_agy_runner(tmp_path):
    from ati.config import load_models

    p = tmp_path / "models.yaml"
    p.write_text(
        "- id: judge\n  runner: agy\n  model: gemini-3.7-flash-high\n"
        "  json_schema: config/schemas/judgment.json\n  sandbox: true\n",
        encoding="utf-8",
    )
    m = load_models(p)[0]
    assert m.runner == "agy"
    assert m.json_schema == "config/schemas/judgment.json"
    assert m.sandbox is True


def test_load_models_rejects_agy_entry_without_model(tmp_path):
    from ati.config import ConfigError, load_models

    p = tmp_path / "models.yaml"
    p.write_text("- id: judge\n  runner: agy\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="requires 'model'"):
        load_models(p)


def test_build_judge_runner_uses_models_yaml_entry(tmp_path):
    from ati.cli import build_judge_runner
    from ati.runners.agy import AgyRunner

    p = tmp_path / "models.yaml"
    p.write_text(
        "- id: gem-judge\n  runner: agy\n  model: gemini-3.7-flash-high\n",
        encoding="utf-8",
    )
    runner = build_judge_runner("gem-judge", p, None, tmp_path, None)
    assert isinstance(runner, AgyRunner)


def test_build_judge_runner_falls_back_to_api_for_unknown_id(tmp_path):
    from ati.cli import build_judge_runner
    from ati.runners.api import ApiRunner

    p = tmp_path / "models.yaml"
    p.write_text("- id: other\n  runner: cli\n  command: [\"x\"]\n", encoding="utf-8")
    runner = build_judge_runner(
        "anthropic/claude-sonnet-5", p, "http://x", tmp_path, "key"
    )
    assert isinstance(runner, ApiRunner)


def test_pairwise_command_reports_when_nothing_to_compare(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir(exist_ok=True)
    (tmp_path / "config" / "models.yaml").write_text(
        "- id: gemini-judge\n  runner: agy\n  model: gemini-3.7-flash-high\n",
        encoding="utf-8",
    )
    (tmp_path / "config" / "prompts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "prompts" / "pairwise.md").write_text("T", encoding="utf-8")
    (tmp_path / "corpus").mkdir(exist_ok=True)
    (tmp_path / "corpus" / "manifest.json").write_text(
        json.dumps({"passages": []}), encoding="utf-8"
    )
    rc = main(["pairwise", "--run", "empty", "--judge-model", "gemini-judge"])
    assert rc == 0
    assert "0 comparisons" in capsys.readouterr().out


def test_resolve_api_key_reads_the_models_own_env_var(monkeypatch):
    from ati.cli import resolve_api_key

    monkeypatch.setenv("OPENCODE_API_KEY", "opencode-secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-secret")
    m = ModelConfig(
        id="m", runner="api", model="x", api_key_env="OPENCODE_API_KEY"
    )
    assert resolve_api_key(m) == "opencode-secret"


def test_resolve_api_key_falls_back_when_unset(monkeypatch):
    from ati.cli import resolve_api_key

    monkeypatch.delenv("NOPE_KEY", raising=False)
    m = ModelConfig(id="m", runner="api", model="x", api_key_env="NOPE_KEY")
    assert resolve_api_key(m, "fallback") == "fallback"


def test_pairwise_judge_gets_the_pairwise_schema_not_the_rubric_one(tmp_path):
    """A verdict forced into the rubric schema has no 'winner' field, so every
    comparison would be discarded and the stage would report zero silently."""
    from ati.cli import build_judge_runner

    p = tmp_path / "models.yaml"
    p.write_text(
        "- id: gemini-judge\n  runner: agy\n  model: gemini-3.7-flash-high\n"
        "  json_schema: config/schemas/judgment.json\n",
        encoding="utf-8",
    )
    runner = build_judge_runner(
        "gemini-judge", p, None, tmp_path, None,
        json_schema="config/schemas/pairwise.json",
    )
    assert runner._schema == "config/schemas/pairwise.json"


def test_judge_runner_keeps_its_declared_schema_by_default(tmp_path):
    from ati.cli import build_judge_runner

    p = tmp_path / "models.yaml"
    p.write_text(
        "- id: gemini-judge\n  runner: agy\n  model: gemini-3.7-flash-high\n"
        "  json_schema: config/schemas/judgment.json\n",
        encoding="utf-8",
    )
    runner = build_judge_runner("gemini-judge", p, None, tmp_path, None)
    assert runner._schema == "config/schemas/judgment.json"
