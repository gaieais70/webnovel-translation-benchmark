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
