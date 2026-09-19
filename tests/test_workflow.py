"""The repeatable workflow end to end: run passes, status, waiting judges, bench.

Hermetic: subjects use the manual runner (it writes prompt files and makes no
call), and the judge is replaced with an in-memory runner. Nothing here can
reach agy or any network endpoint.
"""

import json

import pytest

import ati.cli as cli
import ati.waiting as waiting
from ati.config import AXES
from ati.corpus import sha256_text
from ati.runners.base import RunResult
from ati.runstage import run_output_path

QUOTA = (
    "agy error: API error (attempt 7): RESOURCE_EXHAUSTED (code 429): Individual "
    "quota reached. Resets in 1h17m35s."
)
SOURCE = {"p1": "가" * 300, "p2": "나" * 300}


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for key in ("OPENROUTER_API_KEY", "COMMANDCODE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    (tmp_path / "config" / "prompts").mkdir(parents=True)
    (tmp_path / "config" / "schemas").mkdir()
    (tmp_path / "config" / "models.yaml").write_text(
        "- id: s1\n  runner: manual\n"
        "- id: s2\n  runner: manual\n"
        "- id: j\n  role: judge\n  runner: agy\n  model: gemini-x\n",
        encoding="utf-8",
    )
    (tmp_path / "config" / "tasks.yaml").write_text(
        "- id: t\n  type: consistency\n  passages: [p1, p2]\n", encoding="utf-8"
    )
    (tmp_path / "config" / "ati.yaml").write_text("judge_model: j\n", encoding="utf-8")
    (tmp_path / "config" / "weights.yaml").write_text(
        "accuracy: 0.25\nterminology: 0.20\nregister: 0.15\n"
        "fluency: 0.15\nfidelity: 0.15\nsanitization: 0.10\n",
        encoding="utf-8",
    )
    (tmp_path / "config" / "slop_list.txt").write_text("a testament to\n", encoding="utf-8")
    for name in ("translate", "judge", "pairwise"):
        (tmp_path / "config" / "prompts" / f"{name}.md").write_text(name, encoding="utf-8")
    for name in ("judgment", "pairwise"):
        (tmp_path / "config" / "schemas" / f"{name}.json").write_text("{}", encoding="utf-8")
    passages = []
    (tmp_path / "corpus" / "passages").mkdir(parents=True)
    for pid, text in SOURCE.items():
        (tmp_path / "corpus" / "passages" / f"{pid}.txt").write_text(text, encoding="utf-8")
        passages.append({
            "id": pid, "lang": "ko", "genre": "x", "source_url": "u", "chapter": pid,
            "start": 0, "end": len(text), "sha256": sha256_text(text),
            "has_reference": False,
        })
    (tmp_path / "corpus" / "manifest.json").write_text(
        json.dumps({"passages": passages}), encoding="utf-8"
    )
    return tmp_path


def seed_translations(root, models=("s1", "s2")):
    for model in models:
        for pid in SOURCE:
            path = run_output_path(root / "runs", "r1", model, f"t.{pid}.0")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "model_id": model, "task_id": "t", "task_type": "consistency",
                "passage_id": pid, "iteration": 0,
                "text": f"{model} renders {pid} into English. " * 30,
                "source_chars": 300, "latency_s": 1.0, "cost_usd": None,
                "refused": False, "unavailable": False, "error": None,
            }), encoding="utf-8")


class Judge:
    """Scripted judge: each call returns the next (text, error)."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def run(self, system_prompt, source_text):
        self.calls += 1
        text, error = self.script.pop(0) if self.script else (rubric_reply(), None)
        return RunResult(text=text, latency_s=0.0, error=error)


def rubric_reply():
    return json.dumps({
        "scores": {axis: 4 for axis in AXES},
        "rationales": {axis: "ok" for axis in AXES},
        "term_table": {"청명": "Chung Myung"},
    })


def use_judge(monkeypatch, judge):
    monkeypatch.setattr(cli, "build_judge_runner", lambda *a, **k: judge)
    slept = []
    monkeypatch.setattr(waiting.time, "sleep", slept.append)
    return slept


def test_every_stage_but_add_and_fetch_needs_a_run_id(project, capsys):
    assert cli.main(["report"]) == 2
    assert "needs --run" in capsys.readouterr().out


def test_ati_run_translates_subjects_only_and_retries_in_passes(project, capsys):
    """Without --model, the judges are left out: gemini-judge would otherwise
    have translated every chapter through Antigravity."""
    assert cli.main(["run", "--run", "r1", "--passes", "3"]) == 0
    out = capsys.readouterr().out
    assert "== pass 3 of 3 ==" in out  # the manual runner fails until answered
    translated = {p.parent.name for p in (project / "runs" / "r1").rglob("*.json")}
    assert translated == {"s1", "s2"}
    assert not list((project / "runs" / "r1" / ".active").glob("*.active"))


def test_passes_stop_once_nothing_is_left_to_retry(project, capsys):
    seed_translations(project)
    assert cli.main(["run", "--run", "r1", "--passes", "3"]) == 0
    out = capsys.readouterr().out
    assert "== pass 2 of 3 ==" not in out
    assert "total executed: 0" in out


def test_an_unknown_model_id_is_an_error_not_an_empty_run(project, capsys):
    assert cli.main(["run", "--run", "r1", "--model", "s9"]) == 2
    assert "unknown model id" in capsys.readouterr().out


def test_status_shows_progress_without_calling_anything(project, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_judge_runner", lambda *a, **k: pytest.fail("no calls"))
    seed_translations(project, models=("s1",))
    assert cli.main(["status", "--run", "r1"]) == 0
    out = capsys.readouterr().out
    assert "run r1: 2 models x 2 instances" in out
    assert "translations: 2 ok, 0 refused, 0 failed, 2 missing" in out
    assert "judgments: 2 missing, 0 stale (judge j" in out


def test_judge_wait_sits_out_the_quota_and_finishes(project, monkeypatch, capsys):
    seed_translations(project)
    judge = Judge(("", QUOTA))
    slept = use_judge(monkeypatch, judge)
    assert cli.main(["judge", "--run", "r1", "--wait"]) == 0
    assert slept == [4655 + waiting.RESET_MARGIN_S]
    assert len(list((project / "judgments" / "r1").rglob("*.json"))) == 4
    assert "stopping after 0 judgments" in capsys.readouterr().out


def test_judge_without_wait_exits_nonzero_when_it_stops_early(project, monkeypatch):
    seed_translations(project)
    judge = Judge(("", QUOTA))
    slept = use_judge(monkeypatch, judge)
    assert cli.main(["judge", "--run", "r1"]) == 1
    assert judge.calls == 1 and slept == []


def test_pairwise_target_runs_batches_until_reached(project, monkeypatch, capsys):
    seed_translations(project)
    verdicts = [(json.dumps({"winner": w, "reason": "r"}), None) for w in "AB" * 4]
    use_judge(monkeypatch, Judge(*verdicts))
    argv = ["pairwise", "--run", "r1", "--target", "2", "--max-comparisons", "1"]
    assert cli.main(argv) == 0
    assert len(list((project / "pairwise" / "r1").glob("*.json"))) == 2
    assert "target 2 reached" in capsys.readouterr().out


def test_bench_reports_but_fails_when_judging_stopped(project, monkeypatch, capsys):
    seed_translations(project)
    use_judge(monkeypatch, Judge(("", QUOTA)))
    assert cli.main(["bench", "--run", "r1", "--skip-run"]) == 1
    assert (project / "results" / "leaderboard.md").exists()
    assert "leaderboard is partial" in capsys.readouterr().out


def test_recheck_is_a_dry_run_unless_asked(project, capsys):
    seed_translations(project, models=("s1",))
    path = run_output_path(project / "runs", "r1", "s1", "t.p1.0")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["text"] = "도운찬은 그만 할 말을 잃고 말았다. " * 20
    path.write_text(json.dumps(data), encoding="utf-8")
    before = path.read_text(encoding="utf-8")
    assert cli.main(["recheck", "--run", "r1"]) == 0
    assert "1 records would be reclassified" in capsys.readouterr().out
    assert path.read_text(encoding="utf-8") == before
    assert cli.main(["recheck", "--run", "r1", "--apply"]) == 0
    assert json.loads(path.read_text(encoding="utf-8"))["text"] == ""
