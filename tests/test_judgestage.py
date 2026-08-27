import json

from ati.config import AXES
from ati.corpus import Passage, passage_path, sha256_text
from ati.judgestage import judge_run, judgment_output_path
from ati.runners.fake import FakeRunner
from ati.runstage import run_output_path


def valid_judge_reply(score=4, term_table=None):
    payload = {
        "scores": {axis: score for axis in AXES},
        "rationales": {axis: "ok" for axis in AXES},
    }
    if term_table is not None:
        payload["term_table"] = term_table
    return json.dumps(payload)


def seed_run(tmp_path, instance_id="t.cn-01.0", task_type="passage", **overrides):
    text = "源文"
    p = Passage(
        id="cn-01",
        lang="zh",
        genre="x",
        source_url="u",
        chapter="1",
        start=0,
        end=len(text),
        sha256=sha256_text(text),
        has_reference=False,
    )
    path = passage_path(tmp_path / "corpus", p.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

    payload = {
        "model_id": "m1",
        "task_id": "t",
        "task_type": task_type,
        "passage_id": "cn-01",
        "iteration": 0,
        "text": "English output",
        "source_chars": len(text),
        "latency_s": 1.0,
        "tokens_in": 10,
        "tokens_out": 5,
        "cost_usd": 0.001,
        "refused": False,
        "unavailable": False,
        "error": None,
    }
    payload.update(overrides)
    out = run_output_path(tmp_path / "runs", "r1", "m1", instance_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload), encoding="utf-8")
    return p


def common_args(tmp_path, passage, judge_runner):
    return dict(
        judge_runner=judge_runner,
        judge_id="judge-model",
        run_id="r1",
        runs_root=tmp_path / "runs",
        judgments_root=tmp_path / "judgments",
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        template="TEMPLATE",
    )


def test_judge_run_writes_a_judgment_per_run_file(tmp_path):
    p = seed_run(tmp_path)
    runner = FakeRunner(responses=[valid_judge_reply()])
    count = judge_run(**common_args(tmp_path, p, runner))

    assert count == 1
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["scores"]["accuracy"] == 4
    assert data["judge_id"] == "judge-model"
    assert data["judge_error"] is None


def test_judge_run_skips_already_judged_instances(tmp_path):
    p = seed_run(tmp_path)
    judge_run(**common_args(tmp_path, p, FakeRunner(responses=[valid_judge_reply()])))
    second = FakeRunner(responses=[valid_judge_reply(score=1)])
    count = judge_run(**common_args(tmp_path, p, second))
    assert count == 0
    assert second.calls == []


def test_judge_run_records_judge_error_after_retries(tmp_path):
    p = seed_run(tmp_path)
    runner = FakeRunner(responses=["bad", "bad", "bad"])
    judge_run(**common_args(tmp_path, p, runner))
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["judge_error"] is not None
    assert data["scores"] == {}
    assert len(runner.calls) == 3


def test_judge_run_skips_unavailable_and_errored_runs(tmp_path):
    p = seed_run(tmp_path, unavailable=True, error="hash mismatch", text="")
    runner = FakeRunner(responses=[valid_judge_reply()])
    judge_run(**common_args(tmp_path, p, runner))
    assert runner.calls == []
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["skipped"] is True


def test_judge_run_records_refusal_without_judging(tmp_path):
    p = seed_run(tmp_path, refused=True, text="I can't help with that.")
    runner = FakeRunner(responses=[valid_judge_reply()])
    judge_run(**common_args(tmp_path, p, runner))
    assert runner.calls == []
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["refused"] is True
    assert data["scores"] == {}


def test_judge_run_requests_term_table_for_consistency_tasks(tmp_path):
    p = seed_run(tmp_path, task_type="consistency")
    runner = FakeRunner(
        responses=[valid_judge_reply(term_table={"云澈": "Yun Che"})]
    )
    judge_run(**common_args(tmp_path, p, runner))
    _, prompt = runner.calls[0]
    assert "term_table" in prompt
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["term_table"] == {"云澈": "Yun Che"}


def test_judge_run_flags_overlong_quotes_in_rationales(tmp_path):
    p = seed_run(tmp_path)
    payload = json.loads(valid_judge_reply())
    payload["rationales"]["accuracy"] = 'says "one two three four five six seven eight nine ten"'
    runner = FakeRunner(responses=[json.dumps(payload)])
    judge_run(**common_args(tmp_path, p, runner))
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["quote_violations"] == ["accuracy"]


def test_judge_run_passes_a_reference_translation_when_one_exists(tmp_path):
    """has_reference was carried in the manifest but never read, so the
    calibration the methodology describes was not actually happening."""
    text = "源文"
    p = Passage(
        id="cn-01", lang="zh", genre="x", source_url="u", chapter="1",
        start=0, end=len(text), sha256=sha256_text(text), has_reference=True,
    )
    path = passage_path(tmp_path / "corpus", p.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    ref = tmp_path / "corpus" / "references" / "cn-01.txt"
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text("PUBLISHED REFERENCE", encoding="utf-8")

    seed_run(tmp_path)
    runner = FakeRunner(responses=[valid_judge_reply()])
    judge_run(**common_args(tmp_path, p, runner))
    _, prompt = runner.calls[0]
    assert "PUBLISHED REFERENCE" in prompt
    assert "REFERENCE TRANSLATION" in prompt


def test_judge_run_omits_the_reference_when_absent(tmp_path):
    p = seed_run(tmp_path)
    runner = FakeRunner(responses=[valid_judge_reply()])
    judge_run(**common_args(tmp_path, p, runner))
    _, prompt = runner.calls[0]
    assert "REFERENCE TRANSLATION" not in prompt
