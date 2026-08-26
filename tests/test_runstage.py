import json

from ati.corpus import Passage, passage_path, sha256_text
from ati.runners.fake import FakeRunner
from ati.runstage import run_model, run_output_path
from ati.tasks import Task, TaskInstance


def make_passage(tmp_path, pid, text):
    p = Passage(
        id=pid,
        lang="zh",
        genre="xianxia",
        source_url="u",
        chapter="1",
        start=0,
        end=len(text),
        sha256=sha256_text(text),
        has_reference=False,
    )
    path = passage_path(tmp_path / "corpus", pid)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return p


def test_run_model_writes_one_file_per_instance(tmp_path):
    passage = make_passage(tmp_path, "cn-01", "源文")
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]
    runner = FakeRunner(responses=["English output"])

    count = run_model(
        runner=runner,
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )

    assert count == 1
    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["text"] == "English output"
    assert data["model_id"] == "m1"
    assert data["task_id"] == "t"
    assert data["task_type"] == "passage"
    assert data["passage_id"] == "cn-01"
    assert data["iteration"] == 0
    assert data["source_chars"] == 2
    assert data["error"] is None


def test_run_model_skips_completed_instances(tmp_path):
    passage = make_passage(tmp_path, "cn-01", "源文")
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]

    first = FakeRunner(responses=["first"])
    run_model(
        runner=first,
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )

    second = FakeRunner(responses=["second"])
    count = run_model(
        runner=second,
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )

    assert count == 0
    assert second.calls == []
    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    assert json.loads(out.read_text(encoding="utf-8"))["text"] == "first"


def test_run_model_force_overwrites(tmp_path):
    passage = make_passage(tmp_path, "cn-01", "源文")
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]
    common = dict(
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )
    run_model(runner=FakeRunner(responses=["first"]), **common)
    count = run_model(runner=FakeRunner(responses=["second"]), force=True, **common)
    assert count == 1
    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    assert json.loads(out.read_text(encoding="utf-8"))["text"] == "second"


def test_run_model_records_runner_error_and_continues(tmp_path):
    p1 = make_passage(tmp_path, "cn-01", "一")
    p2 = make_passage(tmp_path, "cn-02", "二")
    task = Task(id="t", type="passage", passage_ids=["cn-01", "cn-02"])
    instances = [
        TaskInstance(task=task, passage_id="cn-01", iteration=0),
        TaskInstance(task=task, passage_id="cn-02", iteration=0),
    ]
    runner = FakeRunner(error="provider exploded")

    count = run_model(
        runner=runner,
        model_id="m1",
        instances=instances,
        passages={p1.id: p1, p2.id: p2},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )

    assert count == 2  # both attempted, neither aborted the run
    for pid in ("cn-01", "cn-02"):
        out = run_output_path(tmp_path / "runs", "r1", "m1", f"t.{pid}.0")
        assert json.loads(out.read_text(encoding="utf-8"))["error"] == "provider exploded"


def test_run_model_marks_unavailable_when_passage_hash_mismatches(tmp_path):
    passage = make_passage(tmp_path, "cn-01", "原文")
    passage_path(tmp_path / "corpus", "cn-01").write_text("changed", encoding="utf-8")
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]
    runner = FakeRunner(responses=["should not be called"])

    run_model(
        runner=runner,
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )

    assert runner.calls == []
    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["unavailable"] is True
    assert "hash" in data["error"].lower()


def test_run_model_sets_task_id_on_manual_runner(tmp_path):
    passage = make_passage(tmp_path, "cn-01", "源")
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]

    class SpyManual(FakeRunner):
        def __init__(self):
            super().__init__(responses=["x"])
            self.task_ids = []

        def set_task_id(self, task_id):
            self.task_ids.append(task_id)

    runner = SpyManual()
    run_model(
        runner=runner,
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )
    assert runner.task_ids == ["t.cn-01.0"]


def test_is_usage_limit_recognizes_quota_errors():
    from ati.runstage import is_usage_limit

    assert is_usage_limit('HTTP 429: {"type":"GoUsageLimitError"}')
    assert is_usage_limit("5-hour usage limit reached")
    assert not is_usage_limit("The read operation timed out")
    assert not is_usage_limit(None)


def test_run_model_retries_a_previously_failed_task(tmp_path):
    """A recorded error is not finished work; a rerun must redo it."""
    passage = make_passage(tmp_path, "cn-01", "源文")
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]
    common = dict(
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )
    run_model(runner=FakeRunner(error="HTTP 429: usage limit"), **common)

    second = FakeRunner(responses=["recovered translation"])
    count = run_model(runner=second, **common)
    assert count == 1
    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["text"] == "recovered translation"
    assert data["error"] is None


def test_run_model_still_skips_successful_tasks(tmp_path):
    passage = make_passage(tmp_path, "cn-01", "源文")
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]
    common = dict(
        model_id="m1",
        instances=instances,
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )
    run_model(runner=FakeRunner(responses=["first"]), **common)
    second = FakeRunner(responses=["second"])
    assert run_model(runner=second, **common) == 0
    assert second.calls == []


def test_run_model_stops_the_whole_model_on_a_usage_limit(tmp_path):
    """Every remaining call would hit the same limit, so stop instead."""
    p1 = make_passage(tmp_path, "cn-01", "一")
    p2 = make_passage(tmp_path, "cn-02", "二")
    task = Task(id="t", type="passage", passage_ids=["cn-01", "cn-02"])
    instances = [
        TaskInstance(task=task, passage_id="cn-01", iteration=0),
        TaskInstance(task=task, passage_id="cn-02", iteration=0),
    ]
    runner = FakeRunner(error="HTTP 429: 5-hour usage limit reached")
    count = run_model(
        runner=runner,
        model_id="m1",
        instances=instances,
        passages={p1.id: p1, p2.id: p2},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )
    assert count == 1
    assert len(runner.calls) == 1
