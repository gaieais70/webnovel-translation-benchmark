import json

from ati.corpus import Passage, passage_path, sha256_text
from ati.runners.fake import FakeRunner
from ati.runstage import came_up_short, ran_away, run_model, run_output_path
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


def test_runaway_output_is_recorded_as_a_failure(tmp_path):
    """A runner with no token ceiling can loop; that is not a translation.

    gpt-5.6-luna returned 141,842 characters for a 6,987-character chapter
    through the opencode CLI, which exposes no max_tokens. Left unflagged it
    was written as a clean run and would have been handed to the judge and
    scored on the rubric like real prose.
    """
    source = "源" * 1000
    passage = make_passage(tmp_path, "cn-01", source)
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]
    runner = FakeRunner(responses=["x" * 9000])  # 9.0x the source

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

    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "ran away" in data["error"]
    assert data["text"] == ""


def test_a_normal_expansion_is_not_flagged(tmp_path):
    """Korean-to-English ran 1.85x-2.28x across this corpus; that must pass."""
    source = "源" * 1000
    passage = make_passage(tmp_path, "cn-01", source)
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]
    runner = FakeRunner(responses=["x" * 2280])  # 2.28x, the widest seen

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

    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["error"] is None
    assert len(data["text"]) == 2280


def test_short_sources_are_exempt_from_the_expansion_check():
    """The ratio is meaningless when the source is a few characters long."""
    assert not ran_away("English output", 2)
    assert not ran_away("x" * 900, 199)
    # at the floor the check applies again
    assert ran_away("x" * 900, 200)


def test_stub_output_is_recorded_as_a_failure(tmp_path):
    """The mirror of the runaway guard: output far too short to be prose.

    grok-4.6 returned 152 characters for an 8,705-character chapter - the
    agent announcing it would load a skill, then stopping when that tool was
    denied. Nothing flagged it, so it was judged as a translation and dragged
    the model's fidelity score down to 2.5.
    """
    source = "源" * 1000
    passage = make_passage(tmp_path, "cn-01", source)
    task = Task(id="t", type="passage", passage_ids=["cn-01"])
    instances = [TaskInstance(task=task, passage_id="cn-01", iteration=0)]
    runner = FakeRunner(responses=["I'll load the translation skill and apply"])

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

    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "came up short" in data["error"]
    assert data["text"] == ""


def test_a_refusal_is_not_treated_as_short_output():
    """On the refusal task type a short answer IS the measurement."""
    assert came_up_short("x" * 10, 1000, refused=False)
    assert not came_up_short("x" * 10, 1000, refused=True)


def test_short_sources_are_exempt_from_the_minimum_check():
    assert not came_up_short("x", 199)
    assert came_up_short("x", 200)


# --- provider blocks and untranslated output (run 2026-09) -------------------


def _run_one(tmp_path, runner, source="가" * 400):
    passage = make_passage(tmp_path, "ko-01", source)
    task = Task(id="t", type="consistency", passage_ids=["ko-01"])
    instances = [TaskInstance(task=task, passage_id="ko-01", iteration=0)]
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
    out = run_output_path(tmp_path / "runs", "r1", "m1", "t.ko-01.0")
    return json.loads(out.read_text(encoding="utf-8"))


def test_a_moderation_cut_is_a_refusal_not_a_translation(tmp_path):
    """Xiaomi appends this to a translation it stops partway through."""
    cut = "He drew his sword. " * 50 + (
        "AndThe request was rejected because it was considered high risk"
    )
    data = _run_one(tmp_path, FakeRunner(responses=[cut]))
    assert data["refused"] is True
    assert data["blocked_by_provider"] is True
    assert data["text"] == ""
    assert data["error"] is None


def test_an_inspection_failure_is_a_refusal_not_a_route_error(tmp_path):
    """Alibaba's content inspection answers with an HTTP 400."""
    runner = FakeRunner(
        error='HTTP 400: {"error":{"message":"<400> InternalError.Algo.DataInspectionFailed"}}'
    )
    data = _run_one(tmp_path, runner)
    assert data["refused"] is True
    assert data["blocked_by_provider"] is True
    assert data["error"] is None


def test_a_refused_task_is_not_retried(tmp_path):
    runner = FakeRunner(error="HTTP 400: DataInspectionFailed")
    _run_one(tmp_path, runner)
    again = FakeRunner(responses=["x" * 900])
    _run_one(tmp_path, again)
    assert again.calls == []


def test_the_source_echoed_back_is_not_a_translation(tmp_path):
    source = "도운찬은 그만 할 말을 잃고 말았다. " * 20
    data = _run_one(tmp_path, FakeRunner(responses=[source]), source=source)
    assert data["text"] == ""
    assert "source script" in data["error"]


def test_english_with_a_few_source_terms_is_still_a_translation(tmp_path):
    text = "Cheongmyeong (청명) looked up at the sky over Mount Hua. " * 20
    data = _run_one(tmp_path, FakeRunner(responses=[text]), source="가" * 400)
    assert data["error"] is None
    assert data["text"] == text


def test_the_finish_reason_is_recorded(tmp_path):
    class Finishing(FakeRunner):
        def run(self, system_prompt, source_text):
            from dataclasses import replace

            return replace(super().run(system_prompt, source_text), finish_reason="stop")

    data = _run_one(tmp_path, Finishing(responses=["An English translation. " * 40]))
    assert data["finish_reason"] == "stop"


def test_zhipus_content_filter_is_a_refusal(tmp_path):
    runner = FakeRunner(
        error="provider error in a 200 response: {'message': 'Output data may "
        "contain inappropriate content. For details see ...', 'code': '1301'}"
    )
    data = _run_one(tmp_path, runner)
    assert data["refused"] is True
    assert data["blocked_by_provider"] is True
    assert data["error"] is None


def test_a_content_filter_finish_is_recorded_as_a_provider_block(tmp_path):
    class Filtered(FakeRunner):
        def run(self, system_prompt, source_text):
            from dataclasses import replace

            base = super().run(system_prompt, source_text)
            return replace(base, text="", refused=True, finish_reason="content_filter")

    data = _run_one(tmp_path, Filtered(responses=["x"]))
    assert data["refused"] is True
    assert data["blocked_by_provider"] is True


def test_a_usage_limit_is_reported_to_the_caller(tmp_path):
    hits = []
    passage = make_passage(tmp_path, "ko-01", "가" * 400)
    task = Task(id="t", type="consistency", passage_ids=["ko-01"])
    run_model(
        runner=FakeRunner(error="HTTP 429: You've reached your 5-hour usage limit"),
        model_id="m1",
        instances=[TaskInstance(task=task, passage_id="ko-01", iteration=0)],
        passages={passage.id: passage},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
        on_usage_limit=hits.append,
    )
    assert hits == ["HTTP 429: You've reached your 5-hour usage limit"]


# --- recheck: today's guards applied to yesterday's records -------------------


def _record(tmp_path, name, **fields):
    payload = {
        "model_id": "m1", "task_id": "t", "task_type": "consistency",
        "passage_id": name, "iteration": 0, "text": "An English chapter. " * 40,
        "source_chars": 400, "refused": False, "unavailable": False, "error": None,
    }
    payload.update(fields)
    path = run_output_path(tmp_path / "runs", "r1", "m1", f"t.{name}.0")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_recheck_finds_what_a_newer_guard_would_catch(tmp_path):
    from ati.runstage import recheck

    cut = "He drew his sword. " * 50 + "The request was rejected because it was considered high risk"
    blocked = _record(tmp_path, "ko-01", text=cut)
    echoed = _record(tmp_path, "ko-02", text="도운찬은 그만 할 말을 잃고 말았다. " * 20)
    fine = _record(tmp_path, "ko-03")
    before = {p: p.read_text(encoding="utf-8") for p in (blocked, echoed, fine)}

    changes = recheck("r1", tmp_path / "runs")
    assert {p.stem for p, _ in changes} == {"t.ko-01.0", "t.ko-02.0"}
    assert all(p.read_text(encoding="utf-8") == before[p] for p in before)  # dry run

    recheck("r1", tmp_path / "runs", apply=True)
    data = json.loads(blocked.read_text(encoding="utf-8"))
    assert data["refused"] is True and data["blocked_by_provider"] is True
    assert data["text"] == "" and data["error"] is None
    data = json.loads(echoed.read_text(encoding="utf-8"))
    assert "source script" in data["error"] and data["text"] == ""
    assert fine.read_text(encoding="utf-8") == before[fine]
    assert recheck("r1", tmp_path / "runs") == []  # idempotent


def test_recheck_never_undoes_a_provider_block(tmp_path):
    """A block's evidence is discarded when it is recorded, so the guards cannot
    re-derive it; re-applying them must not turn it back into a translation."""
    from ati.runstage import recheck

    _record(tmp_path, "ko-01", text="", refused=True, blocked_by_provider=True)
    assert recheck("r1", tmp_path / "runs") == []


def test_a_runner_that_raises_fails_its_task_not_the_run(tmp_path):
    """An exception from one call used to end the run for every model queued."""

    class Exploding:
        calls = 0

        def run(self, system_prompt, source_text):
            Exploding.calls += 1
            if Exploding.calls == 1:
                raise AttributeError("'str' object has no attribute 'get'")
            return FakeRunner(responses=["An English translation. " * 40]).run(
                system_prompt, source_text
            )

    first = make_passage(tmp_path, "ko-01", "가" * 400)
    second = make_passage(tmp_path, "ko-02", "나" * 400)
    task = Task(id="t", type="consistency", passage_ids=["ko-01", "ko-02"])
    instances = [TaskInstance(task=task, passage_id=p, iteration=0) for p in ("ko-01", "ko-02")]
    executed = run_model(
        runner=Exploding(),
        model_id="m1",
        instances=instances,
        passages={first.id: first, second.id: second},
        corpus_root=tmp_path / "corpus",
        system_prompt="SYS",
        runs_root=tmp_path / "runs",
        run_id="r1",
    )
    assert executed == 2
    failed = json.loads(
        run_output_path(tmp_path / "runs", "r1", "m1", "t.ko-01.0").read_text(encoding="utf-8")
    )
    assert "runner raised AttributeError" in failed["error"]
    ok = json.loads(
        run_output_path(tmp_path / "runs", "r1", "m1", "t.ko-02.0").read_text(encoding="utf-8")
    )
    assert ok["error"] is None
