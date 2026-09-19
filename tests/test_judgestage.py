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


def test_judge_run_rejudges_a_run_that_has_since_succeeded(tmp_path):
    """A judgment recorded as skipped because the translation failed is stale
    once that translation succeeds; skipping on file existence alone would
    leave the repaired work permanently unjudged."""
    p = seed_run(tmp_path, error="timed out", text="", unavailable=False)
    judge_run(**common_args(tmp_path, p, FakeRunner(responses=[valid_judge_reply()])))
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    assert json.loads(out.read_text(encoding="utf-8"))["skipped"] is True

    seed_run(tmp_path, error=None, text="a recovered translation")
    runner = FakeRunner(responses=[valid_judge_reply()])
    judge_run(**common_args(tmp_path, p, runner))
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["skipped"] is False
    assert data["scores"]["accuracy"] == 4


def test_judge_run_leaves_a_current_judgment_alone(tmp_path):
    p = seed_run(tmp_path)
    judge_run(**common_args(tmp_path, p, FakeRunner(responses=[valid_judge_reply()])))
    second = FakeRunner(responses=[valid_judge_reply(score=1)])
    assert judge_run(**common_args(tmp_path, p, second)) == 0
    assert second.calls == []


def test_a_scored_judgment_is_rejudged_when_its_run_becomes_unusable(tmp_path):
    """Staleness has to work in both directions.

    grok-4.6 returned a 152-character stub for an 8,705-character chapter. It
    was judged and scored before the guard existed; reclassifying the run as a
    failure afterwards left the scores in place, because only the
    skipped-then-succeeded direction was checked. Those scores then counted
    toward the model's composite with nothing to clear them.
    """
    p_ = seed_run(tmp_path)
    judge_run(**common_args(tmp_path, p_, FakeRunner(responses=[valid_judge_reply()])))
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    assert json.loads(out.read_text(encoding="utf-8"))["scores"]["accuracy"] == 4

    # the run is later reclassified as a failure
    run_path = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    data = json.loads(run_path.read_text(encoding="utf-8"))
    data["error"] = "output came up short (152 characters); not a translation"
    data["text"] = ""
    run_path.write_text(json.dumps(data), encoding="utf-8")

    runner = FakeRunner(responses=[valid_judge_reply()])
    count = judge_run(**common_args(tmp_path, p_, runner))

    assert count == 1
    assert runner.calls == []  # nothing to judge; it is recorded as skipped
    refreshed = json.loads(out.read_text(encoding="utf-8"))
    assert refreshed["skipped"] is True
    assert refreshed["scores"] == {}


def test_a_judgment_is_stale_when_the_translation_changes(tmp_path):
    """Re-running a model replaces the text; the old scores must not survive.

    gpt-5.6-luna and muse-spark were re-run through a different provider with
    --force. Both stayed clean and scored, so the skipped/unusable check saw
    nothing wrong, and their opencode-route scores would have been reported
    against Command Code translations.
    """
    p_ = seed_run(tmp_path)
    judge_run(**common_args(tmp_path, p_, FakeRunner(responses=[valid_judge_reply()])))

    run_path = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.0")
    data = json.loads(run_path.read_text(encoding="utf-8"))
    data["text"] = "A completely different translation from another provider."
    run_path.write_text(json.dumps(data), encoding="utf-8")

    runner = FakeRunner(responses=[valid_judge_reply(score=2)])
    count = judge_run(**common_args(tmp_path, p_, runner))

    assert count == 1
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    assert json.loads(out.read_text(encoding="utf-8"))["scores"]["accuracy"] == 2


def test_an_unchanged_translation_is_not_rejudged(tmp_path):
    p_ = seed_run(tmp_path)
    judge_run(**common_args(tmp_path, p_, FakeRunner(responses=[valid_judge_reply()])))
    second = FakeRunner(responses=[valid_judge_reply(score=1)])
    assert judge_run(**common_args(tmp_path, p_, second)) == 0
    assert second.calls == []


class ScriptedRunner:
    """Returns each scripted (text, error) pair in turn."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def run(self, system_prompt, source_text):
        from ati.runners.base import RunResult

        self.calls.append((system_prompt, source_text))
        text, error = self._script.pop(0)
        return RunResult(text=text, latency_s=0.0, error=error)


def test_a_usage_limit_stops_the_stage_after_one_call(tmp_path):
    """A judge on a personal subscription must not be asked again for every
    remaining item once it has said it is out of quota."""
    p = seed_run(tmp_path, instance_id="t.cn-01.0")
    seed_run(tmp_path, instance_id="t.cn-01.1")
    runner = FakeRunner(error="exit 1: RESOURCE_EXHAUSTED: quota exceeded")
    count = judge_run(**common_args(tmp_path, p, runner))
    assert count == 0
    assert len(runner.calls) == 1
    assert not (tmp_path / "judgments").exists()


def test_two_unreachable_items_in_a_row_stop_the_stage(tmp_path):
    p = seed_run(tmp_path, instance_id="t.cn-01.0")
    seed_run(tmp_path, instance_id="t.cn-01.1")
    seed_run(tmp_path, instance_id="t.cn-01.2")
    runner = FakeRunner(error="exit 1: connection reset")
    judge_run(**common_args(tmp_path, p, runner))
    # three attempts on each of the first two items, none on the third
    assert len(runner.calls) == 6


def test_one_unreachable_item_is_skipped_unrecorded_and_the_stage_continues(tmp_path):
    p = seed_run(tmp_path, instance_id="t.cn-01.0")
    seed_run(tmp_path, instance_id="t.cn-01.1")
    down = ("", "exit 1: connection reset")
    runner = ScriptedRunner([down, down, down, (valid_judge_reply(), None)])
    count = judge_run(**common_args(tmp_path, p, runner))
    assert count == 1
    first = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    second = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.1")
    assert not first.exists()  # a rerun picks it up
    assert second.exists()


def test_a_judge_error_is_not_mistaken_for_a_finished_judgment(tmp_path):
    p = seed_run(tmp_path)
    judge_run(**common_args(tmp_path, p, FakeRunner(responses=["bad", "bad", "bad"])))
    again = FakeRunner(responses=[valid_judge_reply()])
    count = judge_run(**common_args(tmp_path, p, again))
    assert count == 1
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["judge_error"] is None
    assert data["scores"]["accuracy"] == 4


def test_a_refusal_judgment_stays_current(tmp_path):
    """Refused runs were rewritten on every pass and the stage never idled."""
    p = seed_run(tmp_path, refused=True, text="")
    judge_run(**common_args(tmp_path, p, FakeRunner(responses=[])))
    count = judge_run(**common_args(tmp_path, p, FakeRunner(responses=[])))
    assert count == 0


def test_a_scored_judgment_goes_stale_when_its_run_becomes_a_refusal(tmp_path):
    p = seed_run(tmp_path)
    judge_run(**common_args(tmp_path, p, FakeRunner(responses=[valid_judge_reply()])))
    seed_run(tmp_path, refused=True, text="")
    count = judge_run(**common_args(tmp_path, p, FakeRunner(responses=[])))
    assert count == 1
    out = judgment_output_path(tmp_path / "judgments", "r1", "m1", "t.cn-01.0")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["refused"] is True
    assert data["scores"] == {}


def test_a_judge_runner_that_raises_is_not_asked_again_and_stops_the_stage(tmp_path):
    """A defect in the runner repeats on every call; for the Antigravity judge
    each repeat is another request against the operator's account."""

    class Raising:
        def __init__(self):
            self.calls = 0

        def run(self, system_prompt, source_text):
            self.calls += 1
            raise AttributeError("'list' object has no attribute 'get'")

    p = seed_run(tmp_path, instance_id="t.cn-01.0")
    seed_run(tmp_path, instance_id="t.cn-01.1")
    seed_run(tmp_path, instance_id="t.cn-01.2")
    runner = Raising()
    assert judge_run(**common_args(tmp_path, p, runner)) == 0
    assert runner.calls == 2  # one per item, then the stage stops
    assert not (tmp_path / "judgments").exists()


def test_a_torn_run_file_is_skipped_and_the_stage_carries_on(tmp_path, capsys):
    """A lane still writing must not kill a judge pass reading beside it."""
    p = seed_run(tmp_path)
    torn = run_output_path(tmp_path / "runs", "r1", "m1", "t.cn-01.1")
    torn.write_text('{"model_id": "m1", "te', encoding="utf-8")
    runner = FakeRunner(responses=[valid_judge_reply()])
    assert judge_run(**common_args(tmp_path, p, runner)) == 1
    assert "skipping unreadable" in capsys.readouterr().out
    assert not judgment_output_path(
        tmp_path / "judgments", "r1", "m1", "t.cn-01.1"
    ).exists()
