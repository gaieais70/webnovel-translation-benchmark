import pytest

from ati.tasks import TASK_TYPES, Task, TaskError, expand, load_tasks


def test_task_types_are_the_three_from_the_spec():
    assert TASK_TYPES == ("passage", "consistency", "refusal")


def test_load_tasks_parses_entries(tmp_path):
    p = tmp_path / "tasks.yaml"
    p.write_text(
        """
- id: cn-passages
  type: passage
  passages: [cn-01, cn-02]
- id: cn-series
  type: consistency
  passages: [cn-ch01, cn-ch02]
""",
        encoding="utf-8",
    )
    tasks = load_tasks(p)
    assert [t.id for t in tasks] == ["cn-passages", "cn-series"]
    assert tasks[0].type == "passage"
    assert tasks[1].passage_ids == ["cn-ch01", "cn-ch02"]


def test_load_tasks_rejects_unknown_type(tmp_path):
    p = tmp_path / "tasks.yaml"
    p.write_text("- id: x\n  type: poetry\n  passages: [a]\n", encoding="utf-8")
    with pytest.raises(TaskError, match="unknown task type"):
        load_tasks(p)


def test_load_tasks_rejects_empty_passage_list(tmp_path):
    p = tmp_path / "tasks.yaml"
    p.write_text("- id: x\n  type: passage\n  passages: []\n", encoding="utf-8")
    with pytest.raises(TaskError, match="at least one passage"):
        load_tasks(p)


def test_expand_produces_one_instance_per_passage_per_iteration():
    tasks = [Task(id="t", type="passage", passage_ids=["a", "b"])]
    instances = expand(tasks, iterations=3)
    assert len(instances) == 6
    assert instances[0].instance_id == "t.a.0"
    assert instances[-1].instance_id == "t.b.2"
    assert {i.iteration for i in instances} == {0, 1, 2}


def test_expand_rejects_zero_iterations():
    tasks = [Task(id="t", type="passage", passage_ids=["a"])]
    with pytest.raises(TaskError, match="iterations must be at least 1"):
        expand(tasks, iterations=0)
