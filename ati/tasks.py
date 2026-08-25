"""Task suite definitions and expansion into per-iteration work items."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

TASK_TYPES: tuple[str, ...] = ("passage", "consistency", "refusal")


class TaskError(Exception):
    """Raised when the task suite is malformed."""


@dataclass(frozen=True)
class Task:
    id: str
    type: str
    passage_ids: list[str]


@dataclass(frozen=True)
class TaskInstance:
    """One passage, translated once. The unit of work for the run stage."""

    task: Task
    passage_id: str
    iteration: int

    @property
    def instance_id(self) -> str:
        return f"{self.task.id}.{self.passage_id}.{self.iteration}"


def load_tasks(path: Path) -> list[Task]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or []
    tasks: list[Task] = []
    seen: set[str] = set()
    for entry in raw:
        task_id = entry.get("id")
        if not task_id:
            raise TaskError("every task entry requires 'id'")
        if task_id in seen:
            raise TaskError(f"duplicate task id: {task_id}")
        seen.add(task_id)

        task_type = entry.get("type")
        if task_type not in TASK_TYPES:
            raise TaskError(f"unknown task type {task_type!r} in task {task_id}")

        passages = entry.get("passages") or []
        if not passages:
            raise TaskError(f"task {task_id} needs at least one passage")

        tasks.append(Task(id=task_id, type=task_type, passage_ids=list(passages)))
    return tasks


def expand(tasks: list[Task], iterations: int) -> list[TaskInstance]:
    if iterations < 1:
        raise TaskError("iterations must be at least 1")
    instances: list[TaskInstance] = []
    for task in tasks:
        for passage_id in task.passage_ids:
            for i in range(iterations):
                instances.append(
                    TaskInstance(task=task, passage_id=passage_id, iteration=i)
                )
    return instances
