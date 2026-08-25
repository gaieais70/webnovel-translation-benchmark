"""Run stage: translate every task instance with one model.

Idempotent by construction - an instance whose output file already exists is
skipped, so adding a model to an existing run costs only that model's calls.
"""

from __future__ import annotations

import json
from pathlib import Path

from ati.corpus import HashMismatch, Passage, PassageMissing, load_passage
from ati.tasks import TaskInstance


def run_output_path(
    runs_root: Path, run_id: str, model_id: str, instance_id: str
) -> Path:
    safe_model = model_id.replace("/", "__")
    return Path(runs_root) / run_id / safe_model / f"{instance_id}.json"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_model(
    runner,
    model_id: str,
    instances: list[TaskInstance],
    passages: dict[str, Passage],
    corpus_root: Path,
    system_prompt: str,
    runs_root: Path,
    run_id: str,
    force: bool = False,
) -> int:
    executed = 0
    for instance in instances:
        out_path = run_output_path(
            runs_root, run_id, model_id, instance.instance_id
        )
        if out_path.exists() and not force:
            continue

        base = {
            "model_id": model_id,
            "task_id": instance.task.id,
            "task_type": instance.task.type,
            "passage_id": instance.passage_id,
            "iteration": instance.iteration,
            "unavailable": False,
        }

        passage = passages.get(instance.passage_id)
        if passage is None:
            _write(
                out_path,
                {
                    **base,
                    "text": "",
                    "source_chars": 0,
                    "latency_s": 0.0,
                    "tokens_in": None,
                    "tokens_out": None,
                    "cost_usd": None,
                    "refused": False,
                    "unavailable": True,
                    "error": f"passage {instance.passage_id} not in manifest",
                },
            )
            executed += 1
            continue

        try:
            source_text = load_passage(corpus_root, passage)
        except (HashMismatch, PassageMissing) as exc:
            _write(
                out_path,
                {
                    **base,
                    "text": "",
                    "source_chars": 0,
                    "latency_s": 0.0,
                    "tokens_in": None,
                    "tokens_out": None,
                    "cost_usd": None,
                    "refused": False,
                    "unavailable": True,
                    "error": str(exc),
                },
            )
            executed += 1
            continue

        # Manual runner needs the instance id to pair prompt and response files.
        set_task_id = getattr(runner, "set_task_id", None)
        if callable(set_task_id):
            set_task_id(instance.instance_id)

        result = runner.run(system_prompt, source_text)
        _write(
            out_path,
            {
                **base,
                "text": result.text,
                "source_chars": len(source_text),
                "latency_s": result.latency_s,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "cost_usd": result.cost_usd,
                "refused": result.refused,
                "error": result.error,
            },
        )
        executed += 1
    return executed
