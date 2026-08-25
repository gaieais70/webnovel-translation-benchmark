"""Command line entry point: fetch, run, judge, report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

import httpx

from ati.config import ModelConfig, load_defaults, load_models, load_weights
from ati.corpus import (
    HashMismatch,
    PassageMissing,
    load_manifest,
    load_passage,
    passage_path,
    sha256_text,
)
from ati.judgestage import judge_run
from ati.intake import DEFAULT_CHARS, KINDS, IntakeError, add_files
from ati.report import aggregate, anchor_drift, is_comparable, render_markdown
from ati.runners.api import ApiRunner
from ati.runners.cli import CliRunner
from ati.runners.manual import ManualRunner
from ati.runstage import run_model
from ati.tasks import expand, load_tasks
from ati.textmetrics import load_slop_list

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_ITERATIONS = 1


def build_runner(model: ModelConfig, api_key: str | None, manual_root: Path):
    if model.runner == "api":
        return ApiRunner(
            endpoint=model.endpoint or DEFAULT_ENDPOINT,
            model=model.model or model.id,
            api_key=api_key or "",
            temperature=model.temperature,
        )
    if model.runner == "cli":
        return CliRunner(command=model.command or [])
    return ManualRunner(root=manual_root, model_id=model.id)


def cmd_add(args) -> int:
    """Ingest local chapter files into the corpus and the task suite."""
    try:
        result = add_files(
            files=args.files,
            work=args.work,
            lang=args.lang,
            genre=args.genre,
            kind=args.type,
            url=args.url,
            corpus_root=Path(args.corpus),
            manifest_path=Path(args.manifest),
            tasks_path=Path(args.tasks),
            chars=args.chars,
            has_reference=args.reference,
        )
    except IntakeError as exc:
        print(f"error: {exc}")
        return 1

    for pid in result.added:
        print(f"  added      {pid}")
    for pid in result.unchanged:
        print(f"  unchanged  {pid}")
    print(
        f"{len(result.added)} added, {len(result.unchanged)} unchanged; "
        f"task '{result.task_id}' updated in {args.tasks}"
    )
    return 0


def cmd_fetch(args) -> int:
    """Verify local corpus against the manifest.

    Automatic downloading is deliberately not implemented: sources vary and
    scraping them is the operator's decision, not the tool's. This reports
    what is missing or changed so the operator can act.
    """
    passages = load_manifest(Path(args.manifest))
    missing, changed, ok = [], [], 0
    for p in passages:
        try:
            load_passage(Path(args.corpus), p)
            ok += 1
        except PassageMissing:
            missing.append(p)
        except HashMismatch:
            changed.append(p)

    print(f"verified: {ok}/{len(passages)}")
    for p in missing:
        print(f"  MISSING  {p.id}  {p.source_url} chapter {p.chapter} "
              f"chars {p.start}-{p.end} -> {passage_path(Path(args.corpus), p.id)}")
    for p in changed:
        print(f"  CHANGED  {p.id}  source no longer matches manifest hash; "
              "this passage is unavailable for scoring")
    return 0 if not missing and not changed else 1


def cmd_run(args) -> int:
    models = load_models(Path(args.models))
    if args.model:
        models = [m for m in models if m.id in args.model]
    tasks = load_tasks(Path(args.tasks))
    if args.task:
        tasks = [t for t in tasks if t.id in args.task]
    instances = expand(tasks, iterations=args.iterations)
    passages = {p.id: p for p in load_manifest(Path(args.manifest))}
    system_prompt = Path(args.prompt).read_text(encoding="utf-8")
    api_key = os.environ.get("OPENROUTER_API_KEY")

    total = 0
    for model in models:
        runner = build_runner(model, api_key, Path(args.manual))
        executed = run_model(
            runner=runner,
            model_id=model.id,
            instances=instances,
            passages=passages,
            corpus_root=Path(args.corpus),
            system_prompt=system_prompt,
            runs_root=Path(args.runs),
            run_id=args.run,
            force=args.force,
        )
        print(f"{model.id}: {executed} executed, {len(instances) - executed} skipped")
        total += executed
    print(f"total executed: {total}")
    return 0


def cmd_judge(args) -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    judge_runner = ApiRunner(
        endpoint=args.judge_endpoint,
        model=args.judge_model,
        api_key=api_key or "",
        temperature=0.0,
    )
    passages = {p.id: p for p in load_manifest(Path(args.manifest))}
    template = Path(args.judge_prompt).read_text(encoding="utf-8")
    judged = judge_run(
        judge_runner=judge_runner,
        judge_id=args.judge_model,
        run_id=args.run,
        runs_root=Path(args.runs),
        judgments_root=Path(args.judgments),
        passages=passages,
        corpus_root=Path(args.corpus),
        template=template,
        force=args.force,
    )
    print(f"judged: {judged}")
    return 0


def cmd_report(args) -> int:
    weights = load_weights(Path(args.weights))
    slop_list = load_slop_list(Path(args.slop))
    anchors = {
        m.id: m.anchor for m in load_models(Path(args.models)) if m.anchor
    }

    results = aggregate(
        run_id=args.run,
        runs_root=Path(args.runs),
        judgments_root=Path(args.judgments),
        weights=weights,
        slop_list=slop_list,
        anchors=anchors,
    )

    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)

    previous: dict[str, float] = {}
    prev_path = results_dir / "anchors.json"
    if prev_path.exists():
        previous = json.loads(prev_path.read_text(encoding="utf-8"))

    drift = anchor_drift(results, previous)
    comparable = is_comparable(drift)
    metadata = {"judge_id": args.judge_model, "date": date.today().isoformat()}

    (results_dir / f"{args.run}.json").write_text(
        json.dumps(
            {
                "run_id": args.run,
                "metadata": metadata,
                "weights": weights,
                "anchor_drift": drift,
                "comparable": comparable,
                "results": [asdict(r) for r in results],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (results_dir / "leaderboard.md").write_text(
        render_markdown(args.run, results, drift, comparable, metadata),
        encoding="utf-8",
    )
    prev_path.write_text(
        json.dumps(
            {
                r.model_id: r.composite
                for r in results
                if r.anchor and r.composite is not None
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"wrote {results_dir / f'{args.run}.json'}")
    print(f"wrote {results_dir / 'leaderboard.md'}")
    if not comparable:
        print("WARNING: anchor drift exceeds threshold; run is not comparable "
              "to previous runs")
    return 0


def cmd_bench(args) -> int:
    """Run the whole pipeline: fetch, run, judge, report."""
    print("== fetch ==")
    if cmd_fetch(args) != 0:
        print(
            "stopping: corpus is incomplete or has changed. Scoring against a "
            "broken corpus would produce meaningless results."
        )
        return 1

    if not args.skip_run:
        print("== run ==")
        cmd_run(args)

    if not args.skip_judge:
        print("== judge ==")
        cmd_judge(args)

    print("== report ==")
    return cmd_report(args)


def main(argv: list[str] | None = None) -> int:
    # Shared options live on a parent parser so they are accepted both before
    # and after the subcommand: `ati --run r1 report` and `ati report --run r1`.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--manifest", default="corpus/manifest.json")
    common.add_argument("--corpus", default="corpus")
    common.add_argument("--models", default="config/models.yaml")
    common.add_argument("--tasks", default="config/tasks.yaml")
    common.add_argument("--weights", default="config/weights.yaml")
    common.add_argument("--slop", default="config/slop_list.txt")
    common.add_argument("--prompt", default="config/prompts/translate.md")
    common.add_argument("--judge-prompt", default="config/prompts/judge.md")
    common.add_argument("--judge-model", default=None)
    common.add_argument("--judge-endpoint", default=None)
    common.add_argument("--iterations", type=int, default=None)
    common.add_argument("--model", action="append", help="limit to model id")
    common.add_argument("--task", action="append", help="limit to task id")
    common.add_argument("--defaults", default="config/ati.yaml")
    common.add_argument("--runs", default="runs")
    common.add_argument("--judgments", default="judgments")
    common.add_argument("--results", default="results")
    common.add_argument("--manual", default="manual")
    common.add_argument("--run", default="latest", help="run id")
    common.add_argument("--force", action="store_true")

    parser = argparse.ArgumentParser(
        prog="ati",
        description="Artificial Translation Index",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser(
        "fetch",
        help="verify the local corpus against the manifest",
        parents=[common],
    )

    sub.add_parser("run", help="translate every task instance", parents=[common])

    sub.add_parser("judge", help="score run outputs", parents=[common])
    sub.add_parser(
        "report", help="aggregate into results and leaderboard", parents=[common]
    )

    add_parser = sub.add_parser(
        "add", help="ingest local chapter files into the corpus", parents=[common]
    )
    add_parser.add_argument("files", nargs="+", help="chapter text files")
    add_parser.add_argument("--work", required=True, help="short work slug, e.g. skyfall")
    add_parser.add_argument("--lang", required=True, help="source language, e.g. zh")
    add_parser.add_argument("--genre", required=True, help="e.g. xianxia")
    add_parser.add_argument("--type", required=True, choices=list(KINDS))
    add_parser.add_argument("--url", required=True, help="source URL")
    add_parser.add_argument("--chars", type=int, default=DEFAULT_CHARS)
    add_parser.add_argument("--reference", action="store_true")

    bench_parser = sub.add_parser(
        "bench", help="fetch, run, judge, and report in one command", parents=[common]
    )
    bench_parser.add_argument("--skip-run", action="store_true")
    bench_parser.add_argument("--skip-judge", action="store_true")

    args = parser.parse_args(argv)
    if not args.command:
        parser.print_usage()
        return 2

    defaults = load_defaults(Path(args.defaults))
    if args.judge_model is None:
        args.judge_model = defaults.get("judge_model", DEFAULT_JUDGE_MODEL)
    if args.judge_endpoint is None:
        args.judge_endpoint = defaults.get("judge_endpoint", DEFAULT_ENDPOINT)
    if args.iterations is None:
        args.iterations = int(defaults.get("iterations", DEFAULT_ITERATIONS))

    handlers = {
        "add": cmd_add,
        "fetch": cmd_fetch,
        "run": cmd_run,
        "judge": cmd_judge,
        "report": cmd_report,
        "bench": cmd_bench,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
