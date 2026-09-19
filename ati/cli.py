"""Command line entry point: fetch, run, judge, report."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from datetime import date
from pathlib import Path

from ati.config import (
    ConfigError,
    ModelConfig,
    load_defaults,
    load_env_file,
    load_models,
    load_weights,
    requires_antigravity,
)
from ati.bradleyterry import bootstrap_ci, fit, to_elo
from ati.corpus import (
    HashMismatch,
    PassageMissing,
    load_manifest,
    load_passage,
    passage_path,
)
from ati.fileio import read_json, write_json_atomic, write_text_atomic
from ati.judgestage import judge_run
from ati.intake import DEFAULT_CHARS, KINDS, IntakeError, add_files
from ati.pairwise import load_comparisons, run_pairwise
from ati.report import (
    aggregate,
    anchor_drift,
    is_comparable,
    pairwise_records,
    render_markdown,
)
from ati.runners.api import ApiRunner
from ati.runners.agy import AgyRunner, CallGate
from ati.runners.cli import CliRunner
from ati.runners.manual import ManualRunner
from ati.runstage import run_model
from ati.tasks import expand, load_tasks
from ati.textmetrics import load_slop_list

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_ITERATIONS = 1


def resolve_api_key(model: ModelConfig, fallback: str | None = None) -> str | None:
    """The credential for this model's provider.

    Read per model so a pool mixing opencode-go and OpenRouter does not send
    one provider's key to the other and collect 401s.
    """
    return os.environ.get(model.api_key_env) or fallback


def build_alternate_runner(model: ModelConfig, api_key: str | None):
    """The second route for a model that declares one, or None.

    Anything the fallback entry does not name is inherited from the primary,
    so a free-to-paid switch on the same provider says only `model:`.
    """
    spec = model.fallback
    if not spec:
        return None
    key_env = spec.get("api_key_env", model.api_key_env)
    return ApiRunner(
        endpoint=spec.get("endpoint") or model.endpoint or DEFAULT_ENDPOINT,
        model=spec["model"],
        api_key=os.environ.get(key_env) or api_key or "",
        temperature=model.temperature,
        timeout_s=model.timeout_s,
        extra_params=spec.get("params", model.params),
        stream=bool(spec.get("stream", model.stream)),
    )


def build_runner(model: ModelConfig, api_key: str | None, manual_root: Path):
    # load_models already refuses this; checked again because a ModelConfig
    # can be built without going through it, and the cost of getting it wrong
    # is the operator's Google account.
    if model.runner != "agy" and requires_antigravity(model.id, model.model):
        raise ConfigError(
            f"model {model.id}: Gemini may only be reached through Antigravity"
        )
    if model.runner == "api":
        return ApiRunner(
            endpoint=model.endpoint or DEFAULT_ENDPOINT,
            model=model.model or model.id,
            api_key=resolve_api_key(model, api_key) or "",
            temperature=model.temperature,
            timeout_s=model.timeout_s,
            extra_params=model.params,
            fallback=build_alternate_runner(model, api_key),
            stream=model.stream,
        )
    if model.runner == "cli":
        return CliRunner(command=model.command or [], timeout_s=model.timeout_s)
    if model.runner == "agy":
        return AgyRunner(
            model=model.model or model.id,
            json_schema=model.json_schema,
            sandbox=model.sandbox,
            timeout_s=model.timeout_s,
            # Always gated outside tests. See CallGate for what it enforces.
            gate=CallGate(),
        )
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


def build_judge_runner(
    judge_id, models_path, endpoint, manual_root, api_key, json_schema=None
):
    """Resolve the judge against models.yaml, falling back to a plain API call.

    The judge was previously hardcoded to ApiRunner, which made a
    subscription-backed CLI judge impossible. Declaring it in models.yaml like
    any other model also gets the secondary judge for free.
    """
    try:
        declared = {m.id: m for m in load_models(Path(models_path))}
    except (OSError, ConfigError):
        declared = {}
    if judge_id in declared:
        model = declared[judge_id]
        # Each stage enforces its own output shape. The judge entry pins the
        # rubric schema; a pairwise verdict forced into that shape contains no
        # "winner" field at all, so every comparison would be discarded and
        # the stage would report zero work with no error.
        if json_schema is not None:
            model = replace(model, json_schema=json_schema)
        return build_runner(model, api_key, Path(manual_root))
    # An undeclared judge becomes a plain API call, which for Gemini would be
    # exactly the route the operator has ruled out.
    if requires_antigravity(judge_id):
        raise ConfigError(
            f"judge {judge_id}: Gemini may only be reached through Antigravity; "
            f"declare it in models.yaml with runner: agy"
        )
    return ApiRunner(
        endpoint=endpoint or DEFAULT_ENDPOINT,
        model=judge_id,
        api_key=api_key or "",
        temperature=0.0,
    )


def cmd_judge(args) -> int:
    # Passed explicitly, as the pairwise stage does. One judge entry can serve
    # both stages, and whichever schema it declares suits at most one of them:
    # a rubric reply forced into the pairwise shape has no 'scores', so every
    # judgment would fail its schema check.
    judge_runner = build_judge_runner(
        args.judge_model,
        args.models,
        args.judge_endpoint,
        Path(args.manual),
        os.environ.get("OPENROUTER_API_KEY"),
        json_schema=args.judge_schema,
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


def cmd_pairwise(args) -> int:
    judge_runner = build_judge_runner(
        args.judge_model,
        args.models,
        args.judge_endpoint,
        Path(args.manual),
        os.environ.get("OPENROUTER_API_KEY"),
        json_schema=args.pairwise_schema,
    )
    declared = load_models(Path(args.models))
    pool = [m.id for m in declared if m.runner != "agy"]
    if args.model:
        pool = [m for m in pool if m in args.model]

    passages = {p.id: p for p in load_manifest(Path(args.manifest))}
    template = Path(args.pairwise_prompt).read_text(encoding="utf-8")

    existing = load_comparisons(Path(args.pairwise), args.run)
    ratings = to_elo(fit(existing)) if existing else {}

    executed = run_pairwise(
        judge_runner=judge_runner,
        run_id=args.run,
        runs_root=Path(args.runs),
        pairwise_root=Path(args.pairwise),
        corpus_root=Path(args.corpus),
        passages=passages,
        template=template,
        models=pool,
        max_comparisons=args.max_comparisons,
        ratings=ratings,
    )
    print(f"{executed} comparisons executed")

    comparisons = load_comparisons(Path(args.pairwise), args.run)
    if comparisons:
        anchor = next((m.id for m in declared if m.anchor == "weak"), None)
        final = to_elo(fit(comparisons), anchor_id=anchor)
        for model_id, rating in sorted(final.items(), key=lambda kv: -kv[1]):
            print(f"  {rating:7.0f}  {model_id}")
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
        previous = read_json(prev_path)
        if previous is None:
            print(f"WARNING: {prev_path} is unreadable; anchor drift not checked")
            previous = {}

    drift = anchor_drift(results, previous)
    comparable = is_comparable(drift)
    metadata = {"judge_id": args.judge_model, "date": date.today().isoformat()}

    write_json_atomic(
        results_dir / f"{args.run}.json",
        {
            "run_id": args.run,
            "metadata": metadata,
            "weights": weights,
            "anchor_drift": drift,
            "comparable": comparable,
            "results": [asdict(r) for r in results],
        },
    )
    comparisons = load_comparisons(Path(args.pairwise), args.run)
    pairwise_ratings, pairwise_intervals = {}, {}
    anchor = None
    if comparisons:
        anchor = next(
            (m.id for m in load_models(Path(args.models)) if m.anchor == "weak"), None
        )
        pairwise_ratings = to_elo(fit(comparisons), anchor_id=anchor)
        pairwise_intervals = bootstrap_ci(comparisons, anchor_id=anchor)

    write_text_atomic(
        results_dir / "leaderboard.md",
        render_markdown(
            args.run,
            results,
            drift,
            comparable,
            metadata,
            pairwise_ratings,
            pairwise_intervals,
            anchor,
            pairwise_records(comparisons) if comparisons else None,
        ),
    )
    write_text_atomic(
        prev_path,
        json.dumps(
            {
                r.model_id: r.composite
                for r in results
                if r.anchor and r.composite is not None
            },
            indent=2,
        ),
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
    common.add_argument("--pairwise", default="pairwise")
    common.add_argument("--pairwise-prompt", default="config/prompts/pairwise.md")
    common.add_argument("--pairwise-schema", default="config/schemas/pairwise.json")
    common.add_argument("--judge-schema", default="config/schemas/judgment.json")
    common.add_argument("--max-comparisons", type=int, default=50)
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
        "pairwise", help="blind head-to-head comparison", parents=[common]
    )
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

    # Before any credential lookup, so a key in .env is visible to
    # resolve_api_key. Names only - the values must never reach stdout.
    loaded = load_env_file()
    if loaded:
        print(f"loaded from .env: {', '.join(sorted(loaded))}")

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
        "pairwise": cmd_pairwise,
        "report": cmd_report,
        "bench": cmd_bench,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
