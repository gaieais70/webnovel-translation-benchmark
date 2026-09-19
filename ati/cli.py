"""Command line entry point: fetch, run, judge, report."""

from __future__ import annotations

import argparse
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
    subjects,
)
from ati.activity import active_markers, run_activity
from ati.bradleyterry import bootstrap_ci, fit, to_elo
from ati.corpus import (
    HashMismatch,
    PassageMissing,
    load_manifest,
    load_passage,
    passage_path,
)
from ati.fileio import write_json_atomic, write_text_atomic
from ati.fingerprint import (
    code_fingerprint,
    input_fingerprints,
    judge_fingerprint,
    judge_route,
    pairwise_prompt_frame,
    rubric_prompt_frame,
)
from ati.judgestage import judge_run, pending_judgments
from ati.intake import DEFAULT_CHARS, KINDS, IntakeError, add_files
from ati.pairwise import (
    DEFAULT_MAX_COMPARISONS,
    load_comparisons,
    pending_comparisons,
    run_pairwise,
)
from ati.report import (
    aggregate,
    anchor_drift,
    anchors_file,
    is_comparable,
    pairwise_records,
    read_anchor_baseline,
    render_markdown,
)
from ati.runners.api import ApiRunner
from ati.runners.agy import AgyRunner, CallGate
from ati.runners.cli import CliRunner
from ati.runners.manual import ManualRunner
from ati.runstage import recheck, run_model
from ati.status import render_status, run_progress
from ati.tasks import expand, load_tasks
from ati.textmetrics import load_slop_list
from ati.waiting import Pass, drive

DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5"
DEFAULT_ITERATIONS = 1


def resolve_api_key(model: ModelConfig) -> str | None:
    """The credential for this model's provider, from its own variable only.

    There is deliberately no fallback to another variable. Falling back to
    OPENROUTER_API_KEY sent OpenRouter's key to Command Code whenever
    COMMANDCODE_API_KEY was unset; the 401 that came back was recorded as the
    model failing, and on a row with a fallback route it silently switched
    provider.
    """
    return os.environ.get(model.api_key_env) or None


def build_alternate_runner(model: ModelConfig):
    """The second route for a model that declares one, or None.

    Anything the fallback entry does not name is inherited from the primary,
    so a free-to-paid switch on the same provider says only `model:`.
    """
    spec = model.fallback
    if not spec:
        return None
    if requires_antigravity(spec.get("model")):
        raise ConfigError(
            f"model {model.id}: Gemini may only be reached through Antigravity, "
            f"not as a fallback route"
        )
    key_env = spec.get("api_key_env", model.api_key_env)
    key = os.environ.get(key_env)
    if not key:
        # The primary route may still serve every call, so this is a warning:
        # but a fallback without its key can only ever collect a 401.
        print(f"WARNING: {model.id}: fallback key {key_env} is not set")
    return ApiRunner(
        endpoint=spec.get("endpoint") or model.endpoint or DEFAULT_ENDPOINT,
        model=spec["model"],
        api_key=key or "",
        temperature=model.temperature,
        timeout_s=model.timeout_s,
        extra_params=spec.get("params", model.params),
        stream=bool(spec.get("stream", model.stream)),
    )


def build_runner(model: ModelConfig, manual_root: Path):
    # load_models already refuses this; checked again because a ModelConfig
    # can be built without going through it, and the cost of getting it wrong
    # is the operator's Google account.
    if model.runner != "agy" and requires_antigravity(model.id, model.model):
        raise ConfigError(
            f"model {model.id}: Gemini may only be reached through Antigravity"
        )
    if model.runner == "api":
        key = resolve_api_key(model)
        if not key:
            raise ConfigError(
                f"model {model.id}: {model.api_key_env} is not set "
                f"(export it or add it to .env)"
            )
        return ApiRunner(
            endpoint=model.endpoint or DEFAULT_ENDPOINT,
            model=model.model or model.id,
            api_key=key,
            temperature=model.temperature,
            timeout_s=model.timeout_s,
            extra_params=model.params,
            fallback=build_alternate_runner(model),
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
            strip_headers=not args.keep_headers,
        )
    except IntakeError as exc:
        print(f"error: {exc}")
        return 1

    for pid in result.added:
        print(f"  added      {pid}")
    for pid in result.unchanged:
        print(f"  unchanged  {pid}")
    for pid in result.restored:
        print(f"  restored   {pid}  (passage text rewritten; matches the manifest hash)")
    print(
        f"{len(result.added)} added, {len(result.unchanged)} unchanged, "
        f"{len(result.restored)} restored; "
        f"task '{result.task_id}' updated in {args.tasks}"
    )
    return 0


def cmd_fetch(args) -> int:
    """Verify the local corpus against the manifest. It downloads nothing.

    Automatic downloading is deliberately not implemented: sources vary and
    scraping them is the operator's decision, not the tool's. Passage text is
    restored by running `ati add` again on the original source files with
    the same options; a file that reproduces the manifest hash is written
    back. This reports what is missing or changed, and how to get it back.
    """
    passages = load_manifest(Path(args.manifest))
    missing, changed, no_reference, ok = [], [], [], 0
    for p in passages:
        try:
            load_passage(Path(args.corpus), p)
            ok += 1
        except PassageMissing:
            missing.append(p)
        except HashMismatch:
            changed.append(p)
        if p.has_reference and not (
            Path(args.corpus) / "references" / f"{p.id}.txt"
        ).exists():
            no_reference.append(p)

    print(f"verified: {ok}/{len(passages)}")
    for p in missing:
        source = f"from {p.source_file} " if p.source_file else ""
        print(f"  MISSING  {p.id}  {source}({p.source_url} chapter {p.chapter}, "
              f"chars {p.start}-{p.end}) -> {passage_path(Path(args.corpus), p.id)}")
    for p in changed:
        print(f"  CHANGED  {p.id}  local text no longer matches the manifest hash; "
              "this passage is unavailable for scoring")
    for p in no_reference:
        print(f"  NO REFERENCE  {p.id}  has_reference is set but "
              f"{Path(args.corpus) / 'references' / f'{p.id}.txt'} is missing")
    if missing or changed:
        print(
            "To restore passage text, run `ati add` again on the original source "
            "files with the same --work, --type, --chars (and --keep-headers if "
            "used): each file that reproduces its manifest hash is written back."
        )
    return 0 if not missing and not changed and not no_reference else 1


def select_models(roster: list[ModelConfig], requested: list[str] | None):
    """The models a stage works on: those named with --model, else every subject.

    Judges are never included by default (see config.subjects). A name that
    is not in the roster is an error; it used to select nothing, silently.
    """
    if not requested:
        return subjects(roster)
    known = {m.id for m in roster}
    unknown = [m for m in requested if m not in known]
    if unknown:
        raise ConfigError(f"unknown model id(s): {', '.join(unknown)}")
    return [m for m in roster if m.id in requested]


def cmd_run(args) -> int:
    """Translate; `--passes N` retries failures, `--wait` outlasts usage limits.

    `--passes 3` is the retry policy run 2026-09 applied through lane.sh:
    later passes only redo tasks that failed (completed and refused ones are
    skipped), and passes stop early once one has nothing left to redo.
    """
    models = select_models(load_models(Path(args.models)), args.model)
    tasks = load_tasks(Path(args.tasks))
    if args.task:
        tasks = [t for t in tasks if t.id in args.task]
    instances = expand(tasks, iterations=args.iterations)
    passages = {p.id: p for p in load_manifest(Path(args.manifest))}
    system_prompt = Path(args.prompt).read_text(encoding="utf-8")

    # Every runner is built before the first call, so a missing key is
    # reported up front rather than as a 401 recorded against the model.
    runners = {}
    not_run: list[str] = []
    for model in models:
        try:
            runners[model.id] = build_runner(model, Path(args.manual))
        except ConfigError as exc:
            print(f"{model.id}: not run - {exc}")
            not_run.append(model.id)
    passes = max(1, args.passes)

    def invocation() -> Pass:
        limits: list[str] = []
        done = 0
        for number in range(1, passes + 1):
            if passes > 1:
                print(f"== pass {number} of {passes} ==")
            this_pass = 0
            for model in models:
                if model.id not in runners:
                    continue
                executed = run_model(
                    runner=runners[model.id],
                    model_id=model.id,
                    instances=instances,
                    passages=passages,
                    corpus_root=Path(args.corpus),
                    system_prompt=system_prompt,
                    runs_root=Path(args.runs),
                    run_id=args.run,
                    # --force means "redo everything once", not once per pass.
                    force=args.force and number == 1,
                    on_usage_limit=limits.append,
                )
                print(
                    f"{model.id}: {executed} executed, "
                    f"{len(instances) - executed} skipped"
                )
                this_pass += executed
            done += this_pass
            if this_pass == 0:
                break  # nothing failed last pass, so nothing to retry
        print(f"total executed: {done}")
        return Pass(done, limits[-1] if limits else None)

    label = ",".join(m.id for m in models) or "none"
    with run_activity(Path(args.runs), args.run, label):
        if args.wait:
            completed = drive(invocation, wait=True, finished=lambda _p: True)
        else:
            completed = invocation().stopped is None
    if not_run:
        print(f"not run: {', '.join(not_run)}")
        return 1
    return 0 if completed else 1


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
    except OSError:
        # No roster file at all. A roster that exists but does not load is
        # a ConfigError and stops here: read as "no judge declared", it would
        # send a declared judge's id to the default endpoint instead.
        declared = {}
    if judge_id in declared:
        model = declared[judge_id]
        # Each stage enforces its own output shape. The judge entry pins the
        # rubric schema; a pairwise verdict forced into that shape contains no
        # "winner" field at all, so every comparison would be discarded and
        # the stage would report zero work with no error.
        if json_schema is not None:
            model = replace(model, json_schema=json_schema)
        return build_runner(model, Path(manual_root))
    # An undeclared judge becomes a plain API call, which for Gemini would be
    # exactly the route the operator has ruled out.
    if requires_antigravity(judge_id):
        raise ConfigError(
            f"judge {judge_id}: Gemini may only be reached through Antigravity; "
            f"declare it in models.yaml with runner: agy"
        )
    if not api_key:
        raise ConfigError(
            f"judge {judge_id} is not declared in models.yaml, so it would be "
            f"called at {endpoint or DEFAULT_ENDPOINT} with OPENROUTER_API_KEY, "
            f"which is not set"
        )
    return ApiRunner(
        endpoint=endpoint or DEFAULT_ENDPOINT,
        model=judge_id,
        api_key=api_key or "",
        temperature=0.0,
    )


def judge_fingerprint_for(args, json_schema: str | None, prompt_frame: str) -> str:
    """The fingerprint of the judge a stage is about to use. Makes no calls."""
    try:
        declared = {m.id: m for m in load_models(Path(args.models))}
    except OSError:
        declared = {}
    route = judge_route(
        args.judge_model,
        declared.get(args.judge_model),
        args.judge_endpoint or DEFAULT_ENDPOINT,
        json_schema=json_schema,
    )
    return judge_fingerprint(args.judge_model, route, prompt_frame)


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
    fingerprint = judge_fingerprint_for(
        args, args.judge_schema, rubric_prompt_frame(template)
    )
    print(f"judge {args.judge_model}, fingerprint {fingerprint}")

    forced = {"left": args.force}

    def one_pass() -> Pass:
        reasons: list[str] = []
        judged = judge_run(
            judge_runner=judge_runner,
            judge_id=args.judge_model,
            run_id=args.run,
            runs_root=Path(args.runs),
            judgments_root=Path(args.judgments),
            passages=passages,
            corpus_root=Path(args.corpus),
            template=template,
            # --force redoes everything once, not on every pass of a loop.
            force=forced.pop("left", False),
            judge_fingerprint=fingerprint,
            on_stop=reasons.append,
        )
        print(f"judged: {judged}")
        return Pass(judged, reasons[0] if reasons else None)

    if args.wait or args.follow:
        # Repeat until a pass finds nothing to judge; with --follow, keep
        # going while an `ati run` process is still producing translations.
        completed = drive(
            one_pass, wait=args.wait, active=_run_is_active(args) if args.follow else _never
        )
    else:
        completed = one_pass().stopped is None
    return 0 if completed else 1


def _never() -> bool:
    return False


def _run_is_active(args):
    return lambda: bool(active_markers(Path(args.runs), args.run))


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
    pool = [m.id for m in select_models(declared, args.model) if m.runner != "agy"]

    passages = {p.id: p for p in load_manifest(Path(args.manifest))}
    template = Path(args.pairwise_prompt).read_text(encoding="utf-8")
    fingerprint = judge_fingerprint_for(
        args, args.pairwise_schema, pairwise_prompt_frame(template)
    )
    print(f"judge {args.judge_model}, fingerprint {fingerprint}")

    # --target N: keep running batches of --max-comparisons, refitting the
    # ratings between batches so pairing stays adaptive, until N comparisons
    # exist or nothing is left to compare. --wait alone finishes one batch,
    # waiting out usage limits. Without either, one batch, as before.
    target = args.target
    if target is None and args.wait:
        target = len(load_comparisons(Path(args.pairwise), args.run)) + args.max_comparisons

    def one_pass() -> Pass:
        existing = load_comparisons(Path(args.pairwise), args.run)
        budget = args.max_comparisons
        if target is not None:
            budget = min(budget, target - len(existing))
            if budget <= 0:
                print(f"{len(existing)} comparisons on disk; target {target} reached")
                return Pass(0)
        reasons: list[str] = []
        executed = run_pairwise(
            judge_runner=judge_runner,
            run_id=args.run,
            runs_root=Path(args.runs),
            pairwise_root=Path(args.pairwise),
            corpus_root=Path(args.corpus),
            passages=passages,
            template=template,
            models=pool,
            max_comparisons=budget,
            ratings=to_elo(fit(existing)) if existing else {},
            judge_id=args.judge_model,
            judge_fingerprint=fingerprint,
            on_stop=reasons.append,
        )
        print(f"{executed} comparisons executed")
        return Pass(executed, reasons[0] if reasons else None)

    if target is not None:
        completed = drive(one_pass, wait=args.wait)
    else:
        completed = one_pass().stopped is None

    comparisons = load_comparisons(Path(args.pairwise), args.run)
    if comparisons:
        anchor = next((m.id for m in declared if m.anchor == "weak"), None)
        final = to_elo(fit(comparisons), anchor_id=anchor)
        for model_id, rating in sorted(final.items(), key=lambda kv: -kv[1]):
            print(f"  {rating:7.0f}  {model_id}")
    return 0 if completed else 1


def cmd_status(args) -> int:
    """Where a run stands: translations, judgments, comparisons. No calls."""
    roster = load_models(Path(args.models))
    models = [m.id for m in select_models(roster, args.model)]
    tasks = load_tasks(Path(args.tasks))
    if args.task:
        tasks = [t for t in tasks if t.id in args.task]
    instance_ids = [i.instance_id for i in expand(tasks, iterations=args.iterations)]
    progress = run_progress(args.run, Path(args.runs), models, instance_ids)

    judgments = comparisons = None
    try:
        rubric = Path(args.judge_prompt).read_text(encoding="utf-8")
        pairwise_template = Path(args.pairwise_prompt).read_text(encoding="utf-8")
    except OSError:
        rubric = pairwise_template = None
    if rubric is not None:
        fingerprint = judge_fingerprint_for(args, args.judge_schema, rubric_prompt_frame(rubric))
        pending = pending_judgments(
            args.run, Path(args.runs), Path(args.judgments),
            judge_id=args.judge_model, judge_fingerprint=fingerprint,
            warn=lambda _msg: None,
        )
        judgments = {
            "missing": sum(1 for *_, why in pending if why == "missing"),
            "stale": sum(1 for *_, why in pending if why == "stale"),
            "judge_id": args.judge_model,
            "fingerprint": fingerprint,
        }
        pair_fp = judge_fingerprint_for(
            args, args.pairwise_schema, pairwise_prompt_frame(pairwise_template)
        )
        recorded = load_comparisons(Path(args.pairwise), args.run)
        comparisons = {
            "count": len(recorded),
            "pairs": len({tuple(sorted((a, b))) for a, b, _ in recorded}),
            "stale": len(
                pending_comparisons(
                    Path(args.pairwise), args.run, Path(args.runs),
                    judge_id=args.judge_model, judge_fingerprint=pair_fp,
                )
            ),
        }
    print(
        render_status(
            args.run,
            progress,
            len(instance_ids),
            len(active_markers(Path(args.runs), args.run)),
            judgments,
            comparisons,
        )
    )
    return 0


def cmd_recheck(args) -> int:
    """Re-apply the current output guards to records already on disk."""
    changes = recheck(args.run, Path(args.runs), apply=args.apply)
    for path, what in changes:
        print(f"  {path.parent.name} {path.stem}: {what}")
    verb = "reclassified" if args.apply else "would be reclassified (dry run; --apply to write)"
    print(f"{len(changes)} records {verb}")
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

    # Read before anything in results/ is rewritten: a legacy anchors.json is
    # recognised as this run's own by the results file beside it.
    anchors_path = results_dir / "anchors.json"
    baseline_path = Path(args.baseline) if args.baseline else anchors_path
    baseline_run, previous = read_anchor_baseline(baseline_path, args.run)
    if baseline_run:
        print(f"anchor baseline: run {baseline_run} ({baseline_path})")
    else:
        print(
            "anchor baseline: none - drift not checked. Pass --baseline with an "
            "earlier run's anchors.json to check it."
        )

    drift = anchor_drift(results, previous)
    comparable = is_comparable(drift)
    metadata = {
        "judge_id": args.judge_model,
        "date": date.today().isoformat(),
        **report_provenance(args),
        "anchor_baseline": baseline_run,
    }

    comparisons = load_comparisons(Path(args.pairwise), args.run)
    pairwise_ratings, pairwise_intervals = {}, {}
    anchor = None
    if comparisons:
        anchor = next(
            (m.id for m in load_models(Path(args.models)) if m.anchor == "weak"), None
        )
        pairwise_ratings = to_elo(fit(comparisons), anchor_id=anchor)
        pairwise_intervals = bootstrap_ci(comparisons, anchor_id=anchor)
        rated = len(pairwise_ratings)
        metadata["pairwise_coverage"] = {
            "comparisons": len(comparisons),
            "pairs": len({tuple(sorted((a, b))) for a, b, _ in comparisons}),
            "possible_pairs": rated * (rated - 1) // 2,
        }

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
            docs_prefix=_docs_prefix(results_dir),
        ),
    )
    write_json_atomic(
        anchors_path, anchors_file(args.run, results, baseline_run, previous)
    )

    print(f"wrote {results_dir / f'{args.run}.json'}")
    print(f"wrote {results_dir / 'leaderboard.md'}")
    if metadata.get("pending_judgments"):
        print(
            f"WARNING: {metadata['pending_judgments']} outputs have no current "
            "judgment; run `ati judge` before treating this ranking as final"
        )
    if not comparable:
        print("WARNING: anchor drift exceeds threshold; run is not comparable "
              "to previous runs")
    return 0


def _docs_prefix(results_dir: Path) -> str:
    """Relative link from the leaderboard to docs/, wherever it is written.

    Hardcoded as ../docs/, the links broke as soon as a leaderboard was
    written anywhere but results/ - every archived one has dead links.
    """
    try:
        rel = os.path.relpath(Path("docs").resolve(), Path(results_dir).resolve())
    except ValueError:  # a different drive on Windows
        return "../docs/"
    return rel.replace(os.sep, "/") + "/"


def report_provenance(args) -> dict:
    """What produced a report: input hashes, judge fingerprints, code hash.

    Recorded so a published ranking can be tied to the exact roster, corpus,
    prompts and code behind it. None of it enters any score.
    """
    inputs = input_fingerprints(
        {
            "models": args.models,
            "weights": args.weights,
            "tasks": args.tasks,
            "manifest": args.manifest,
            "translate_prompt": args.prompt,
            "judge_prompt": args.judge_prompt,
            "pairwise_prompt": args.pairwise_prompt,
            "judge_schema": args.judge_schema,
            "pairwise_schema": args.pairwise_schema,
            "slop_list": args.slop,
            "defaults": args.defaults,
        }
    )
    provenance: dict = {"inputs": inputs, "code": code_fingerprint()}
    try:
        rubric_template = Path(args.judge_prompt).read_text(encoding="utf-8")
        pair_template = Path(args.pairwise_prompt).read_text(encoding="utf-8")
    except OSError:
        return provenance  # no prompts, so no fingerprint to compare against
    rubric_fp = judge_fingerprint_for(
        args, args.judge_schema, rubric_prompt_frame(rubric_template)
    )
    provenance["judge_fingerprints"] = {
        "rubric": rubric_fp,
        "pairwise": judge_fingerprint_for(
            args, args.pairwise_schema, pairwise_prompt_frame(pair_template)
        ),
    }
    provenance["pending_judgments"] = len(
        pending_judgments(
            args.run,
            Path(args.runs),
            Path(args.judgments),
            judge_id=args.judge_model,
            judge_fingerprint=rubric_fp,
            warn=lambda _msg: None,  # aggregate has already reported them
        )
    )
    return provenance


def cmd_bench(args) -> int:
    """Run the whole pipeline: fetch, run, judge, report."""
    print("== fetch ==")
    if cmd_fetch(args) != 0:
        print(
            "stopping: corpus is incomplete or has changed. Scoring against a "
            "broken corpus would produce meaningless results."
        )
        return 1

    incomplete = []
    if not args.skip_run:
        print("== run ==")
        if cmd_run(args) != 0:
            incomplete.append("translation")

    if not args.skip_judge:
        print("== judge ==")
        if cmd_judge(args) != 0:
            incomplete.append("judging")

    print("== report ==")
    status = cmd_report(args)
    # The report is still written - partial results are worth seeing - but a
    # pipeline that stopped early must not exit as though it had finished.
    if incomplete:
        print(
            f"WARNING: {' and '.join(incomplete)} stopped before the end; the "
            "leaderboard is partial. Rerun the same command to resume."
        )
        return 1
    return status


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
    common.add_argument(
        "--max-comparisons", type=int, default=DEFAULT_MAX_COMPARISONS,
        help="pairwise: comparisons per batch",
    )
    # No default: a forgotten --run used to write to runs/latest and read
    # results/latest.json without a word.
    common.add_argument("--run", default=None, help="run id, e.g. 2026-10")
    common.add_argument("--force", action="store_true")
    common.add_argument(
        "--passes", type=int, default=1,
        help="run: passes over the suite; later passes redo only failures",
    )
    common.add_argument(
        "--wait", action="store_true",
        help="run/judge/pairwise: wait out usage limits and resume",
    )
    common.add_argument(
        "--follow", action="store_true",
        help="judge: keep going while an `ati run` on this run is still active",
    )
    common.add_argument(
        "--target", type=int, default=None,
        help="pairwise: keep running batches until this many comparisons exist",
    )
    common.add_argument(
        "--baseline",
        default=None,
        help="report: an earlier run's anchors.json to measure anchor drift "
        "against (default: <results>/anchors.json)",
    )

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
    sub.add_parser(
        "status",
        help="progress of a run: translations, judgments, comparisons (no calls)",
        parents=[common],
    )
    recheck_parser = sub.add_parser(
        "recheck",
        help="re-apply the current output guards to run records (no calls)",
        parents=[common],
    )
    recheck_parser.add_argument(
        "--apply", action="store_true", help="write the changes (default: dry run)"
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
    add_parser.add_argument(
        "--keep-headers",
        action="store_true",
        help="do not strip scraped reader furniture from the top of each file",
    )

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

    handlers = {
        "add": cmd_add,
        "fetch": cmd_fetch,
        "run": cmd_run,
        "judge": cmd_judge,
        "pairwise": cmd_pairwise,
        "report": cmd_report,
        "bench": cmd_bench,
        "status": cmd_status,
        "recheck": cmd_recheck,
    }
    if args.command not in ("add", "fetch") and not args.run:
        print(f"error: `ati {args.command}` needs --run, e.g. --run 2026-10")
        return 2
    try:
        defaults = load_defaults(Path(args.defaults))
        if args.judge_model is None:
            args.judge_model = defaults.get("judge_model", DEFAULT_JUDGE_MODEL)
        if args.judge_endpoint is None:
            args.judge_endpoint = defaults.get("judge_endpoint", DEFAULT_ENDPOINT)
        if args.iterations is None:
            args.iterations = int(defaults.get("iterations", DEFAULT_ITERATIONS))
        return handlers[args.command](args)
    except ConfigError as exc:
        # A configuration problem - including a refused Gemini route - is
        # reported as one line, before any model call, not as a traceback.
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
