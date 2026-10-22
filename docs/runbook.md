# Runbook — a benchmark run, end to end

How to operate a run from an empty corpus to a published leaderboard. Every
command here is resumable: rerunning it skips work already done. Every stage
except `add` and `fetch` needs `--run`; use `YYYY-MM` (e.g. `2026-10`).

Why things are the way they are is in `docs/run-notes-2026-08.md` and
`docs/run-notes-2026-09.md`; this page is only what to do. Run `2026-09` was
driven by the scripts in `scripts/2026-09/`, kept as that run's provenance;
the commands below replace them (`lane.sh` → `ati run --passes 3`,
`judge_driver.py` → `--wait`/`--follow`/`--target`, `status.py` →
`ati status`, `reclassify.py` → `ati recheck`).

## 0. Before you start

```bash
pip install -e ".[dev]"
python -m pytest -q          # hermetic; makes no model calls
```

- **Keys** go in `.env` (gitignored; see `.env.example`), one per provider:
  `COMMANDCODE_API_KEY`, `OPENROUTER_API_KEY`. A model's key is read only from
  its own `api_key_env`. A missing key is reported before any call and that
  model is not run; nothing ever falls back to another provider's key.
- **Antigravity** (`agy`) must be installed and signed in to the operator's
  account. It is the only route to Gemini: the config refuses a Gemini slug on
  any other runner. This repository never reads or writes `~/.gemini`.
- **Nothing else** needs setting up. `config/ati.yaml` holds exactly three
  defaults (`judge_model`, `judge_endpoint`, `iterations`); every other setting
  is a flag, and the file refuses keys it does not know.

## 1. Corpus

```bash
ati add raw/<work>/chapter_*.txt --work <slug> --lang ko --genre murim \
    --type consistency --url <source url>
ati fetch
```

- Clean the source files first. The intake strips common web-reader furniture
  from the top of each file (`--keep-headers` turns that off), but anything
  else in the file will be translated and judged. See run-notes-2026-09 §1.
- `ati add` writes passage text to `corpus/passages/` (gitignored) and
  pointers and hashes to `corpus/manifest.json` and `config/tasks.yaml`.
  **Commit those two files before the run starts** - they are the run's
  inputs and contain no source text.
- Lost the passage text (fresh clone, wiped directory)? `ati fetch` lists what
  is missing and from which source file. Run the same `ati add` command again:
  every file that reproduces its manifest hash is written back ("restored").

## 2. Roster

`config/models.yaml` lists the models under test and the judges.

- Judges carry `role: judge` and are never translation subjects. An `agy` row
  is never a subject unless named with `--model`.
- One weak anchor (`anchor: weak`) is required for pinned head-to-head
  ratings and for drift detection. `anchor: strong` is supported; none is
  configured today.
- **Reasoning effort** in `params` is configured, not verified: providers
  ignore fields they do not understand. After the first chapters land, check
  `reasoning_tokens` in the run records.
- **Provider output caps** bind silently. After the first pass, look for
  `finish_reason: length` at a round number (4,096, 5,120, ...) and set
  `params.max_tokens` for that row (run-notes-2026-09 §4). An endpoint that
  takes `max_completion_tokens` instead (OpenAI's own) reports that limit in
  the error; it includes reasoning tokens.
- **Command Code rows must stream** (`stream: true`): its gateway drops a
  buffered request after ~125 s. `timeout_s` (default 900) bounds a silence on
  a stream; a stream is also cut at one hour in total.
- Changing a model's params mid-run changes its measurement. Re-run that
  model in full with `--force` so it is measured under one set of conditions.

## 3. Translate

Run one lane per provider, in parallel, each with its own model list:

```bash
ati run --run 2026-10 --passes 3 --wait --model laguna-s-2.1 --model hy3 ...
ati run --run 2026-10 --passes 3 --wait --model solar-pro4 ...
```

- `--passes 3` is the retry policy: later passes redo only failures and stop
  early when nothing is left. Refusals are not retried; they are measurements.
- `--wait` waits out a usage limit and resumes (Command Code has a 5-hour
  window that a full max-effort run exhausts). Without it, rerun the same
  command after the reset.
- While a lane works it keeps a heartbeat marker in `runs/<run>/.active/`;
  `ati judge --follow` uses it to know translations are still coming.
- `ati status --run 2026-10` shows per-model ok / refused / failed / missing,
  average latency, output ratio and which route answered. It makes no calls.
- If an output guard is added or changed mid-run, `ati recheck --run 2026-10`
  shows which existing records the current guards would reclassify;
  `--apply` rewrites them. The judge stage then re-judges or skips them.

## 4. Judge (rubric)

```bash
ati judge --run 2026-10 --wait --follow
```

The judge is Gemini through Antigravity, on the operator's own Google
account. That account cannot be re-provisioned, so these protections are
always on and must not be weakened:

- **CallGate**: one agy call at a time across every process on the machine,
  at least 10 s between calls, fair turn-taking between processes, and every
  call made from an empty working directory.
- The prompt tells the judge to use no tools; `--print-timeout` matches the
  row's `timeout_s`; the schema path is absolute.
- The stage **stops at the first usage-limit answer**, after two unreachable
  items in a row, or after three items in a row with no parseable answer.
- `--wait` then sleeps exactly the "Resets in XhYmZs" Antigravity states,
  plus 3 minutes (an hour if no time is given, 10 minutes for other stops),
  and gives up after 6 stops in a row that made no progress. At most one
  refused call reaches the account per quota window.
- `--follow` keeps the stage polling (every 5 min) while an `ati run` lane is
  still active on the run, so judging can overlap translation.

**Quota.** Observed: ~180 judge calls per rolling 5-hour window (run-notes-2026-09
§6). The rubric stage costs one call per output (20 models x 11 chapters = 220)
plus retries; plan on two windows.

**Changing the judge, its prompt or its schema** makes every judgment it
produced stale (each records a judge fingerprint), and the next `ati judge`
re-judges them - one call each. `ati status` shows the stale count before you
spend anything. Judgments that predate fingerprints are treated as current.

## 5. Head-to-head

```bash
ati pairwise --run 2026-10 --target 300 --max-comparisons 25 --wait
```

- Two judge calls per comparison (both orders): 300 comparisons is 600 calls,
  three to four quota windows. Ratings are refitted between batches of
  `--max-comparisons`, so pairing stays adaptive.
- The rubric and pairwise stages can run at the same time - the gate queues
  them - but they then share one quota window. Running them one after the
  other in a single terminal is simpler to reason about.
- Do not publish or quote partial standings (run-notes-2026-08: they had to
  be retracted once).

## 6. Report

```bash
ati report --run 2026-10
```

- Writes `results/<run>.json` and `results/leaderboard.md`. No model calls.
- The results file records a hash of every input (roster, tasks, manifest,
  prompts, schemas, weights, slop list), a hash of the harness code, and both
  judge fingerprints.
- **Anchor drift** is measured against the previous run's
  `results/anchors.json`. The leaderboard says which run it compared against,
  or that drift was not checked. To compare against an archived run, pass
  `--baseline archive/<run>/results/anchors.json`. Drift across different
  corpora says little: expect "not comparable" when the work changes.
- Check the leaderboard's notes: "Not yet judged" means `ati judge` has work
  left; "Judge errors" means some composites rest on fewer items.
- Reweighting is free: edit `config/weights.yaml`, rerun `ati report`.

## 7. Close the run

1. Write `docs/run-notes-<run>.md`: what was measured, what broke, what changed.
2. Archive the published artifacts into `archive/<run>-<work>/`: `results/`,
   `pairwise/<run>/`, and copies of `config/models.yaml`, `config/tasks.yaml`
   and `corpus/manifest.json`. `runs/` and `judgments/` contain translation
   text and are never committed.
3. Leave `results/anchors.json` where it is: the next run's report reads it as
   its baseline.
4. In the archived `leaderboard.md`, change its `../docs/` links to
   `../../../docs/`: the copy sits two directories deeper than `results/`.
5. Point the README's leaderboard link at the archived leaderboard.

Working directories (`runs/`, `judgments/`, `pairwise/`, `results/`,
`manual/`) are gitignored; only archived copies are published.

## Troubleshooting

| Symptom | Meaning | Action |
|---|---|---|
| `<model>: not run - X_API_KEY is not set` | the key is missing | add it to `.env`; rerun |
| HTTP 524 from Command Code | buffered request dropped at ~125 s | `stream: true` on the row |
| `finish_reason: length` at a round number | output token cap reached | `params.max_tokens` on the row (`params.max_completion_tokens` where the endpoint takes that) |
| `stopping after N judgments: ... RESOURCE_EXHAUSTED ... Resets in ...` | judge quota spent | nothing: `--wait` resumes; or rerun after the reset |
| `N items in a row got no parseable answer` | the judge's answers stopped matching the schema | check `agy` and the prompt before rerunning |
| `6 stops in a row without progress; leaving it for the operator` | the limit is not lifting | check the account; rerun later |
| status shows stale judgments you did not expect | judge, prompt or schema changed | revert the change, or accept a re-judge |
