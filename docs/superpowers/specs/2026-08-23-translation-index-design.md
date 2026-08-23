# Artificial Translation Index — Design

Date: 2026-08-23
Status: approved for planning

## Purpose

Rank AI models on their ability to translate web serial fiction (Chinese,
Japanese, Korean → English), so hobbyists can pick a model for translating
texts that have no official translation.

The project publishes **rankings and methodology**. It does not publish
translations, and it does not redistribute source text.

## Scope

In v1:

- A Python CLI that runs a fixed task suite against configured models,
  scores the outputs with a judge model, and emits a Markdown leaderboard.
- A committed rubric and weighting scheme others can reweight.
- A corpus manifest of pointers that rebuilds the corpus locally.

Deferred, with the design recorded and a stub in place:

- Blind pairwise tie-breaking among top models.
- A generated HTML leaderboard. Markdown first; HTML once there are real
  results worth looking at.

## Non-goals

- Producing or hosting translations of any work.
- Reference-based automatic metrics (COMET, chrF). No ML dependencies.
- Evaluating non-fiction or non-CJK→EN pairs.

## Prior art

[EQ-Bench Creative Writing v3](https://eqbench.com/creative_writing.html)
is the closest existing benchmark in shape — LLM-judged creative output,
rubric score plus anchored pairwise Elo, deterministic slop and repetition
metrics, disclosed judge bias. It evaluates English writing only and has no
translation benchmark, so this project does not duplicate it. Four of its
design choices are adopted below: calibration anchors, deterministic
side-metrics, multiple iterations, and the bias controls on pairwise
comparison. Its published cost of roughly $10 per model for a full rubric
plus Elo pass is the budget anchor for this project.

## Legal and ethical constraints

These are requirements, not preferences:

1. No source text, and no fan or official translation text, is ever
   committed to the repository or included in published output.
2. The corpus manifest stores only: source URL, chapter identifier,
   character offset range, sha256 of the extracted passage, and metadata
   (language, genre, length, whether a reference translation exists).
3. `fetch` rebuilds the local corpus from the manifest and verifies each
   passage against its hash. Corpus files are gitignored.
4. The judge is instructed to write rationales without quoting more than a
   short phrase. Published rationales are checked against a maximum quoted
   span length before release.
5. The public-domain control set is exempt from (1) and may be committed.

## Architecture

Four stages. Each reads the previous stage's output from disk and writes
JSON. Each is separately runnable and resumable, so one failed provider
does not force a full rerun.

```
corpus/ (local, gitignored)
   |  run
   v
runs/<run_id>/<model_id>/<task_id>.<iter>.json
   |  judge
   v
judgments/<run_id>/<model_id>/<task_id>.<iter>.json
   |  report
   v
results/<run_id>.json  +  results/leaderboard.md
```

### Runners

Model access is heterogeneous: OpenRouter API, subscription-backed CLIs
(Claude Code, opencode), and occasionally a web-only model. One interface
absorbs all of it.

```python
class Runner(Protocol):
    def run(self, system_prompt: str, source_text: str) -> RunResult: ...

@dataclass
class RunResult:
    text: str
    latency_s: float
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None   # None for subscription access
    refused: bool
    error: str | None
```

Implementations:

- **ApiRunner** — OpenAI-compatible HTTP, used for OpenRouter and any
  direct provider endpoint. Reports real token counts and cost.
- **CliRunner** — subprocess driven by a command template in config.
  Source text on stdin, translation on stdout. Covers `claude -p`,
  `opencode run`, and anything added later without new code.
- **ManualRunner** — writes each prompt to `manual/<model>/<task>.prompt`
  and reads back `manual/<model>/<task>.response`. Escape hatch for
  web-only models. Skips tasks with no response file present.

`cost_usd = None` propagates to the report as **N/A**, never as zero.
Subscription models therefore appear in the quality ranking but not in the
value ranking.

Models are declared in `config/models.yaml`:

```yaml
- id: anthropic/claude-opus-5
  runner: api
  endpoint: openrouter
  model: anthropic/claude-opus-5
- id: claude-code-opus-5
  runner: cli
  command: ["claude", "-p", "--model", "opus"]
- id: anchor-strong
  runner: api
  endpoint: openrouter
  model: <pinned at first run>
  anchor: strong
- id: anchor-weak
  runner: api
  endpoint: openrouter
  model: <pinned at first run>
  anchor: weak
```

### Calibration anchors

Two models are marked as anchors — one known-strong, one known-weak — and
run in every evaluation. Their purpose is to detect drift, not to be
ranked: if the judge, the prompt, or a provider changes behavior between
runs, the anchors move and the report flags it.

Anchor scores are published alongside the leaderboard for every run. A run
whose anchor composite differs by more than 5 points from the previous run
is marked **not comparable** to prior runs in the report, rather than
silently compared.

Anchor model choices are pinned at first run and changed only with a new
methodology version.

### Tasks

`config/tasks.yaml` declares the suite. Three task types:

**passage** — one excerpt of roughly 800 characters, translated in a single
call. v1 corpus: 4 passages per source language (CN, JP, KR) plus 3
public-domain control passages = 15 passage tasks.

**consistency** — a series of 10 consecutive chapters from one work, each
translated in an **independent call with no shared context**. This measures
whether a model renders names, honorifics, and setting-specific terms
(cultivation ranks, skills, titles) the same way every time — the failure
mode hobbyists hit hardest. One series per source language.

**refusal** — 6 passages containing violence, sexual content, or otherwise
dark material typical of web serials. Produces two numbers: hard refusal
rate, and a sanitization score for outputs that translated but softened.

Cost and speed are not tasks. Latency, tokens, and cost are recorded on
every call in every task and aggregated at report time.

All models receive the same system prompt, stored in
`config/prompts/translate.md` and hashed into the run record.

### Iterations

`--iterations N` repeats every task N times, default 1. Per-axis scores
average across iterations, and the report shows the standard deviation so a
model that is erratic is visible as erratic rather than as its mean. A
subset of tasks can be run at higher N via `--iterations-tasks`.

Generation temperature is fixed in `config/models.yaml` per model, default
0.3 — translation rewards determinism more than creative writing does.

### Judging

A single pinned judge model, temperature 0, returning structured JSON
against a schema. The judge never evaluates its own outputs: if a model
under test shares the judge's identity, its judgments come from the
secondary judge instead.

Each judged output returns, per axis, an integer 1–5 and a one-line
rationale. Axes and their 1/3/5 anchors live in `docs/rubric.md` and are
included verbatim in the judge prompt.

| Axis | Measures |
|---|---|
| accuracy | Does the English convey what the source says? |
| terminology | Are names and setting terms rendered consistently? |
| register | Is character voice, formality, and tone preserved? |
| fluency | Does it read as natural English prose? |
| fidelity | Absence of omissions, additions, and hallucinated content |
| sanitization | Absence of softening or censoring of source content |

`terminology` on a **passage** task is judged within the passage. On a
**consistency** task it is computed, not judged: the judge extracts a term
table per chapter, and the harness scores stability as the fraction of
source terms rendered identically across all 10 chapters. This is
deterministic arithmetic over judge-extracted tables, which is more
trustworthy than asking a judge to hold 10 chapters in mind at once.

Reference translations, where the manifest records that one exists, are
shown to the judge as a calibration anchor — not as ground truth to match.

**Judge validation.** The top 5 models by composite are re-judged by a
secondary judge model. The report publishes the agreement rate between
judges. Judge self-preference is disclosed, not corrected: the judge's
own family is named in the report so readers can discount accordingly. A
human spot-check of a sampled subset is part of the methodology and is
recorded in `docs/spot-checks.md` when performed.

### Deterministic text metrics

Computed from the English output with no judge involved, and therefore
free and perfectly reproducible:

- **slop** — frequency of phrases from `config/slop_list.txt`, a list of
  LLM-tell constructions ("a testament to", "couldn't help but", "a mix
  of X and Y") plus translationese markers common in MTL output.
- **repetition** — summed frequency of the top common words, bigrams, and
  trigrams, normalized by output length. Higher means more repetitive.

Both are reported per model as standalone columns, normalized per 1,000
output words. Neither enters the composite; they are diagnostic signals
that a reader can weigh directly, and they are the cheapest defense against
a judge that rewards fluent-sounding filler.

The slop list is seeded from
[EQ-Bench's published approach](https://eqbench.com/about.html) and extended
with translation-specific entries as they are observed.

### Scoring

Composite is quality only, 0–100:

```
composite = 100 * sum(weight[axis] * (score[axis] - 1) / 4)
```

Default weights in `config/weights.yaml`:

| axis | weight |
|---|---|
| accuracy | 0.25 |
| terminology | 0.20 |
| fluency | 0.15 |
| register | 0.15 |
| fidelity | 0.15 |
| sanitization | 0.10 |

Reported alongside the composite, never folded into it:

- **refusal rate** — percentage of refusal tasks hard-refused.
- **cost** — USD per 1,000 source characters, or N/A.
- **speed** — median seconds per 1,000 source characters.
- **value** — composite per USD per 1,000 source characters. Omitted for
  models with no cost data.
- **slop**, **repetition** — as above.

Weights are configurable so a reader who cares only about censorship or
cost can recompute the ranking from committed results without rerunning
anything.

### Pairwise stage (deferred — design only)

`ati/pairwise.py` ships as an interface that raises `NotImplementedError`.
When implemented it works as follows, so that v1's data format does not
need to change:

- Only the top N models by composite (default 5) are paired, on the
  passage tasks only.
- The judge sees two anonymized translations of the same source passage and
  returns a winner plus a margin of 1–5.
- Every pairing runs in **both directions** (A vs B and B vs A) and the
  results are averaged, cancelling position bias.
- Outputs are **truncated to a fixed character count** before comparison,
  cancelling length bias.
- Ratings use **Glicko weighted by win margin**, with the two calibration
  anchors pinned to fixed ratings so adding a model does not shift the
  existing field.

Pairwise output is a separate rating column in the report, never merged
into the rubric composite.

### Report

`results/<run_id>.json` is the machine-readable record: per model, per
task, per iteration, per axis scores, plus run metadata (date, judge model
and version, prompt hash, weights hash, corpus manifest hash, anchor
scores). `results/leaderboard.md` is generated from it — a ranked table,
a per-axis breakdown, the diagnostic columns, and a methodology section
pointing at the rubric.

Every published run states which model versions were tested and on what
date, because model behavior changes under a stable name.

## CLI

```
ati fetch                        # rebuild local corpus from manifest, verify hashes
ati run [--model ID] [--task ID] [--iterations N]
ati judge [--run RUN_ID]
ati report [--run RUN_ID]
```

`run` and `judge` are idempotent: a task whose output file already exists
is skipped unless `--force`. Adding one new model is therefore an O(1)
operation rather than a full rerun.

## Repository layout

```
ati/
  __init__.py
  cli.py            # argument parsing, stage dispatch
  runners/          # base.py, api.py, cli.py, manual.py
  corpus.py         # manifest load, fetch, hash verification
  tasks.py          # task loading and expansion
  judge.py          # judge prompt construction, response parsing
  scoring.py        # aggregation, weights, term stability, value metric
  textmetrics.py    # slop and repetition
  report.py         # Markdown generation
  pairwise.py       # stub: interface only, raises NotImplementedError
config/
  models.yaml  tasks.yaml  weights.yaml  slop_list.txt
  prompts/translate.md  prompts/judge.md
corpus/
  manifest.json     # committed
  passages/         # gitignored
docs/
  rubric.md  methodology.md  spot-checks.md
results/
tests/
```

## Error handling

- A runner failure records `error` on the run file and continues to the
  next task. It does not abort the run.
- A model whose failed tasks exceed 20% of its suite is excluded from the
  published ranking and listed separately as incomplete.
- Judge responses that fail schema validation are retried twice, then
  recorded as `judge_error` and excluded from that model's axis average,
  with the exclusion count shown in the report.
- A hash mismatch during `fetch` is fatal for that passage: the source has
  changed, and the task is marked unavailable rather than silently scored
  against different text.

## Testing

Pure logic only, with a fake runner and canned judge responses. No live API
calls in tests.

- Score aggregation and weighting, including axis exclusion on judge error.
- Iteration averaging and standard deviation.
- Term-stability computation across a synthetic 10-chapter term table.
- Slop and repetition metrics against fixture text with known counts.
- Manifest hashing and mismatch detection.
- Runner interface conformance for all three implementations.
- Cost handling: `None` propagates to N/A and never to zero.
- Anchor drift detection: a synthetic prior run triggers the
  not-comparable flag.
- Report generation from a fixture results file.

## Decisions deferred to implementation

- Which specific works form the corpus. Selection criteria: publicly
  readable, representative of a common genre, at least 10 consecutive
  chapters available for the consistency series.
- Which model serves as primary judge, and which as secondary. Pinned at
  first run.
- Which models serve as the strong and weak calibration anchors. Pinned at
  first run.
