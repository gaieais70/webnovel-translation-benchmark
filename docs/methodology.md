# Methodology

## What this measures

How well AI models translate serialized web fiction from Chinese, Japanese,
and Korean into English. It is aimed at people translating works that have no
official translation.

## What this publishes

Rankings and methodology. No source text and no translation text is
published or committed — the repository stores only pointers and hashes.
Nothing downloads a corpus: anyone reproducing a run obtains the source
chapters themselves and runs `ati add` on them, which writes back every
passage whose text reproduces the hash in the manifest. `ati fetch` verifies
the result.

## Pipeline

0. `ati add` — register locally saved chapter files: hash, slice, and record
   pointers. Source text stays in the gitignored corpus directory.
1. `ati fetch` — verify the local corpus against `corpus/manifest.json`.
2. `ati run` — every model under test translates every passage, once per
   iteration. Judges declared in the roster are not translation subjects.
3. `ati judge` — a pinned judge model scores each output 1–5 on six axes.
4. `ati pairwise` — the same judge compares translations head to head.
5. `ati report` — aggregate into `results/` and `results/leaderboard.md`.

Stages are idempotent. Adding one model costs only that model's calls.
`ati bench` runs steps 1–3 and 5 in one command; `ati status` shows where a
run stands. How a run is operated is in [the runbook](runbook.md).

Every result file records what produced it: a hash of each input (roster,
task suite, manifest, prompts, schemas, weights), a hash of the harness code,
and the judge fingerprints described under *Judge bias*.

## What counts as a translation

An output is scored only if it is a translation of the whole passage. The run
stage records, rather than scores:

- **failures** — output shorter than half the source or longer than four
  times it, output mostly in the source's script, a generation the provider
  cut off (`finish_reason` `length` or `error`), or a route that never
  answered. A failure is recorded, not scored, and is attempted again on the
  next pass: published runs use `ati run --passes 3`, so each task gets up to
  three attempts before its failure is final. (Within one attempt, failures
  that never reached a model — HTTP 5xx, a gateway timeout, a dropped
  connection or stream — are retried up to three times; a read timeout is
  not, because the model may still be generating.);
- **refusals** — the model declining, or the *provider's* moderation layer
  stopping the translation (a `content_filter` finish, or a known moderation
  message in place of or at the end of the text). Refusals are not retried and
  are reported in the refusal rate, not averaged into the composite.

The leaderboard's **Scored** column shows how many passages each composite
rests on. A model scored on fewer passages was graded on a subset — often
missing exactly the passages that broke models — and its composite is not
the same measurement as one over every passage. A model that **failed more
than 20%** of its tasks (refusals do not count as failures) is moved to an
*Incomplete* table and left out of the ranking altogether.

Reasoning effort is set per model in the roster (`params`), at each model's
top rung. It is configured, not verified: providers ignore request fields
they do not understand, so a model could run at its default while the roster
says maximum. The `reasoning_tokens` recorded with every output is the
evidence of what actually happened.

## Axes and weights

Six axes, defined with anchors in [the rubric](rubric.md): accuracy,
terminology, register, fluency, fidelity, sanitization. Weights live in
`config/weights.yaml` and are published with every run — reweight and rerun
`ati report` to recompute the ranking without rerunning any model.

Terminology on consistency tasks is **computed, not judged**: the judge
extracts a term table per chapter and the harness measures how many recurring
terms were rendered identically across all of them.

## What is reported outside the composite

Refusal rate, cost per 1,000 source characters, seconds per 1,000 source
characters, value, slop, and repetition. These sit beside the composite
rather than inside it, because their importance depends entirely on the
reader — censorship is disqualifying for some and irrelevant to others.

**Cost N/A means unknown, not free.** Subscription-backed access reports no
per-call cost, so those models rank on quality but not on value.

## Calibration anchors

A known-weak model (`hy3`) runs in every evaluation as a calibration anchor.
The design calls for a known-strong anchor as well, and the roster supports
one (`anchor: strong`), but none is configured yet — so drift is detected at
the weak end only. Anchors are not there to be ranked — they detect drift. If
an anchor's composite moves more than 5 points from the previous run, the
judge, the prompt, or a provider has changed behavior, and the report marks
the run **not comparable** to earlier runs rather than charting them together.

The leaderboard names the run it measured drift against, or says drift was
not checked when no earlier run's anchors were available. Drift is only
meaningful between runs on the same corpus with the same judge; a run on a
new work is expected to read as not comparable.

## Head-to-head ratings

Rubric scores compress: judges cluster absolute ratings in a narrow band, and
near-equal models become indistinguishable. A second pass compares
translations directly and fits a Bradley–Terry model to the results.

Bradley–Terry rather than Elo or Glicko, because those are sequential
algorithms whose output depends on the order matches were played, and a
benchmark holds every comparison at once. The fit is order-independent and
reproducible from the same data.

Controls on each comparison:

- **Both orders.** Every pair is judged twice with the positions swapped. A
  judge that reverses itself is recorded as a tie, not averaged into noise.
- **Ties are allowed.** Two renderings of one sentence are often equally good;
  forcing a winner manufactures noise.
- **No truncation.** EQ-Bench truncates to control length bias in free-form
  creative writing. Translations of one source passage are naturally similar
  in length, and truncating would conceal the omissions the fidelity axis
  exists to catch.
- **Adaptive pairing.** Closest-rated, least-compared pairs are judged first,
  so an interrupted run still produces a meaningful ranking. The schedule
  takes one passage from each pair in turn before a second from any, so the
  budget spreads across the whole field; ratings are refitted between
  batches.
- **95% bootstrap intervals** are published with every rating. Where the top
  two intervals overlap, the report says so: the ordering is not established.
  The report also states how many comparisons, over how many of the possible
  model pairs, the ratings rest on.

Which translations are compared:

- A model enters only with at least one scorable output; each pair is
  compared only on passages both translated successfully. A model missing the
  hard chapters is therefore compared on an easier subset — check its W–L–T
  record before believing its rating.
- A comparison whose source plus both translations exceed 60,000 characters
  is skipped rather than truncated, and the skip is counted in the stage's
  output.
- One translation per model and passage is compared. With more than one
  iteration per passage, the head-to-head stage uses only the last; every run
  so far has used one iteration.

Ratings use a scale on which 400 points is 10:1 odds. Only the offset is
arbitrary — it is pinned to the weak calibration anchor so numbers stay
comparable as models join or leave the pool.

## Judge bias

A single pinned judge, structured JSON output validated against a schema;
malformed responses are retried twice and then excluded rather than scored as
zero, and the stage stops if three items in a row get no usable answer. The
judge never evaluates its own outputs. A judge not declared in the roster is
called at temperature 0; a declared API judge uses its row's temperature
(0.3 unless the row sets one). The Antigravity CLI exposes no temperature
setting, so a Gemini judge reached through it runs at the CLI's default.

Every judgment and comparison records a **judge fingerprint**: a hash of the
judge's id, the route behind it (model, endpoint and sampling settings; for
Antigravity, the no-tools instruction and the output schema) and the prompt
frame. Changing any of them makes the affected judgments stale, and the next
`ati judge` re-judges them, so one leaderboard never mixes two judges or two
prompts. Judgments made before fingerprints existed (all of run `2026-09`)
are taken as current.

Self-preference is **disclosed, not corrected**: the judge's model family is
named in every report so readers can discount accordingly. A secondary-judge
agreement check on the top five models is planned but not yet implemented;
no run so far has one. Human spot-checks are recorded in
[spot-checks.md](spot-checks.md).

When the judge is an agent rather than a completion endpoint (Gemini through
the Antigravity CLI), it is told in every prompt to use no tools: an agentic
judge that runs commands is no longer judging only the text in front of it,
and on run `2026-09` every call in which it did so stalled until timeout.

## Known limitations

- LLM judges compress absolute scores toward the middle. Anchors detect
  drift but do not fix compression. The head-to-head ratings are the remedy
  and are published beside the composite; they are not folded into it.
- One judge serves both the rubric and the head-to-head stage, so their
  agreement measures that judge agreeing with itself, not two methods
  converging.
- Refusal detection is a keyword heuristic over the start of an output.
- Model behavior changes under a stable name. Every run records its date.

## Prior art

The design of the anchors, the deterministic slop and repetition metrics, and
the bias controls of the pairwise stage follow
[EQ-Bench Creative Writing v3](https://eqbench.com/creative_writing.html),
which does the same job for English creative writing.
