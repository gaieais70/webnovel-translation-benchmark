# Methodology

## What this measures

How well AI models translate serialized web fiction from Chinese, Japanese,
and Korean into English. It is aimed at people translating works that have no
official translation.

## What this publishes

Rankings and methodology. No source text and no translation text is
published or committed — the repository stores only pointers and hashes, and
`ati fetch` rebuilds a corpus locally for anyone reproducing a run.

## Pipeline

0. `ati add` — register locally saved chapter files: hash, slice, and record
   pointers. Source text stays in the gitignored corpus directory.
1. `ati fetch` — verify the local corpus against `corpus/manifest.json`.
2. `ati run` — every model translates every passage, once per iteration.
3. `ati judge` — a pinned judge model scores each output 1–5 on six axes.
4. `ati report` — aggregate into `results/` and `results/leaderboard.md`.

Stages are idempotent. Adding one model costs only that model's calls.

`ati bench` runs steps 1–4 in one command.

## What counts as a translation

An output is scored only if it is a translation of the whole passage. The run
stage records, rather than scores:

- **failures** — output shorter than half the source or longer than four
  times it, output mostly in the source's script, a generation the provider
  cut off (`finish_reason` `length` or `error`), or a route that never
  answered. Each gets up to three attempts before it is final;
- **refusals** — the model declining, or the *provider's* moderation layer
  stopping the translation (a `content_filter` finish, or a known moderation
  message in place of or at the end of the text). Refusals are not retried and
  are reported in the refusal rate, not averaged into the composite.

The leaderboard's **Scored** column shows how many passages each composite
rests on. A model scored on fewer passages was graded on a subset — often
missing exactly the passages that broke models — and its composite is not
the same measurement as one over every passage.

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

One known-strong and one known-weak model run in every evaluation. They are
not there to be ranked — they detect drift. If an anchor's composite moves
more than 5 points between runs, the judge, the prompt, or a provider has
changed behavior, and the report marks the run **not comparable** to earlier
runs rather than charting them together.

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
  so an interrupted run still produces a meaningful ranking.
- **95% bootstrap intervals** are published with every rating. Where the top
  two intervals overlap, the report says so: the ordering is not established.

Ratings use a scale on which 400 points is 10:1 odds. Only the offset is
arbitrary — it is pinned to the weak calibration anchor so numbers stay
comparable as models join or leave the pool.

## Judge bias

A single pinned judge, structured JSON output validated against a schema;
malformed responses are retried twice and then excluded rather than scored as
zero. The judge never evaluates its own outputs. API judges run at
temperature 0; the Antigravity CLI exposes no temperature setting, so a
Gemini judge reached through it runs at the CLI's default.

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
  drift but do not fix compression; pairwise tie-breaking, which would, is
  deferred (see `ati/pairwise.py`).
- Refusal detection is a keyword heuristic over the start of an output.
- Model behavior changes under a stable name. Every run records its date.

## Prior art

The design of the anchors, the deterministic slop and repetition metrics, and
the bias controls planned for the pairwise stage follow
[EQ-Bench Creative Writing v3](https://eqbench.com/creative_writing.html),
which does the same job for English creative writing.
