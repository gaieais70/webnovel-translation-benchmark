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

1. `ati fetch` — verify the local corpus against `corpus/manifest.json`.
2. `ati run` — every model translates every passage, once per iteration.
3. `ati judge` — a pinned judge model scores each output 1–5 on six axes.
4. `ati report` — aggregate into `results/` and `results/leaderboard.md`.

Stages are idempotent. Adding one model costs only that model's calls.

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

## Judge bias

A single pinned judge, temperature 0, structured JSON output validated
against a schema; malformed responses are retried twice and then excluded
rather than scored as zero. The judge never evaluates its own outputs.

Self-preference is **disclosed, not corrected**: the judge's model family is
named in every report so readers can discount accordingly. The top 5 models
are re-judged by a secondary judge and the agreement rate is published.
Human spot-checks are recorded in [spot-checks.md](spot-checks.md).

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
