# Artificial Translation Index

A benchmark ranking AI models on translating serialized web fiction
(Chinese, Japanese, Korean → English), for people translating works that
have no official translation.

**This project publishes rankings, not translations.** No source text or
translation text is committed or published.

- [Leaderboard](results/leaderboard.md)
- [Methodology](docs/methodology.md)
- [Rubric](docs/rubric.md)

## Usage

Install:

```bash
pip install -e ".[dev]"
```

Add chapter text you have saved locally. This hashes each file, slices it,
writes the manifest entry, and registers the task — no JSON by hand:

```bash
ati add ./raw/ch*.txt --work skyfall --lang zh --genre xianxia --type consistency --url https://example.com/skyfall
```

`--type passage` takes the first 800 characters of each file; `--type
consistency` keeps whole chapters for the multi-chapter terminology test;
`--type refusal` is for passages with violent or sexual content.

Then run everything:

```bash
ati bench --run 2026-08
```

That chains fetch → run → judge → report and writes
`results/leaderboard.md`. The individual stages remain available
(`ati fetch`, `ati run`, `ati judge`, `ati report`) and every stage is
resumable — work already done is skipped.

Then run the head-to-head pass, which is separate because it is the expensive
part:

```bash
ati pairwise --run 2026-08 --max-comparisons 200
```

Every comparison is checkpointed as it completes. If the judge subscription
hits a usage limit the stage stops cleanly and prints how far it got; rerun
the same command later and it resumes without repeating work. Rerun
`ati report` afterwards to fold the ratings into the leaderboard.

Set `OPENROUTER_API_KEY` for API-backed models. Subscription-backed models
are configured as `runner: cli` in `config/models.yaml` and report cost as
N/A. Defaults live in `config/ati.yaml`.

## Reweighting

Edit `config/weights.yaml` and rerun `ati report`. No model calls are made.
