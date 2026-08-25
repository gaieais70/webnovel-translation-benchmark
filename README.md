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

```bash
pip install -e ".[dev]"
```

```bash
ati fetch
```

```bash
ati run --run 2026-08 --iterations 1
```

```bash
ati judge --run 2026-08
```

```bash
ati report --run 2026-08
```

Set `OPENROUTER_API_KEY` for API-backed models. Subscription-backed models
are configured as `runner: cli` in `config/models.yaml` and report cost as
N/A.

## Reweighting

Edit `config/weights.yaml` and rerun `ati report`. No model calls are made.
