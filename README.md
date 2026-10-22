# Webnovel Translation Benchmark

**Which AI should translate your web novel, and what will it cost?** This
benchmark ranks AI models (Gemini, Claude, GPT, DeepSeek, Kimi, Qwen, GLM and
more) on translating serialized web fiction into English, for people
translating Korean, Chinese and Japanese web novels that have no official
translation. Every model gets the same chapters, the same prompt and its
maximum reasoning setting, and every result comes with its cost per chapter.

**[See the ranking →](https://gaieais70.github.io/webnovel-translation-benchmark/)**

**This project publishes rankings, not translations.** No source text or
translation text is committed or published.

## Latest results

Run `2026-09`: 28 models, eleven chapters of 화산귀환 (*Return of the Mount Hua
Sect*, a Korean murim novel), Korean → English. Quality is an AI judge's score
out of 100; cost is one chapter (about 7,000 Korean characters) at the
provider's list price, reasoning included.

| Model | Quality | Cost per chapter | Per 100 chapters |
|---|---|---|---|
| Gemini 3.8 Flash ¹ | 88.0 | $0.083 | $8.28 |
| Claude Opus 5 | 84.0 | not recorded | – |
| Claude Sonnet 5 | 83.8 | not recorded | – |
| Kimi K3 | 83.0 | $0.505 | $50.51 |
| GLM-5.3 | 82.0 | $0.237 | $23.67 |
| GPT-5.6 Luna | 80.3 | $0.075 | $7.47 |
| Muse Spark 1.2 | 79.6 | $0.0022 | $0.22 |
| DeepSeek V4 Pro | 78.5 | $0.047 | $4.70 |

¹ Judged by a model of its own family (Gemini 3.7 Flash). An independent judge
(Grok 4.6) still ranked it first among the models it re-checked, by 2.8 points
instead of 6.5; read it as among the best, not clearly first.

All 28 models, per-quality scores and the head-to-head ratings:
[leaderboard](archive/2026-09-hwasan/results/leaderboard.md) ·
[run notes](docs/run-notes-2026-09.md). Earlier run:
[`2026-08`](archive/2026-08-ghost-commute/results/leaderboard.md).

**Now testing:** Claude Opus 5.5, Sonnet 5.5 and Haiku 5.5; GPT-6.1 Sol, GPT-6
Astra and GPT-6 Luna; Gemini 4 Argon. The Gemini 3.7 judge has been retired,
so the next edition re-judges every model with a panel of two judges from
different families (Claude Haiku 5.5 and Gemini 3.8 Flash) and shows each
judge's score beside the average.

- [Methodology](docs/methodology.md)
- [Rubric](docs/rubric.md): accuracy, terminology, register, fluency,
  fidelity, and sanitization (whether violence and crude language survive)
- [Runbook](docs/runbook.md): operating a run end to end

## Usage

Install and check:

```bash
pip install -e ".[dev]"
python -m pytest -q
```

Add chapter text you have saved locally. This hashes each file, slices it,
writes the manifest entry, and registers the task — no JSON by hand:

```bash
ati add ./raw/ch*.txt --work skyfall --lang zh --genre xianxia --type consistency --url https://example.com/skyfall
```

`--type passage` takes the first 800 characters of each file; `--type
consistency` keeps whole chapters for the multi-chapter terminology test;
`--type refusal` is for passages with violent or sexual content. Running the
same command again restores any passage text that has gone missing.

Then, for a new run id:

```bash
ati run --run 2026-10 --passes 3 --wait     # translate; retry failures; outlast usage limits
ati judge --run 2026-10 --wait              # rubric scores
ati pairwise --run 2026-10 --target 300 --wait   # head-to-head, the expensive part
ati report --run 2026-10                    # results/<run>.json and results/leaderboard.md
ati status --run 2026-10                    # progress at any point; makes no calls
ati cost --run 2026-10                      # list-price cost so far, and what is left
```

A panel of judges each writes under its own directory and is combined at the
report:

```bash
ati judge --run 2026-10 --judge-model haiku-judge --judgments judgments/haiku-judge --wait
ati report --run 2026-10 --panel haiku-judge --panel gemini-3.8-judge
```

`ati bench --run 2026-10` chains fetch → run → judge → report (not pairwise).
Every stage is resumable — work already done is skipped — and a stage that
stops on a usage limit keeps what it finished. `--wait` waits the limit out
and resumes. See [the runbook](docs/runbook.md) for lanes, quotas and the
judge's safeguards.

The public site is built from an archived run:

```bash
python scripts/build_site.py --run 2026-09 --archive archive/2026-09-hwasan
```

## Configuration

- `config/models.yaml` — the roster. Models are reached with `runner: api`
  (any OpenAI-compatible endpoint), each with its own `api_key_env`;
  `runner: cli` for a subscription CLI (the Claude Code and Codex CLIs, run
  with their own agent prompt replaced and every tool off); and `runner: agy`,
  the Antigravity CLI, the only route Gemini may take. Each row declares its
  list `price`, and CLI rows ask for structured output (`output_format`) so
  every record carries token counts.
- `config/ati.yaml` — the default judge and iteration count, nothing else.
- `config/weights.yaml` — axis weights.
- Keys go in `.env` (see `.env.example`).

## Reweighting

Edit `config/weights.yaml` and rerun `ati report`. No model calls are made.
