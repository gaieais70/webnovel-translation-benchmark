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

Run `2026-09`: 34 models, eleven chapters of 화산귀환 (*Return of the Mount Hua
Sect*, a Korean murim novel), Korean → English. Quality is the average of two AI
judges (Claude Haiku 5.5 and Gemini 3.8 Flash), out of 100; cost is one chapter
(about 7,000 Korean characters) at the provider's list price, reasoning included.

| Model | Quality | Cost per chapter | Per 100 chapters |
|---|---|---|---|
| Claude Opus 5.5 | 86.4 | $1.288 | $128.83 |
| Claude Sonnet 5.5 | 83.5 | $0.846 | $84.59 |
| GPT-6 Astra | 83.4 | $0.814 | $81.40 |
| Gemini 3.8 Flash ¹ | 82.9 | $0.083 | $8.28 |
| GPT-6.1 Sol | 81.9 | $0.170 | $17.02 |
| GLM-5.3 | 79.8 | $0.237 | $23.67 |
| Kimi K3 | 78.5 | $0.505 | $50.51 |
| GPT-6 Luna | 77.5 | $0.0070 | $0.70 |
| Muse Spark 1.2 | 76.9 | $0.0022 | $0.22 |

¹ Also one of the two judges, so one of its scores comes from a judge of its own
family (Claude Haiku 5.5 is in the same position). The two judges' composites
correlate at 0.98. The head-to-head ratings order the top models differently
(Claude Sonnet 5.5 first, Claude Opus 5.5 sixth) and their ranges overlap, so read
the top five as a close group.

All 34 models, each judge's score, per-quality scores and the head-to-head ratings:
[leaderboard](archive/2026-09-hwasan-panel/results/leaderboard.md) ·
[run notes](docs/run-notes-2026-09.md). Earlier results:
[`2026-09` under the retired Gemini 3.7 judge](archive/2026-09-hwasan/results/leaderboard.md),
[`2026-08`](archive/2026-08-ghost-commute/results/leaderboard.md).

**Now testing:** Gemini 4 Argon. Claude Opus 5, Sonnet 5 and Haiku 4.5 have no
recorded cost, because their September records carried no token counts.

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
python scripts/build_site.py --run 2026-09 --archive archive/2026-09-hwasan-panel
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
