# Artificial Translation Index

A benchmark ranking AI models on translating serialized web fiction
(Chinese, Japanese, Korean → English), for people translating works that
have no official translation.

**This project publishes rankings, not translations.** No source text or
translation text is committed or published.

- [Latest published leaderboard](archive/2026-08-ghost-commute/results/leaderboard.md)
  (run `2026-08`; run `2026-09` is being judged)
- [Methodology](docs/methodology.md)
- [Rubric](docs/rubric.md)
- [Runbook](docs/runbook.md) — operating a run end to end

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
```

`ati bench --run 2026-10` chains fetch → run → judge → report (not pairwise).
Every stage is resumable — work already done is skipped — and a stage that
stops on a usage limit keeps what it finished. `--wait` waits the limit out
and resumes. See [the runbook](docs/runbook.md) for lanes, quotas and the
judge's safeguards.

## Configuration

- `config/models.yaml` — the roster. Models are reached with `runner: api`
  (any OpenAI-compatible endpoint; Command Code and OpenRouter today), each
  with its own `api_key_env`. The judge is Gemini through the Antigravity CLI
  (`runner: agy`), the only route Gemini may take. `runner: cli` (a local CLI
  such as `opencode.cmd run`, prompt on stdin) and `runner: manual` (prompt
  files answered by pasting a response) are supported but unused by the
  current roster. Subscription access reports cost as N/A.
- `config/ati.yaml` — the default judge and iteration count, nothing else.
- `config/weights.yaml` — axis weights.
- Keys go in `.env` (see `.env.example`).

## Reweighting

Edit `config/weights.yaml` and rerun `ati report`. No model calls are made.
