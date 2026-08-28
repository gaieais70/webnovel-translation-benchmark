# Artificial Translation Index

Run `2026-08` — 2026-08-27

Judge: `gemini-judge`. Scores are LLM-judged against a fixed rubric; see [the rubric](../docs/rubric.md) and [the methodology](../docs/methodology.md).

## Ranking

| # | Model | Composite | Refusal % | Cost /1k ch | Sec /1k ch | Value | Slop | Repetition |
|---|---|---|---|---|---|---|---|---|
| 1 | deepseek-v4-pro | 85.7 | 0.0 | sub | 56.3 | N/A | 0.39 | 0.120 |
| 2 | muse-spark-1.2-contributor | 82.8 | 0.0 | N/A | 10.4 | N/A | 0.38 | 0.121 |
| 3 | grok-4.6 | 82.7 | 0.0 | 0.0109 | 24.6 | 7615 | 0.20 | 0.123 |
| 4 | glm-5.3-flash | 80.6 | 0.0 | sub | 69.3 | N/A | 0.10 | 0.111 |
| 5 | gpt-5.6-luna | 80.1 | 0.0 | N/A | 5.0 | N/A | 0.52 | 0.124 |
| 6 | deepseek-v4-flash | 75.2 | 0.0 | sub | 4.8 | N/A | 0.31 | 0.117 |
| 7 | mimo-v2.5-pro | 72.3 | 0.0 | sub | 12.9 | N/A | 0.36 | 0.117 |
| 8 | hy3 _weak anchor_ | 71.4 | 0.0 | sub | 42.4 | N/A | 0.16 | 0.116 |
| 9 | minimax-m3 | 63.5 | 0.0 | sub | 7.2 | N/A | 0.38 | 0.117 |
| 10 | longcat-2.0 | 63.0 | 0.0 | sub | 18.8 | N/A | 0.39 | 0.118 |
| 11 | inkling-small | 49.0 | 0.0 | N/A | 7.6 | N/A | 0.42 | 0.123 |

## Per-axis scores

| Model | accuracy | terminology | register | fluency | fidelity | sanitization |
|---|---|---|---|---|---|---|
| deepseek-v4-pro | 4.6 ±0.6 | 2.6 ±0.3 | 5.0 ±0.0 | 5.0 ±0.0 | 5.0 ±0.0 | 5.0 ±0.0 |
| muse-spark-1.2-contributor | 4.2 ±0.6 | 2.6 ±0.3 | 5.0 ±0.0 | 5.0 ±0.0 | 4.9 ±0.3 | 5.0 ±0.0 |
| grok-4.6 | 4.2 ±0.6 | 2.9 ±0.3 | 4.9 ±0.3 | 4.6 ±0.5 | 5.0 ±0.0 | 5.0 ±0.0 |
| glm-5.3-flash | 4.1 ±0.6 | 2.4 ±0.4 | 5.0 ±0.0 | 5.0 ±0.0 | 4.8 ±0.4 | 5.0 ±0.0 |
| gpt-5.6-luna | 4.1 ±0.8 | 2.5 ±0.5 | 4.9 ±0.3 | 5.0 ±0.0 | 4.6 ±0.6 | 5.0 ±0.0 |
| deepseek-v4-flash | 3.7 ±1.0 | 2.1 ±0.8 | 5.0 ±0.0 | 5.0 ±0.0 | 4.4 ±0.6 | 5.0 ±0.0 |
| mimo-v2.5-pro | 3.5 ±0.5 | 2.3 ±0.7 | 4.5 ±0.5 | 4.5 ±0.7 | 4.5 ±0.5 | 5.0 ±0.0 |
| hy3 | 3.4 ±0.5 | 2.9 ±0.5 | 4.5 ±0.7 | 3.9 ±0.8 | 4.5 ±0.5 | 5.0 ±0.0 |
| minimax-m3 | 2.9 ±0.5 | 2.2 ±0.6 | 4.3 ±0.6 | 4.2 ±0.6 | 4.0 ±0.9 | 5.0 ±0.0 |
| longcat-2.0 | 3.0 ±0.7 | 2.0 ±1.0 | 4.3 ±0.7 | 4.0 ±0.9 | 4.2 ±0.6 | 5.0 ±0.0 |
| inkling-small | 2.4 ±0.5 | 2.1 ±1.2 | 3.1 ±0.6 | 3.1 ±0.6 | 3.3 ±1.2 | 5.0 ±0.0 |

## Notes

- **Cost `sub`** means a subscription covered the call, so it billed zero. That is not the same as the model being free to run. **`N/A`** means the access path reports no cost at all. Neither is a price.
- **Slop** counts LLM-tell and translationese phrases per 1,000 words. **Repetition** is top-n-gram concentration. Neither enters the composite; both are diagnostics.
- Model behavior changes under a stable name. These results describe the versions tested on 2026-08-27.
