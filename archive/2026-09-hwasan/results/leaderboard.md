# Artificial Translation Index

Run `2026-09` — 2026-09-19

Judge: `gemini-judge`. Scores are LLM-judged against a fixed rubric; see [the rubric](../../../docs/rubric.md) and [the methodology](../../../docs/methodology.md).

Anchor drift: not checked - no earlier run's anchors were found.

## Ranking

| # | Model | Composite | Scored | Refusal % | Cost /1k ch | Sec /1k ch | Value | Slop | Repetition |
|---|---|---|---|---|---|---|---|---|---|
| 1 | gpt-5.6-luna | 80.3 | 11/11 | 0.0 | N/A | 59.9 | N/A | 0.33 | 0.111 |
| 2 | muse-spark-1.2-contributor | 79.6 | 11/11 | 0.0 | N/A | 5.0 | N/A | 0.90 | 0.112 |
| 3 | deepseek-v4-pro | 78.5 | 11/11 | 0.0 | N/A | 22.4 | N/A | 0.75 | 0.109 |
| 4 | qwen3.8-flash | 73.2 | 10/11 | 9.1 | N/A | 66.3 | N/A | 0.64 | 0.106 |
| 5 | deepseek-v4-flash | 72.7 | 11/11 | 0.0 | N/A | 19.8 | N/A | 0.41 | 0.105 |
| 6 | glm-5.2 | 71.5 | 8/11 | 27.3 | N/A | 18.8 | N/A | 0.73 | 0.108 |
| 7 | hy4-preview | 71.3 | 11/11 | 0.0 | N/A | 57.5 | N/A | 0.72 | 0.114 |
| 8 | glm-5.3-flash | 69.5 | 11/11 | 0.0 | N/A | 65.1 | N/A | 0.36 | 0.104 |
| 9 | hy3 _weak anchor_ | 65.1 | 11/11 | 0.0 | N/A | 21.5 | N/A | 0.51 | 0.105 |
| 10 | nemotron-3-ultra | 59.7 | 11/11 | 0.0 | N/A | 2.1 | N/A | 0.85 | 0.099 |
| 11 | mimo-v2.5-pro | 58.1 | 9/11 | 18.2 | N/A | 12.2 | N/A | 0.71 | 0.105 |
| 12 | mimo-v2.5 | 57.6 | 10/11 | 9.1 | N/A | 17.7 | N/A | 0.79 | 0.104 |
| 13 | minimax-m3 | 54.2 | 11/11 | 0.0 | N/A | 4.8 | N/A | 0.59 | 0.103 |
| 14 | qwen3.8-27b | 54.0 | 11/11 | 0.0 | N/A | 33.4 | N/A | 0.72 | 0.109 |
| 15 | solar-pro4 | 51.4 | 11/11 | 0.0 | 0.0006 | 20.6 | 82358 | 0.51 | 0.103 |
| 16 | nemotron-3-super-120b | 40.0 | 11/11 | 0.0 | sub | 10.8 | N/A | 0.61 | 0.107 |
| 17 | inkling-small | 31.8 | 11/11 | 0.0 | N/A | 7.6 | N/A | 0.68 | 0.106 |
| 18 | dots-3-note-preview | 22.7 | 8/11 | 18.2 | sub | 5.9 | N/A | 1.03 | 0.107 |
| 19 | laguna-s-2.1 | 17.7 | 11/11 | 0.0 | N/A | 11.6 | N/A | 0.77 | 0.106 |
| 20 | nemotron-3.5-lightning | 16.2 | 11/11 | 0.0 | 0.0002 | 38.8 | 94163 | 0.67 | 0.103 |

## Per-axis scores

| Model | accuracy | terminology | register | fluency | fidelity | sanitization |
|---|---|---|---|---|---|---|
| gpt-5.6-luna | 4.5 ±0.5 | 2.8 ±0.3 | 5.0 ±0.0 | 4.9 ±0.3 | 3.7 ±0.9 | 4.9 ±0.3 |
| muse-spark-1.2-contributor | 4.1 ±0.7 | 2.5 ±0.9 | 4.9 ±0.3 | 4.6 ±0.5 | 4.9 ±0.3 | 5.0 ±0.0 |
| deepseek-v4-pro | 4.1 ±0.7 | 2.1 ±0.5 | 4.8 ±0.4 | 4.8 ±0.4 | 5.0 ±0.0 | 4.9 ±0.3 |
| qwen3.8-flash | 3.7 ±0.8 | 1.9 ±0.8 | 4.6 ±0.7 | 4.9 ±0.3 | 4.7 ±0.5 | 5.0 ±0.0 |
| deepseek-v4-flash | 3.6 ±1.0 | 1.9 ±0.5 | 4.6 ±0.6 | 4.9 ±0.3 | 4.6 ±0.5 | 5.0 ±0.0 |
| glm-5.2 | 3.4 ±0.9 | 2.2 ±1.0 | 4.5 ±0.5 | 4.8 ±0.4 | 4.6 ±0.5 | 5.0 ±0.0 |
| hy4-preview | 3.6 ±0.8 | 2.2 ±0.7 | 4.3 ±0.7 | 4.7 ±0.4 | 4.4 ±0.8 | 4.9 ±0.3 |
| glm-5.3-flash | 3.5 ±0.8 | 2.0 ±1.0 | 4.6 ±0.6 | 4.6 ±0.6 | 4.3 ±0.9 | 4.8 ±0.4 |
| hy3 | 3.2 ±0.4 | 2.0 ±0.7 | 4.0 ±0.7 | 4.0 ±0.7 | 4.7 ±0.6 | 5.0 ±0.0 |
| nemotron-3-ultra | 3.0 ±0.4 | 2.0 ±0.6 | 3.8 ±0.6 | 4.0 ±0.6 | 3.9 ±0.9 | 4.8 ±0.4 |
| mimo-v2.5-pro | 2.9 ±0.3 | 1.5 ±0.3 | 3.9 ±0.7 | 4.1 ±0.7 | 4.2 ±0.6 | 4.7 ±0.7 |
| mimo-v2.5 | 2.6 ±0.5 | 1.4 ±0.6 | 3.8 ±0.6 | 4.4 ±0.7 | 4.3 ±0.6 | 4.9 ±0.3 |
| minimax-m3 | 2.6 ±0.6 | 1.5 ±0.9 | 3.8 ±0.6 | 4.4 ±0.8 | 3.5 ±0.8 | 4.6 ±0.5 |
| qwen3.8-27b | 2.5 ±0.7 | 2.4 ±1.0 | 3.1 ±0.3 | 3.1 ±0.3 | 4.4 ±0.6 | 4.8 ±0.6 |
| solar-pro4 | 2.8 ±0.6 | 1.3 ±0.7 | 3.3 ±0.4 | 3.2 ±0.7 | 4.5 ±0.8 | 4.6 ±0.8 |
| nemotron-3-super-120b | 2.0 ±0.4 | 1.6 ±0.6 | 2.9 ±0.3 | 2.9 ±0.3 | 3.2 ±0.9 | 4.4 ±0.8 |
| inkling-small | 1.7 ±0.6 | 1.7 ±0.9 | 2.5 ±0.5 | 2.5 ±0.5 | 1.7 ±0.9 | 4.7 ±0.6 |
| dots-3-note-preview | 1.0 ±0.0 | 1.4 ±0.3 | 2.4 ±0.5 | 2.5 ±0.7 | 2.1 ±0.8 | 3.2 ±0.8 |
| laguna-s-2.1 | 1.0 ±0.0 | 1.1 ±0.3 | 1.9 ±0.5 | 2.7 ±0.7 | 1.3 ±0.4 | 3.5 ±1.2 |
| nemotron-3.5-lightning | 1.2 ±0.4 | 1.2 ±0.7 | 1.8 ±0.6 | 1.4 ±0.5 | 1.4 ±0.5 | 4.4 ±1.2 |

## Head-to-head (Bradley–Terry)

Ratings from blind pairwise comparison, judged in both orders. 400 points is 10:1 odds; only the offset is arbitrary. Ties are allowed and a judge that reverses itself when the order flips is recorded as a tie.

| Model | Rating | 95% interval | W–L–T |
|---|---|---|---|
| gpt-5.6-luna | 1119 | 1083 – 1241 | 20–0–5 |
| muse-spark-1.2-contributor | 1119 | 1088 – 1245 | 21–1–3 |
| deepseek-v4-pro | 1097 | 1053 – 1196 | 20–1–7 |
| deepseek-v4-flash | 1089 | 1037 – 1181 | 19–1–8 |
| qwen3.8-flash | 1021 | 946 – 1116 | 19–8–4 |
| hy3 | 1000 | pinned | 17–8–8 |
| glm-5.3-flash | 986 | 912 – 1051 | 15–8–10 |
| glm-5.2 | 963 | 884 – 1018 | 13–8–11 |
| hy4-preview | 945 | 855 – 1021 | 15–12–6 |
| mimo-v2.5-pro | 924 | 841 – 975 | 12–12–9 |
| nemotron-3-ultra | 917 | 815 – 969 | 12–12–7 |
| minimax-m3 | 885 | 795 – 935 | 9–13–10 |
| mimo-v2.5 | 871 | 778 – 903 | 6–12–14 |
| qwen3.8-27b | 861 | 754 – 902 | 8–14–11 |
| solar-pro4 | 861 | 748 – 891 | 8–14–11 |
| nemotron-3-super-120b | 818 | 704 – 850 | 7–18–6 |
| inkling-small | 784 | 659 – 812 | 5–20–5 |
| dots-3-note-preview | 719 | 564 – 722 | 2–23–2 |
| laguna-s-2.1 | 710 | 547 – 704 | 1–22–2 |
| nemotron-3.5-lightning | 699 | 540 – 694 | 0–22–3 |

Based on 300 comparisons covering 190 of 190 model pairs.

> The intervals for **gpt-5.6-luna** and **muse-spark-1.2-contributor** overlap, so their ordering is not established by this data. More comparisons would be needed to separate them.

## Notes

- **Cost `sub`** means a subscription covered the call, so it billed zero. That is not the same as the model being free to run. **`N/A`** means the access path reports no cost at all. Neither is a price.
- **Slop** counts LLM-tell and translationese phrases per 1,000 words. **Repetition** is top-n-gram concentration. Neither enters the composite; both are diagnostics.
- Model behavior changes under a stable name. These results describe the versions tested on 2026-09-19.
