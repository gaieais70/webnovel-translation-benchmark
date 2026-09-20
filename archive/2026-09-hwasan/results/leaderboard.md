# Artificial Translation Index

Run `2026-09` — 2026-09-20

Judge: `gemini-judge`. Scores are LLM-judged against a fixed rubric; see [the rubric](../../../docs/rubric.md) and [the methodology](../../../docs/methodology.md).

Anchor drift: not checked - no earlier run's anchors were found.

## Ranking

| # | Model | Composite | Scored | Refusal % | Cost /1k ch | Sec /1k ch | Value | Slop | Repetition |
|---|---|---|---|---|---|---|---|---|---|
| 1 | gemini-3.8-flash | 88.0 | 11/11 | 0.0 | N/A | 4.9 | N/A | 0.55 | 0.108 |
| 2 | gpt-5.6-luna | 80.3 | 11/11 | 0.0 | N/A | 59.9 | N/A | 0.33 | 0.111 |
| 3 | muse-spark-1.2-contributor | 79.6 | 11/11 | 0.0 | N/A | 5.0 | N/A | 0.90 | 0.112 |
| 4 | muse-spark-1.3-contributor | 79.3 | 11/11 | 0.0 | N/A | 12.9 | N/A | 0.31 | 0.110 |
| 5 | deepseek-v4-pro | 78.5 | 11/11 | 0.0 | N/A | 22.4 | N/A | 0.75 | 0.109 |
| 6 | deepseek-v4.1-flash | 74.7 | 11/11 | 0.0 | N/A | 27.1 | N/A | 0.53 | 0.105 |
| 7 | qwen3.8-flash | 73.2 | 10/11 | 9.1 | N/A | 66.3 | N/A | 0.64 | 0.106 |
| 8 | deepseek-v4-flash | 72.7 | 11/11 | 0.0 | N/A | 19.8 | N/A | 0.41 | 0.105 |
| 9 | glm-5.2 | 71.5 | 8/11 | 27.3 | N/A | 18.8 | N/A | 0.73 | 0.108 |
| 10 | hy4-preview | 71.3 | 11/11 | 0.0 | N/A | 57.5 | N/A | 0.72 | 0.114 |
| 11 | glm-5.3-flash | 69.5 | 11/11 | 0.0 | N/A | 65.1 | N/A | 0.36 | 0.104 |
| 12 | hy3 _weak anchor_ | 65.1 | 11/11 | 0.0 | N/A | 21.5 | N/A | 0.51 | 0.105 |
| 13 | nemotron-3-ultra | 59.7 | 11/11 | 0.0 | N/A | 2.1 | N/A | 0.85 | 0.099 |
| 14 | mimo-v2.5-pro | 58.1 | 9/11 | 18.2 | N/A | 12.2 | N/A | 0.71 | 0.105 |
| 15 | mimo-v2.5 | 57.6 | 10/11 | 9.1 | N/A | 17.7 | N/A | 0.79 | 0.104 |
| 16 | minimax-m3 | 54.2 | 11/11 | 0.0 | N/A | 4.8 | N/A | 0.59 | 0.103 |
| 17 | qwen3.8-27b | 54.0 | 11/11 | 0.0 | N/A | 33.4 | N/A | 0.72 | 0.109 |
| 18 | solar-pro4 | 51.4 | 11/11 | 0.0 | 0.0006 | 20.6 | 82358 | 0.51 | 0.103 |
| 19 | nemotron-3-super-120b | 40.0 | 11/11 | 0.0 | sub | 10.8 | N/A | 0.61 | 0.107 |
| 20 | inkling-small | 31.8 | 11/11 | 0.0 | N/A | 7.6 | N/A | 0.68 | 0.106 |
| 21 | dots-3-note-preview | 22.7 | 8/11 | 18.2 | sub | 5.9 | N/A | 1.03 | 0.107 |
| 22 | laguna-s-2.1 | 17.7 | 11/11 | 0.0 | N/A | 11.6 | N/A | 0.77 | 0.106 |
| 23 | nemotron-3.5-lightning | 16.2 | 11/11 | 0.0 | 0.0002 | 38.8 | 94163 | 0.67 | 0.103 |

## Per-axis scores

| Model | accuracy | terminology | register | fluency | fidelity | sanitization |
|---|---|---|---|---|---|---|
| gemini-3.8-flash | 5.0 ±0.0 | 2.6 ±0.0 | 5.0 ±0.0 | 5.0 ±0.0 | 5.0 ±0.0 | 5.0 ±0.0 |
| gpt-5.6-luna | 4.5 ±0.5 | 2.8 ±0.3 | 5.0 ±0.0 | 4.9 ±0.3 | 3.7 ±0.9 | 4.9 ±0.3 |
| muse-spark-1.2-contributor | 4.1 ±0.7 | 2.5 ±0.9 | 4.9 ±0.3 | 4.6 ±0.5 | 4.9 ±0.3 | 5.0 ±0.0 |
| muse-spark-1.3-contributor | 4.3 ±0.4 | 1.9 ±0.6 | 5.0 ±0.0 | 4.9 ±0.3 | 4.9 ±0.3 | 5.0 ±0.0 |
| deepseek-v4-pro | 4.1 ±0.7 | 2.1 ±0.5 | 4.8 ±0.4 | 4.8 ±0.4 | 5.0 ±0.0 | 4.9 ±0.3 |
| deepseek-v4.1-flash | 3.9 ±0.8 | 1.9 ±0.6 | 4.8 ±0.4 | 4.8 ±0.4 | 4.5 ±0.8 | 5.0 ±0.0 |
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
| gemini-3.8-flash | 1229 | 1205 – 1357 | 34–0–0 |
| gpt-5.6-luna | 1138 | 1097 – 1231 | 26–2–7 |
| muse-spark-1.2-contributor | 1138 | 1095 – 1234 | 27–3–5 |
| deepseek-v4-flash | 1102 | 1059 – 1181 | 25–4–9 |
| deepseek-v4-pro | 1099 | 1046 – 1189 | 25–5–7 |
| muse-spark-1.3-contributor | 1071 | 1016 – 1147 | 23–6–10 |
| deepseek-v4.1-flash | 1060 | 1000 – 1151 | 23–7–10 |
| qwen3.8-flash | 1028 | 949 – 1096 | 23–12–5 |
| hy3 | 1000 | pinned | 19–12–10 |
| glm-5.3-flash | 988 | 922 – 1048 | 17–12–12 |
| glm-5.2 | 956 | 888 – 1008 | 16–15–12 |
| hy4-preview | 946 | 868 – 1008 | 18–18–8 |
| nemotron-3-ultra | 919 | 827 – 970 | 15–19–9 |
| mimo-v2.5-pro | 917 | 838 – 958 | 15–20–9 |
| minimax-m3 | 887 | 791 – 925 | 11–19–11 |
| mimo-v2.5 | 869 | 783 – 913 | 8–19–14 |
| solar-pro4 | 855 | 767 – 887 | 8–20–11 |
| qwen3.8-27b | 852 | 762 – 884 | 8–21–11 |
| nemotron-3-super-120b | 814 | 712 – 844 | 7–24–6 |
| inkling-small | 779 | 661 – 804 | 5–27–5 |
| dots-3-note-preview | 717 | 586 – 730 | 2–30–2 |
| laguna-s-2.1 | 699 | 569 – 712 | 1–31–2 |
| nemotron-3.5-lightning | 694 | 546 – 701 | 0–30–3 |

Based on 445 comparisons covering 253 of 253 model pairs.

> The intervals for **gemini-3.8-flash** and **gpt-5.6-luna** overlap, so their ordering is not established by this data. More comparisons would be needed to separate them.

## Notes

- **Cost `sub`** means a subscription covered the call, so it billed zero. That is not the same as the model being free to run. **`N/A`** means the access path reports no cost at all. Neither is a price.
- **Slop** counts LLM-tell and translationese phrases per 1,000 words. **Repetition** is top-n-gram concentration. Neither enters the composite; both are diagnostics.
- Model behavior changes under a stable name. These results describe the versions tested on 2026-09-20.
