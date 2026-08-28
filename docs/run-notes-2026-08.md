# Run notes — `2026-08`

Operational findings from the first full run of the 11-chapter
`ghost-commute-consistency` task. Everything here is measured, not inferred.
Per-call data lives in `runs/` and `judgments/`; this file records what the
numbers add up to, which is not recoverable from the artifacts alone.

## 1. The same model name is not the same model across providers

`glm-5.3-flash` was reached three ways. It behaved differently each time:

| Route | ch01 | ch03 / ch08 |
|---|---|---|
| opencode | clean, 28,190 output tokens | timeout ×3 |
| Command Code | — | ch03 timeout 900s; ch08 truncated at 45,153 |
| OpenRouter | **truncated at 42,846** | not attempted |

The opencode build is the least verbose of the three and therefore scores on
the most chapters. OpenRouter generated 52% more on a chapter opencode
completed cleanly. Provider is part of the measurement, not a detail below it.

hy3 has the same problem in a different form: opencode serves `hy3`, Command
Code serves `tencent/hy3-paid`. Different SKU, unknown whether same weights.

## 2. Routing, per provider

| Provider | Endpoint | Notes |
|---|---|---|
| opencode | `/zen/go/v1/chat/completions` | `muse-spark-1.2-contributor` and `gpt-5.6-luna` return **HTTP 500** to every request including a bare `{model, messages}` payload — server-side, unrelated to usage, persistent across 33+ attempts. `grok-4.6` is refused: `not supported for format oa-compat`. |
| opencode CLI | `opencode.cmd run` | Reaches all three. But it is an **agent loop**, not a completion: 4.3 calls/chapter, 5.0:1 input:output from re-sent context, and **no `max_tokens` ceiling** — which is how gpt-5.6-luna emitted 119,112 characters on ch09. |
| Command Code | `api.commandcode.ai/provider/v1` | Works for fast models. **524** (`Upstream model provider is temporarily unavailable`) on slow upstreams: `tencent/hy3-paid` 6/11, `xai/grok-4.6` after 333s. Separate **429 concurrency limit** — four parallel jobs triggered it. Returns **no `cost` field**. |
| OpenRouter | `/api/v1/chat/completions` | `x-ai/grok-4.6` 11/11 clean, 146s avg. Cheapest `glm-5.3-flash` at $0.07/$0.25 but the most verbose build. |

On Windows the CLI runner must invoke `opencode.cmd`, not `opencode`: npm
installs an extensionless shell script alongside the batch shim, and
`subprocess.run` without a shell can only execute the latter. The bare name
fails `WinError 2` on every call, which in the run records is indistinguishable
from the model failing.

## 3. Token profiles differ by an order of magnitude

Output tokens per chapter, for roughly the same amount of visible English
(~16,000 characters):

| Model | Output tokens/chapter | Hidden reasoning share |
|---|---|---|
| gpt-5.6-luna | ~3,500 | negligible |
| muse-spark-1.2-contributor | ~7,300 | moderate |
| grok-4.6 | ~9,900 | ~66% |
| inkling-small | ~16,000 | ~83% |
| glm-5.3-flash | 28,000–42,846 | 86–90% |

OpenRouter's telemetry confirms this is generation, not infrastructure: one
glm-5.3-flash call spent **3s routing and 607s generating** — ~70 tokens/sec,
a normal rate, producing an abnormal volume. Its two failed chapters are the
model, established across four routes, three token budgets and two timeouts.

## 4. Two passages break models regardless of family

| Passage | Chars | Models broken |
|---|---|---|
| **ch09** | 6,987 | 6 — deepseek-v4-flash (timeout), mimo-v2.5-pro (truncated), longcat-2.0 (timeout ×2), gpt-5.6-luna (runaway ×2), inkling-small (truncated) |
| **ch03** | 8,705 | 4 — glm-5.3-flash (timeout ×3), minimax-m3 (runaway ×2), inkling-small (truncated), grok-4.6 (CLI route only) |

Neither is the longest passage. Every failure is **over-generation**, never
under-generation. Unexamined: whether something in the content induces this.
Decide before publishing whether these are scored passages or documented
degenerate cases.

## 5. Single attempts are not reliable on reasoning-heavy models

Same model, same chapter, same budget, different outcome:

- `glm-5.3-flash` ch02: 25,446 tokens (truncated) → 22,788 (clean)
- `inkling-small` ch03: 25,296 (clean) → 46,338 (truncated)
- `glm-5.3-flash` ch01: 28,190 on opencode → 42,846 on OpenRouter

`iterations: 1` is cheap but a single failure is not proof of a limit, and a
single success is not proof of capability.

## 6. Harness defects found and fixed

All were producing results that looked like model failures.

| Defect | Fix |
|---|---|
| Token budget starved short chapters (flat 16,384 reasoning allowance) | 2.5/char + 24,576, clamp 49,152 |
| 600s timeout cut off generation the budget permitted | 900s |
| HTTP 5xx treated as permanent | retry `(500, 502, 503, 504)` |
| Cloudflare **524** outside the retry set | added; body explicitly says to retry |
| Runaway generation scored as translation | `MAX_OUTPUT_EXPANSION = 4.0` |
| Stub output scored as translation | `MIN_OUTPUT_RATIO = 0.5`, refusals exempt |
| `content: null` raised `TypeError` out of `run_model`, killing the whole run | coerced to `""` |
| Judgment staleness checked one direction only | symmetric |
| Judgments not tied to the text they scored | SHA-256 pinning |

The last one mattered most: re-running a model with `--force` left its old
judgment looking current, so the previous route's scores would have been
reported against the new translations with nothing in the output showing it.

## 7. Cost

opencode, 355 billed calls, $7.42 total — of which **$3.82 was a separate
coding project**, not this benchmark. Benchmark spend was $3.60 across 221
calls.

`grok-4.6` alone was $2.08 of that — 58% — entirely because of the agent
route's re-sent context (837,992 input tokens against deepseek-v4-pro's
50,503 for comparable work). The same model on OpenRouter, as a bare
completion: **11/11 instead of 9/11, 146s instead of 253s, ~$0.76**.

Command Code reports no cost, so any model routed there shows `N/A`. Real
per-model costs for the opencode models are computable from the usage export
and are not currently wired into the report.
