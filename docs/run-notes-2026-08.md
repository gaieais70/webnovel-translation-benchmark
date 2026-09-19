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

## GLM-5.3-Flash ch03: the prompt is not the cause

Tested after an outside suggestion that the ch03/ch08 failures were bad
prompting rather than the model. They are not.

Same passage, same route (OpenRouter), same 46,338-token budget, one variable
changed - a curb appended to translate.md telling the model to begin the
translation immediately and not deliberate, plan, or outline first:

    baseline  lat=343s  fin=error   out=30,173 tok  out_chars=0
    curbed    lat=807s  fin=length  out=46,338 tok  out_chars=0

The curb made it worse. 53% more output tokens, 2.4x the latency, and still
not one visible character. Instructing the model to skip deliberation fed the
loop instead of stopping it.

Four independent lines now agree the cause is the model meeting this passage:

  1. 9 of 11 chapters succeed on the unmodified prompt. A wrong prompt fails
     uniformly; this fails on exactly ch03 and ch08.
  2. ch03 fails on all four routes tried (opencode, Command Code, OpenRouter,
     CLI).
  3. Given a 131,072-token ceiling - 2.7x the normal budget, ~19x the input -
     it ran 1,784s, consumed every token, and produced out_chars=0. It never
     began writing. Cost $0.033 to learn.
  4. Artificial Analysis independently profiles GLM 5.3 at ~47k tokens average
     with ~34k of reasoning, second-highest of the models they track.

Consequence: translate.md is unchanged, so the eleven-model leaderboard needs
no re-run and stays comparable. ch03 and ch08 are recorded as degenerate cases
for this model, not as harness or prompt defects. 9/11 stands.

Method note: this is the shape to reuse. Changing translate.md to rescue one
model would have changed conditions for all eleven and forced a full re-run -
so the hypothesis had to be tested against a copy, never by editing the real
prompt in place.

## The pairwise judge has an 80% first-position bias

Measured over the first 148 of 300 comparisons. Slots are assigned
alphabetically, so slot A is arbitrary with respect to quality.

    forward order:  A=118  B=25  tie=5     <- first-shown won 80% of the time
    reverse order:  A=58   B=84  tie=6

Across both orders, first position wins 59% and second 37%. Judging in a
single order would have produced a confident, meaningless ranking.

Resolution verified correct on every observed cell:

    fwd  rev -> outcome   n    meaning
      A    B -> a        84    both orders chose model a
      B    A -> b        24    both orders chose model b
      A    A -> tie      29    judge chose the SLOT, not the model
      B    B -> tie       1    judge chose the SLOT, not the model
      A  tie -> tie       6    explicit tie verdict
    tie    A -> tie       5    explicit tie verdict

Note `reverse` is recorded in the PRESENTATION frame, not the model frame.
forward=A with reverse=A means the judge picked whichever translation was
shown first, both times - a slot preference, correctly discarded as a tie.
Reading it as the model frame makes agreement look like disagreement.

30 of 148 (20%) resolved as slot-preference ties and were discarded rather
than credited. The 108 decisive comparisons are those where the judge named
the same translation regardless of position. That subset is the trustworthy one.

Coverage confound to apply when reading pairwise standings: models with
missing chapters are compared on fewer passages, and the chapters they are
missing are the hard ones. glm-5.3-flash (3 passages, missing ch03+ch08),
inkling-small (3, missing ch03+ch09) and minimax-m3 (3, missing ch03) are all
graded on an easier subset than the eight models with 4 passages. Check
coverage before believing a win rate.

## Pairwise complete: 300 comparisons, rho = 0.936 against the rubric

Final Bradley-Terry ratings (hy3 is the weak anchor, pinned at 1000):

    1337  deepseek-v4-pro       100%  43-0-17   7 psg   rubric 1
    1260  glm-5.3-flash          92%  34-3-21   6 psg   rubric 4  (inflated)
    1234  muse-spark-1.2-contr   83%  30-6-16   6 psg   rubric 2
    1194  grok-4.6               75%  27-9-17   7 psg   rubric 3
    1187  gpt-5.6-luna           79%  23-6-24   7 psg   rubric 5
    1082  deepseek-v4-flash      52%  22-20-18  7 psg   rubric 6
    1000  hy3 (anchor)           35%  14-26-13  7 psg   rubric 8
     980  mimo-v2.5-pro          31%  12-27-14  7 psg   rubric 7
     910  longcat-2.0            20%   8-33-12  7 psg   rubric 10
     799  minimax-m3              9%   4-42-6   6 psg   rubric 9
     740  inkling-small           4%   2-47-4   6 psg   rubric 11

Spearman rho = 0.936, max displacement 2 places. The rubric and head-to-head
comparison measure substantially the same thing. Outcomes: 52% a, 21% b,
27% tie. Forward slot-A win rate 75%, so the position bias held to the end
and the both-order control remained necessary throughout.

Do not trust a partial pairwise run. At 148/300 gpt-5.6-luna was 17-0 and
ranked 1st, ahead of deepseek-v4-pro; it finished 6-6 over the back half at
23-6 and 4th. deepseek-v4-pro was the only undefeated record that survived
(43-0). Preliminary standings were reported and had to be retracted.

glm-5.3-flash's 2nd place is a coverage artifact: 6 passages against 7 for the
models above it, missing ch03 and ch08 - the two passages that break models.
It is graded on the easy subset. Same caveat applies to minimax-m3 and
inkling-small at 6 passages.

## Where both methods are blind: the tail, not the mean

Operator read ch05 from gpt-5.6-luna and muse-spark side by side and preferred
muse-spark, describing luna as catching some specific terminology but being
"pretty horrible" when it errs. The completed pairwise agrees (muse-spark 3rd
at 83%, luna 4th at 79%), but the mechanism is worth recording.

Diagnostic on that chapter: luna renders a term as "numinous" twice where
muse-spark writes "spirit" (4x) and "divine" (1x). Arguably the more precise
term, and a judge scoring terminology in isolation credits it - but "a
candidate numinous being" is prose a reader stops on. Wins the term, loses
the sentence.

Aggregate text statistics cannot see this. Same chapter, same source:

    metric              luna    muse-spark
    words               2574    2444
    words >=10 chars    119     120        (muse-spark uses MORE)
    type-token ratio    32.4%   32.9%
    words/sentence      9.8     8.5

Essentially indistinguishable. The defect is sparse and local - a handful of
wrecked sentences out of 264 - against a statistically identical background.
The rubric compresses a chapter to six 1-5 scores; the pairwise judge forms a
whole-chapter impression. Both average the tail away. A reader does not.

Proposed but NOT yet built or approved: a third measurement scoring the tail -
one translation at a time, judge extracts the N worst sentences with severity
ratings, models ranked by error density and worst-case severity rather than
overall impression. Would run on the same passages as pairwise for
comparability.
