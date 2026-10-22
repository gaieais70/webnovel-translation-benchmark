# Run notes — `2026-09`

Operational findings from run `2026-09`: twenty-three models translating
eleven consecutive chapters of 화산귀환 (*Return of the Mount Hua Sect*,
chapters 410–420, Korean → English), judged by Gemini through Antigravity.
Twenty were scored first; three more were added on 2026-09-19 under the same
conditions (§10), and one of them is Gemini itself, which is why §11 exists. Everything
here is measured, not inferred. Per-call data lives in `runs/` and
`judgments/`; this file records what the numbers add up to.

This run is **not comparable to `2026-08`**. The corpus is a different work in
a different genre (murim, not modern fantasy), eleven models are new, several
models changed provider, and the judge's prompt carries a new line (§6).

## 1. Corpus preparation

The source files were scraped from a web reader and carried its furniture.
Each chapter opened with a heading cluster — the site title (`화산귀환 - 411화`),
the reader's font-size widget (`글자` / `-` / `16px` / `+` / `기본`) and a
numbered chapter heading — and six of the eleven ended with a lone `1`, a
page counter.

The operator asked for this to be cleaned by a Sonnet 5 agent before any model
saw the text. The agent wrote cleaned copies to `raw/hwasan/` and left the
originals untouched. Verified independently afterwards:

- every file is a **pure deletion** of the original — no line edited, added
  or reordered (difflib opcodes, all `delete`);
- no widget token, numbered heading, title line or bare counter survives
  anywhere in any file;
- the intake stage's own header stripper removes nothing further.

The numbered chapter heading was removed along with the rest of the cluster.
Its number is off by one against the file name (`chapter_411.txt` opens with
`410화.`), which would otherwise have been translated as content.

| Passage | Source file | Characters |
|---|---|---|
| hwasan-ch01 … ch11 | chapter_410 … chapter_420 | 6,179 – 7,910 |

## 2. Routing

The operator directed that translations go through Command Code. Its live
catalogue serves sixteen of the twenty models, including `deepseek-v4-flash`
and `glm-5.2`, which the roster had placed on OpenRouter; both were moved.
Four models are not served by Command Code at all and stayed on OpenRouter:
`solar-pro4`, `nemotron-3-super-120b`, `nemotron-3.5-lightning`,
`dots-3-note-preview`.

## 3. Command Code drops any buffered request that runs past ~125 seconds

Every 524 (`Upstream model provider is temporarily unavailable`) in the first
hour arrived at a total latency of 392–395 s: three attempts of ~125 s each
plus the 5 s and 10 s backoffs. Every call that *succeeded* on Command Code
finished inside ~125 s — hy3's 374 s and mimo-v2.5's 251 s "successes" were
one or two 524s followed by a retry that happened to finish in time.

So the 524 is not the upstream being unavailable. It is the gateway abandoning
a request that has not completed within its window, and a model that reasons
longer than that at maximum effort cannot finish a chapter however often it is
retried. Before the fix: qwen3.8-flash 0 of 8, hy3 2 of 6.

Streaming fixes it. With `stream: true`, reasoning deltas flow for the whole
generation and the gateway never sees an idle request. Measured on the same
chapter that had failed every time: qwen3.8-flash ran **721 s**, streamed
59,331 reasoning tokens and 15,506 characters of translation, and finished
cleanly. hy3 went from 2/11 to 11/11. Every Command Code row now streams.

A stream can still be cut: twice the gateway closed one cleanly with no
finish reason and no `[DONE]` (glm-5.3-flash, 78 s in). That is treated as a
dropped connection and retried, not scored.

## 4. Provider defaults that bind silently

- **MiniMax M3 on Command Code has a ~5,120-token default output cap.**
  Chapters 3 and 10 stopped at exactly 5,120 and 5,119 tokens with
  `finish_reason: length`. The row now sends `max_tokens: 65536` (probed and
  accepted); both chapters then completed.
- **Nemotron 3 Ultra on Command Code has a 4,096-token default output cap.**
  Chapter 8 stopped at exactly 4,096; every chapter that completed sat just
  under it (3,354–4,029). Same fix, same probe.
- **Inkling Small on Command Code has the same 4,096-token cap**, and at
  maximum effort it spent every one of those tokens reasoning: all eleven
  chapters stopped at exactly 4,096 with `finish_reason: length` and no
  translation at all. Same fix.
- A scan of every model's maximum output for clustering just under 4,096 /
  8,192 / 16,384 / 32,768 / 65,536 found no other cap that had bound.
- **Command Code has a 5-hour usage window**, and a full max-effort run
  exhausts it: `You've reached your 5-hour usage limit for your plan. Your
  limit resets at ...`. The run stage stops a model on it rather than burning
  the queue; the remaining chapters were finished after the reset.
- **OpenRouter free routes sometimes answer HTTP 200 with an error object in
  place of `choices`.** It used to be recorded as `malformed response:
  'choices'`, discarding the provider's message. It is now reported with that
  message and treated as a route failure, so a model with a paid fallback
  uses it.

## 5. Output that passed every guard and was not a translation

The length guards (`came_up_short` below 0.5×, `ran_away` above 4.0×) let
through three kinds of output that are not translations. Found by reading the
endings of every chapter whose ratio fell well below the ~2× norm:

- **Provider moderation, mid-translation.** Xiaomi's route stops a
  translation partway and appends *"The request was rejected because it was
  considered high risk"* — mimo-v2.5 ch10, mimo-v2.5-pro ch05 and ch08.
  Alibaba's answers HTTP 400 `DataInspectionFailed` — qwen3.8-flash ch10.
  These chapters have bandits, threats and killing in them. For someone
  translating the work this is a refusal: the route will not translate the
  chapter. They are now recorded as refusals (`blocked_by_provider: true`),
  count toward the refusal rate, are not scored, and are not retried.
- **The source echoed back.** solar-pro4 returned chapter 4 in Korean, at a
  length ratio of exactly 1.00. Output whose letters are mostly in the
  source's script is now a failed translation.
- **Generation cut short by the provider, reported as a normal finish.**
  nemotron-3.5-lightning's free route hit exactly 600 s on eight of eleven
  chapters — three returned nothing, four stopped mid-sentence, one had
  already degenerated; dots-3-note-preview stopped one mid-sentence after
  32 s. Neither was flagged, because only `finish_reason: length` was checked
  and the finish reason was not recorded at all. It is now recorded on every
  run; `error` is treated as a route failure (and uses a paid fallback where
  one exists), `content_filter` as a refusal. Both models were re-run in full
  under the corrected harness so each is measured under one set of conditions.
  On the re-run nemotron-3.5-lightning completed all eleven: nine on the free
  route, two (ch04, ch07) on the paid fallback after the free route ended the
  generation with `finish_reason: error`. The `route` field on each record
  says which.

Where the refusals fell — nine in all, every one a provider's moderation
layer rather than a model declining:

| Passage | Blocked for |
|---|---|
| ch01 | glm-5.2 |
| ch05 | mimo-v2.5-pro |
| ch06 | dots-3-note-preview |
| ch08 | mimo-v2.5-pro |
| ch09 | glm-5.2 |
| **ch10** | **dots-3-note-preview, glm-5.2, mimo-v2.5, qwen3.8-flash** |

ch10 (source chapter 419) tripped four different providers' filters — Zhipu,
Xiaomi, Alibaba and dots' upstream. No model refused on its own account.

Records written before these guards existed were re-checked against them by
`scripts/2026-09/reclassify.py`, run after every lane pass that predated them;
eight records changed in all (seven provider blocks, one echoed source).

Degenerate output that is the model's own — nemotron-3.5-lightning filling a
chapter with `- - -` or `Y Y YY Y Analysis` inside the length window — is
left to the judge, which scores it as what it is.

## 6. The judge, and the Google account behind it

Gemini is reached only through the official Antigravity CLI (`agy`) on the
operator's own account, by the operator's rule. The account is the one thing
in this project that cannot be re-provisioned, so every agy call now passes
through a gate (`CallGate`, `ati/runners/agy.py`):

- **one call at a time, machine-wide** — a cross-process file lock, so two
  stages started together queue instead of running concurrently;
- **a 10-second gap** between the end of one call and the start of the next;
- **fair between processes** — whoever made the last call waits 2 s longer,
  so a second waiting stage always gets the next turn (without this, the
  rubric stage starved a pairwise process for twenty minutes);
- **an empty working directory**, so an agent running with
  `--dangerously-skip-permissions` has no `.env`, corpus or run output in reach;
- **stop, don't retry, on a usage limit** — the rubric stage now halts on the
  first usage-limit error or two consecutive unreachable items; it used to
  retry every item three times and move on, which after a quota hit would have
  sent hundreds of refused requests in a row;
- config refuses a Gemini slug on any runner but `agy` — Command Code's
  catalogue lists `google/gemini-*`, so one mistyped row would otherwise have
  routed the judge through a gateway.

**The judge was running shell commands.** agy is an agent, not a completion
endpoint. In the first 30 judge calls, 3 started a shell command mid-judgment,
and those 3 were exactly the calls that hung until the timeout — every hang,
no successful call. agy's own `PreToolUse` deny hook would block this, but it
loads only in a workspace the operator has trusted in their Antigravity
settings, which this project does not edit. Instead every judge prompt now
opens with one line telling the judge to use no tools and answer from the text.
In the calls after that change: no tool use, no hangs, 30–95 s per call.

That line changes the judge's prompt, so the 35 judgments made before it were
moved to `judgments/2026-09-pre-notools/` (kept, not deleted) and re-judged.
Every judgment in `judgments/2026-09/` was made under one prompt.

**Quota sets the pace, not throughput.** After roughly 180 judge calls in
four hours, Antigravity answered `RESOURCE_EXHAUSTED (code 429): Individual
quota reached ... Resets in 59m36s` — consistent with a rolling five-hour
window that opened with the first call. The stage stopped on that first
refusal. The driver (`scripts/2026-09/judge_driver.py`) reads the stated
reset time and waits exactly that long plus three minutes, so at most one
refused call reaches the account per window. It runs the rubric stage and then
the pairwise stage in one process, so the two never split a window.

Two harness defects in the judge path, both caught before they cost a call:

- `gemini-judge` declares the pairwise schema, and the rubric stage used the
  judge's declared schema — every rubric reply would have been forced into
  `{winner, reason}` and failed parsing. The rubric stage now passes
  `--judge-schema` explicitly.
- A rubric judgment recorded as `judge_error` counted as current, so one bad
  response left the item unscored for good. It is now re-judged.

## 7. Retry policy

Each lane makes up to three passes; later passes only retry chapters that
failed. Route failures (524, disconnects, cut streams) are retried because
they are the provider, not the model. Model failures — output that came up
short, ran away, or was truncated — get the same three attempts and are then
recorded as failures. The "Scored" column in the leaderboard shows how many
chapters each composite rests on.

## 8. Results

Published leaderboard: [archive/2026-09-hwasan/results/leaderboard.md](../archive/2026-09-hwasan/results/leaderboard.md).
220 outputs: 210 scored, 9 refused (all provider moderation, §5), 1 failed
(dots-3-note-preview ch07, an upstream HTTP 400 on all three attempts). 210
rubric judgments, 300 head-to-head comparisons covering all 190 model pairs,
no judge errors.

    #  model                        composite  scored  BT rating  W-L-T
    1  gpt-5.6-luna                     80.3   11/11      1119    20-0-5
    2  muse-spark-1.2-contributor       79.6   11/11      1119    21-1-3
    3  deepseek-v4-pro                  78.5   11/11      1097    20-1-7
    4  qwen3.8-flash                    73.2   10/11      1021    19-8-4
    5  deepseek-v4-flash                72.7   11/11      1089    19-1-8
    6  glm-5.2                          71.5    8/11       963    13-8-11
    7  hy4-preview                      71.3   11/11       945    15-12-6
    8  glm-5.3-flash                    69.5   11/11       986    15-8-10
    9  hy3 (weak anchor)                65.1   11/11      1000    17-8-8
   10  nemotron-3-ultra                 59.7   11/11       917    12-12-7
   11  mimo-v2.5-pro                    58.1    9/11       924    12-12-9
   12  mimo-v2.5                        57.6   10/11       871    6-12-14
   13  minimax-m3                       54.2   11/11       885    9-13-10
   14  qwen3.8-27b                      54.0   11/11       861    8-14-11
   15  solar-pro4                       51.4   11/11       861    8-14-11
   16  nemotron-3-super-120b            40.0   11/11       818    7-18-6
   17  inkling-small                    31.8   11/11       784    5-20-5
   18  dots-3-note-preview              22.7    8/11       719    2-23-2
   19  laguna-s-2.1                     17.7   11/11       710    1-22-2
   20  nemotron-3.5-lightning           16.2   11/11       699    0-22-3

**The two methods agree: Spearman ρ = 0.982, maximum displacement 3 places.**
The top three and the bottom six are the same in both. As in 2026-08, one
judge produced both, so this is partly the judge agreeing with itself, not two
independent instruments converging.

- **The top two are not separated.** gpt-5.6-luna and muse-spark tie on the
  Bradley–Terry scale (1119 each, intervals 1083–1241 and 1088–1245) and sit
  0.7 composite points apart. gpt-5.6-luna leads the rubric on accuracy (4.5)
  but has the lowest fidelity of the top ten (3.7): it drops or compresses
  content where muse-spark (4.9) and deepseek-v4-pro (5.0) do not.
- **Where they disagree, it is the anchor and the middle.** hy3 is 9th on the
  rubric but 6th head to head; glm-5.2 and hy4-preview are the reverse. Their
  intervals overlap heavily — the middle of the field is not ordered by this
  data beyond "between 900 and 1020".
- **glm-5.2 scores on 8 of 11 chapters.** Its three refusals include ch10, the
  hardest-filtered passage; its composite is over an easier subset.
- **The bottom three are floor-bound.** With 0–2 wins in ~25 comparisons the
  fit is regularised toward the floor, and each point rating sits just above
  its own bootstrap interval. Read them as "last three", not as precise
  values.

**Position bias** is smaller than in 2026-08 but present: the translation
shown first won 59% of decisive verdicts (80% in 2026-08). 27 of 300
comparisons were slot-preference ties — the judge picked the same position in
both orders — and were discarded rather than credited.

**Not comparable to 2026-08.** Different work, different genre, eleven new
models. The anchor moved from 71.4 to 65.1; on a different corpus that says
nothing about drift. The report records drift as not checked.

**Not done in this run:** the secondary-judge agreement check (not
implemented); human spot-checks (`docs/spot-checks.md` is for a person to
fill in); verification that `reasoning_effort: max` took effect on every route
— the `reasoning_tokens` on each run record is the evidence.

## 9. Harness changes made during and after the run

Everything above was fixed during the run (commit `5f6520b`). After the run
closed, a cleanup pass (a DeepSeek v4.1 audit, implemented by an Opus agent in
an isolated worktree and verified to reproduce this run's numbers exactly)
added generic tooling so a future run needs none of `scripts/2026-09/`:
`ati run --passes N --wait`, `ati judge --wait --follow`, `ati pairwise
--target N --wait`, `ati status`, `ati recheck`, atomic record writes, judge
fingerprints on new judgments, and input and code fingerprints in
`results/<run>.json`. See [the runbook](runbook.md). It also found that
`ati run` without `--model` would have sent the judge rows through the
translation stage — including Gemini through Antigravity; judges now carry
`role: judge` and are never translation subjects.

## 10. Three models added after the first twenty were scored

`muse-spark-1.3-contributor` and `deepseek-v4.1-flash` (Command Code), and
`gemini-3.8-flash` — Gemini as a translation *subject*, through Antigravity,
the only route this project reaches Gemini by. Same corpus, prompts, guards
and judge as the rest of the run; they entered late, nothing else changed.
All three completed 11/11 with no failures. Gemini was the fastest row in the
run at 37s per chapter; deepseek-v4.1-flash the slowest of the three at 204s.

Head-to-head was extended from 300 to 445 comparisons so every one of the new
253 model pairs is covered, and all 33 new outputs were judged on the rubric.

Final ordering (rubric composite, then Bradley-Terry rating):

     1  gemini-3.8-flash            88.0   1229   34-0-0   <- see §11
     2  gpt-5.6-luna                80.3   1138   26-2-7
     3  muse-spark-1.2-contributor  79.6   1138   27-3-5
     4  muse-spark-1.3-contributor  79.3   1071   23-6-10
     5  deepseek-v4-pro             78.5   1099   25-5-7
     6  deepseek-v4.1-flash         74.7   1060   23-7-10

Spearman rho between the two methods is 0.978 over all 23 models, maximum
displacement 4 places. Position bias fell again: the first-shown translation
won 57% of decisive verdicts (59% at 300 comparisons, 80% in 2026-08); 34 of
445 comparisons resolved as slot-preference ties and were discarded.

muse-spark 1.3 lands a shade below 1.2 on both measures while writing
markedly cleaner prose (slop 0.31 against 0.90). deepseek v4.1-flash beats
v4-flash on the rubric and trails it head to head; both intervals overlap.

## 11. The judge favours its own family - measured, not assumed

gemini-3.8-flash tops both rankings, scored by gemini-3.7-flash-high. That is
not an independent measurement, and the numbers look like it: a flat 5.0 with
**zero variance** on five of six axes where no other model exceeds 4.5 on
accuracy, and 34 head-to-head wins with **no losses and not one tie**, in a
field where every other leader ties several times.

So the check the methodology has always described was finally run. grok-4.6 -
not a subject in this run, and no relation to either the judge or the models
it scores - re-judged the same translations against the same rubric. Composite
over the judge-scored axes only, so both judges are measured the same way
(these are not the published composites, which use computed term stability):

    model                 Gemini judge   grok judge   change
    gemini-3.8-flash            100.0         93.5     -6.5
    deepseek-v4-pro              90.0         92.6     +2.6
    gpt-5.6-luna                 90.8         92.0     +1.2
    deepseek-v4.1-flash          88.3         88.8     +0.5

Both judges rank gemini-3.8-flash first, so the result is not invented. But
its lead over the next model falls from **+9.2 to +0.9**. The Gemini judge
gave its own family 5.0 on accuracy and fidelity; the independent judge gave
4.7 and 4.5, and simultaneously raised gpt-5.6-luna's fidelity from 3.7 to
4.1 on the same text. About nine tenths of the margin is the judge.

Read the top row as "among the best", not as a nine-point lead. The
leaderboard and the published page both carry that warning.

**Scope.** The first pass covered four models (44 judgments) and 2 of 20
comparisons before Command Code's limit stopped it. The rest ran on
2026-09-23: three more models on the rubric (muse-spark 1.2 and 1.3, hy3; 33
judgments) and the remaining 18 comparisons among five. By then Command Code
had no credit left at all (§14), so grok moved to OpenRouter
(`x-ai/grok-4.6`). xAI is that slug's only provider, so the model is the same
one; the route is not, and a route change changes the judge fingerprint and
would have marked the first 44 judgments stale. The new pass was scoped with
`--model` to the three missing rows instead, so both sets stand. Cost $3.77,
against an estimate of $1.50.

On the five judged axes (the basis of §13), all seven models:

    model                   Gemini judge   grok judge   change
    gemini-3.8-flash              100.0         95.3     -4.7
    muse-spark-1.3                 93.5         89.9     -3.6
    deepseek-v4-pro                90.9         92.5     +1.6
    muse-spark-1.2                 90.3         91.2     +0.9
    gpt-5.6-luna                   89.1         90.6     +1.5
    deepseek-v4.1-flash            87.6         88.8     +1.2
    hy3                            75.1         84.4     +9.3

gemini-3.8-flash still ranks first under grok, by +2.8, against +6.5 over the
next of these seven under Gemini. Only one other model loses ground:
muse-spark-1.3, which Gemini ranks second of seven and grok fifth. hy3's +9.3
is mostly grok's compression (70.6% of its axis scores are a flat 5.0 against
Gemini's 51.9%), which lifts weak rows most.

Head to head under grok (20 comparisons, hy3 anchored at 1000):
gemini-3.8-flash 1265, gpt-5.6-luna 1239, muse-spark-1.2 1239,
deepseek-v4-pro 1190. The same leader, by a margin that twenty comparisons
among five models cannot separate from its neighbours.

## 12. Harness defects found while adding the three models

- **`ati run` with no `--model` would have translated through the judges** -
  including Gemini on the operator's own Antigravity account. Only the
  explicit model lists in the lane scripts prevented it. Judges now carry
  `role: judge`, agy rows are never subjects unless named, and a test asserts
  both.
- **No API-routed judge could ever have worked.** The judge stages pass an
  empty system prompt, and xAI answers HTTP 400 "system message must have
  content". The empty message is now omitted. The Gemini judge never hit this:
  the agy runner concatenates system and user text.
- **`sonnet-judge` has never been usable.** Command Code serves Claude only on
  `/provider/v1/messages` in the Anthropic Messages shape; this harness speaks
  the OpenAI-compatible shape. It is kept declared, with a note.
- **The head-to-head pool excluded agy rows**, so gemini-3.8-flash - the row
  most in need of checking - was left out of the comparison stage entirely.
  Comparing a translation costs judge calls, not the route that wrote it, so
  the pool is now every non-judge row.

## 13. A second independent judge, and figures that can be reproduced

- **jev-1.13 scored all 243 scorable outputs** through OpenRouter's decisions
  endpoint (`typesafe/jev-1.13`), which is not a chat model: each rubric axis
  is a `score` question whose criteria are the five anchors, and it returns a
  probability-weighted value over them rather than a whole number. That value
  is continuous (3.12, not 3); truncating it with `int()` was a bug caught in
  the first probe, where four of five axes disagreed with their own
  distributions. Terminology is not scored - jev returns no term table - so it
  is compared on the five judged axes. Cost: $0.11.
- **It confirms the self-preference, more strongly than grok.** On the five
  axes, gemini-3.8-flash leads the next model by +9.1 under Gemini and +2.8
  under grok. Under jev it does not lead: deepseek-v4-pro edges it, and across
  the field glm-5.3-flash scores highest.
- **Compression is now measured.** Gemini gives a flat 5.0 on 46.6% of every
  axis score in the run. On the four models all three judges scored: Gemini
  79.1%, grok 75.0%, jev 4.5%. The first independent check was nearly as
  compressed as the judge it checked.
- **jev detects bias; it does not rank.** Its top eight models fall within
  2.6 points and every 95% bootstrap interval overlaps every other, and it
  scores the weakest models 25-33 points above Gemini. Its ordering at the
  top is noise, and it is not a replacement leaderboard.
- **§11's figures stand, on their basis.** The cross-judge table was first
  computed in a session scratchpad that did not survive. The method is the
  weighted mean of the axes, each normalised 1 -> 0 and 5 -> 100; on all six
  axes, including judged terminology, it reproduces both of §11's columns
  exactly, and the +9.2 -> +0.9 lead. The page now uses five axes so that jev
  can sit beside the others, which is why it says +2.8. Both are reproducible
  from `scripts/2026-09/cross_judge.py`.

## 14. Five more models: the middle tier and the Claude column

- **The tiers.** The routing sheet marks one break - the blank row 16 - not
  three tiers. The operator split what sits below it into middle (glm-5.3 at
  $0.91/$2.86 per 1M tokens, kimi-k3 at $1.70/$8.50) and high (grok-4.6 and
  Qwen3.8-Max, both $2.00/$6.00), which stay held. Prices are OpenRouter list,
  2026-09-20; by 2026-09-23 glm-5.3 was $0.84/$2.64 and kimi-k3 $3.00/$15.00.
- **Three routes were checked for the middle tier, and it waited.** Command
  Code serves both but its weekly limit was spent until 2026-09-21 21:40.
  opencode (`opencode.ai/zen/go/v1`) still lists both, but the workspace
  balance is zero. OpenRouter would have cost about $10 at list price, and
  the operator keeps OpenRouter use to a minimum, so both started on Command
  Code after the reset.
- **Command Code then ran out of money, not time.** At 13:42 on 2026-09-22
  every model on the account began answering HTTP 400 "You have insufficient
  credits". That is a third limit, separate from the 5-hour and weekly
  windows: a monthly dollar bucket that refills only at the billing cycle or
  by purchase, with no balance endpoint to check it. Max-effort reasoning had
  emptied it - about 90% of the run's Command Code output tokens are
  reasoning, and glm-5.3 alone spent 55k-126k per chapter. glm-5.3 stood at
  10/11 and kimi-k3 at 2/11. `ati cost` (below) puts the run's Command Code
  translations at $8.59 at list price, which fits a $10 bucket once grok
  judging and overwritten retries are added.
- **The middle tier finished on opencode Go**, the operator's own
  subscription (dollar allowances per 5 hours, week and month). kimi-k3's two
  Command Code chapters were quarantined in `runs/quarantine-2026-09/` and
  regenerated there, so all eleven share one provider; glm-5.3 keeps its ten
  Command Code chapters and took ch10 from Go. `route` on each record says
  which provider wrote it. The model weights are the same on both routes; the
  serving stack is not, which is the same caveat the roster already makes
  about opencode in 2026-08. Go's 5-hour allowance stopped the pair twice on
  2026-09-23 (about 15:45 and 20:14), and `--wait` waited out the first. The
  machine went down unexpectedly during the second wait, so the chain was
  restarted on 2026-09-26; it translated the last two chapters in seven
  minutes. Cost at list price: glm-5.3 $2.60, kimi-k3 $5.56.
- **The Claude column runs on the local Claude Code CLI.** Command Code
  answers MODEL_NOT_IN_PLAN for every Claude model, so the Messages-shaped
  runner its `/provider/v1/messages` wants would not have helped. The CLI
  runs on the operator's own subscription with `--system-prompt` (which
  replaces the agent's prompt rather than appending to it), every tool
  denied, and an empty working directory. It is a different route from every
  other row, so these are not directly comparable to the API rows. All three
  were translated 11/11 by 2026-09-21 19:45; sonnet-5 averaged 1,040 s per
  chapter against 127 for opus-5 and 144 for haiku-4.5. The operator then
  asked for everything outstanding to be finished, so on 2026-09-23 Gemini
  judged all 33, and they joined head to head after the middle tier
  (+120 comparisons, 525 -> 645).
- **Nine defects, nearly all of them a limit the harness could not see:**
  - `--wait` read a duration ("Resets in 1h17m35s") but not an instant
    ("resets at 2026-09-22T01:40:52.743Z"), so a weekly limit backed off one
    hour and gave up six hours into a thirty-two hour wait.
  - The CLI runner read only stderr on failure; `claude -p` reports its limit
    on stdout, so twenty failures were recorded as a bare "exit 1: ".
  - The CLI's "session limit" matched no usage-limit marker, so it was
    retried as an ordinary failure instead of waited out.
  - Its reset is a wall-clock time in a named zone ("resets 12am
    (America/New_York)"), which `--wait` also could not read.
  - `load_models` never passed `system_prompt_flag` through, so the three
    Claude rows would have run underneath the coding agent's prompt. A test
    now requires every roster key to map to a real field.
  - `--model` reached the judge stage's parser but not `judge_run`, so a
    scoped judging pass judged the whole run - on the operator's own Gemini
    quota.
  - A test compared the real clock with a fixed reset date, and began failing
    once that date came within twelve hours.
  - The middle-tier waiter was launched as a detached process twice, and both
    times it was gone by morning. It is a Scheduled Task instead.
  - That task ran its script in a visible console, and closing the window
    ended the run seven chapters in (0xC000013A, about 23:47). It now starts
    hidden.
- **Found once the credit ran out:**
  - The credit message matched the usage-limit markers, so `--wait` treated
    an empty balance as a window and retried it hourly. A balance does not
    reopen on its own; it now ends the run at once (97c4a96).
  - Progress was counted in calls attempted, and a call that fails on a limit
    is still an attempt, so every stop logged "1 in a row" and the give-up
    count never fired: nine hours of hourly retries, 13:42 to 23:01, that
    would have gone on indefinitely. Progress is now finished work (77e1eb3).
  - Nothing could say what a batch would cost before it ran. `ati cost`
    prices each row's recorded tokens at list price and projects what is left
    from the average finished chapter. It is a floor: a retry overwrites the
    record of the attempt it replaced.
- **Three glm-5.3 chapters needed a second attempt.** ch06 spent the
  provider's whole default of 131,072 output tokens on reasoning and produced
  no text; ch10 did the same on Go. Both were left to the retry passes at the
  same cap and finished there (101,834 and 103,281 reasoning tokens; the
  other chapters took 52k-122k). ch07 came back with zero reasoning tokens in
  69 s, on the same route and request, so it was not produced at max effort;
  it is kept in `runs/quarantine-2026-09/` and was regenerated at 79,636.
  kimi-k3's ch07 on Go reports no reasoning count at all, but it emitted
  45,851 tokens for about 4k of text, so it reasoned like the others; the
  route omitted the breakdown on that call.
- **Results.** The chain finished on 2026-09-27: 28 models, 645 comparisons
  covering all 378 pairs. Archived on 2026-10-22 in
  `archive/2026-09-hwasan/` (until then the archive held the 23-model
  version), with list-price cost per chapter added to the report (§15).

## 15. The frontier column, a retired judge, and cost on the leaderboard

- **Gemini 3.7 Flash is gone.** On 2026-10-22 agy answered
  `gemini-3.7-flash-high` with "not recognized as a known model", so nothing
  can be added to the Gemini-judged ranking above; it is closed as it stands.
  By the operator's decision the run is re-judged in full by a panel of two
  families, Haiku 5.5 (`haiku-judge`, Claude CLI) and Gemini 3.8 Flash
  (`gemini-3.8-judge`, agy), each on both stages, each under its own
  directories, averaged by `ati report --panel`. Each judge is also a
  subject, so its own row and its family's are flagged, and every judge's
  composite is shown beside the panel's.
- **Seven rows added:** Claude Opus, Sonnet and Haiku 5.5 on the Claude CLI;
  GPT-6.1-Sol, GPT-6-Astra and GPT-6-Luna on the Codex CLI under the
  operator's ChatGPT subscription (no OpenAI key is held, and the metered-API
  rows drafted on 2026-10-10 never ran); Gemini 4 Argon on agy, deferred by
  the operator because agy's quota was spent until about 2026-10-25.
- **Codex as a bare translator.** Its base instructions are replaced from a
  file (`{system_prompt_file}`), and its skills list, environment, permission
  and plugin blocks, AGENTS.md, tools and user config are switched off. Two
  keys first tried for this (`include_skills_usage_instructions`,
  `include_plugins_instructions`) are not ones Codex knows, and it ignores
  unknown keys silently; `codex exec --json` reported it. The working switches
  were each checked with `codex debug prompt-input`. That mattered: the
  operator has a skill named novel-translation-engine, which Codex would
  otherwise have listed to the model. Four chapters written before the fix
  are in `runs/quarantine-2026-09/`.
- **Cost is recorded for every route.** The CLI rows of this run recorded no
  tokens, so opus-5, sonnet-5 and haiku-4.5 have no cost. New CLI rows ask for
  structured output (`claude-json`, `codex-jsonl`) and record tokens, cached
  input and reasoning. The leaderboard prices every record one way - the
  route's own list figure where it gives one (the Claude CLI's matches
  OpenRouter list to the cent), otherwise tokens at the roster's OpenRouter
  list price - and shows list $ per chapter and output tokens per chapter.
  CLI rows carry their harness's input overhead (about 12k cached tokens on
  Claude, 7k on Codex per call).
- **Two limit-detection defects fixed before they cost anything.** A CLI
  failure kept the first 200 characters of stderr, where Codex prints a
  banner and the Claude CLI now prints a tool notice, so neither's limit
  message survived. And Codex's "usage limit ... Try again in 3 hours" matched
  the transient marker "try again", so a spent quota would have burned every
  pass in seconds. Explicit limit wording now wins.
