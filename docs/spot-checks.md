# Human spot-checks

Judge scores are validated by human review of a sampled subset. Each entry
records what was checked and whether the judge's score held up.

| Date | Run | Model | Task | Judge score | Human assessment | Agreed? |
|---|---|---|---|---|---|---|
| 2026-08 | `2026-08` | gpt-5.6-luna vs muse-spark-1.2-contributor | ghost-commute ch05, read side by side | head-to-head: muse-spark 3rd (83%), luna 4th (79%) | preferred muse-spark; luna catches some terminology but is "pretty horrible" when it errs | yes, on the ordering |

Detail of the 2026-08 check, including why neither the rubric nor the
head-to-head judge sees the kind of error that decided it, is in
[run-notes-2026-08.md](run-notes-2026-08.md) ("Where both methods are
blind").
