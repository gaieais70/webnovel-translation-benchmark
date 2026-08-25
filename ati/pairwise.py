"""Pairwise tie-breaking stage. Deferred - interface only.

Recorded here so v1's data format does not need to change when this ships.

Intended design:

- Only the top N models by composite (default 5) are paired, on passage
  tasks only.
- The judge sees two anonymized translations of the same source passage and
  returns a winner plus a margin of 1-5.
- Every pairing runs **bidirectionally** (A vs B, then B vs A) and the
  results are averaged, cancelling position bias.
- Outputs are **truncated** to a fixed character count before comparison,
  cancelling length bias.
- Ratings use **Glicko weighted by win margin**, with the two calibration
  **anchor** models pinned to fixed ratings so that adding a model does not
  shift the existing field.

Pairwise output is a separate rating column in the report. It is never
merged into the rubric composite.
"""

from __future__ import annotations


class PairwiseNotImplemented(NotImplementedError):
    """Raised because the pairwise stage is deferred past v1."""


def rank_pairwise(judge_runner, run_id: str, runs_root, top_n: int = 5) -> dict:
    """Return model_id -> Glicko rating. See module docstring for the design.

    Bidirectional comparison, truncated outputs, Glicko with margin
    weighting, anchor models pinned.
    """
    raise PairwiseNotImplemented(
        "The pairwise stage is deferred past v1. See ati/pairwise.py for the "
        "intended design."
    )
