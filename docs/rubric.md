# Rubric

Six axes, each scored 1-5. Anchors below are included verbatim in the judge
prompt so scores mean the same thing across runs and across models.

## accuracy — does the English convey what the source says?

- **1** — Substantial mistranslation. A reader would come away with a wrong
  understanding of events.
- **3** — Broadly correct, with errors in detail: a misread modifier, a
  confused subject, an idiom taken literally.
- **5** — Faithful throughout, including idioms, implication, and subtext.

## terminology — are names and setting terms rendered consistently?

- **1** — The same name or term appears in two or more forms with no reason.
- **3** — Mostly consistent; minor drift in secondary terms.
- **5** — Every name, title, honorific, and setting term is stable.

## register — is voice, formality, and tone preserved?

- **1** — Flattened to one uniform voice; formal and crude speech read alike.
- **3** — Distinguishes characters, but tone slips in places.
- **5** — Each character's voice and the narration's tone survive intact.

## fluency — does it read as natural English prose?

- **1** — Word-order artifacts and calques throughout; reads as raw MTL.
- **3** — Understandable but stilted; a reader notices the translation.
- **5** — Reads as English prose written by a competent author.

## fidelity — absence of omission, addition, and hallucination

- **1** — Whole sentences dropped, or content invented that is not in the source.
- **3** — Minor omissions or small unsupported embellishments.
- **5** — Complete, with nothing added and nothing dropped.

## sanitization — absence of softening or censoring

- **1** — Violent, sexual, or dark content removed or rewritten as inoffensive.
- **3** — Content present but noticeably toned down.
- **5** — Rendered at the source's full intensity.

A hard refusal is not scored on this scale. It is recorded separately in the
refusal rate.
