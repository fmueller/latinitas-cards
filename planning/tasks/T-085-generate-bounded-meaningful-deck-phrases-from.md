---
id: T-085-generate-bounded-meaningful-deck-phrases-from
title: Generate bounded meaningful deck phrases from reviewed patterns
status: todo
priority: high
spec_ref: specs/v0.2.1.md#deck-based-phrase-and-grammar-practice
dependencies:
    - T-084-define-reviewed-deck-lexical-inputs-and
updated_at: "2026-10-09T20:05:15Z"
---

# T-085-generate-bounded-meaningful-deck-phrases-from Generate bounded meaningful deck phrases from reviewed patterns

## Description

Generate deterministic phrase candidates from the reviewed input contract,
preserving evidence and alternatives for subsequent recipe and export tasks.

## Acceptance

- Generate all four construction families, varying noun case/number, adjective agreement
  and verb person/number/tense only where reviewed data supports them.
- Enforce government, agreement and selected-sense compatibility; report skipped
  candidates with actionable reasons, never silently infer missing forms or naturalness.
- Consume saved selections/limits; deterministic ordering and bounded output prevent
  combinatorial growth and do not auto-select every variant.
- Bound traversal as well as emitted output; a large selection with a small limit
  must not eagerly materialize the Cartesian product. Use T-084 constituent bindings
  for stable candidate keys before review, preserving distinct senses with equal glosses.
- Candidate preview exposes source identities, senses, pattern identity/version,
  chosen forms, supplied function words, evidence and review state. Candidates are
  proposals, not export-approved cards.
- Fixtures contrast cum amico/ad amicum, genitive constructions, adjective agreement
  and short clauses with incorrect government, incompatible senses and unsupported forms.
- Include at least two supported verb-form choices with independently reviewed expected
  analyses and a subject–verb disagreement negative control. Declare the supported
  subset rather than claiming exhaustive conjugation support.
- Pass the mandatory ruff/mypy/pytest chain.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes
