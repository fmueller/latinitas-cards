---
id: T-208-parse-confirmed-verb-variants-and-report
title: Parse confirmed verb variants and report exhaustive skips
status: todo
priority: high
spec_ref: specs/v0.5.1.md#verb-parsing-and-skip-reports
dependencies:
    - T-207-approve-deck-workflow-contract-and-calibration
updated_at: "2026-11-05T22:11:24Z"
---

# T-208-parse-confirmed-verb-variants-and-report Parse confirmed verb variants and report exhaustive skips

## Description

Deliver FR-4 by extending source_extraction/principal_parts and focused command
modules, not growing cli.py. Own extraction consumers before nominal routing.
The maintainer approved the expanded FR-4 planning amendment, not implementation
authority; preserve the inactive spec and blocked contract gate.

## Acceptance

- Confirm normalization in both lexical and principal-parts fields for trailing
  HTML, parenthesized usage, + Dat./+ Abl., mit Akk. and X = Y lines. Preserve raw
  evidence and rule/fragment provenance. Meaningful (Perf.), coeptum (in|ceptum),
  number/gender and actual alternatives must not be stripped indiscriminately.
- Compound pipes use a separate option from pipe_alternatives, remain displayed
  and are ignored for canonical lexical/answer equivalence, never split into
  objects. Follow T-206's approved answer-surface contract, not an assumed grader.
- ab|sūmere, re|ferre, ad|esse, esse (est - sunt), timēre, placēre, dīcere (PPP)
  and iubēre (infinitive) generate cards without any source-note edits, under
  confirmed profiles and applicable recorded review. Cover both recipes and assert
  dīcere PPP/iubēre infinitive eligibility/output. Ignore annotations for parsing
  but retain source fields/provenance, verified before/after; no cleaned substitutes.
  Pipes remain displayed and pipe-free answer equivalence succeeds. Preserve
  existing macron/identity policy and review gates.
- Reviewed loquī/loquor/locūtus sum and audēre/audeō/ausus sum generate cards with
  German deponent/semi-deponent role labels. Cover gaudēre/gaudeō/gāvīsus sum
  (+ Abl.) and retain actual evidence; never invent active forms or infer PPP
  from three-part count. Preserve mixed placuī / placitus sum perfect alternatives.
- Assert expected parsed roles, preserved alternatives and usage separation for
  timēre/placēre appended text, esse usage hints, NBSP, absent roles and in|cipere
  with parenthesized in|ceptum, even if later linguistic eligibility is withheld.
  For the required reviewed timēre/placēre fixtures, card generation is mandatory;
  skip diagnostics cannot satisfy acceptance. Ambiguous negative controls still
  fail closed with a specific reason naming the fragment.
- Reject ambiguous boundaries/conflicting pipe policies. Keep decet/oportet/āiō/
  ōdisse fixtures excluded with existing codes; keep empty certiōrem facere/necesse
  (est) at missing_principal_parts and mis-tagged dīligentia/prae|ceptum non-verbal.
  Preserve old-profile output, fourth-role review and source checksums.
- Expose complete preview --json, readable table and CSV report with note key,
  lemma, primary code/related reasons, fragments and class/scope provenance.
  Deterministically aggregate distinct source notes, not recipe/role counts;
  partially productive notes are not fully skipped. Full modes never truncate.
- Reconcile the reported unchanged-policy baseline of 115 productive/100 skipped
  Verb notes (65/24/2/6/2/1 codes) only against the exact available source/profile.
  Post-fix totals instead enumerate recovered/residual notes and sum to 215 if that
  snapshot is available; JSON/table/CSV agree. Without private access, record this
  count check not evaluable and use redistributable characterization/recovery fixtures.

## Verification Notes

Use contrasting fixtures via preview/generate, compare source checksums and
unchanged-profile goldens; assert card output/labels and canonical equivalence,
not just absence of crashes. T-216 owns native display/equivalence acceptance and
exact-source baseline-versus-recovery accounting. Run uv run ruff check → uv run mypy → uv run pytest -v;
restart the full chain after failure. Follow all spec compatibility/permission rules.

## Implementation Notes
