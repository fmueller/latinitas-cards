---
id: T-032-review-extraction-coverage-and-correct
title: Review extraction coverage and correct demonstrated contract defects
status: todo
priority: high
spec_ref: specs/v0.1.0.md#extraction-content-review
dependencies:
    - T-031-require-evidence-backed-field-suggestions-and
updated_at: "2026-09-27T08:56:30Z"
---

# T-032-review-extraction-coverage-and-correct Review extraction coverage and correct demonstrated contract defects

## Description

Adjudicate representative content against the existing principal_parts.py layout and
normalization contract before expanding parsing. Fix only demonstrated in-contract bugs.

## Acceptance

- Record population/per-category counts, reproducible stratified sample selection,
  reviewed counts, findings, and unreviewed remainder for generated, skipped, incomplete,
  unsupported and ambiguous entries. State a synthetic-only boundary if applicable;
  never claim universal or private-deck coverage from that sample.
- Distinguish mapping error, extraction defect, genuinely missing data, unsupported
  layout, and unresolved role. Commit only synthetic or independently sanitized examples.
- Correct reproduced in-contract extraction/normalization defects with discriminating
  regressions and independent expectations, preserving display/comparison values.
  If no additional defect is found, document that outcome rather than inventing code work.
- Fix the reproduced markup-only eligibility defect: amo — amare — <b></b> — amatum
  produced eight notes/no skips, including a blank perfect answer and empty recognition
  target. Normalize semantic display text once before eligibility, retain raw provenance
  separately, and escape at rendering without weakening T-026's safe-HTML contract.
- Regress markup-only and comment-only parts, encoded whitespace, required leading
  omissions, and decode-once/double-entity inputs. Assert actual normalized eligibility,
  skip/review classification and nonblank rendered answers, not raw-string presence.
- Never infer PPP from supine or vice versa, shift omitted roles, or select alternatives
  silently. Broader formats/alternatives/omissions/mixed hints/calibration go to v0.2.0.
- Preserve historical German terminology approval; identify newly introduced wording
  that needs review and explain coverage limits and deferred cases.

## Verification Notes

- Review adjudications against the supported matrix. Run the mandatory ruff/mypy/pytest
  chain for any code changes. Record evidence and limits when executed.

## Implementation Notes
- Use improved confirmed mapping from T-031; do not import or commit private deck outputs.
