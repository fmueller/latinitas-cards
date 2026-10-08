---
id: T-081-make-form-parsing-docs-usable
title: Make form-parsing docs usable without reading tests
status: completed
priority: low
spec_ref: specs/v0.2.0.md#calibrated-form-parsing
dependencies: []
updated_at: "2026-10-08T01:25:54Z"
---

# T-081-make-form-parsing-docs-usable Make form-parsing docs usable without reading tests

## Description

Found by the T-069 new-user walkthrough. docs/contextual-form-parsing.md has no complete
accepted-decision example, refers to fixtures that live only in tests, mentions internal task
IDs and Python callers, and points to `REFERENCE_CARD_CSS` in source instead of the reference
note type doc. Not release-blocking.

## Acceptance

- A sanitized example cases.json produces at least one exportable exercise.
- User docs link the reference CSS section and omit internal task IDs and Python-only instructions.

## Verification Notes

- Docs-first RED: the new CLI regression failed first because the sample was
  missing, then because the unreviewed sample was ineligible. GREEN: explicit
  independently authored sample decisions, bound to fingerprints from supported
  CLI preview, produced the expected facts; 13 focused tests passed. No production
  implementation changed. Early test-authoring corrections aligned withheld
  status with the existing contract and avoided Rich error-line wrapping.
- Installed CLI in a clean temporary directory: documented preview/export commands
  produced exactly one exercise and one CSV row with the documented contextual
  ID, puella/nominativus/pluralis facts, and optional gender withheld. Checked exact
  fields, no Personal Notes, approval refusal/new-scheduling disclosure, unchanged
  payload on overwrite refusal, and stale-context withholding with zero export rows.
- Guide links and heading anchors passed, with no internal task IDs, Python-only
  instructions or source-CSS references. No rendered output changed; no new native
  Anki provisioning or client-verification claim was made.
- Dedicated code-simplifier: no changes. Separate General and Python code-reviewer
  lanes each returned "No concrete task-relevant findings." Candidate validation
  confirmed no candidates; fresh disposition verification found no unresolved or
  newly introduced task-relevant issues. No dispositions or follow-ups required.
  Security omitted: no production trust boundary, input behavior or data-write
  behavior changed. Other framework/database lanes were not materially affected.
- Final mandatory chain: uv run ruff check passed; uv run mypy passed (89 source
  files); uv run pytest -v passed (925 tests). git diff --check passed.

## Implementation Notes

- Added docs/examples/form-parsing-cases.json and a complete supported-CLI guide,
  including exact context/profile/evidence/review binding and published CSS link.
  The accepted judgments demonstrate sample mechanics, not expert truth or
  calibration. Retained independent contextual namespace, manual fresh-import
  setup, schema/template versions, user-data protection and unsupported operations.
- Pinned specs/v0.2.0.md#calibrated-form-parsing. Fetched origin/main contains the
  accepted T080 commit; no file transfers. Owned only T081. Release task T068 and
  off-spec work remain excluded; the orchestrator owns the adversarial next round.
- 2026-10-08T01:25:54Z: verification pass
- 2026-10-08T01:25:54Z: Docs-first CLI sample and regression delivered; explicit sample judgments only, no calibration/native provisioning claims; reviewed and final chain passed.
