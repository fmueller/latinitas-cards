---
id: T-089-validate-integrated-v0-2-1-workflows-and-release
title: Validate integrated v0.2.1 workflows and release readiness
status: todo
priority: low
spec_ref: specs/v0.2.1.md#review-identity-and-export
dependencies:
    - T-088-export-approved-phrase-exercises-through-managed
    - T-066-show-reviewed-form-translations
    - T-067-automatic-light-dark-appearance
updated_at: "2026-10-09T20:05:15Z"
---

# T-089-validate-integrated-v0-2-1-workflows-and-release Validate integrated v0.2.1 workflows and release readiness

## Description

Close the cross-feature acceptance matrix and document readiness without publishing
a release. This is a validation/documentation gate, not authorization to tag,
deploy or publish.

## Acceptance

- Run an offline CLI-only synthetic-deck walkthrough spanning reviewed inputs, bounded
  phrases, both recipes, per-claim corrections, sidecar reuse, inspection/check,
  deterministic CSV and managed withdrawal. Include stale evidence, ambiguous readings
  and tampered exports as negative controls.
- Start with an ordinary synthetic deck and supply missing lexical evidence through
  documented CLI/file import, not Python fixtures or prebuilt internal objects. Verify
  the initial CSV import/adoption handoff against captured destination fixtures,
  retained-slot updates and blocked lifecycle plans with explicit manual-resolution
  instructions, not a claim of automatic retirement or a native import check.
- Cover both principal-part recipes and form parsing with shared decisions, Latin/German
  terminology and translations where applicable; verify unchanged identities/slots and
  renewed managed payload approval for presentation/answer changes.
- Include both phrase recipes under Latin/German terminology; presentation changes keep
  canonical claims/reviews and identities unchanged. Validate phrase layout/style setup
  and fail-closed old-digest handling as well as current-schema updates.
- Confirm T-067 native Desktop and AnkiMobile appearance evidence before calling auto
  verified. If unavailable, record an explicit blocker, not emulation as native proof.
- Update user docs/changelog and readiness evidence with supported constructions, limits,
  unresolved blockers and native scope. Preserve v0.2.0 released metadata until a
  separately authorized release task.
- Run taskrail validate/coverage, mandatory ruff/mypy/pytest and mise run check.
  Record timestamps and actual results, not committed gitignored artifact paths.
- Do not activate v0.3.0, publish to PyPI, create tags or publish GitHub releases.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes

- Decomposition review, 2026-10-09: three independent medium-mode reviews covered
  linguistic/pedagogical completeness, architecture/sequencing and verification/CLI
  usability. All required findings were incorporated into T-064–T-067/T-084–T-089:
  input authoring and early identity bindings; designated targets/contextual ambiguity;
  verb/deponent/defective fixtures; schema/slot ownership; bounded package inspection;
  atomic batches; independent approval-gate controls; and truthful managed capabilities.
- Optional suggestions were also adopted: shared-review milestones, bounded compatibility
  constraints/traversal, phrase terminology integration and a native appearance checklist.
  No implementation or native verification is claimed by this planning change.
