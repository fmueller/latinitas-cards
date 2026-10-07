---
id: T-068-publish-v0-2-0
title: Publish v0.2.0
status: todo
priority: high
spec_ref: specs/v0.2.0.md#goals
dependencies:
    - T-069-check-v0-2-0-gaps-and-drift
    - T-070-keep-unapproved-divergent-fields-out-of-baseline
    - T-071-record-explicit-decisions-for-unselected-tag-conflicts
    - T-072-label-unreviewed-fourth-role-as-linguistic-review
    - T-073-decide-representative-deck-native-acceptance
    - T-074-refresh-cli-help-for-v0-2-0-workflows
    - T-075-capture-snapshots-and-first-adoption-from-cli
    - T-076-verify-managed-csv-desktop-import-dialog
    - T-077-preserve-morphology-across-setup-reconfigure
    - T-078-close-managed-lifecycle-classification-test-gaps
    - T-079-add-extraction-parsing-fixtures-and-review-counts
    - T-080-harden-managed-handoff-attestation-and-wording
    - T-081-make-form-parsing-docs-usable
    - T-082-persist-explicitly-selected-no-write
updated_at: "2026-10-06T20:02:15Z"
---

# T-068-publish-v0-2-0 Publish v0.2.0

## Description

Publish the completed managed-update, extraction, form-parsing, and morphology
work from specs/v0.2.0.md using the annotated-tag and GitHub-release process of
v0.1.0/v0.1.1 (see docs/release-v0.1.1.md and T-047). Owner authorization covers
release metadata commits, main pushes, the tag, and GitHub publication only.
PyPI publishing stays in specs/v0.3.0.md (T-038).

## Acceptance

- Bump pyproject.toml and uv.lock project version to 0.2.0; move CHANGELOG
  `[Unreleased]` entries under a dated `## [0.2.0]` section, leave an empty
  `[Unreleased]`, and update comparison links.
- Update README release banner and tag-based install instructions to v0.2.0;
  update specs/README.md to mark v0.2.0 completed.
- Add docs/release-v0.2.0.md with readiness evidence, practical limitations, and
  a current dependency-advisory assessment.
- Update tests/unit/release_candidate_test.py metadata assertions for v0.2.0
  (red before the bump, green after).
- Pass mandatory ruff/mypy/pytest, `mise run check`, and exact-commit CI on
  Python 3.13/3.14 before tagging.
- Publish one annotated v0.2.0 tag and matching GitHub release (not draft or
  prerelease, no uploaded assets); do not publish to PyPI.
- Verify fresh-tag `uv sync --locked` install, installed distribution metadata
  reports 0.2.0, and `latinitas-cards --help` runs; record publication evidence
  separately.

## Verification Notes

- Pending.

## Implementation Notes

- This task does not activate v0.2.1 or implement T-038 PyPI publishing.
- The 2026-10-07 orchestration restart tests first, implements confirmed findings
  and the remaining v0.2.0 tasks sequentially, then repeats adversarial testing.
  These implementation tasks precede publication; dependency ordering does not
  authorize a release or replace the final testing and publication gates.
