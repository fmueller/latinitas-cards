---
id: T-068-publish-v0-2-0
title: Publish v0.2.0
status: completed
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
    - T-083-restore-native-managed-absence-verification
updated_at: "2026-10-08T20:42:28Z"
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

- Step 1: Fetch confirmed HEAD and origin/main at accepted
  [e13bef1](https://github.com/fmueller/latinitas-cards/commit/e13bef1a5643160db9e59a19ce4ecbf49df865e6).
  Validate/next selected only T-068, all dependencies completed, no active owner.
  Spec remains v0.2.0, SHA-256
  `f54ab9cc84246cbbb8f7fa3399a7bec30012d09e3092c7b13c6499f09a1657ed`.
  Unshallowed history and reused docs/release-v0.1.1.md/T-047 procedure;
  historical release commits have no thread provenance trailers by repo policy.
- Step 2: Metadata RED failed at 0.1.1 != 0.2.0 before bump; GREEN passed
  after project/lock, changelog, README, index and readiness updates. Lock diff
  changes only project version; no runtime or dependency changes.
- Step 3: Initial ruff/mypy/pytest passed (90 files, 926 tests). Fresh native
  Anki 26.9.3 managed gate passed 17 cases on 2026-10-08. Report hash
  `45f2d6c46c7f5e15b10575e459a0688461de0e028837164db57f1cdb65935d6a`;
  script hash `2be32b1bd9617db35f03d577cd7fd34301c7fce2ec63b2129239f499d1a59f14`.
  Application-source bindings match accepted implementation; GUI/Mobile evidence
  is prior, not repeated. Readiness preserves exact native scope and waiver.
- Step 4: Dedicated Task loaded code-simplifier; simplified spec-index assertion
  and wrapped index text only. Focused metadata test and format check passed;
  parent inspected diff and reran the focused test successfully.
- Step 5: Parallel independent General and Security Tasks loaded code-reviewer
  and mapped guidance (General ECC code-reviewer; Security ECC security-reviewer,
  security-review companion and common security rules). Security lane selected
  for advisory/readiness assessment. Python specialist omitted: assertion-only
  metadata test changes, covered by General; no runtime language changes.
  Database/framework lanes omitted: no application, schema or persistence changes.
  General: "No concrete task-relevant findings." Fresh candidate-validation Task
  validated SEC-T068-001, rejected none, no duplicates.
- Validated SEC-T068-001 verbatim: "The release readiness document’s CLTK source
  citation is broken, preventing readers from checking the stated model-download
  behavior." Evidence: docs/release-v0.2.0.md:94 linked cltk/cltk/blob/main;
  nonexistent branch returned 404; master source resolved with REUSE_RESOURCES.
- Step 6: SEC-T068-001 fixed using an immutable CLTK commit citation. New
  assertion failed before doc fix, then passed after it. Pinned raw URL returned
  HTTP 200, source line 62 uses REUSE_RESOURCES, pyproject reports CLTK 2.5.1.
  No deferrals. Optional urllib3 alerts #141/#142/#143 remain open, two high and
  one moderate; source and fresh lock-export assessment documents conditional
  network reachability, not vulnerability-free certification or exploit proof.
- Step 7: Final exact ruff/mypy/pytest chain passed (90 files, 926 tests), then
  mise run check passed with format and all policy/mutation guards. Fresh Task
  loaded code-reviewer in disposition-verification mode: SEC-T068-001 resolved;
  "No concrete task-relevant findings." One review/fix/recheck cycle.
- Step 8: Preparation verification passed at 2026-10-08T20:37:32Z;
  publication then completed as recorded below. Evidence is separate from the
  immutable released commit. Final Taskrail closure follows fresh local gates.

## Publication evidence — 2026-10-08

- Published at 2026-10-08T20:40:39Z (22:40:39 Europe/Berlin):
  [v0.2.0 GitHub release](https://github.com/fmueller/latinitas-cards/releases/tag/v0.2.0).
  API confirms not draft, not prerelease, assets empty. No PyPI publication,
  deployment, other release, spec activation or unrelated shared-state action.
- One annotated [v0.2.0 tag](https://github.com/fmueller/latinitas-cards/tree/v0.2.0),
  tag object `073e0b30b2c3a4678822fe04058d2a018ed5325f`, targets
  [3b3616c](https://github.com/fmueller/latinitas-cards/commit/3b3616cef6c948da24d77214f64f2204a7031ba0).
  Remote tag object and peeled commit match; tagger is the maintainer.
  No published ref was rewritten. The candidate main push contains accepted
  e13bef1; publication-evidence commit follows it without changing the tag.
- [Exact-release-commit CI](https://github.com/fmueller/latinitas-cards/actions/runs/37840918189)
  completed successfully before tag creation: Python 3.13 and 3.14 each passed
  926 tests, lint/type/format and policy/infrastructure guards passed.
  Trigger was the normal authorized main push, not manual workflow dispatch.
- Fresh remote-tag clone passed README `uv sync --locked`; installed distribution
  metadata is 0.2.0. Root, managed and form-parsing help all exit 0 without optional
  CLTK/analyzer setup. HEAD equals released commit, local tag is annotated,
  and checkout remains clean.
- All 34 application-source hashes in the fresh native report match both the
  released checkout and the fresh tag clone; all 84 artifact hashes verify.
  Native script/application/dependency inputs are unchanged by release metadata.
  The 17-case retest therefore binds the final tagged application; no new GUI,
  Mobile or representative-deck test is claimed. Prior evidence and narrow waiver
  remain unchanged, with exact limits in docs/release-v0.2.0.md.
- Current advisory assessment remains three open optional urllib3 alerts,
  two high and one moderate, patched by 2.8.0 but still locked at 2.7.0.
  Default offline workflows exclude it; optional streaming resource downloads
  reach affected paths conditionally. Exposure is documented, not dismissed.

## Implementation Notes

- This task does not activate v0.2.1 or implement T-038 PyPI publishing.
- The 2026-10-07 orchestration restart tests first, implements confirmed findings
  and the remaining v0.2.0 tasks sequentially, then repeats adversarial testing.
  These implementation tasks precede publication; dependency ordering does not
  authorize a release or replace the final testing and publication gates.
- 2026-10-08T20:37:32Z: verification pass
- 2026-10-08T20:42:28Z: verification pass
- 2026-10-08T20:42:28Z: Published immutable v0.2.0 GitHub release/tag; fresh installation and exact-release CI pass. Evidence recorded separately; optional annotation advisories remain documented and unresolved.
