---
id: T-089-validate-integrated-v0-2-1-workflows-and-release
title: Validate integrated v0.2.1 workflows and release readiness
status: completed
priority: low
spec_ref: specs/v0.2.1.md#review-identity-and-export
dependencies:
    - T-088-export-approved-phrase-exercises-through-managed
    - T-066-show-reviewed-form-translations
    - T-067-automatic-light-dark-appearance
updated_at: "2026-10-11T16:47:57Z"
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

- 2026-10-11: source/base is accepted origin/main d1beff10fe8ca7f27486ea751c161468fb8df5a9,
  with T088 af9050e136033a238cbe2c780b2412a8e3f3aab3 verified as an ancestor.
  Pinned v0.2.1 spec SHA-256 is
  0b2150fc9778a3d7c0b058a5d0961a819ee283f148569323c32eee07ac3776b4.
  Independent validate/next matched T089 before start at 16:14:06 UTC.
- Offline walkthrough 16:14–16:33 UTC: ordinary documented CSV/lexical file,
  25 explicit input decisions, four families/six contrasting constructions,
  eight explicit exercises across both phrase recipes, authored correction,
  sidecar reuse, deterministic exports/checks and unchanged source manifest.
  Captured/adopted eight closed destination-fixture notes, approved/emitted/
  observed 12 retained operations (unresolved 0), reconciled version 3.
  Principal both recipes: reviewed regular/deponent/multi-sense translations,
  separate PPP/supine profiles, absent defective row rejected, Latin/German
  identity/slot checks and 28 fresh approved payload operations on three notes.
  Parsing: three accepted features, optional gender withheld, unchanged canonical
  claims/identity and Latin/German terminology, one fresh approved operation.
  These other managed lanes stopped at approval. SQL fixture application and
  backend fixture construction are not native CSV import/appearance/restore proof.
- Negative controls passed: governed-case/agreement/verb-number selection rejected;
  tampered export/stale evidence check rejected; ambiguous nominative/vocative
  single-case recognition withheld; old receipt rejected; historical CSS v2
  capture rejected; eight unsupported retirements could not be approved.
  Full provenance, commands, results, hashes and manual-resolution boundaries
  are in docs/release-v0.2.1.md and the T089 work thread.
- TDD applicability: documentation/validation only, no runtime change or new
  production test. Harness authoring errors were corrected, not claimed as RED.
  No verified task-local product failure or product follow-up was discovered.
- Workflow-v3: dedicated Task loaded code-simplifier; narrow documentation changes
  accepted, no evidence removed. Separate parallel General and Security Tasks
  loaded code-reviewer plus lane guidance. Security: "No concrete task-relevant
  findings." General candidate T089-1 (final timestamps missing pre-finalization)
  was rejected by fresh candidate-validation as premature: task still in_progress,
  no final lifecycle claimed. Final gate evidence is recorded below; actual
  lifecycle times will be recorded only after writes. No validated findings.
- Fresh code-reviewer disposition-verification confirmed T089-1's required evidence
  now recorded and the premature rejection appropriate, ending "No concrete
  task-relevant findings." One review cycle; no unresolved product findings.
  A final independent count check corrected the documentation from 26 to 25 input
  decisions; principal canonical claim/review equality also passed. General plus
  Security cover documentation correctness and data-loss/trust boundaries; no
  Python/framework/database lane added because production/storage contracts did not change.
- Final gates, 16:42:59–16:44:02 UTC on unchanged accepted runtime and reviewed
  documentation: uv run ruff check PASS; uv run mypy PASS (106 files);
  uv run pytest -v PASS (1,022 tests, 29.98s); mise run check PASS (including
  format check, full tests and commit/mutation policy suites); git diff --check
  PASS; Taskrail validate state valid; coverage 7/7 linked, 6/7 implemented before
  completion. Structural coverage is not executed/native acceptance coverage.
- Native Desktop/AnkiMobile auto acceptance remains pending under the maintainer's
  dated 2026-10-11 implementation-loop deferral in docs/morphology-native-verification.md.
  No unexecuted native lane is marked clean/passed/covered. No browser substitute,
  publication, version/tag/release/deploy or v0.3.0 activation is claimed.

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
- 2026-10-11T16:47:47Z: verification pass
- 2026-10-11T16:47:57Z: Implementation-only readiness: verify pass at 2026-10-11T16:47:47Z after full gates and independent reviews. Native Desktop/AnkiMobile auto acceptance remains pending per dated maintainer deferral; fixture SQL handoff is not native import/safety acceptance. No publication or version/spec activation.
