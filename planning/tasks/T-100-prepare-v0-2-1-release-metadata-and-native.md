---
id: T-100-prepare-v0-2-1-release-metadata-and-native
title: Prepare v0.2.1 release metadata and native readiness
status: completed
priority: medium
spec_ref: specs/v0.2.1.md#goals
dependencies: []
updated_at: "2026-10-14T03:27:39Z"
---

# T-100-prepare-v0-2-1-release-metadata-and-native Prepare v0.2.1 release metadata and native readiness

## Description

Prepare the explicitly authorized v0.2.1 release using the repository's prior
annotated-tag/GitHub-release process, after accepted implementation and native
Linux Desktop verification. This task prepares and delivers a reviewed candidate;
publication is separately executed after exact-commit CI is verified.

## Acceptance

- Bump project metadata and the project's uv.lock entry to 0.2.1 without changing
  dependency versions. Move pending user-focused changelog entries under dated
  0.2.1, leave empty Unreleased, update comparison/release links and README's
  tag-install instructions. Update existing release metadata assertions with TDD.
- Refresh v0.2.1 readiness and the current native verification record with the
  parent-accepted fresh replay on cac2d7ff99a533c39a8bd83a733b4b905c0851c8.
  Distinguish Linux synthetic acceptance from Mobile (explicitly unverified),
  macOS/Windows, sync, live collections, general scheduling and expert language
  calibration. Historical reports/deferrals stay historical, not overwritten.
- Record 32 native lanes, 112 notes/320 cards, 104 same-card theme cycles plus
  six supplements, 54 observed retained effects/zero unresolved, exact native
  restore, stock backend/GUI positive and expected-negative controls, and CSS v4
  manual installation/fresh-capture limits. Do not call screenshots proof of
  table preservation. No automatic migration or additional safety guarantees.
- Clearly distinguish original adversarial queue clean at 4/5 from separate
  native acceptance runs: run 1 found repaired defects, run 2 runtime passed
  but filesystem evidence finalization failed, fresh replay 3 was accepted with
  all 54 archive hashes and 4937 member hashes independently verified.
- Assess and disclose the four current optional annotation dependency alerts
  using fresh lock exports/current advisories and bounded authoritative sources.
  Do not remediate dependencies, dismiss alerts or assert vulnerability-free
  safety. Default offline install excludes affected libraries.
- Scoped mutation survivor and 1050-test unmutated baseline are report-only
  evidence, not a completed full mutation campaign or efficacy certification.
- Run workflow-v3 via autonomous-backlog, dedicated simplification/independent
  review/disposition checks, final Ruff/mypy/pytest and mise run check. Complete
  Taskrail lifecycle and push a focused candidate to origin/main. Do not tag,
  create a GitHub release, publish PyPI, deploy or activate v0.3.0 in this task.

## Verification Notes

- Accepted product revision: cac2d7ff99a533c39a8bd83a733b4b905c0851c8.
- Pinned spec SHA256: aabfdf060b1d90074a672b8cd29875677857b3d2ccb1f4455e09ade0e9de5ca4.
- CSS v4 SHA256: 18cd5b7d2e5b50f90ac017db225d49f7ff11fa57a35058487300e4b56457945e.
- Accepted replay report and checkpoint inventory are in the parent thread's
  review artifacts; cite durable source/replay thread and hashes in committed
  notes, not concrete gitignored planning artifact paths.

## Implementation Notes

### Workflow-v3 evidence — 2026-10-14

1. Understand: fetched origin/main and independently confirmed clean base
   ce0429edb016520f4ba37103a04747031c688907. Taskrail validate returned
   state valid; next --json selected only T100, on-spec, before start.
   Read the full task and checked the pinned spec hash. Unshallowed before
   inspecting prior release commit 3b3616c and its T068 documentation method.
   Ran mise run setup and installed maintainer hooks before committing.
   Scope is candidate delivery only; runtime/CSS/dependencies/spec are unchanged.
2. Strict TDD: updated the existing release metadata test, then ran
   uv run pytest tests/unit/release_candidate_test.py -k versioned_for_v021 -q.
   RED: one failed, expected assertion '0.2.0' == '0.2.1'. Minimal project/lock,
   changelog, README and readiness changes made the three file tests GREEN.
   Historical audit assertions remain in the same test; no new test file.
3. Initial checks: Ruff passed, mypy passed (107 files), pytest -v passed
   (1050 tests, 57.98s). No production/UI change or new native replay was made.
4. Simplify: dedicated Task loaded code-simplifier; no useful simplification,
   no edits or rejected suggestions. Three focused tests/diff check passed.
5. Independent review: separate parallel Tasks loaded code-reviewer in General,
   Python and Security lanes. General used ECC code-reviewer/common review;
   Python used ECC python-reviewer/python-patterns; Security used ECC
   security-reviewer/security-review/cloud-infrastructure-security.
   General is mandatory; Python covers metadata/test contract; Security covers
   conditional advisory and safety disclosures. Database omitted: no schema,
   SQL or persistence changes. Framework/ML/network omitted: no runtime,
   framework, model or infrastructure behavior change. Two specialist lanes
   fit the default budget. Each returned verbatim:
   "No concrete task-relevant findings."
   A fresh candidate-validation Task loaded code-reviewer and confirmed no
   candidate IDs to validate/deduplicate and no actionable omitted candidate.
6. Dispositions: no concrete findings, fixes or deferrals; no invented rows.
7. Recheck: final Ruff passed, mypy passed (107 source files), pytest -v
   passed (1050 tests, 50.92s); mise run check passed, including formatting,
   1050 tests (49.71s) and all policy-guard suites. Fresh disposition review
   must finish before lifecycle finalization. One review cycle, no fix cycle.
   uv build produced 0.2.1 sdist/wheel; isolated locked-default wheel install
   reported 0.2.1 and CLI --help passed; neither urllib3 nor fsspec installed.
8. Finalize: verify/complete and focused candidate push follow the completed
   review gates. Publication and exact-candidate remote CI remain parent-owned.

### Evidence boundaries and remaining risks

Current native evidence is the parent's accepted fresh replay 3, not new evidence
from this metadata task. Durable parent/replay links and inventory/report/media
hashes are in docs/release-v0.2.1.md. All 54 archives were independently checked;
4937 manifest-listed member hashes exclude checkpoint 00 (no member manifest).
54 retained effects are operations, not notes; media is not table proof.
Original non-native queue 4/5, native run 1 repaired defects, run 2 evidence
failure, and replay 3 acceptance remain separate. Scoped survivor/selector
failure and unmutated baseline are report-only, not a full mutation campaign.

Fresh GitHub Dependabot/advisory API checks and locked default/CPU/GPU exports
confirmed four optional annotation alerts (3 high, 1 moderate), default exclusion
of urllib3/fsspec, and affected versions in both extras. Bounded upstream Stanza,
Requests and fsspec source reads support the conditional exposure disclosure.
No dependency remediation, alert dismissal or exploit test was performed.
Mobile/macOS/Windows/sync/live collections/general scheduling/expert calibration
remain unverified. CSS v4 manual setup, old-binding refusal, fresh capture,
renewed payload approval and unsupported structural changes remain unchanged.
No tag/release/PyPI/deployment or v0.3 activation is part of T100 completion.
- 2026-10-14T03:27:39Z: verification pass
- 2026-10-14T03:27:39Z: Fresh disposition verification clear; no concrete task-relevant findings. Candidate checks complete. Publication remains pending parent exact-commit CI and separate publication; T100 completes candidate preparation only.
