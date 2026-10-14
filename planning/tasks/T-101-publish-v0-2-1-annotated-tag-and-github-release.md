---
id: T-101-publish-v0-2-1-annotated-tag-and-github-release
title: Publish v0.2.1 annotated tag and GitHub release
status: completed
priority: medium
spec_ref: specs/v0.2.1.md#goals
dependencies: []
updated_at: "2026-10-14T03:49:44Z"
---

# T-101-publish-v0-2-1-annotated-tag-and-github-release Publish v0.2.1 annotated tag and GitHub release

## Description

Publish v0.2.1 under the maintainer's explicit authorization to continue through
release. Follow the existing annotated-tag and GitHub release process, after
accepted T100 candidate preparation and native Linux Desktop signoff. No PyPI,
uploaded release assets, deployment or v0.3.0 activation is authorized.

## Acceptance

- Fetch and reconcile the accepted candidate and remote ownership; confirm
  no v0.2.1 tag/release exists before creating either. Never rewrite a remote tag
  or blindly retry an uncertain publication operation; inspect actual remote state.
- Finalize only necessary public release metadata (README/index/readiness and
  release assertions) with workflow-v3 review and final checks. Preserve accepted
  runtime, locked dependencies, spec scope and native/advisory limitations.
- Pass final Ruff/mypy/pytest, mise run check, and exact-tag-target GitHub Build
  lint/type/policy plus Python 3.13/3.14 before tagging. Verify other applicable
  current checks. No manually triggered deployment or mutation campaign.
- Create exactly one annotated v0.2.1 tag under maintainer identity and push that
  tag, then publish a matching non-draft/non-prerelease GitHub release without
  assets. Use concise user-focused notes and explicit practical limitations,
  including CSS v4 manual setup/fresh capture and Mobile unverified status.
- Verify remote annotated tag dereferences to the accepted tested commit and
  GitHub release points to that tag. Perform fresh-tag clone, locked installation,
  distribution version 0.2.1 and CLI help smoke; no optional annotation extra.
- Record immutable tag/CI/publication/install evidence separately after release,
  without changing the tag. Complete Taskrail lifecycle only after successful
  publication verification and mandatory checks. Deliver focused evidence notes
  and CLI-generated state to origin/main; report actual final remote state.
- Original adversarial queue stays at 4/5; native replay is separate. Preserve
  four optional dependency alerts and scoped mutation limitations. No universal
  native safety, sync, scheduling, expert-language or Mobile claim.

## Verification Notes

- Parent accepted T100 candidate de0bdeae3f3b5e12752deca72bb99f23d2b0a0f2.
- Exact candidate Build passed, run 38506454208; CodeQL 38506453374 and
  dependency graph 38506460009 passed. Changed tag-target commits need fresh CI.
- Pinned spec SHA256: aabfdf060b1d90074a672b8cd29875677857b3d2ccb1f4455e09ade0e9de5ca4.
- Parent accepted native replay on cac2d7ff99a533c39a8bd83a733b4b905c0851c8;
  T100 changed metadata/docs/tests only, not native runtime or CSS.

## Implementation Notes

- Step 1: Fetched origin/main at accepted planning base
  [749e8c8](https://github.com/fmueller/latinitas-cards/commit/749e8c8611dae98f8169de6c7012fc7edd0a8153),
  unshallowed before history reads, verified the pinned spec hash above.
  Independent Taskrail validate/next selected only T101 on the active spec,
  with no active task. T038 remains off-spec and untouched. Remote v0.2.1 tag
  and GitHub release were absent. Read full T068 acceptance, publication
  procedure and release commit history; hooks installed under maintainer identity.
- Step 2: Existing metadata test RED failed on the current-release assertion;
  minimal README, spec index and readiness updates made all three focused
  tests GREEN. No artificial TDD for prose. Runtime/CSS, lock and active spec
  contents are unchanged; no new native replay or advisory remediation claimed.
- Step 3: Initial Ruff/mypy/pytest passed: 107 source files, 1,050 tests
  (41.44s). Focused release-candidate tests: three passed (5.57s).
- Step 4: Dedicated Task loaded code-simplifier and made no changes; metadata
  has no accidental complexity needing an abstraction or wording change.
- Step 5: Separate parallel General and Security Tasks loaded code-reviewer
  and mapped ECC guidance; Security loaded security-review and common security
  rules. Security selected for public safety/advisory claims. Python omitted:
  assertions only, covered by General; database/framework omitted: no runtime,
  schema, persistence or framework changes. Security returned "No concrete
  task-relevant findings."
- General candidate G-1 verbatim: "The metadata presents v0.2.1 as published
  while the remote has no v0.2.1 tag or GitHub release and T101 remains
  `in_progress`." Fresh candidate-validation Task rejected G-1, no duplicates
  or validated findings. The observation is correct during preparation but
  the explicitly requested T068-style sequence finalizes metadata before
  exact-target CI/tagging so the immutable release contains final metadata.
  Publication is not reported as delivered until remote verification succeeds.
  Changing the tagged metadata afterward would contradict the required method.
- Step 6: No validated findings or deferrals; no fix cycle required. The short
  prepublication interval of release wording on main is an explicit staged
  limitation, not proof that publication already occurred. T101 stays active
  until successful publication/install verification and final checks.
- Step 7: Final exact Ruff/mypy/pytest chain passed (107 source files,
  1,050 tests, 34.33s); mise run check passed including formatting, 1,050 tests
  and all policy/mutation guard suites. Fresh disposition-verification Task
  loaded code-reviewer and verified G-1 rejection, unchanged safety boundaries
  and no newly introduced task-relevant issues. No unresolved findings.
  Taskrail validate and diff whitespace checks passed; pinned spec unchanged.
  Publication gates and exact-tag-target CI remain to be executed before tagging.

## Publication evidence — 2026-10-14

- Published at 2026-10-14T03:46:38Z (05:46:38 Europe/Zagreb):
  [v0.2.1 GitHub release](https://github.com/fmueller/latinitas-cards/releases/tag/v0.2.1).
  API confirms tagName v0.2.1, not draft, not prerelease, assets empty.
  Exactly one annotated tag and one matching release were created; no assets,
  PyPI, deployments, workflow dispatch, tag rewrite or spec activation.
- Immutable target:
  [5c3f35f](https://github.com/fmueller/latinitas-cards/commit/5c3f35fbb8c5cf55e41fd5046ac16eb7288c4ee5).
  Remote tag object `9aa11f81d85062d7c42d3166860778ebc04594d4` and peeled
  commit match local annotated tag and tested target. Tagger is Felix Müller,
  the maintainer. Target contains accepted planning base and T100 candidate;
  source/CSS, dependencies and pinned spec are unchanged from T100.
  Publication evidence is a later main commit; the tag is never edited.
- Exact-target [Build](https://github.com/fmueller/latinitas-cards/actions/runs/38507739355)
  passed before tagging: Python 3.13/3.14 tests, lint/type/format and policy
  guards. [CodeQL](https://github.com/fmueller/latinitas-cards/actions/runs/38507738872)
  Python/actions analyses also passed. API confirms both run head SHAs and
  all six checks successful; no additional external statuses. CI callback was
  independently inspected, not treated as new authorization. Automatic exact-SHA
  main event subscription enabled; no manual dispatch or deployment.
- No dependency-graph run was triggered on this metadata-only target. Lock and
  dependency inputs are unchanged from T100's successful
  [dependency graph run](https://github.com/fmueller/latinitas-cards/actions/runs/38506460009).
  Fresh API inspection still shows four open annotation alerts: #141/#142/#143
  urllib3 and #144 fsspec, three high and one moderate; no remediation/dismissal.
- Fresh remote-tag shallow clone passed README `uv sync --locked`, installed
  distribution version 0.2.1 and `uv run latinitas-cards --help` (exit 0).
  HEAD and annotated tag object match the remote target/object above; checkout
  clean, pinned spec hash unchanged. No annotation extra; urllib3/fsspec absent.
  This is fresh-tag installation evidence, distinct from T100 wheel smoke.
- Release notes preserve CSS v4 MANUAL setup, separately bound old-layout
  resolution, fresh closed capture/evidence, renewed individual approvals,
  blocked withdrawal/manual creation-retirement limits. Native acceptance stays
  bounded synthetic Linux; Mobile/macOS/Windows/sync/live collections unverified.
  Original adversarial 4/5 remains separate; scoped mutation remains report-only.
  No new native replay or universal safety/scheduling/expert-language claim.
- Step 8: After successful publication and fresh-tag installation verification,
  repeated exact Ruff/mypy/pytest passed (107 files, 1,050 tests, 35.65s),
  then mise run check passed (1,050 tests, 35.17s, formatting and all guards).
  Taskrail validate passed. CLI verify/complete now follows these final gates;
  focused evidence/state delivery on main does not alter the immutable tag.
- 2026-10-14T03:49:44Z: verification pass
- 2026-10-14T03:49:44Z: Verified public v0.2.1 release and annotated immutable target; fresh-tag install and exact CI pass. No PyPI/assets/deployment/spec activation. Publication evidence delivered separately without tag edits.
