---
id: T-101-publish-v0-2-1-annotated-tag-and-github-release
title: Publish v0.2.1 annotated tag and GitHub release
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#goals
dependencies: []
updated_at: "2026-10-14T03:31:10Z"
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
