---
id: T-173-publish-v0-4-0-maintainer-release-gate
title: Publish v0.4.0 (maintainer release gate)
status: completed
priority: low
spec_ref: specs/v0.4.0.md#delivery-order
dependencies:
    - T-172-validate-integrated-v0-4-0-workflows-offline-and
    - T-154-capture-v0-4-0-characterization-goldens
    - T-155-share-the-hardened-xml-reader-and-add-an-adapter
    - T-156-add-the-language-table-and-english-terminology
    - T-157-localize-fixed-card-strings-and-english-reviewed
    - T-158-normalize-perseus-epidoc-tei-structure-citations
    - T-159-preserve-tei-notes-apparatus-and-editorial-markup
    - T-160-build-the-accepted-fact-view-and-consume-latinitas
    - T-161-import-aligned-parallel-translations-as-annotation
    - T-162-define-the-corpus-generation-profile-and-corpus
    - T-163-derive-corpus-exercise-identities-and-candidates
    - T-164-render-the-corpus-v1-note-type-and-layout-content
    - T-165-read-known-lemmas-from-a-confirmed-deck-profile
    - T-166-rank-candidates-and-persist-sticky-selections
    - T-167-deselect-and-restore-selected-corpus-exercises
    - T-168-preview-export-and-inspect-corpus-exercises
    - T-169-plan-managed-updates-for-corpus-exercises
    - T-170-document-corpus-first-tei-and-english-workflows
    - T-171-prepare-the-v0-4-0-native-acceptance-kit
updated_at: "2026-11-01T09:59:55Z"
---

# T-173-publish-v0-4-0-maintainer-release-gate Publish v0.4.0 (maintainer release gate)

## Description

Maintainer release gate (Delivery Order step 10). Created blocked with the reason
"maintainer release gate" so the selector never picks it; only the maintainer unblocks
it after every other v0.4.0 task is complete and authorizes publication. Requires
network, a native Anki installation and release secrets.

Fixed Contracts in `specs/v0.4.0.md` are binding and take precedence over this note; file a blocked follow-up task instead of amending them. Run the mandatory `uv run ruff check`, `uv run mypy`, `uv run pytest -v` chain before verify/complete.

## Acceptance

- Native Linux Anki Desktop evidence per `#native-acceptance`, recorded from the kit.
- A real Perseus smoke run: Caesar BG (phi0448.phi001) and the Aeneid (phi0690.phi003)
  normalize and validate; record source digests and passage counts; any failure is filed
  as a follow-up before release.
- Licence/provenance notes for any parallel translation used in docs or fixtures.
- The v0.3.0 PyPI procedure (`docs/releasing.md`): green Build on the candidate, one
  annotated `v0.4.0` tag, approved `pypi` deployment, PyPI hashes equal build artifacts,
  GitHub release, evidence recorded in a follow-up commit.
- A missing prerequisite blocks the task naming the item.

## Verification Notes

- Publication succeeded. See `docs/release-v0.4.0.md` for candidate,
  native/source boundaries, workflow, approval and every-file hash evidence.
- Final evidence validation: exact Ruff, mypy, pytest chain passed with
  3,232 tests passed, seven skipped and one existing serializer warning
  (593.63 seconds). Independent review reported no remaining findings.

## Implementation Notes

- 2026-10-27T23:11:02Z: maintainer release gate
- 2026-11-01T08:08:22Z: 2026-11-01: maintainer authorized the full v0.4.0 release after all other active-spec tasks completed; confirmed current PyPI publisher fmueller/latinitas-cards release.yml environment pypi and approved README/docs corrections. Existing protected tag-only workflow remains mandatory.
- Released candidate `3d51ee680ea68ef9d521970178706cc3eb977d57`; annotated tag
  object `4e05ec66bf23f8f4e012b83767293ccc6b527f82` remains unchanged. All eight
  candidate checks passed. Release run 41290520404 passed all five jobs,
  including exact-version PyPI installation/help and absence of NLP extras.
- The integration could not approve the environment (HTTP 403); the maintainer
  reported personally approving it, then protected publication proceeded.
  Approval-history API returned an empty list, not independent actor evidence.
  Every PyPI wheel/sdist filename and SHA256 matched the reviewed build
  artifact and independently downloaded registry bytes before GitHub release.
- Maintainer accepted existing low-severity Dependabot alert 146 risk before
  tagging. Its package/manifest/exposure remain unverified after API HTTP 403;
  no dependency or alert state changed. No blanket safety claim is made.
- 2026-11-01T09:59:55Z: verification pass
- 2026-11-01T09:59:55Z: Published v0.4.0 on PyPI and GitHub after maintainer protected approval; exact artifact/registry file hashes match, full final validation and independent review passed, evidence recorded without moving tag. Existing low alert146 risk accepted with scope unverified; active spec stays v0.4.0.
