---
id: T-038-publish-releases-on-pypi
title: Publish v0.3.0 tag, GitHub release and PyPI distributions (maintainer gate)
status: blocked
priority: medium
spec_ref: specs/v0.3.0.md#pypi-publishing
dependencies:
    - T-102-capture-characterization-goldens-for-legacy
    - T-103-define-the-cli-result-envelope-exit-codes
    - T-104-define-versioned-corpus-records-with-a-canonical
    - T-105-validate-normalized-corpora-with-corpus-validate
    - T-106-segment-sentences-and-tokenize-passages-with-the
    - T-107-normalize-plain-text-sources-with-manifest
    - T-108-extract-usfx-parsing-from-cli-py-into-an-adapter
    - T-109-normalize-usfx-sources-into-the-corpus-model-with
    - T-110-preserve-supported-usfx-markup-and-notes-and
    - T-111-verify-full-clementine-vulgate-normalization
    - T-112-route-legacy-vulgate-commands-through-the-adapter
    - T-113-define-annotation-sidecar-records-and-the
    - T-114-report-stale-orphaned-and-unmatched-annotations-in
    - T-115-record-corpus-claim-decisions-through-claim
    - T-116-define-the-versioned-automation-policy-file-and
    - T-117-plan-annotation-patches-against-policy-and-approve
    - T-118-apply-planned-annotation-operations-atomically
    - T-119-enforce-destructive-conflicting-and-actor-rules
    - T-120-report-effective-configuration-and-guarantee-no
    - T-121-complete-package-metadata-notices-and-distribution
    - T-122-add-the-tag-only-trusted-publishing-release
    - T-123-document-pypi-installation-extras-and-the
    - T-124-validate-integrated-v0-3-0-workflows-offline-and
    - T-125-join-corpus-claim-decisions-into-corpus-views
    - T-126-prepare-v0-3-0-orchestrated-loop
    - T-127-repair-registered-schema-export-for-corpus-record
    - T-129-reject-segmentation-outputs-aliasing-corpus-and
    - T-130-report-duplicate-corpus-claim-decisions-without
    - T-131-reject-segmentation-outputs-aliasing-the-default
    - T-132-ignore-stale-bindings-before-live-decision-duplicates
    - T-133-reject-annotation-apply-aliases-to-reserved-writer
    - T-134-reject-apply-aliases-to-resolved-default-manifest
    - T-135-correct-obsolete-legacy-usfx-hardening
    - T-136-reject-segment-aliases-to-resolved-default
    - T-137-verify-styled-noninteractive-setup-diagnostics
    - T-138-reconcile-dependabot-maintenance-batch
    - T-139-reject-multiline-publishing-capabilities-in
    - T-140-reject-plan-lock-hardlinks-to-nested-default
    - T-141-reject-multiline-oidc-write-outside-the-release
    - T-142-reject-literal-block-oidc-write-in-workflow-guard
    - T-143-reject-equivalent-pypi-environment-forms-outside
    - T-144-reject-multiline-environment-object-names-in-non
    - T-145-patch-optional-fsspec-dependency-for-advisory-144
    - T-146-keep-characterization-subprocesses-runnable-during
    - T-147-inspect-only-active-code-in-mutation-cli-contract
updated_at: "2026-10-14T05:43:05Z"
---

# T-038-publish-releases-on-pypi Publish v0.3.0 tag, GitHub release and PyPI distributions (maintainer gate)

## Description

Maintainer release gate for v0.3.0, implementing the "Maintainer release gate"
paragraph of `specs/v0.3.0.md#pypi-publishing` and Delivery Order step 8. It is the
only v0.3.0 task that needs the maintainer, network access to GitHub and PyPI, or
repository settings, and it runs only after every other v0.3.0 task is completed and
the accepted 0.3.0 candidate is on `origin/main`. The task is kept blocked with reason
"maintainer release gate" so autonomous loops never select it; only the maintainer
unblocks it. It follows the T-101 annotated-tag and GitHub-release precedent and is the
first release published through `release.yml`; it reopens no v0.2.x native gate and
cites no Anki or Mobile evidence.

## Acceptance

- Maintainer prerequisites are confirmed and recorded with dates: the PyPI project name
  `latinitas-cards` is free or owned; a pending (or active) Trusted Publisher exists for
  `fmueller/latinitas-cards`, workflow `release.yml`, environment `pypi`; the GitHub
  environment `pypi` exists with a non-empty required-reviewer list and tag-only
  deployment (checked through the GitHub API).
- The maintainer confirms the `data/lat-clementine.descriptor.json` values (edition
  "Biblia Sacra Vulgata Clementina", publisher "eBible.org", url
  "https://ebible.org/find/details.php?id=latVUC", license "Public Domain") and whether
  `data/lat-clementine.usfx.xml` derives from the eBible.org latVUC file; the answer is
  recorded in `docs/license-compatibility-audit.md` before tagging.
- The accepted candidate commit passed Build (lint/type/policy guards, Python 3.13 and
  3.14 tests and the `package` job) and CodeQL where enabled in repository settings.
  Actual remote state shows no `v0.3.0` tag, no `v0.3.0` GitHub release and no PyPI
  version 0.3.0. Only necessary release metadata (for example the changelog date) may
  change before tagging, with the full checks and fresh exact-target CI.
- Exactly one annotated `v0.3.0` tag is created under the maintainer identity on the
  accepted commit and pushed. A remote tag is never rewritten, deleted or re-pushed, and
  an uncertain publication step is inspected, never blindly retried.
- The tag's `release.yml` run passes `build` (tag/version and tag-on-`main` checks,
  `check-dist.sh`) and `verify` (`check-install.sh` on 3.13 and 3.14 from the uploaded
  artifact). The maintainer approves the `pypi` environment deployment (the approval
  mechanism); `publish` uploads those artifacts through Trusted Publishing, and
  `verify-pypi` installs `latinitas-cards==0.3.0`, runs `--help` and finds no extras.
- The SHA-256 of every 0.3.0 file in the PyPI JSON API equals the `build` artifact's.
  A matching non-draft, non-prerelease GitHub release for `v0.3.0` has concise
  user-focused notes and practical limits: experimental `annotate` extras and the pip
  CPU-index caveat, format version 1 before 1.0, Linux and macOS only, the bundled
  Vulgate kept out of the distributions, no native Anki retest because Anki-facing
  output is unchanged.
- Tag object, peeled commit, workflow run URLs, deployment approval, PyPI file hashes
  and install evidence are recorded in a follow-up commit on `main` without moving the
  tag. The Taskrail lifecycle (`verify`, `complete`) runs only after publication is
  verified and the ruff/mypy/pytest chain and `mise run check` pass.
- A missing prerequisite keeps the task blocked with `taskrail block` naming the exact
  missing item. TestPyPI, an API token or a manual upload is never a substitute.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.
- Record the candidate commit and its Build/CodeQL run URLs, the prerequisite
  confirmation dates, the tag object and peeled commit, the `release.yml` run URL and
  job results, the deployment approver, and each dist file name with its build and PyPI
  SHA-256.

## Implementation Notes

- Dependencies are every other v0.3.0 task, including follow-ups filed during the loop;
  add any such follow-up as a dependency before unblocking. Follow-ups filed by T-124 do
  not depend on T-124; the maintainer adds each v0.3.0 task created after 2026-10-14 to
  this task's `dependencies` before unblocking. id and `spec_ref` unchanged.
- T-126 is preparation, not release authorization; it must be completed along with all
  implementation and integrated-validation work. Keep this task blocked until the
  maintainer attends and explicitly unblocks it; no autonomous worker may release it.
- v0.3.0 follow-up rule. (a) A defect in completed work that the current task builds on is
  fixed inside the current task, with a regression test, not filed as a follow-up: the
  selector would run the follow-up only after every remaining higher-priority or lower-id
  task. (b) Follow-ups are filed only for work that no remaining v0.3.0 task needs. They
  use `--priority medium` or `high`, never `low`, so they run before T-124. (c) A follow-up
  that needs a maintainer decision (for example a Fixed Contracts disagreement) is created
  and then `taskrail block`ed with the reason "maintainer decision: ...". T-124 will then
  block at its start check. That is the intended stop.
- Unblock only in an attended session with no autonomous loop running, and run `taskrail
  start T-038-publish-releases-on-pypi` immediately after `taskrail unblock` so no loop can
  select it.
- Read `docs/releasing.md` and `release.yml` before starting; the runbook and workflow
  names are tested to match. Commit under the maintainer identity with hooks installed.
- Compare hashes with `https://pypi.org/pypi/latinitas-cards/0.3.0/json` (`digests.sha256`)
  against the `build` artifact or its job summary.
- If a job fails before upload, inspect the run and fix forward through a new reviewed
  commit and a new version decision by the maintainer; a published version cannot be
  re-uploaded and the tag is never moved.
- After publication, update `specs/README.md` ("v0.3.0 is released") and the matching
  `release_candidate_test.py` assertion in the evidence commit.
- Do not activate another spec or start v0.4.0 work in this task.
- 2026-10-14T05:43:05Z: maintainer release gate: needs PyPI Trusted Publisher, GitHub pypi environment approval and tag publication by the maintainer; unblock only when all other v0.3.0 tasks are completed
- 2026-10-20: maintainer Felix Müller approved the license audit and its declared
  corpus provenance for release preparation; recorded in
  `docs/license-compatibility-audit.md`. This is maintainer acceptance, not an
  independent upstream derivation check or publication authorization. T-038
  remains blocked pending the other release prerequisites and explicit release
  authorization; no tag, publication or environment change was performed.
- 2026-10-20: the maintainer reported completing the PyPI Trusted Publisher and
  GitHub environment setup described in `docs/releasing.md`. PyPI account
  configuration is maintainer-reported, not independently authenticated here.
  Read-only GitHub API checks confirmed environment `pypi`, required reviewer
  `fmueller`, self-review permitted, administrator bypass disabled, and exactly
  one deployment policy: tag `v*`, with no branch policy. These prerequisite
  confirmations do not authorize tagging, deployment approval or publication.
- 2026-10-20: maintainer Felix Müller explicitly confirmed that the bundled
  `data/lat-clementine.usfx.xml` derives from the eBible.org latVUC file.
  Recorded in `docs/license-compatibility-audit.md` as maintainer confirmation,
  not an independent upstream byte comparison or blanket license assurance.
  T-038 remains blocked; this partial prerequisite confirmation does not
  authorize tagging, deployment approval or publication.
- Documentation review G-1, validated and fixed: "docs/normalized-corpus.md
  still presents maintainer confirmation of the bundled corpus provenance as
  pending, contradicting the current provenance audit." Its current link text
  now points to dated confirmation and verification limits. A scratch check
  failed before the correction and passed afterward; the existing provenance
  and packaging metadata tests passed (4 tests). Historical audit unchanged.
- 2026-10-20: resumed attended release preparation at the maintainer's request.
  Added the completed active-spec follow-ups through T-147 as dependencies,
  as required above; T-128 remains excluded future-spec work. Prepared the
  changelog for 0.3.0 with the current date and all pending user-facing fixes.
  The implementation base is fca68427e41196e7650dfdca8f1a3b78c1455ba1 with
  successful Build 39491087705 and CodeQL 39491088352. The metadata commit
  requires fresh exact-head CI before tagging. Read-only checks found no
  v0.3.0 tag, GitHub release or PyPI version (PyPI returned 404 after a
  transient 503); environment pypi still requires fmueller review and permits
  only v* tags. No lifecycle transition or publication was performed;
  explicit authorization to tag and publish remains required.
- Preparation validation: the updated metadata test failed on the old
  2026-10-16 changelog heading, then passed after the date update and folding
  pending fixes into 0.3.0. Ruff, mypy and full pytest passed in sequence
  (2637 passed, 7 skipped, one existing Pydantic warning, 454.33 seconds).
  The dedicated code-simplifier made no changes. Independent General and
  Security reviews, fresh candidate validation and fresh final disposition
  verification each reported "No concrete task-relevant findings." No
  findings or deferrals remained. Follow-up Ruff, mypy, the targeted metadata
  test, diff checking and Taskrail validation passed. T-038 is not verified
  or completed by these preparation checks; publication evidence is absent.
