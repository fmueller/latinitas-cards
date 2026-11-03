---
id: T-205-publish-v0-5-0-through-the-maintainer-release-gate
title: Publish v0.5.0 through the maintainer release gate
status: blocked
priority: low
spec_ref: specs/v0.5.0.md#whole-spec-acceptance-and-final-checks
dependencies:
    - T-184-design-v0-5-0-acceptance-and-pipeline-contracts
    - T-185-design-independently-judged-acceptance-evaluation
    - T-186-approve-v0-5-0-contracts-and-evaluation-policy
    - T-187-implement-versioned-automatic-decisions-and-reuse
    - T-188-check-mapped-deck-evidence-with-latin-rules-and
    - T-189-add-explicit-providers-and-blind-linguistic
    - T-190-gate-complete-card-assertions-on-accepted-evidence
    - T-191-extend-translation-first-forms-and-controlled
    - T-192-select-full-eligible-output-with-resumable
    - T-193-orchestrate-inspectable-stages-with-bounded-resume
    - T-194-explain-pipeline-outcomes-in-human-and-machine
    - T-195-track-export-receipts-and-installed-card
    - T-196-validate-provenance-bound-curated-rule-and
    - T-197-export-curated-application-cards-only-after
    - T-198-present-acceptance-evidence-and-morphology-without
    - T-199-document-v0-5-0-workflows-and-reconcile-cli
    - T-200-evaluate-held-out-automatic-acceptance-against
    - T-201-validate-real-pipeline-and-linux-anki-desktop
    - T-202-audit-v0-5-0-contract-and-implementation-drift
    - T-203-audit-v0-5-0-semantic-and-acceptance-gaps
    - T-204-check-v0-5-0-release-readiness-without-publication
updated_at: "2026-11-02T23:10:47Z"
---

# T-205-publish-v0-5-0-through-the-maintainer-release-gate Publish v0.5.0 through the maintainer release gate

## Description

Maintainer-only publication gate. Only the maintainer may unblock after every release
task and explicit exact-candidate readiness acceptance. Separate publication permission
is mandatory. No autonomous loop may unblock or execute this gate.

## Acceptance

- Reconcile all v0.5.0 tasks, including added release-impacting prerequisites, accepted
  design/policy and exact readiness/evaluation/native/audit revisions. Preserve T-179's
  historical drafting lifecycle; do not treat it as absent implementation evidence or
  silently complete it.
- Obtain explicit publication authority, candidate CI success and release credentials/
  protected-environment approval. Missing prerequisites or changed hashes block release.
- Follow docs/releasing.md for tag/build/PyPI/GitHub publication, verify published hashes
  against candidate artifacts and record actual outcomes. Do not overwrite releases or
  infer deployment authority.
- Record publication evidence/limitations separately from planned acceptance. A narrowed
  release cannot claim that original rejected scope passed.

## Verification Notes

Maintainer inspects exact candidate/runbook. Run mandatory validation before verify/
complete. Planning, activation, design approval and readiness alone do not authorize
external writes.

## Implementation Notes

- 2026-11-02T23:10:47Z: maintainer release gate: readiness acceptance and separate publication permission required
