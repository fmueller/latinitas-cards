---
id: T-187-implement-versioned-automatic-decisions-and-reuse
title: Implement versioned automatic decisions and reuse bindings
status: completed
priority: high
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-186-approve-v0-5-0-contracts-and-evaluation-policy
updated_at: "2026-11-17T06:11:02Z"
---

# T-187-implement-versioned-automatic-decisions-and-reuse Implement versioned automatic decisions and reuse bindings

## Description

P50-2 decision storage and strict-reader integration. Own focused claim/sidecar
adapters and stage-result bindings, not a parallel review system. Apply the approved
P50-1 contract and the spec's execution-isolation/final-validation requirements.

## Acceptance

- Human and automatic acceptance remain distinct after serialization, reload and
  regeneration. Preserve historical human decisions and reject unsupported versions.
- Bind reusable successes to source content, extraction, task, language, configuration
  and producer/analyzer/prompt/checker/policy versions. Correctness changes invalidate
  affected results without discarding valid unrelated work.
- Persist stage outcomes, alternatives, objections and evidence. Malformed, stale,
  failed or unknown records cannot authorize export or destination writes.
- Implement the application-owned policy registry and strict support-registry
  bindings from the accepted contract. Ship no enabled automatic policies before
  T-200's separate empirical/maintainer gates. Disabled-policy all-checks-pass
  decisions remain withheld through reload/reuse; forged accepted records and
  requested profile policy tuples cannot grant authority. Recheck withdrawal.
- Keep would-accept evaluation data in isolated non-learning artifacts. Positive
  automatic tests use isolated synthetic registry fixtures, not a public bypass
  or a production-enabled fixture. Human approvals keep their distinct bindings.
- Strict deck/corpus readers consume only explicitly upgraded/adapted records.
  Preserve old valid human sidecars and unchanged deck/corpus output compatibility.

## Verification Notes

Use public readers and preview/export interfaces for old/new round trips, changed
source/checker/policy/language, malformed/unsupported versions and unchanged reuse.
Run the mandatory ruff, mypy, pytest chain before verify/complete.

## Implementation Notes

Implemented separate strict acceptance/stage records, canonical noncircular
commitments, immutable artifact storage and dependency-aware reuse. Public readers
revalidate current bindings and authority for preview, export and manual handoff.
Human approvals retain their own immutable checks and do not inherit automatic
policy authority. Explicit deck/corpus adapters preserve historical human claims,
authored corrections, decisions and source-sidecar bytes; existing readers and
output paths remain unchanged.

The application registry ships only v050-disabled revision 1 with no enabled
scope. All-checks-pass automatic records still withhold on reload/export/handoff.
Synthetic enabled policies are isolated test fixtures, not a public bypass.
Model-check execution, linguistic rules and empirical enablement remain with
later tasks; no empirical/reference/study or publication work was performed.

TDD established missing-module failures and green public-interface coverage.
Independent high-risk review reproduced five defects before fixes: corpus anchor
conflicts, authored phrase combination approval, versioned projection identity,
legacy artifact containment and malformed evidence. All five are independently
verified fixed. Strengthened forgery/containment tests also failed under a
temporary verification bypass and passed after restoration. Dedicated
simplification and scoped fix review completed; no required finding remains.
One adversarial round and one of two permitted fix rounds were used; the reserved
materially-new-evidence follow-up remains unused.

Focused tests passed (74). Independent adapter/corpus checks passed (43), with
additional public probes confirming unchanged history, regeneration, human
policy-only independence, malformed-record withholding and static symlink
containment. Offline public smoke confirmed disabled automatic withholding,
human export and language-drift rejection without provider/destination writes.
Reload relies on caller-supplied current source/task bindings; static containment
is not a race-resistant filesystem guarantee, and handoff does not repair
installed content. Task-local completion does not imply independent integrated
or release acceptance.
- 2026-11-17T05:37:48Z: verification pass
- 2026-11-17T05:37:57Z: Implementation and independent scoped review complete; verification passed at 2026-11-17T05:37:48Z. Final Ruff, mypy (209 files), pytest (3384 passed, 7 skipped), format and planning checks passed. Task-local completion only; no successor started, automatic policy enablement, empirical approval or publication.

### Independent acceptance correction

Integrated acceptance on f115e01 identified one remaining strict-reader defect:
hash-consistent artifact bytes with duplicate JSON keys lost the contradictory
value before schema validation. Shared decoding now applies the existing unique
key mechanism to artifact bytes as well as records, before canonicalization.
Nonfinite constants and float overflow also reject at that boundary; valid
generic JSON arrays, scalars and finite floats retain their prior semantics.

The public regression preserved decision/check/human commitments while changing
artifact bytes and their exact hash. It failed before the fix and now withholds
preview/export with manual handoff; restoring the original bytes restores export.
Standalone artifact cases independently reject nonfinite values and nested
duplicates. Red checks: 11 failed, 3 valid controls passed; affected green checks:
118 passed. Public offline human/disabled-policy/freshness smoke remains green.

Dedicated simplification found no further change. Scoped independent security
and test verification confirmed the fix, including escaped-equivalent keys,
nested duplicates, both overflow signs, generic finite values and unchanged
record strictness. The coordinator closed the finding with no remaining scoped
blocker. This consumed the reserved materially-new-evidence follow-up and fix
round 2 of 2, without resetting the original high-risk review budget. T187 remains
the same completed task; no successor, judge run or policy enablement was started.
- 2026-11-17T06:11:02Z: verification pass

Final correction gate on unchanged executable inputs passed: Ruff, mypy (209
files), pytest (3398 passed, 7 intentional skips and one existing serializer
warning), formatting (535 files), diff check and Taskrail validation. The
supported verify command reconciled evidence on the completed task at the above
timestamp without reopening it; independent integrated acceptance remains
pending the overseer's exact-candidate recheck.
