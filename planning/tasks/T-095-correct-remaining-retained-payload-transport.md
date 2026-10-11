---
id: T-095-correct-remaining-retained-payload-transport
title: Correct remaining retained-payload transport limitation
status: completed
priority: medium
spec_ref: specs/v0.2.1.md#review-identity-and-export
dependencies: []
updated_at: "2026-10-11T21:12:50Z"
---

# T-095-correct-remaining-retained-payload-transport Correct remaining retained-payload transport limitation

## Description

ROUND3 confirmed a remaining public transport-contract contradiction on
8d33fc885926bea5cb5447493d0b8511d4c344e0. Severity: low (documentation),
not a functional retained-update failure. The spec requires a safe approved
retained-slot payload-update path and accurate manual structural boundaries.

Reproduce with `sed -n '352,368p' docs/destination-snapshots-and-file-transport.md`
and `sed -n '444,451p' docs/destination-snapshots-and-file-transport.md`.
Lines 357-360 correctly permit separately approved retained prompt/answer
changes. Lines 447-450 instead list all slot prompt/answer/guard edits as
unsupported. Expected: distinguish supported retained payload changes from
unsupported guard, eligibility, structural and layout changes.

Fresh synthetic public CLI capture/adopt/plan/approve/emit/observe/reconcile
completed 44 principal/parsing and 24 phrase operations, zero unresolved.
This contradicts the broad prohibition, without certifying native import.

Dedup: completed T094 corrected three explicitly named surfaces (parsing
guidance, reference note-type guidance, principal preview), not this transport
document. Those regressions pass. T048/T059 are historical original contracts;
T080/T081/T089 and T090-T093 do not track this remaining sentence. Do not
reopen or edit completed tasks or widen this into general documentation review.

## Acceptance

- Correct the remaining transport limitation to permit separately approved
  retained principal/parsing/phrase prompt/answer payload updates.
- Preserve unsupported guard/eligibility/slot creation, retirement,
  reactivation, schema migration and APKG-application boundaries.
- Preserve exact CSS/layout binding, manual setup and native matching/tag,
  sync/restore limits. Native appearance remains pending, not verified.
- Cross-check the two passages against the exercised retained workflow;
  no new transport capability, lifecycle change or broad documentation rewrite.

## Verification Notes

- ROUND3 pinned v0.2.1 spec SHA256:
  0b2150fc9778a3d7c0b058a5d0961a819ee283f148569323c32eee07ac3776b4.
- Five fresh closed-storage rehearsals preserve 47 complete card rows and
  47 history rows, source notes, Personal Notes and user tags. These are
  offline protocol checks, not native CSV importer/scheduling/sync proof.
- Local task proposal only; parent acceptance required before planning-only
  commit/push. Tester made no product fix or manual lifecycle/state edit.

## Implementation Notes

- Base and fetched origin/main matched cd1e8e963f93d9d760d4f6a04caec61c8af17cb3;
  initial worktree was clean. Spec SHA256 matched the pin above. Before lifecycle
  writes, `mise exec -- taskrail validate` returned `state valid` and
  `mise exec -- taskrail next --json` selected only T095, `off_spec: false`.
  Started through Taskrail at 2026-10-11T21:04:44Z; no hand-edited statuses/state.
- Reproduced the contradiction using the two Description `sed` commands.
  Changed only the final transport limitation: separately approved prompt/answer
  updates on observed retained principal-part, parsing and phrase slots are
  supported by the offline CSV handoff, not native import certification.
  Unchanged exact guards/layout and all structural/schema/APKG/manual/native
  limits remain explicit. No runtime change, capability expansion or broad rewrite.
- TDD applicability: documentation-only correction; no artificial production
  red/green test. Before/after prose inspection demonstrated the contradiction
  and corrected boundary. Existing runtime contract tests ran unchanged.
- Manual checks: public `uv run latinitas-cards managed --help`,
  `managed approve --help`, and `managed emit --help` confirm operation-bound
  explicit review, offline handoff and operator attestations, not native proof.
  Cross-checked `managed_plans.py` retained payload guards and transport flags
  against spec Review Identity And Export and the previously recorded 68-operation
  protocol evidence. Did not rerun those five protocols or claim new native proof.
- Dedicated Task simplifier loaded personal `code-simplifier`; accepted its
  narrow prose tightening. Re-read actual diff and reran focused checks.
- Independent parallel Tasks loaded personal `code-reviewer`: General routed
  to `ecc/agents/code-reviewer.md` (no companions); Python routed to
  `ecc/agents/python-reviewer.md` plus `ecc/skills/python-patterns/REFERENCE.md`.
  Python selected for retained-control contract alignment. Security/database/UI
  lanes omitted: no changes to their implementation or invariants. No budget excess.
  Both exact conclusions: "No concrete task-relevant findings."
- Fresh candidate-validation Task loaded `code-reviewer`, inspected the diff
  and accepted the empty candidate set: "No concrete task-relevant findings."
- First fresh disposition verifier raised this candidate verbatim:

  > FINDING T-095-DOC-1 — domain
  > Severity: low
  > Evidence: `docs/destination-snapshots-and-file-transport.md:448–449` newly says retained prompt/answer payload updates “are supported.” `docs/managed-csv-native-verification.md:131` still says “Slot prompt/answer/guard changes remain unsupported,” while its surrounding limitations scope native evidence to specific API settings and state that other import paths need separate checks.
  > Finding: The new transport-contract wording can be read as claiming native support for retained prompt/answer updates, but the native verification document still explicitly lists those updates as unsupported.
  > Failure/impact: A reader could treat the managed workflow’s offline plan/approval/emit support as evidence that the native import operation is supported, despite the stated limitation on native evidence.
  > Recommended direction: Clarify in this transport document that the supported scope is the managed review/CSV handoff contract and does not itself establish native application support; retain the existing native verification and matching/tag-only conditions.

- A separate fresh candidate-validation Task loaded `code-reviewer` and
  rejected T-095-DOC-1 as a concrete contradiction: the section already says
  "not native import certification" and the native document records narrower
  historical evidence. Disposition: not reproducible as a concrete support claim;
  nevertheless reinforced task-local wording with "offline CSV handoff" and
  "This is not native import certification." Historical native doc untouched.
  No deferred findings or production TDD fix; re-read the corrected boundary.
- Second fresh disposition-verification Task loaded `code-reviewer` and
  confirmed: "T095’s documentation acceptance is met. No concrete task-relevant
  findings." It resolved the wording risk and preserved the rejected original
  contradiction disposition; six focused files passed (156 tests). One narrow
  clarification/recheck cycle, no unresolved findings.
- Implementer focused command, before/after simplification and clarification:
  `uv run pytest tests/unit/managed_plans_test.py tests/unit/managed_application_test.py -q`:
  67 passed on each run. `git diff --check` clean throughout.
- Final exact repository chain after fresh verification:
  `uv run ruff check` — All checks passed!
  `uv run mypy` — Success: no issues found in 107 source files.
  `uv run pytest -v` — 1045 passed in 40.81s.
- Native appearance remains 2026-10-11 maintainer-deferred/pending, not passed.
  ROUND3/5 remains NOT CLEAN; parent owns ROUND4 after acceptance. No second
  task, spec-priority-dependency-gate change, tags, releases or deployment.
- 2026-10-11T21:12:43Z: verification pass
- 2026-10-11T21:12:50Z: Documentation boundary corrected and independently reviewed; verification pass at 2026-10-11T21:12:43Z after exact repository chain. Native appearance remains pending.
