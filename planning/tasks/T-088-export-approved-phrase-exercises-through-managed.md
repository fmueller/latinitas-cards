---
id: T-088-export-approved-phrase-exercises-through-managed
title: Export approved phrase exercises through managed CSV updates
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#review-identity-and-export
dependencies:
    - T-087-persist-phrase-exercise-identity-and-explicit
updated_at: "2026-10-09T20:05:15Z"
---

# T-088-export-approved-phrase-exercises-through-managed Export approved phrase exercises through managed CSV updates

## Description

Connect approved phrase selections to deterministic CSV and existing managed
lifecycle contracts; keep export safety distinct from linguistic review.

## Acceptance

- Export only approved selected exercises using existing LatinitasID, recipe/slot,
  ordering and escaping contracts; unchanged reruns give byte-identical CSV.
- Preserve lexical source notes and user-owned fields/tags. Changed payloads require
  renewed approval; withdrawing an exported phrase uses managed retirement rather than
  assuming CSV omission removes an Anki note.
- Initial creation uses ordinary approved CSV plus explicit manual import and baseline
  adoption; managed creation remains unsupported. Own a narrowly scoped safe extension
  for approved answer/presentation field updates on retained existing slots, including
  principal-part, parsing and phrase payloads. Keep identities, eligibility and slots
  unchanged; preserve conflict handling, observation and reconciliation safeguards.
- Exercise plan/approve/emit/observe/reconcile for supported retained-slot changes.
  Recipe deselection/withdrawal still produces explicit blocked lifecycle plans under
  the CSV transport; document reviewed manual resolution and fresh observation rather
  than claiming automatic retirement/deletion. No tag-only retirement or approval bypass.
- Test that linguistic approvals alone do not authorize managed writes; stale payload
  approval, structural creation, changed eligibility/slots and retirement stay blocked.
  No direct Anki integration or new destination-state protocol.
- Own destination setup/binding integration for the versioned phrase layout and T-067
  CSS changes. Provide a reviewed manual setup/reconciliation walkthrough; mismatched
  old template/style digests fail closed, never silently rebind scheduled templates.
- Shared read-only check passes fresh phrase CSV and supplied package artifacts,
  failing tampered, stale, unreviewed or identity/slot-mismatched content with nonzero
  exit. Inspection writes no files or destination state; package generation is not required.
- Document runnable CLI selection, review, preview, export and managed-update examples,
  including manual Anki import safety and limits.
- Pass the mandatory ruff/mypy/pytest chain.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes

- Current managed_plans.py excludes slot fields from compatible content writes and
  marks creation/non-retain effects unsupported. This task must prove the retained-slot
  extension with before/after and negative controls, not assume the existing transport
  already supports all generated answer updates.
