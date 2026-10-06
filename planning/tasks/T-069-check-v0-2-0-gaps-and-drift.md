---
id: T-069-check-v0-2-0-gaps-and-drift
title: Check v0.2.0 gaps, inconsistencies, and spec drift
status: completed
priority: high
spec_ref: specs/v0.2.0.md#goals
dependencies: []
updated_at: "2026-10-06T20:17:27Z"
---

# T-069-check-v0-2-0-gaps-and-drift Check v0.2.0 gaps, inconsistencies, and spec drift

## Description

Before publishing v0.2.0 (T-068), review the shipped implementation against
specs/v0.2.0.md for gaps, internal inconsistencies, and drift. Cover every
Potential Features area (managed update plans, note/card lifecycle, destination
tag ownership, safe update application, pre-release migration boundary,
evidence-led extraction, calibrated form parsing, morphology themes) plus the
Caution, Delivery Order And Open Evidence, and Explicitly Excluded sections.

## Acceptance

- Run `taskrail coverage` and the taskrail-gap review; record structural and
  semantic gap findings per spec area.
- Compare each spec area's promises with code, CLI help, tests, and completed
  task notes; flag unimplemented, partially implemented, or over-claimed items.
- Check drift: behavior or CLI surface that contradicts or exceeds the spec,
  and spec text that no longer matches what shipped.
- Check consistency across README, CHANGELOG `[Unreleased]`, docs/, specs/README.md,
  AGENTS.md, and command help (names, options, defaults, limitations).
- Confirm explicitly excluded items are not shipped or claimed.
- Fix small doc/spec inconsistencies in place; create follow-up tasks via
  `taskrail task new` for anything larger, marking which ones block the release.
- Record a summary of findings and their disposition in Verification Notes.

## Verification Notes

- Review run 2026-10-06: `taskrail coverage` reported 8/8 areas covered and implemented,
  0 uncovered, 5 orphans (all intentionally on v0.2.1/v0.3.0). Semantic review used four
  area reviewers (managed updates/tags/application/migration; lifecycle; extraction and
  form parsing; themes plus doc/help consistency) and four lens reviewers (data-safety
  attacker, skeptical evidence auditor, new-user walkthrough, README/CHANGELOG editorial
  and link check), with an adversarial re-review of the one code fix. Findings that
  drove code or task changes were reproduced independently before acting.
- Managed update plans: plan/approve/emit/observe/reconcile implemented, fail closed; all
  nine named acceptance fixtures present. Blocking defect reproduced: partial observation
  folded unapproved divergent fields into the baseline -> fixed in T-070. Snapshot capture
  and first adoption are library/JSON only -> disclosed limitation, T-075.
- Destination tag ownership: implemented and tested; removal decision is approval of the
  full-tag-set operation. Unselected tag conflicts recorded as suppressed without a
  decision (reproduced) -> T-071, blocking.
- Safe update application: native evidence covers Anki's backend import API only, not the
  Desktop dialog -> CHANGELOG limitation, T-076. Managed path cannot change slot
  prompt/answer fields -> CHANGELOG limitation. Attestation/control-char/wording -> T-080.
- Pre-release migration boundary: rejection and library fresh-start planner implemented;
  CHANGELOG over-claim of a fresh-start command removed; legacy-transition doc updated.
- Note and card lifecycle: retire/reactivate/add are plan-only and refused at apply; spec
  and docs now say so. Test and classification gaps -> T-078.
- Evidence-led extraction: layout matrix and fixtures published; stale "future work" text
  in docs/source-extraction-fixtures.md fixed. Unreviewed fourth role mislabelled as source
  evidence and not counted as a warning -> T-072, blocking.
- Calibrated form parsing: per-claim review, auto-acceptance disabled. Principal-part claim
  review is Python-API-only -> CHANGELOG limitation. Fixture/count gaps -> T-079; doc
  usability -> T-081.
- Morphology themes: schema versioning, themes, escaping and static default verified.
  Native acceptance used synthetic content without screenshots, not a representative deck
  -> T-073, blocking (maintainer decision). Setup reset of hand-edited morphology -> T-077.
- Caution / Explicitly Excluded: no excluded item shipped or claimed.
- Consistency fixes in place: spec Summary, lifecycle plan-only note, Delivery Order
  tense and native-verification prose; README (workflow heading, quick-start
  `--approve-fresh-import`, managed and theme sections, form-parsing native status,
  docs index, link fix); CHANGELOG [Unreleased] rewritten in plain language with a
  Limitations section; AGENTS.md command set/architecture/active spec; reference-note-type,
  destination-snapshots, legacy-transition and source-extraction docs; T-068 acceptance
  (no version in `--help`). All links in edited files checked. CLI help drift -> T-074.
- Release-blocking follow-ups added as T-068 dependencies: T-070, T-071, T-072, T-073.
  Non-blocking: T-074 through T-081. Mandatory ruff/mypy/pytest chain passed (840 tests).

## Implementation Notes

- Review only; release metadata and publication belong to T-068.
- 2026-10-06T20:17:27Z: verification pass
