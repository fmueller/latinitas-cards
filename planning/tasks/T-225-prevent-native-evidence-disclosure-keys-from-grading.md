---
id: T-225-prevent-native-evidence-disclosure-keys-from-grading
title: Prevent native evidence disclosure keys from grading cards
status: completed
priority: high
spec_ref: specs/v0.5.0.md#semantic-color-layers
dependencies:
    - T-198-present-acceptance-evidence-and-morphology-without
updated_at: "2026-11-19T16:16:05Z"
---

# T-225-prevent-native-evidence-disclosure-keys-from-grading Prevent native evidence disclosure keys from grading cards

## Description

Close T-198's keyboard-accessible answer-evidence requirement under
specs/v0.5.0.md#semantic-color-layers. On candidate 76977b9 with official Linux
Anki 26.09.4 / Qt 6.11.0, actual reviewer input demonstrated that Space graded
Good and advanced the card while the open disclosure's SUMMARY had focus.
Tab followed by Enter also graded instead of toggling evidence. Native DOM,
card identities and new revlog rows distinguish this from browser-only behavior.
This task owns only that native presentation regression, not T-200 evaluation
or T-201 whole-spec acceptance.

## Acceptance

- Read actual Anki reviewer event/shortcut contracts before choosing the fix.
  Tab reaches and leaves evidence controls; focused Enter and Space toggle only
  the intended disclosure without grading, card advancement or new revlog rows.
- Preserve normal intentional grading outside disclosure controls, nested and
  neighboring interactive controls, question/answer isolation, per-card origin,
  user-owned fields/tags, linguistic acceptance and legacy characterization.
- Retain discriminating failing/passing checks and real native pointer/keyboard
  evidence with focused SUMMARY, details states and same-card/history controls.
  Account for changed rendered approval bindings without fabricating authority.
- Complete scoped simplification, independent review/dispositions, final
  Ruff/mypy/pytest chain and native independent acceptance before delivery.

## Verification Notes

- Original native failure: SUMMARY focused after pointer expansion; Space changed
  reviewer card 1795100626583 to 1795100626582 and appended a Good revlog row.
  The earlier Tab/Enter probe appended a Good row for card 1795100626581.
  Retain original evidence separately from corrected controls; these reviews are
  disposable setup only, never personal collection or T-201 acceptance evidence.

## Implementation Notes

- Kept the disclosure in Qt's input focus context and handled its own unmodified
  Enter/Space locally. Blocked ordinary text editing without global listeners;
  preserved Tab traversal, descendant controls and browser modifier defaults.
- Three correction rounds retained: inputmode=none failed native testing; removing
  that attribute passed; the final Shift guard passed scoped regression/review.
  Dedicated simplification and high-risk independent native, test/binding and
  outcome review found no remaining scoped blocker.
- Final frozen chain: Ruff passed; mypy passed on 236 files; pytest reported
  3776 passed, 7 skipped and one warning. A prior combined replay test failed then
  passed alone; its cause remains unestablished. The frozen full run passed it.
- Actual Linux Anki 26.09.4 / Qt 6.11.0 German and English pointer, Enter/Space,
  repeat, Tab/Shift-Tab and ordinary editing checks preserved card/history during
  disclosure interaction. Intentional pointer-leave grading appended exactly one
  row. Temporary nested/neighbor controls activated without toggling the panel.
- Changed exact rendered HTML invalidates old curated rendered approvals while
  rule authority, identities and fronts remain unchanged. All demonstration
  approvals are synthetic engineering controls, not linguistic authority.
- IME, drag/drop and native assistive-technology semantics remain unverified.
  T-201 whole-spec acceptance and T-195 genuine native manual-resolution/history
  integration remain outstanding; no evaluation or policy enablement occurred.
- 2026-11-19T15:51:42Z: verification pass
- 2026-11-19T15:51:42Z: Task-local native regression fix independently accepted; full T201/evaluation and genuine native invalidation/manual-resolution remain outstanding
- 2026-11-19T16:16:05Z: verification pass
