---
id: T-062-verify-morphology-presentation-on-ankimobile-and
title: Verify morphology presentation on AnkiMobile and Desktop
status: completed
priority: high
spec_ref: specs/v0.2.0.md#morphology-themes
dependencies:
    - T-053-render-safe-principal-part-comparisons-with
updated_at: "2026-10-06T19:58:44Z"
---

# T-062-verify-morphology-presentation-on-ankimobile-and Verify morphology presentation on AnkiMobile and Desktop

## Description

Perform native-client acceptance of new answer-side comparisons and themes. Browser previews alone cannot close this gate.

## Acceptance

- Record AnkiMobile client/version and iPhone/iPad devices as the primary targets, and Anki Desktop client/version as secondary. Required unavailable device checks remain explicit open gates.
- Verify answer reveal, touch expansion, complete four-role comparison with accepted further explanation, explicit absent/withheld roles, and readable muted/monochrome light/dark presentation. Core answer and compact comparison remain visible regardless of expansion.
- Verify completion and recognition prompts remain Latin-first and theme changes preserve semantic claims and note/card/template identities. Deliver the reference note types (fields, templates and CSS from the authoritative contract) in a carrier package, import the exporter CSVs through the native dialog, and check the same effective theme in preview/imported notes. Maintainer decision 2026-10-06: hand-typed GUI note-type setup is waived for this gate.
- Record native details/summary behavior and select the tested disclosure or static fully readable fallback; broken controls cannot hide required information. Static fallback still requires native reveal/readability evidence. Do not introduce untested JavaScript.
- Run existing-recipe checks when the direct prerequisite is complete; form-parsing-exercises owns affected retests after its later changes. Bind evidence to tested content/setup and repeat affected checks for release-candidate changes.
- Capture and inspect representative native screenshots and observed interactions. Record limitations before claiming client compatibility or release readiness.

## Verification

Provide native device/client evidence and the fallback decision. Run the mandatory validation chain for any code fixes and repeat affected client checks.

### Partial native evidence and open gate — 2026-10-05

- Fetched origin/main and confirmed containment of accepted T-061 baseline
  `3086b7c`; HEAD matched it, worktree was clean, Taskrail validate passed and
  next --json selected exactly this task under the pinned active spec. T-053
  was completed. Started only this task through Taskrail.
- See `docs/morphology-native-verification.md` for source/setup hashes, actual
  commands, fixture boundaries, inspected native captures and all remaining
  matrix cells. Actual Anki Desktop 26.09.3 ran on Linux x86_64 orb Desktop
  with a disposable empty profile; Qt library availability was diagnosed and
  resolved. No user/live collection, sync or scheduled setup was touched.
- Native reviewer smoke exited 0: only perfect-completion ordinal 2 in
  muted/light/static, programmatic answer reveal, legible accepted explanation,
  four-role comparison and explicit absent/withheld rows. Both front/answer PNGs
  were inspected. API seeding is not manual setup/import parity; recognition
  card creation is not presentation proof. No JS or product code changed.
- Linux has no available iPhone/iPad AnkiMobile client/device/version, iOS tools
  or USB device bus. Required Mobile native reveal/touch/readability and the
  remaining Desktop/theme/disclosure/manual setup/import/identity matrix remain
  open. Static stays the unverified-client default, not a Mobile certification.
  T-060 retains ownership of affected later retests; this task is not complete.
- Workflow steps 1–3: evidence-only investigation; no production behavior change
  and no RED/GREEN claim. The scratch fixture's initial eligible-card count
  assumption failed before GUI use (supine also generates slots); corrected to
  four, preserving the assertion of absent/absent/present/withheld statuses.
  Focused morphology/relationship checks: 38 passed. Full exact chain: ruff
  passed, mypy passed (84 source files), pytest -v passed (825 tests).
- Step 4: dedicated code-simplifier loaded its skill, inspected the doc, script,
  report/captures and all hashes; no changes. Step 5: independent General lane
  loaded code-reviewer and General guidance; returned verbatim "No concrete
  task-relevant findings." Specialist lanes omitted because tracked changes
  are evidence Markdown/lifecycle only, with no code/API/schema/trust-boundary
  changes. Fresh candidate-validation loaded code-reviewer; no supplied
  candidate IDs, but identified the status wording issue below.
- Validated finding, verbatim: "`docs/morphology-native-verification.md:3` says
  “T-062 remains blocked,” but the Taskrail task and `planning/STATE.md` both
  record it as `in_progress`, with `blockers: []`
  (`planning/tasks/T-062-verify-morphology-presentation-on-ankimobile-and.md:4`;
  `planning/STATE.md:8–10`). The acceptance gate is clearly unfulfilled, but
  the sentence can be read as claiming Taskrail has already recorded a blocked
  status. Clarify that the *acceptance gate* is blocked and the Taskrail task
  remains active pending a formal block."
- Step 6 disposition: fixed the evidence doc to say "Native acceptance remains
  blocked", distinguishing acceptance from Taskrail lifecycle. Formal Taskrail
  fail/block follows the reviewed evidence gates; no fabricated completion or
  deferred findings. This prose correction needs no product test regression.
- Step 7: reran the full exact chain after the wording disposition: ruff passed,
  mypy passed (84 files), pytest -v passed (825). Taskrail validate and diff
  whitespace check passed. Fresh disposition-verification loaded code-reviewer,
  returned "Prior finding — RESOLVED" and "No new task-relevant findings."
  That reviewer inspected doc/task/spec/script but could not independently
  inspect the report/captures; the earlier General reviewer and simplifier did.
  One review cycle; no product fixes, unresolved findings or false native tests.
- Step 8: after review gates, Taskrail verification at 2026-10-05T05:57:54Z
  recorded fail and Taskrail block recorded the unavailable native-device
  reason. Taskrail validate passed; 0 active, 1 blocked, 2 todo. Never ran
  complete, started a second task or created another thread. Parent loop must
  stop on this selected blocker even though another task is eligible.

### Native acceptance — 2026-10-06

- Unblocked and started through Taskrail with the maintainer available for native
  testing. Maintainer decision: the hand-typed GUI note-type setup is waived and a
  carrier `.apkg` built from the authoritative contract delivers the note types;
  acceptance wording updated accordingly.
- `scripts/build-native-acceptance-kit.py` (new, anki 26.9.3) built a carrier package
  plus exporter CSVs: six theme/appearance/comparison variants, a retheme update, and two
  form-parsing decks. Claims are stipulated fixture reviews. A disposable-backend dry run
  verified import mapping, counts and identity preservation before handoff.
- The maintainer imported into a separate `Latinitas Test` deck tree in the backed-up main
  profile on Anki Desktop 26.09.2 (bb0dd6d1) and synced to AnkiMobile 25.09 on an iPhone 16
  Pro Max (iOS 26) and an iPad Air M4 (iPadOS 26). Every Desktop check D1–D7 and Mobile
  check M1–M7 passed: Latin-first fronts, real reveal, all four roles with explicit
  absent/withheld rows, all theme/appearance combinations, native disclosure
  click/touch, client dark mode, the parsing decks, and retheme identity plus Personal
  Notes preservation.
- Fallback decision: native details/summary is a verified option on the tested clients.
  Static remains the profile default for untested clients. No JavaScript was added.
- Limitations: no screenshots (maintainer decision); observations are maintainer-
  reported; synthetic stipulated content; only the listed clients. Full record and hashes
  are in `docs/morphology-native-verification.md`.
- Follow-ups from the session were specified in v0.2.1 as T-064 (CLI claim review and
  inspection), T-065 (terminology language), T-066 (form translations) and T-067
  (automatic appearance).
- Chain: `uv run ruff check` passed, `uv run mypy` passed (87 files), `uv run pytest -v`
  passed (838).

## Implementation Notes

- 2026-10-05T05:57:54Z: verification fail
- 2026-10-05T05:57:54Z: Required native AnkiMobile iPhone/iPad devices/client unavailable in Linux orb. Partial Desktop 26.09.3 completion muted/light/static proof only; finish native Mobile and remaining Desktop/disclosure/manual setup/import parity gates before completion or release readiness.
- 2026-10-06T19:58:44Z: verification pass
