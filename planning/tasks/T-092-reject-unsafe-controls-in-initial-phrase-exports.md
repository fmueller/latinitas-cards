---
id: T-092-reject-unsafe-controls-in-initial-phrase-exports
title: Reject unsafe controls in initial phrase exports and checks
status: completed
priority: medium
spec_ref: specs/v0.2.1.md#review-identity-and-export
dependencies: []
updated_at: "2026-10-11T19:49:33Z"
---

# T-092-reject-unsafe-controls-in-initial-phrase-exports Reject unsafe controls in initial phrase exports and checks

## Description

ROUND2 confirmed a medium-severity fresh-export/check boundary gap on
45a7cbfb7a73db3aab77ab838f94423829dfdfe6, under the active spec's safe deterministic
phrase CSV and review contract. A valid reviewed phrase Reference explanation
containing U+001F is accepted, remains eligible, passes inspection and is exported
as a literal 0x1f byte in PhraseTranslationAnswer. The retained managed planner
rejects the same field with `unsafe control U+001F in managed field
PhraseTranslationAnswer`. No native import or actual native corruption is claimed.

Reproduction: start from the documented CSV/profile/phrase input workflow with
reviewed `cum amico` translation and recognition exercises. List current claims,
copy the answer's complete Reference JSON, replace its explanation with
`unsafe\u001fcontrol`, and submit it using `claims record --selections selected.json
--recipes recipes.json --batch batch.json --json`, with explicit authored evidence.
Then run `claims inspect ... --check --json` and `phrases export inputs.json
selected.json recipes.json NEW.csv --json`. All return 0; parse the CSV and observe
one U+001F in the answer. Expected: reject unsafe generated field content before
successful initial export/check, without silently deleting or altering valid text.

Dedup: T080 protects managed proposal/destination fields and still passes here;
this is the distinct initial phrase export/check path. T090 literal inspection
and T091 styling regressions pass and do not address active controls.

## Acceptance

- Unsafe C0/C1/DEL content cannot obtain a successful fresh phrase export/check;
  use the established managed-field boundary, preserving its allowed tab/CR/LF.
- Exercise both phrase recipes with authored answer controls and a safe asymmetric
  Unicode/quoted/multiline/literal HTML control through the public CLI.
- Rejection preserves existing output bytes and does not change reviews merely
  through read-only inspection. Explain the rejected field/control to the user.
- Existing retained managed-control rejection and literal escaping remain correct.

## Verification Notes

- Independently reproduced twice from clean workspaces on 2026-10-11. Initial
  inspect, record, subsequent inspect and export exits: 0/0/0/0; output has one
  raw U+001F. Evidence and replay are in the ROUND2 synthetic bundle, including
  `confirm_findings.py`, `supervisor-replay/active-control`, and phrase report.
- Source spec SHA-256:
  `0b2150fc9778a3d7c0b058a5d0961a819ee283f148569323c32eee07ac3776b4`.
- Tester filed only; no implementation, task verification or native safety claim.

## Implementation Notes

- Reused the retained managed-field C0/C1/DEL boundary without sanitization.
  Phrase workspace inspection reports field/control diagnostics while remaining
  read-only and allowing review correction; initial export rejects before writes.
- Preserved CSV-reader newlines so allowed embedded CR remains byte-exact during
  artifact comparison rather than producing false drift.
- Public CLI RED: 18 failures (16 unsafe checks incorrectly returned 0, two safe
  CR artifact checks falsely reported drift). GREEN: all 18 passed for translation
  and recognition with NUL, U+0001, VT, FF, U+001F, DEL, U+0080, U+009F and safe
  asymmetric Unicode, quotes, tab/CR/LF and escaped literal HTML.
- Subprocess manual replay confirmed the original U+001F export returned 0 and
  contained one raw 0x1f byte; after the fix both recipes reject check/export with
  actionable answer-field/control diagnostics and no new output. Safe exports
  match original CSV bytes; source, recipe, selection, review and existing output
  bytes remain unchanged. This is offline CLI evidence, not native import proof.
- Dedicated code-simplifier: no edits, 18 focused tests passed. Separate parallel
  General, Security and Python code-reviewer lanes each returned "No concrete
  task-relevant findings." Fresh candidate validation and disposition verification
  agreed; no candidate IDs, fixes or deferred review findings. One review cycle.
- Initial full ruff/mypy/pytest chain passed (1043 tests); fresh phrase, managed,
  inspection, literal and appearance regression set passed (120 tests).
- Native Desktop/AnkiMobile testing remains maintainer-deferred on 2026-10-11;
  auto acceptance remains pending, not passed or covered. ROUND2/5 is NOT CLEAN.
  T093 typed Reference validation and T094 guidance are separate remaining tasks;
  only the parent may start ROUND3 after all three fixes are accepted.
- 2026-10-11T19:49:21Z: verification pass
- 2026-10-11T19:49:33Z: Offline T092 fix completed after final exact ruff/mypy/pytest (1043 pass), independent lanes/candidate validation/disposition verification and byte-preserving CLI replay. Verify run 2026-10-11T19:49:21Z. Native auto acceptance remains maintainer-deferred/pending, not passed or covered; ROUND2/5 NOT CLEAN; T093/T094 remain separate.
