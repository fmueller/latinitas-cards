---
id: T-092-reject-unsafe-controls-in-initial-phrase-exports
title: Reject unsafe controls in initial phrase exports and checks
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#review-identity-and-export
dependencies: []
updated_at: "2026-10-11T19:29:13Z"
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
