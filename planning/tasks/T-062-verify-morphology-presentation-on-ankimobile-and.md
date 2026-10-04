---
id: T-062-verify-morphology-presentation-on-ankimobile-and
title: Verify morphology presentation on AnkiMobile and Desktop
status: todo
priority: high
spec_ref: specs/v0.2.0.md#morphology-themes
dependencies:
    - T-053-render-safe-principal-part-comparisons-with
updated_at: "2026-10-04T23:05:46Z"
---

# T-062-verify-morphology-presentation-on-ankimobile-and Verify morphology presentation on AnkiMobile and Desktop

## Description

Perform native-client acceptance of new answer-side comparisons and themes. Browser previews alone cannot close this gate.

## Acceptance

- Record AnkiMobile client/version and iPhone/iPad devices as the primary targets, and Anki Desktop client/version as secondary. Required unavailable device checks remain explicit open gates.
- Verify answer reveal, touch expansion, complete four-role comparison with accepted further explanation, explicit absent/withheld roles, and readable muted/monochrome light/dark presentation. Core answer and compact comparison remain visible regardless of expansion.
- Verify completion and recognition prompts remain Latin-first and theme changes preserve semantic claims and note/card/template identities. Test the documented manual reference CSS/setup delivery path and the same effective theme in preview/imported notes.
- Record native details/summary behavior and select the tested disclosure or static fully readable fallback; broken controls cannot hide required information. Static fallback still requires native reveal/readability evidence. Do not introduce untested JavaScript.
- Run existing-recipe checks when the direct prerequisite is complete; form-parsing-exercises owns affected retests after its later changes. Bind evidence to tested content/setup and repeat affected checks for release-candidate changes.
- Capture and inspect representative native screenshots and observed interactions. Record limitations before claiming client compatibility or release readiness.

## Verification

Provide native device/client evidence and the fallback decision. Run the mandatory validation chain for any code fixes and repeat affected client checks.
