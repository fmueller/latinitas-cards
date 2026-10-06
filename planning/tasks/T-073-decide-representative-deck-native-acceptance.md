---
id: T-073-decide-representative-deck-native-acceptance
title: Decide representative-deck native acceptance for morphology disclosure
status: todo
priority: high
spec_ref: specs/v0.2.0.md#morphology-themes
dependencies: []
updated_at: "2026-10-06T20:16:22Z"
---

# T-073-decide-representative-deck-native-acceptance Decide representative-deck native acceptance for morphology disclosure

## Description

Found by the T-069 review. specs/v0.2.0.md Delivery Order item 5 asks for AnkiMobile and Desktop
acceptance on a sanitized representative deck before marking disclosure and contrast compatible.
T-062 used a synthetic four-verb kit, captured no native screenshots for the 2026-10-06 run
although its acceptance lists them, and did not exercise a CSS change on an existing scheduled
note type. Needs a maintainer decision. Release-blocking for T-068.

## Acceptance

- Either run the native kit on the sanitized representative deck (tests/fixtures/representative-university-latin.apkg) with screenshots, or record a dated maintainer waiver in specs/v0.2.0.md, T-062 notes, and docs/morphology-native-verification.md.
- docs/morphology-native-verification.md states exact client/OS versions or explicitly notes what was reported only by the maintainer.
- User-facing docs describe disclosure as verified only within the recorded scope.

## Verification Notes

- Pending.

## Implementation Notes


