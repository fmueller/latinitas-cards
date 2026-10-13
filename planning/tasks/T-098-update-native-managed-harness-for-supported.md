---
id: T-098-update-native-managed-harness-for-supported
title: Update native managed harness for supported retained slot content
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#review-identity-and-export
dependencies: []
updated_at: "2026-10-13T21:02:09Z"
---

# T-098-update-native-managed-harness-for-supported Update native managed harness for supported retained slot content

## Description

The documented native managed harness aborts at the supported retained-slot
content case on pinned f8cd5dc85df946aa9a66ac5a74d6edab07cfff64.
With optional anki/aqt 26.9.3 in a disposable Python 3.14.2 environment, run:

```sh
python scripts/check-managed-desktop.py --output-dir /tmp/new-native-evidence --prepare
```

Its backend setup subprocess reaches `scripts/check-managed-anki.py:628` and
fails `assert label == "front-clear"` when `label` is `slot-content` and no
unsupported operations exist. Retained slot answer changes are now supported;
the old negative case still expects them to be rejected. Consequently the
documented Desktop preparation never produces bindings.json or opens the GUI.
This is harness drift, not a product failure to reject an unsafe operation.

Deduplication: completed T-083 corrected a different anchored-absence negative
case. Completed T-088 established supported retained updates. Neither corrects
this slot-content assumption in the advertised native verifier.

## Acceptance

- Update native harness expectations for supported retained answer/prompt changes,
  while retaining genuine structural creation/retirement refusal checks.
- The backend verifier and Desktop `--prepare` finish on current supported
  templates/CSS; do not weaken production policy to satisfy stale assertions.
- A native Desktop retained-content import is observed with exact card/history,
  template/field bindings, Personal Notes and unapproved-content preservation.

## Verification Notes

- ROUND5 reproduced this on 2026-10-13 with Python 3.14.2/SQLite 3.50.4. A local
  setup-only adapter reused the freshly created asymmetric fixture and bypassed
  the obsolete backend negative tail; the unchanged actual Desktop update driver
  then preserved 8 cards and 15 review logs and observed both approved operations.
- Python 3.13.15/SQLite 3.53.1 separately hit `no such collation sequence: unicase`
  in the harness's table reader. That environment issue is not the slot-content
  defect and is not claimed to be a production importer regression.

## Implementation Notes

No product fix was made during acceptance. Keep the native proof boundary:
backend fixture creation does not substitute for the tested GUI CSV import.
