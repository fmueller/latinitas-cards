---
id: T-174-decide-whether-generate-recipe-overrides-should
title: Decide whether generate --recipe overrides should stale reviewed claims
status: blocked
priority: medium
spec_ref: specs/v0.4.0.md#german-and-english-output
dependencies:
    - T-157-localize-fixed-card-strings-and-english-reviewed
updated_at: "2026-10-29T05:30:24Z"
---

# T-174-decide-whether-generate-recipe-overrides-should Decide whether generate --recipe overrides should stale reviewed claims

## Description

Follow-up found during T-157 review; the behavior already exists at db90405 and is not
caused by T-157. `generate --profile P --recipe principal_part_completion` (one recipe
instead of the profile's two) reports every stored decision as "Stale or unmatched
decision ignored" (16 for the v0-4-0 de-light-user fixture), while `preview` without
`--recipe` applies them. `commands/principal_parts.py` writes `selected_recipes` into the
profile override, and claim fingerprints bind the whole deck profile except `morphology`
(`claim_review.py`). Section `#german-and-english-output` restates that fingerprint
contract as unchanged since v0.2.1, so changing the fingerprint inputs needs a maintainer
decision: either a run-time `--recipe` filter is excluded from claim binding, or the
current staleness is documented as intended.

## Acceptance

- The maintainer records which behavior is intended for a `--recipe` override.
- If the binding changes, existing German claim fingerprints and claims-v1 / v0-4-0
  goldens stay byte-identical for runs without `--recipe`, with a regression test for a
  single-recipe run that keeps accepted decisions.
- Otherwise the `--recipe` help and docs state that narrowing recipes stales reviews.

## Verification Notes

- Not started. Reproduced read-only at base and at the T-157 candidate during T-157 review.

## Implementation Notes

- 2026-10-29T05:30:24Z: maintainer decision: whether a generate --recipe override should be excluded from claim fingerprints, which changes the unchanged v0.2.1 fingerprint contract
