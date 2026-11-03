---
id: T-196-validate-provenance-bound-curated-rule-and
title: Validate provenance-bound curated rule and mnemonic drafts
status: todo
priority: medium
spec_ref: specs/v0.5.0.md#curated-grammar-and-mnemonics
dependencies:
    - T-187-implement-versioned-automatic-decisions-and-reuse
updated_at: "2026-11-02T23:10:11Z"
---

# T-196-validate-provenance-bound-curated-rule-and Validate provenance-bound curated rule and mnemonic drafts

## Description

P50-6 versioned curated intake using applicable authored conventions and focused
domain/command modules. Own item schema, validation and pair review; do not silently
recast qa notes. Follow approved contracts and spec execution/validation requirements.

## Acceptance

- Validate stable author keys, precise rule, original mnemonic wording/language,
  examples, caveats/exceptions, tags and course/lecture provenance. Separate rules
  from memory aids, and quoted course wording from suggestions.
- Document/test agent extraction from permitted lecture notes into a candidate file.
  Prefer course aids; optional generated aids stay marked candidates. No mandatory
  hosted processing or automatic authoritative import.
- Preview candidates and record explicit per-item pair approval/rejection. Preserve
  German wordplay; English alternatives need separate review. Changed rule/mnemonic/
  caveat invalidates affected approval.
- Source rights and extraction/provider permission are explicit; a later local model
  does not make earlier hosted extraction local. Automatic form acceptance cannot
  approve curated pairs.

## Verification Notes

Check malformed/missing provenance, quoted/suggested drafts, approved/rejected variants,
stable keys after edits and unchanged approval reuse. Export needs pair AND rendered
approvals; this task cannot authorize it alone. Run the mandatory validation chain.

## Implementation Notes
