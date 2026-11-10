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
domain/command modules. Own item schema, validation and rule/optional-mnemonic review;
do not silently recast qa notes. Follow approved contracts and spec execution/validation
requirements.

## Acceptance

- Validate stable author keys, precise rule, optional original mnemonic wording/language,
  examples, caveats/exceptions, tags and course/lecture provenance. Separate rules
  from memory aids, and quoted course wording from suggestions. Accept rule-only items
  without placeholder mnemonics; record rule/explanation approval independently of
  approval for a language-specific mnemonic and its association with the rule.
- Document/test agent extraction from permitted lecture notes into a candidate file.
  Prefer course aids; optional generated aids stay marked candidates. No mandatory
  hosted processing or automatic authoritative import.
- Preview candidates and record explicit rule and optional-mnemonic approval/rejection.
  Preserve German wordplay; English alternatives need separate review. Changed rule/
  explanation/mnemonic/caveat invalidates affected approval, not unrelated rule-only
  cards on a mnemonic-only edit. Any displayed mnemonic requires approval, not only
  mnemonics used by the recall recipe.
- Source rights and extraction/provider permission are explicit; a later local model
  does not make earlier hosted extraction local. Automatic form acceptance cannot
  approve curated rules or mnemonics.

## Verification Notes

Check malformed/missing provenance, quoted/suggested drafts, approved/rejected variants,
rule-only intake, stable keys after edits and unchanged approval reuse. Contrast approved
English rule/explanation content with no English mnemonic and an approved German mnemonic.
Export needs rule AND rendered approval, plus mnemonic approval whenever one is shown;
this task cannot authorize it alone. Run the mandatory validation chain.

## Implementation Notes
