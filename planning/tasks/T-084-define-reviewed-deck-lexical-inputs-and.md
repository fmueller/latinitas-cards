---
id: T-084-define-reviewed-deck-lexical-inputs-and
title: Define reviewed deck lexical inputs and construction patterns
status: todo
priority: high
spec_ref: specs/v0.2.1.md#deck-based-phrase-and-grammar-practice
dependencies:
    - T-064-record-reviewed-claim-decisions-from-the-cli-for
updated_at: "2026-10-09T20:05:15Z"
---

# T-084-define-reviewed-deck-lexical-inputs-and Define reviewed deck lexical inputs and construction patterns

## Description

Define the bounded, versioned phrase-input contract using confirmed deck profiles
and the shared review sidecar. Reuse source identities and evidence, not a second
review system. No corpus or LLM is required.

## Acceptance

- Select lexical entries/senses by stable deck identity; capture reviewed paradigms,
  gender, verb sense/valency and supported form choices needed by each construction.
  A gloss or principal parts alone is not a complete paradigm.
- Provide a documented versioned file-import or CLI-authoring route for missing
  lexical data/evidence and patterns, with individual review through T-064; no Python
  API or mutation of source notes is required. Use bounded reviewed sense/pattern
  compatibility constraints, not an unbounded semantic inference engine.
- Persist sense/pattern keys and canonical role-labelled constituent bindings before
  saving selections. Define the stable phrase source identity/scope used by the
  existing single-source note-ID contract; never pick one contributing word as owner.
  Reordering selections preserves bindings, but swapping subject/object roles does not.
- Define reviewed patterns for preposition–noun, noun–genitive, verb–noun/short clause
  and adjective–noun, with case government, agreement, semantic compatibility and
  explicit pattern-supplied function words. Ship bounded fixtures for all four families.
- A CLI inspects/selects entries, senses and patterns, with text/JSON diagnostics for
  missing paradigms, uncertain senses, incompatible combinations and unsupported forms.
  Source deck notes are never mutated.
- Persist versioned vocabulary/pattern selections and generation limits for deterministic
  reruns; validate unknown selections and invalid limits.
- Positive and asymmetric negative fixtures distinguish agreement, government and
  valency errors, without network or optional analyzer resources.
- Pass the mandatory ruff/mypy/pytest chain.

## Verification Notes

- Record verification timestamps and results; do not commit gitignored artifact paths.

## Implementation Notes

- Start from profile.py, sources.py, source_extraction.py and claim_review.py;
  use focused domain/command modules, not new logic in cli.py.
