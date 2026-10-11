---
id: T-084-define-reviewed-deck-lexical-inputs-and
title: Define reviewed deck lexical inputs and construction patterns
status: completed
priority: high
spec_ref: specs/v0.2.1.md#deck-based-phrase-and-grammar-practice
dependencies:
    - T-064-record-reviewed-claim-decisions-from-the-cli-for
updated_at: "2026-10-11T11:30:57Z"
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
- Implemented version-1 file imports and `phrases inspect/select` in focused
  modules. `claims list/record --phrase-input` uses the T-064 sidecar, locking,
  fingerprints and individual review; structured corrections come from the
  imported file with evidence, not free-text substitutions or deck mutations.
- Bind confirmed deck identities/scopes, stable sense/pattern/form keys and
  canonical constituent roles before saving. Composite phrase source identity
  includes every role/source binding; the separate object key includes the
  construction and tested sense/form choices. No word is chosen as owner.
- Shipped bounded synthetic fixtures for all four pattern families, explicit
  function words, governed cases, nominal/verb agreement, valency and finite
  compatibility allowlists. Unknown selections and invalid limits fail closed;
  writes are atomic and protect source/profile/manifest/review files and locks.
- Dedicated code-simplifier passes made no edits. Independent parallel General,
  Python and FastAPI/Pydantic lanes found no concrete task-relevant findings.
  Security S-001 (selection output could replace the shared review lock) was
  freshly validated, fixed with direct/symlink/hard-link RED-to-GREEN tests,
  and independently verified RESOLVED. No findings were deferred.
- TDD evidence: missing `phrases` command failed before implementation; initial
  eight tests passed. Strict-version regression rejected boolean/float/string
  versions after demonstrating boolean acceptance. Text inspection regression
  failed on truncated later patterns before full escaped text output. Security
  lock regression failed in all three alias cases, then passed in all three.
- Manual CLI checks on 2026-10-11: unreviewed selection exited 1 without output;
  explicit individual fixture batch recorded 23 claims; reviewed inspection
  passed; all four selections saved; reordered requests were byte-identical;
  wrong governed case exited 1 and retained prior output; source CSV SHA-256
  remained unchanged. Text output exposed all four families. Lock output was
  rejected and subsequent reviewed inspection passed. Existing CSV scope
  bootstrap via `generate --approve-scope --approve-fresh-import` succeeded.
- Validation on 2026-10-11: full ruff/mypy/pytest chain passed with 965 tests,
  including 22 focused phrase-input cases. Full-suite integration updated the
  CLI-group callback test and removed the obsolete empty-Unreleased assumption
  without changing release metadata. Phrase generation, recipe rendering,
  generated-answer approval and export remain later tasks, not this cycle.
- 2026-10-11T11:30:57Z: verification pass
- 2026-10-11T11:30:57Z: Implemented reviewed version-1 lexical/pattern input and deterministic role-bound selections; final mandatory checks pass (965 tests), manual CLI pass, independent review S-001 resolved; phrase generation/render/export remain later tasks.
