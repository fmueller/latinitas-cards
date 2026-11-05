---
id: T-209-generate-reviewed-nominal-completion-and
title: Generate reviewed nominal completion and recognition cards
status: todo
priority: high
spec_ref: specs/v0.5.1.md#nominal-lexical-forms
dependencies:
    - T-208-parse-confirmed-verb-variants-and-report
updated_at: "2026-11-05T22:11:24Z"
---

# T-209-generate-reviewed-nominal-completion-and Generate reviewed nominal completion and recognition cards

## Description

Deliver FR-1 mixed nominal/verb setup, lexical roles and recipes through approved
role/layout contracts. Own profile routing and card/template consumers after parsing.

## Acceptance

- Setup confirms noun/adjective routing from configured tags or a confirmed field
  in the same note type. Missing/conflicting evidence withholds; no shape-only class.
- Support noun nominative/genitive/gender m/f/n/c, number-correct plural-only roles
  and irregular genitive-plural annotations. Cover īnsula, pars and castra without
  singular mislabeling.
- Support aliēnus/ācer, fortis and fēlīx adjective terminations without invented
  forms, in completion and recognition with German role/gender labels.
- Preview separates nominal/verb eligible, withheld and skipped counts. Preserve
  verb output, identities and ordinals; new layouts use explicit setup/migration.

## Verification Notes

Exercise mixed-profile/untagged/conflicting fixtures via setup/preview/generate.
Render both recipes in German/English light/dark and inspect captures. T-216 owns
native ordinal/scheduling evidence. Run mandatory ruff → mypy → pytest chain;
follow full spec compatibility, evidence and permission requirements.

## Implementation Notes
