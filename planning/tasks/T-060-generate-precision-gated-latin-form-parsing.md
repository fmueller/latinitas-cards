---
id: T-060-generate-precision-gated-latin-form-parsing
title: Generate precision-gated Latin form parsing exercises
status: todo
priority: medium
spec_ref: specs/v0.2.0.md#calibrated-form-parsing
dependencies:
    - T-052-generate-supported-principal-part-relationships
    - T-058-expose-deterministic-managed-plans-and-operation
updated_at: "2026-10-04T23:05:46Z"
---

# T-060-generate-precision-gated-latin-form-parsing Generate precision-gated Latin form parsing exercises

## Description

Add the distinct later form-parsing exercise step using the same evaluated claim policy, without treating principal-part comparison as proof of full token analysis.

## Acceptance

- Map encountered forms to lemma and only applicable accepted features such as person/number/tense/mood/voice/case/gender; use Latin-first prompts and explicit required-role eligibility.
- Specify stable object/task/template bindings before generation. Reuse coherent-object identity where appropriate; do not merge unrelated contextual objects or repurpose existing principal-part template slots. Document the supported preview/export and manual reference-setup route for new bindings; refuse incompatible managed application rather than silently provisioning templates.
- Ambiguous forms expose alternatives and review reasons rather than unconditional assertions; unsupported features and withheld claims cannot appear as facts or generate misleading cards.
- Automatic generation requires domain-relevant measured precision and approved per-claim policy. Optional analyzers/LLMs do not make default installation or core operation network-required.
- Preview the accepted claims and skips. Structural card additions under the initial managed CSV scope remain blocked for application; generation does not authorize destination changes.
- Tests include asymmetric ambiguous forms, absent features, and accepted versus rejected claims; principal-part completion/recognition stay unchanged.
- Repeat native update/presentation checks affected by parsing changes before claiming that scope verified or release-ready; earlier existing-recipe evidence cannot certify new behavior. These retests are part of this task's completion, not a reason to delay the initial native gates.

## Verification

Use red/green tests and the mandatory ruff, mypy, pytest -v chain. Record reviewed/evaluated linguistic evidence and affected native retest results for the supported exercise scope.
