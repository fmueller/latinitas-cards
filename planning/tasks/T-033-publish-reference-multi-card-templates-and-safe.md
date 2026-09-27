---
id: T-033-publish-reference-multi-card-templates-and-safe
title: Publish reference multi-card templates and safe import guidance
status: todo
priority: high
spec_ref: specs/v0.1.0.md#reference-note-type-and-import-safety
dependencies:
    - T-030-generate-conditional-sibling-cards-with-stable
updated_at: "2026-09-27T08:56:30Z"
---

# T-033-publish-reference-multi-card-templates-and-safe Publish reference multi-card templates and safe import guidance

## Description

Supply the manual reference schema/templates and safe first/repeat import instructions
for the new model. General automated provisioning remains v0.6.0.

## Acceptance

- Provide exact versioned fields, stable slot registry, copyable guarded front/back
  templates and CSS for both recipes and synthetic missing-form examples, aligned with
  exported columns. Explain shared Personal Notes/tags and independent card schedules.
- Establish one small authoritative generated-schema contract for managed regular fields,
  user-owned Personal Notes, and transport metadata. Tags is a special CSV transport
  column, not a regular note field. Derive column positions/directives and reference
  template fields from the contract; independently assert external ordering/tag mapping,
  exclusion of personal data, and conditional fronts against explicit expectations.
- Keep source field inference/profile preparation/raw provenance with source adapters;
  CSV metadata and escaping belong to transport serialization. This is drift prevention,
  not a reproduced current import failure; no generic adapter framework is required.
- Checklist covers backup, dedicated note type, LatinitasID first, HTML, Update, Note Type
  matching scope, deck/mapping inspection, and absent/unmapped Personal Notes.
- Warn prominently before import/reimport: mapped Tags replaces destination-only manual
  tags even on otherwise unchanged content; parent inheritance is not preservation.
  Give backup/defer options without claiming destination-aware CSV tag merging.
- Document manual sibling-burying settings, separate note/card counts, eligibility-loss
  review and tested boundaries. Broad compatibility stays v0.9.0.
- Preserve completed HTML/Personal Notes safeguards. No tasks for resolved harness,
  environment, scheduling-snapshot or black-screenshot process issues.

## Verification Notes

- Check reference templates/schema against exporter and fixtures. Run applicable docs
  checks and the mandatory code chain if code changes. Native rendering is in T-035.
- Record evidence when executed; current spec template is schematic, not verified output.

## Implementation Notes
- Depends on final field and slot choices in T-030; no automated provisioning required.
