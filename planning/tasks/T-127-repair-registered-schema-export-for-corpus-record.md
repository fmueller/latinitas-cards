---
id: T-127-repair-registered-schema-export-for-corpus-record
title: Repair registered schema export for corpus record unions
status: todo
priority: medium
spec_ref: specs/v0.3.0.md#normalized-jsonl-corpus-model
dependencies:
    - T-107-normalize-plain-text-sources-with-manifest
updated_at: "2026-10-15T23:16:07Z"
---

# T-127-repair-registered-schema-export-for-corpus-record Repair registered schema export for corpus record unions

## Description

Repair the documented schema regeneration command for the registered corpus
record union, under the normalized corpus model's schema drift contract.
`uv run python scripts/export-json-schemas.py` currently raises `KeyError: 'version'`
because `CorpusRecordSchema` is a RootModel union, not a version-bearing header.
The schema drift test already handles this shape; the exporter does not.

Follow-up derived from T-107-normalize-plain-text-sources-with-manifest's verification or discovery.

## Acceptance

- Add a regression that executes the exporter against the complete registry,
  fails on the current implementation, and verifies all expected schema bytes.
- The documented command exits zero and exports every registered v1 document,
  including the corpus record union and corpus manifest, without schema drift.
- Preserve the registered schema names, corpus models and legacy behavior; do not
  redesign the registry or change the pinned spec. Pass the mandatory gates.

## Verification Notes

- Discovered during T-107 on 2026-10-15: documented exporter exits 1 with
  `KeyError: 'version'` at `model.model_fields["version"].default`.
- T-107 generated its two affected schemas using the existing `render_schema`
  registry API; exporter repair is not part of that task's implementation.

## Implementation Notes
