---
id: T-127-repair-registered-schema-export-for-corpus-record
title: Repair registered schema export for corpus record unions
status: completed
priority: medium
spec_ref: specs/v0.3.0.md#normalized-jsonl-corpus-model
dependencies:
    - T-107-normalize-plain-text-sources-with-manifest
updated_at: "2026-10-16T14:53:24Z"
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

- Workflow-v3 understand: fetched origin/main and established accepted base
  `23b508a7db2c9846b482359a898fc7518fa12547`, containing tested T-125
  `2cf5e691964ed806a27861d8abdf38b3d91ca6b4`. Spec SHA256 remained
  `fa3cd4c7ba93cdda1780f75d047372b0211c63adc22e68cd71c51d90b125b704`.
  Before lifecycle writes, Taskrail validate passed; selector chose T-127
  (T-124 was another candidate), with zero active tasks. Started only T-127.
- Strict TDD: documented `uv run python scripts/export-json-schemas.py`
  reproduced exit 1, `KeyError: 'version'`. The new public subprocess
  regression also failed with that exact error before production changes.
  Minimal repair uses declared Literal versions and the existing drift test's
  v1 union fallback, retaining `render_schema`. Required Literal fields also
  previously generated erroneous `vPydanticUndefined` filenames; removed only
  those three untracked files created by this reproduction.
- GREEN: `uv run pytest -v tests/unit/json_schema_drift_test.py` passed all
  11 tests. The exporter exits zero; every registered v1 document (nine total,
  including corpus union and manifest) has the exact committed filename and
  bytes. The regression copies the real exporter into a temporary directory,
  invokes it with the installed package and complete registry, checks the exact
  filename set, and compares each file's bytes to independent committed fixtures.
  No registry, model, generated schema, spec or legacy behavior was changed.
- Initial exact chain: `uv run ruff check` passed; `uv run mypy` passed
  (169 source files); `uv run pytest -v` passed (2478 tests, one warning).
- Dedicated simplifier Task loaded personal `code-simplifier`, inspected the
  diff and ownership, and made no changes: no meaningful simplification.
- Separate parallel read-only General and Python lane Tasks loaded personal
  `code-reviewer`, reviewer-index mappings, ECC General/common rules and
  Python reviewer/python-patterns respectively. Both returned verbatim:
  "No concrete task-relevant findings." General ran diff whitespace checks;
  Python additionally reran the focused suite (11 passed).
  General covers acceptance/contracts; Python covers Pydantic and subprocess
  behavior. Security, Database, framework and other domain lanes were omitted:
  no trust boundary, persistence/database, framework, or domain-model change.
- Fresh candidate-validation Task loaded `code-reviewer` in its designated
  mode and returned: "No concrete task-relevant findings." Zero validated,
  rejected or deduplicated candidates; no findings to fix or defer, so no
  finding-fix TDD was needed. One review/disposition cycle only.
- Final exact ruff -> mypy -> pytest chain passed again (169 typed files,
  2478 tests, one expected Pydantic warning in the float-rejection test).
  Sequential `mise run check` then passed, including format, full pytest
  (2478 passed), and all shell policy suites. No full suites ran concurrently.
  Hooks were installed with `mise exec -- lefthook install` before the first
  commit; maintainer identity and no-attribution commit policy passed.
- Tested implementation revision: `e2fa794e7dce41e3a09e35ed3b7190b2b366741a`.
  Fresh final disposition-verification Task loaded `code-reviewer`, inspected
  that commit against the accepted base and lifecycle audit, and returned
  verbatim: "No concrete task-relevant findings." It confirmed zero unresolved
  or newly introduced issues and independently reran the exporter regression
  (1 passed) and diff whitespace checks. No deferrals or follow-up blockers.
- Final manual exporter run exited zero and left every committed schema
  unchanged. All gates and final disposition verification preceded Taskrail
  verify/complete; T-124 and T-128 remain unstarted, T-038 remains blocked.
- 2026-10-16T14:53:24Z: verification pass
- Lifecycle chronology: started 2026-10-16T14:44:34Z only after selector/pin
  checks; verified pass at the timestamp above only after every gate and final
  reviewer; then completed through Taskrail. Post-completion validate passed,
  with 125 done, zero active, one blocked and two todo. Observed next T-124,
  not started. T-128 remains future-spec todo; T-038 remains maintainer-blocked.
  Pre-delivery fetch still showed the accepted base on origin/main; delivery
  uses focused non-force commits only. No publication or external gate changes.
