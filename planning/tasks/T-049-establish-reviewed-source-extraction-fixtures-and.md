---
id: T-049-establish-reviewed-source-extraction-fixtures-and
title: Establish reviewed source extraction fixtures and supported layouts
status: completed
priority: high
spec_ref: specs/v0.2.0.md#evidence-led-source-extraction
dependencies: []
updated_at: "2026-10-05T00:25:24Z"
---

# T-049-establish-reviewed-source-extraction-fixtures-and Establish reviewed source extraction fixtures and supported layouts

## Description

Use existing skipped/ambiguous evidence to select a bounded set of additional source layouts. Publish reviewed sanitized examples before broadening extraction; do not invent supported layouts from speculation.

## Acceptance

- Inventory available v0.1 evidence and distinguish promised-layout regressions from new support. Missing evidence remains an explicit review requirement.
- Publish a supported-layout matrix and fixtures with independently reviewed expected candidates, role positions, alternatives, explicit omissions, formatting/separators, and embedded hints.
- Include similar-looking counterexamples that must remain unsupported or ambiguous; record exact withholding reasons.
- Include explicit fourth-role PPP versus supine distinctions and deponent/exceptional cases without claiming extraction proves their linguistic analysis.
- Preserve original text/provenance and document representative profile-confirmation examples; no private deck is required.
- Declare the reviewed sample, counting units and denominators, and whether categories overlap. Distinguish wholly skipped entries from generated entries with omitted/withheld-role warnings; overlapping counts must not appear to partition the sample. Do not claim universal coverage.

## Verification

Record review provenance and expected outputs suitable for subsequent regression tests. Fixture expectations must not be derived from the parser under test.

## Implementation and independent review provenance

2026-10-05: Hand-authored 17 profile-conditioned expectation cases in
`tests/fixtures/source-extraction/reviewed.jsonl`, with the support/evidence
matrix and confirmation contracts in `docs/source-extraction-fixtures.md`.
Sources are the committed T-032 CSV, sanitized T-011 boundary documentation,
and explicitly labeled synthetic replays; no private deck was accessed.
Expectations were derived from raw evidence and documented contracts, never
from running the parser. No production code or parser behavior changed.
Documentation/fixture work used concrete content checks, not a claimed TDD
red/green production transition.

Dedicated Task simplifier loaded code-simplifier: no edits recommended; eight
existing extraction-content-review tests passed. Dedicated read-only General
Task loaded code-reviewer, General reviewer and common code-review rule:
"No concrete task-relevant findings." It independently adjudicated all 17
JSONL expectations against original CSV/documentation, including raw values,
roles, alternatives, omissions, hints, reasons and denominators. This is an
agent structural/content review, not Latin-expert approval. General was the
only selected lane: no language code, framework, trust boundary, database,
linguistic-analysis implementation or concurrency contract changed; provenance
and sanitization were covered as content criteria. No extra specialist budget
was needed.

Fresh candidate-validation Task loaded code-reviewer and General/Python
references. Empty supplied candidate sets; it identified this acceptance gap:

> Overlooked acceptance gap: `docs/source-extraction-fixtures.md:6–8` says the independent review is recorded in T-049’s task notes, but the current `planning/tasks/T-049-establish-reviewed-source-extraction-fixtures-and.md` contains no review provenance or review results—only the task description and acceptance criteria. T-049’s verification requires recording review provenance and expected outputs. The review report described in the request may provide that evidence, but it is not present in the task notes the document points to.

Disposition F1: fixed by this record, preserving actual review conclusions and
limits. Before the fix the task ended at its Verification requirement; after
the fix this provenance section records authoring method, independent lanes,
review results and scope. No behavior/test change was needed for this prose
omission. No findings deferred and no new follow-up invented; broader extraction
implementation remains owned by the subsequent pinned-spec tasks.

Initial concrete checks: JSONL decoding and unique IDs (17), status counts
(13 supported, three ambiguous, one unsupported), raw CSV exact matches for
all ten CSV-referenced cases, confirmed role lengths, and two explicit
omissions passed. `uv run pytest tests/unit/extraction_content_review_test.py
-q`: eight passed. The target statuses are not claims of current generation.

Final exact chain after the provenance fix: `uv run ruff check` — all checks
passed; `uv run mypy` — no issues in 65 source files; `uv run pytest -v` —
627 passed in 16.65 seconds. Fresh read-only disposition-verification Task
loaded code-reviewer: "F1 — RESOLVED." and "New task-related findings: None."
It inspected the updated provenance record, fixture/spec acceptance and
reran the focused extraction-content tests (eight passed). One review/fix
cycle; no unresolved or deferred findings. Taskrail finalization follows
these checks and review, not the other way around.

## Implementation Notes

- 2026-10-05T00:25:24Z: verification pass
- 2026-10-05T00:25:24Z: Reviewed source fixtures and matrix published without parser expansion; one provenance finding fixed and verified; exact validation chain passed.
