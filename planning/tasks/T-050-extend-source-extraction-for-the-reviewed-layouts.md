---
id: T-050-extend-source-extraction-for-the-reviewed-layouts
title: Extend source extraction for the reviewed layouts
status: completed
priority: high
spec_ref: specs/v0.2.0.md#evidence-led-source-extraction
dependencies:
    - T-049-establish-reviewed-source-extraction-fixtures-and
updated_at: "2026-10-05T01:04:07Z"
---

# T-050-extend-source-extraction-for-the-reviewed-layouts Extend source extraction for the reviewed layouts

## Description

Implement only the additional extraction rules justified by the reviewed fixture set, using the existing confirmed-profile workflow.

## Acceptance

- Preserve original text and provenance alongside normalized candidates and applied-rule records.
- Keep alternatives unresolved until an explicit selection rule or review; retain omissions in their positions rather than shifting later forms. Preserve mixed hints as evidence, not grammatical assertions.
- Unknown layouts and conflicting hints produce actionable unsupported/ambiguous outcomes.
- Profile confirmation and skip/review output distinguish extraction failure from linguistic uncertainty. Report reviewed/generated/skipped/ambiguous counts using the declared units, denominators, and overlap rules; distinguish wholly skipped entries from generated entries with omitted/withheld-role warnings.
- Exact-output tests cover every accepted layout and its counterexamples, especially omitted middle roles and alternatives. Existing supported layouts remain regressions.
- Preserve evidence for the claim-policy handoff: unresolved alternatives and conflicting role evidence must remain identifiable in generation/preview/export, not silently become unconditional facts in either existing recipe. Verify that handoff once claim-policy and principal-part-analysis are available.

## Verification

Use red/green tests for new behavior. Run uv run ruff check, uv run mypy, and uv run pytest -v in that order; after any failure rerun the full chain.

## Implementation and workflow evidence

2026-10-05: Fetched accepted remote main and confirmed it contained T-048 and
completed T-049. Clean checkout matched the accepted remote revision; no stale
source-checkout state was transferred. Before lifecycle writes, Taskrail validate
passed, next --json selected exactly this task, and the active spec was v0.2.0.
Only this one task cycle was started. Maintainer identity was preserved and
lefthook install was run before committing.

Implemented bounded extraction in source_extraction.py, explicit confirmed-profile
options for pipe candidates and trailing slot-3 poet. evidence, and retained
raw/provenance, ordered positional candidates, hints, and applied rules. Both
existing recipes withhold unresolved targets and mask uncertain completion-front
context. Back context labels uncertain candidates as evidence, not answers.
Export carries safely escaped structured evidence inside the existing Principal
Parts field, without changing the Anki schema or identity. Rejected conflicts
remain raw review evidence. Leading-role omissions remain rejected but preserve
their extracted positional evidence. No authoritative fixture expectations changed.

Preview/setup separate matched extraction from linguistic uncertainty. Counts
distinguish generated entries, wholly skipped entries, overlapping generated-role
warnings, current-entry ambiguity and manifest review events; cards remain separate.
The historical extraction-review strata are unchanged, with new claim warnings
tested separately. Legacy pipe/multiline display remains readable, but unresolved
answers are no longer eligible; the release replay deliberately verifies that
fully unresolved objects are retained for review but not exported as blank cards.

Strict TDD: new fixture tests first failed 23 cases on absent profile/evidence
contracts, then passed all 17 reviewed cases plus six counterexamples. Downstream
both-recipe tests failed on inflated skipped counts (2 versus 1); a strengthened
completion test then failed on alternatives appearing as factual prompt context.
After minimal fixes, focused extraction/generation/export tests passed (145).
The broad gate exposed historical eligibility assertions requiring the intentional
withholding update, plus test typing/line-length issues. Each failed gate was
followed by the complete ruff, mypy, pytest chain, not selective verification.

Dedicated Task simplifier loaded code-simplifier, replaced duplicate alternative
scans with one computed predicate, and reran the full gate (654 passed).
Independent parallel Task lanes loaded code-reviewer with General code-reviewer,
Python python-reviewer/python-patterns, and Security security-reviewer/security
references. General covered domain/count contracts; Python covered typed models
and callers; Security covered untrusted HTML, hidden evidence and output escaping.
No framework, database/migration, concurrency or ML implementation changed, so
those specialist lanes were omitted. Three selected lanes fit the soft budget.

Fresh candidate validation confirmed these exact findings (PY-1 duplicates G-2):

> G-1: When an entry contains both a confirmed `poet.` hint and confirmed pipe alternatives, record every transformation applied; the `if hints` branch currently skips the alternative-extraction rules.

> G-2: Preserve the already-computed extraction evidence when rejecting an explicitly omitted leading role, rather than discarding it in the parse failure.

> S-1: Hint extraction can label the whole principal-part segment as the hint's raw evidence when a valid HTML line boundary uses attributes or another supported block tag.

All three fixed: cumulative applied rules; attachment of computed evidence to
leading-omission failure; conservative rejection of unreviewed attributed-break
and block-tag hint boundaries, preserving full raw evidence without falsely
labelling the whole part as the hint. No extra hint grammar was invented.
Four reproduction cases failed for the expected missing-rules, missing-evidence
and wrongly-accepted-boundary reasons; all 27 extraction tests then passed.
Fresh disposition-verification Task loaded code-reviewer, confirmed G-1/G-2/PY-1/
S-1 RESOLVED, and reported "New task-relevant findings: None." Full gate: 658 passed.

Final self-check found one concrete reporting-unit issue, independently validated
and disposition-verified by a fresh read-only code-reviewer Task:

> COUNT-1: The ambiguity total counts manifest-removal and stale-snapshot review events as current source entries, so it can exceed or misrepresent the declared sample denominator.

Actual two-row manifest export followed by one-row removal reproduced 2 ambiguous
entries for a one-entry sample, despite that entry generating normally. RED failed
2 versus 0; GREEN passed after counting only current row-index memberships and
reporting snapshot/removal events separately. Fresh validator confirmed COUNT-1
VALIDATED/RESOLVED, retained prior resolved dispositions, and found no new issue.
Second/final focused review cycle; nothing deferred or left unresolved.

Manual/render check used actual generated fields with the repository reference
Anki CSS for regular, alternative, hinted and omitted-middle entries, both recipes.
Four objects generated 24 eligible cards, zero wholly skipped and three generated
warning entries. Chromium inspection confirmed masked completion fronts, qualified
evidence on backs, preserved third/fourth positions, readable layout, and hidden
JSON spans (eight spans; every computed display was none). This is a browser
field/template render, not a claim of native Anki-client testing. Temporary service
and browser session were stopped after inspection.

Final fresh gate: uv run ruff check; uv run mypy; uv run pytest -v, in that exact
order, before Taskrail verify/complete. Later claim-policy/principal-part-analysis
handoff remains owned by existing T-051/T-052: consume retained SourceExtraction
and unresolved flags/raw review evidence, and verify calibrated eligibility there.
No new task or broader parser support was invented; no publishing or deployment.

## Implementation Notes

- 2026-10-05T01:04:07Z: verification pass
- 2026-10-05T01:04:07Z: Reviewed positional extraction/evidence shipped; unresolved recipe targets and context withheld; entry and manifest-event units separated; 659-test exact gate and independent disposition review passed. Later claim-policy/analyzer handoff remains T-051/T-052.
