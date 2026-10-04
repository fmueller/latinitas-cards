---
id: T-025-ship-the-authored-note-extraction-agent-skill
title: Ship the authored note extraction agent skill
status: completed
priority: medium
spec_ref: specs/v0.1.1.md#authored-note-extraction-skill
dependencies:
    - T-020-parse-and-validate-authored-note-import-files
    - T-023-preview-and-select-authored-notes-before-export
updated_at: "2026-10-04T11:30:09Z"
---

# T-025-ship-the-authored-note-extraction-agent-skill Ship the authored note extraction agent skill

## Description

Add a repository agent skill, installed for both `.claude/skills/` and
`.agents/skills/`, that extracts authored items from loosely structured Markdown study
notes into the import format and then runs validation and preview.

## Acceptance

- The skill documents key conventions per kind and requires explicit, reviewed keys for
  `qa` items. Contextually distinct form occurrences receive distinct explicit keys;
  correcting a citation never silently renames an existing key.
- It covers varying heading and table column names, bullet-list vocabulary, and answers
  given inline or in a separate solutions section.
- Re-extraction into an existing import file preserves existing keys and `skip` decisions
  and reports new, changed, and missing items.
- It runs validation and preview before reporting, and never writes to Anki directly.
- The skill is corpus-agnostic; its examples use synthetic notes from more than one text
  and contain no private study material.
- Synthetic first-extraction and re-extraction fixtures and expected import files are
  checked in and validate cleanly. Record an extraction/re-extraction exercise with a
  corrected answer, an existing skipped item, a new item, and a missing item; verify stable
  keys, preserved statuses, and the expected new/changed/missing report, then run preview.
- `uv run ruff check`, `uv run mypy`, and `uv run pytest -v` pass.

## Verification Notes

- Strict red/green: `uv run pytest tests/unit/authored_extraction_skill_test.py -q`
  initially failed both tests with missing skill/fixture FileNotFoundError; after
  the minimal skill/fixtures implementation, both passed (2 passed in 0.44s).
- Actual agent extraction/re-extraction ran from synthetic notes, not expected
  JSONL. The checked-in exercise ledger is
  `tests/fixtures/authored-extraction/README.md`. Both actual candidates passed
  authored validate and preview; effective selections 5 and 6, zero duplicates.
  All six existing keys, statuses, and IDs stayed stable; one new, two changed,
  one missing retained, three unchanged matched. The corrected cargo answer was
  also inspected in a section-filtered preview (selection 1).
- Initial exact ruff/mypy/pytest chain: all checks passed, mypy clean in 65
  source files, 508 passed in 9.68s. Final exact chain after independent reviews:
  all checks passed, mypy clean in 65 source files, 508 passed in 7.81s.
- Dedicated Task simplifier loaded code-simplifier and building-skills:
  "No changes made." Targeted tests: 2 passed; diff whitespace check passed.
- Separate parallel General, Python, Security reviewer Tasks each loaded
  code-reviewer and its lane guidance. General used ECC code-reviewer; Python
  used ECC python-reviewer and python-patterns; Security used security-reviewer
  and security-review plus common security/code-review rules. Each concluded
  verbatim: "No concrete task-relevant findings." Each ran focused tests
  (2 passed) and inspected actual extraction outputs. Security covers the
  Markdown trust boundary, candidate preservation, and selection/data-loss risk.
- Database omitted: no database/schema/concurrency changes. Framework, other
  language, ML/RAG, healthcare, network lanes omitted: no applicable changes.
  Three lanes (General plus two specialists) stay within the soft budget.
- Fresh candidate-validation Task loaded code-reviewer: "No candidate IDs were
  submitted. Independently reviewed the T-025 changes against
  `specs/v0.1.1.md#authored-note-extraction-skill`; found no material
  task-relevant issues." No findings to fix, defer, or deduplicate; no rejected
  candidates. One review cycle; no dispositions requiring code changes.
- Fresh disposition-verification Task loaded code-reviewer and building-skills:
  "Status: no unresolved or newly identified task-relevant findings." It reran
  the exact full chain: ruff passed, mypy clean in 65 source files, 508 passed
  in 7.84s, plus actual candidate validation/preview and byte comparisons.
  Its Taskrail PATH limitation was resolved by the owning agent running
  `taskrail validate`: state valid, before verification and after completion.
- Taskrail verification passed at 2026-10-04T11:30:09Z; completed only after
  all reviews, candidate validation, fresh disposition verification, and checks.

## Implementation Notes

- Skill mirrored in .agents and .claude; synthetic first/revised Markdown,
  expected JSONL, expected change report, contract tests, and user docs shipped.
  No binary Markdown extraction/parser, export, or live Anki updates added.
- Missing rows are retained unchanged pending review; an included missing row
  remains preview-eligible. QA review and ambiguous-match blockers are explicit.
- 2026-10-04T11:30:09Z: verification pass
- 2026-10-04T11:30:09Z: Agent-only skill, mirrored installations, synthetic exercise, expected fixtures, tests and docs implemented; reviewed and verified before completion.
