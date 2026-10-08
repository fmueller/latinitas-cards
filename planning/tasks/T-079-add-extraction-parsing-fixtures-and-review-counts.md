---
id: T-079-add-extraction-parsing-fixtures-and-review-counts
title: Add extraction and parsing acceptance fixtures and preview review counts
status: completed
priority: low
spec_ref: specs/v0.2.0.md#calibrated-form-parsing
dependencies: []
updated_at: "2026-10-08T00:59:59Z"
---

# T-079-add-extraction-parsing-fixtures-and-review-counts Add extraction and parsing acceptance fixtures and preview review counts

## Description

Found by the T-069 review. Principal-part preview reports generated/skipped/ambiguous but no
reviewed/withheld-claim count; deponent and middle-omission comparison cases are tested only
on hand-built parses, not end to end through a confirmed profile. Comparison roles are fixed
to four role names. Not release-blocking.

## Acceptance

- Preview reports reviewed/withheld claim counts for the declared sample.
- End-to-end fixtures cover a confirmed three-role deponent profile and a middle omission rendered as absent.

## Verification Notes

- RED: the new preview-count test failed because the claim sample/count lines
  were absent. GREEN: the six count/profile cases passed after retaining the
  current comparisons on the generation result and reporting their assessments.
- Mutation check: rendering absent roles as withheld made all four confirmed
  profile/recipe cases fail at the independently expected absent role label;
  restoring the renderer returned them to green. No renderer change is retained.
- Initial full chain: `uv run ruff check`, `uv run mypy`, `uv run pytest -v`
  passed (89 typed source files; 907 tests). Targeted command/relationship
  modules passed 47 tests; `git diff --check` passed.
- Dedicated code-simplifier Task loaded its skill and made no edits. Separate
  independent General and Python Tasks loaded code-reviewer and their mapped
  guidance (Python also loaded python-patterns). Each returned verbatim:
  "No concrete task-relevant findings." Fresh candidate validation confirmed
  no candidate IDs and no missed acceptance bug. No findings were deferred.
- General covered domain acceptance; Python covered data ownership/typing.
  Security and database lanes were omitted because no trust boundary,
  persistence, schema, or destination-write behavior changed.
- Installed `uv run latinitas-cards preview` on saved confirmed synthetic
  profiles at limit 0: three-role deponent sample reported zero assessments,
  0/3 withheld unassessed roles and six cards; middle-omission sample reported
  zero assessments, 1/3 withheld unassessed roles and four cards. The latter
  retained one generated-warning membership and zero wholly skipped entries.
- Chromium DOM checks and inspected full-page 2x captures covered both recipe
  answers for both profiles: absent unnamed fourth role for sequor; absent
  perfect and withheld supine for dico. No shifted forms or invented claims.
  This is browser fixture evidence, not new native Anki compatibility evidence.

## Implementation Notes

- The existing comparison owner still rechecks candidate, extraction, profile,
  identity and decisions. Generation retains those same typed comparisons for
  generated objects; preview counts assessment records, not unique facts,
  source extraction reviews, displayed assertions, or eligible cards.
- Counts cover the full generated sample even when limit is 0 or 1. Tests use
  independently authored 1/4 accepted and 3/4 withheld assessment expectations
  (accepted, unreviewed, stale and explicitly withheld), ignore an unrelated
  candidate, and separately expect 2/8 withheld roles without assessments.
- Saved three-role profiles preserve the multiword deponent example in its
  declared source slot, not an inferred voice/person analysis. Markup-only
  middle perfect omission remains absent and does not shift the supine.
- Note/card identities, slots, recipes, eligibility, warning/skipped membership,
  comparison role names, claim policy and calibration remain unchanged.
- 2026-10-08T00:59:58Z: verification pass
- 2026-10-08T00:59:59Z: Final exact ruff/mypy/pytest chain passed (907 tests); fresh disposition-verification Task returned no concrete task-relevant findings. Installed CLI and browser fixture checks passed. One task cycle only.
