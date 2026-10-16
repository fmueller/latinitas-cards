---
id: T-128-bound-dense-usfx-markup-retention-without-dropping
title: Bound dense USFX markup retention without dropping required source metadata
status: todo
priority: medium
spec_ref: specs/v0.8.0.md#performance-and-accessibility-hardening
dependencies: []
updated_at: "2026-10-16T07:43:29Z"
---

# T-128-bound-dense-usfx-markup-retention-without-dropping Bound dense USFX markup retention without dropping required source metadata

## Description

Future-hardening follow-up from the T-110 disposition-verification review (S-2),
under `specs/v0.8.0.md#performance-and-accessibility-hardening`.
Retaining one Pydantic markup model per
source element still permits large memory amplification after sparse span
mapping removed the independent per-character allocation problem.

The 2026-10-16 reproduction used 50,000 empty `<w/>` elements and one text
character: 200,053 source bytes, 50,000 preserved entries, and 51,961,856 bytes
of peak-RSS growth. Normalization completed successfully; no OOM, crash or
specified total-memory budget breach was demonstrated. Fresh independent
contract validation rejected this as a current v0.3.0 must-fix violation.
It is future hardening, not a T-110 completion gate and not a fixed issue.

## Acceptance

- Establish a memory-bounded compact or spooled retention/materialization path
  for dense supported and allowed-unknown markup, preserving every required
  element, attribute and canonical span without changing text or hashes.
- Add subprocess regression coverage for dense markup and meaningful metadata,
  not only a large passage with a single entry; record before/after memory
  evidence and verify the complete observable output.
- Preserve preflight validation and no target writes on input error. Account
  for the existing builder, validation and atomic-writer boundaries before
  changing their interfaces; retain the legacy projection and goldens.
- Do not add arbitrary element-count limits or amend the pinned spec. If
  bounding retention requires a contract decision, stop for maintainer review.
- Pass the mandatory ruff/mypy/pytest chain and `mise run check`, dedicated
  simplification, independent General and Security review, finding disposition
  and fresh disposition verification before completing this future task.

## Verification Notes

- Created through `taskrail task new` at 2026-10-16T00:50:52Z. No implementation
  or lifecycle execution of this follow-up occurred in the T-110 cycle.
- Repointed through the supported CLI at 2026-10-16T07:43:29Z to the existing
  v0.8.0 heading, superseding its initial active-v0.3.0 blocker interpretation.
  Taskrail v0.4.0 has no cancellation command; no status or dependency was
  hand-edited. This off-spec todo task was never started.

## Implementation Notes

- Existing task dependencies and priorities were not edited.
- Future-spec planning is owned by
  https://ampcode.com/threads/T-01a143a7-979b-72d9-95b6-d8d6cd94b6c2.
  That thread will refine this draft acceptance and repoint this same task ID
  to its drafted `bounded-memory-dense-usfx-markup-retention` subheading only
  after that heading is delivered. Do not allocate a duplicate follow-up.
- T-127 already owns the registered schema exporter repair; do not duplicate it.
