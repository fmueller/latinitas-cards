---
id: T-128-bound-dense-usfx-markup-retention-without-dropping
title: Bound dense USFX markup retention without dropping required source metadata
status: todo
priority: medium
spec_ref: specs/v0.8.0.md#bounded-memory-dense-usfx-markup-retention
dependencies: []
updated_at: "2026-10-16T07:52:40Z"
---

# T-128-bound-dense-usfx-markup-retention-without-dropping Bound dense USFX markup retention without dropping required source metadata

## Description

Future-hardening follow-up from the T-110 disposition-verification review (S-2),
under `specs/v0.8.0.md#bounded-memory-dense-usfx-markup-retention`.
Retaining one Pydantic markup model per source element still permits large
memory amplification after sparse span mapping removed the independent
per-character allocation problem. Bound the complete normalization path from
passage construction through validation, serialization, and publication;
parser-only allocation reductions do not satisfy this follow-up.

v0.8.0 already owns bounded-memory performance hardening. v0.4.0 introduces
corpus features and v0.9.0 validates release contracts, so neither is the
implementation home for this task. Keep v0.3.0 active and unchanged; this task
must not alter its priorities, dependencies, acceptance, or release gates.

The 2026-10-16 reproduction used 50,000 empty `<w/>` elements and one text
character: 200,053 source bytes, 50,000 preserved entries, and 51,961,856 bytes
of peak-RSS growth. Normalization completed successfully; no OOM, crash or
specified total-memory budget breach was demonstrated. Fresh independent
contract validation rejected this as a current v0.3.0 must-fix violation.
It is future hardening, not a T-110 completion gate and not a fixed issue.

## Acceptance

- Before implementation, commit the spec-required benchmark protocol and
  numerical budgets for absolute peak RSS, baseline-adjusted RSS growth,
  elapsed time, and temporary disk use on a named reference environment.
  Record versions, fresh-process baselines, repeated-run variation, units,
  and fixture sizes. Budgets are prerequisites, not measurements already
  established by S-2. Reproduce the baseline on the accepted implementation.
- Meet a documented end-to-end memory envelope through final publication,
  including builder, validation and canonical-writer peaks. At fixed text
  and depth, working memory stays within a fixed budget as metadata counts
  grow; separately account for text, depth, indexes and bounded buffers.
  Compact/spooled representations are options, not a mandated design;
  reducing only the per-entry constant does not establish bounded retention.
- Run fresh-process before/after resource and correctness checks at 5,000,
  50,000 and 500,000 entries where supported input limits permit. Cover empty
  elements, meaningful nonempty `w` attributes/text, nested `wj`, flattened
  notes/headings, and allowed-unknown metadata. Exercise one dense passage,
  many passages, long sparse-markup text, and the bundled Clementine control.
  Distinguish fixed-text cardinality from growing meaningful payload lanes.
- Preserve all required entries, order, attributes, canonical spans and text,
  note/heading attachments, identity, hashes, accepted aliases, deterministic
  output, and legacy projection goldens. Check independent counts and literal
  values/spans alongside canonical-byte comparisons and corpus validation.
- Document and test supported boundaries on both sides of the 64 MiB input
  and depth-64 limits, plus declared spool/disk boundaries. Do not silently
  cap elements, drop metadata, or reject previously valid dense inputs.
  Any compatibility restriction needs an explicit spec decision and
  maintainer review; do not amend the pinned active spec.
- Preserve diagnostics, exit codes, preflight validation and failure/no-write
  semantics for late malformed/unknown XML, validation failures, exhausted
  temporary storage, and interruption. Existing destinations remain unchanged
  on failure and partial output is never published. Verify temporary-file
  cleanup/recovery, including abrupt interruption, without weakening source
  immutability or atomic publication.
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
- Refined future acceptance and repointed through `taskrail task repoint` at
  2026-10-16T07:52:40Z to the dedicated dense-retention heading. Taskrail
  validated the reference; the generated STATE change only refreshes its
  timestamp. This is planning, not implementation verification or completion.

## Implementation Notes

- Existing task dependencies and priorities were not edited.
- This task preserves the original T-128 ID and S-2 evidence; it supersedes
  the initial active-release blocker interpretation, not the observed risk.
  Resource budgets/reference environment and representation design remain
  explicit future decisions. No implementation is claimed by this refinement.
- T-127 already owns the registered schema exporter repair; do not duplicate it.
