---
id: T-247-enforce-explicit-per-card-exercise-and-nested
title: Enforce explicit per-card exercise and nested-output contracts
status: completed
priority: high
spec_ref: specs/v0.5.0.md#cli-orchestration-and-provider-boundaries
dependencies:
    - T-246-record-approved-post-trial-contracts-and-task
updated_at: "2026-11-27T21:32:29Z"
---

# T-247-enforce-explicit-per-card-exercise-and-nested Enforce explicit per-card exercise and nested-output contracts

## Description

F1 of the approved post-trial package, reviewed revision 4 (revision 5 records the
unchanged receipt). Also implements the shared-evidence and outcome-report headings.
Completed T241–T245 are baseline behavior, not work to recreate. The coordinator's
T246 documentation-only commit discharges F1's spec-commit prerequisite; record its
commit and spec SHA256 before implementation. See docs/v0.5.0-post-trial-plan.md.

Own task construction in pipeline_orchestration and existing pipeline_consumers/
provider value contracts, not a new approval layer. Keep syntax recovery and response
schema redesign out of scope; reuse existing Pydantic models and strict validators.

## Acceptance

- Audit the final requests for all 17 saved reservations through the private trial
  coordinator's custody. Measure instruction/schema/objective/source bytes. Confirm
  one card, fresh messages, full required source/sense evidence and proposal-free
  checking; do not assume batching is broken. If only a sanitized inventory is
  available to the worker, record that boundary rather than claim raw-file access.
- Supply explicit form direction and only its applicable presentation template.
  Remove duplicate ANSWER_CONTRACT instructions only where safe. Do not trim required
  evidence, assertions, alternatives or objections. Phrase direction/rendering is
  unchanged; retain a multi-binding phrase control and English compatibility.
- Define concrete stem/quantity conventions using an example outside the selected
  ten, aligned with the installed Bennett-backed checker. Enforce required feature/
  stem lines and literal concatenation with quantities preserved, without claiming
  arbitrary explanation prose is mechanically verified.
- Validate nested assertion representations as well as the outer model. Malformed
  nested JSON is processing_failed/malformed_output; well-formed deterministic
  exercise violations are distinct contract-violation processing failures. Unsupported
  or disputed Latin stays withheld. Reject duplicate keys, nonfinite values, invalid
  control characters and malformed values without repair or answer-specific exceptions.
- Valid German production/recognition controls stay distinguishable and valid; wrong
  direction, missing lines, inconsistent concatenation and malformed nested values
  get their intended outcomes. A new synthetic counterexample proves rule-based
  behavior. Extend existing isolation tests only if needed, not a redundant matrix.
- Version prompts/checkers and exact objective/request commitments. Old results
  cannot authorize changed tasks or stale-permission dispatch. A public synthetic
  72fdf19-layout store stays readable with original interpretation or explicit stale
  status and unchanged evidence/charges; corruption remains rejected. Never commit
  private trial stores or copied model responses as fixtures.
- Version the report schema for the machine-readable contract-violation reason,
  provide a public reader compatibility control and preserve historical reports.
  Record before/after request sizes. Preserve human/curated approvals, disabled
  policies, German characterization goldens and all release gates.

## Verification Notes

Use implementing-changes: focused red/green, dedicated simplification, independent
conventional review and disposition verification. Run final uv run ruff check →
uv run mypy → uv run pytest -v before Taskrail verify/complete, and record tests
added/removed and runtime. No hosted calls, private collection edits, Anki import,
dependency migration or policy enablement. Commit/push main after local checks;
do not wait for CI. Parent owns successor selection and final CI supervision.

## Implementation Notes

- Base: origin/main fdb0a2724ce9aaf6b0d3590bf03f0c3953dcb3d6, containing the T246
  documentation commit 3147ad29bf11d776cd2b3243d67a14bb90359acf. Spec SHA256
  e7922ded1924bc79019a58493d1e767d198fcb396b7eed1a81f3f48226c461be (recorded
  before implementation).
- Audit boundary: the worker had only the coordinator's sanitized numeric/boolean
  inventory of the 17 saved requests, not raw private files. It reports one task,
  fresh system/user messages, identical two-call user content, the answer contract
  duplicated in the objective and no explicit direction key; request sizes were
  12,238-12,941 bytes. Linguistic evidence sufficiency was not audited from raw text.
- Form tasks now carry `direction` and only that direction's templates, with the
  feature display and context pre-filled. The objective no longer repeats the answer
  contract; chat requests keep it in the system message and `latinitas-json/1`
  requests carry it once as `instruction`. Phrase objectives only lose the duplicate.
  Prompts are versioned proposal/2 and blind/2.
- Public synthetic request sizes before and after (system message unchanged at
  2,753 bytes): verb production 11,817 -> 10,528; noun declension 13,136 -> 11,808;
  recognition 12,644 -> 10,923; translation 12,721 -> 10,913; cloze 12,681 -> 10,807.
  Before/after saved-request sizes for the private 17 need a custody-side check.
- The stem contract names Bennett conventions with public examples (vocare and
  porta), which the coordinator confirmed are outside the selected ten.
- The provider reader validates the documented nested values: malformed JSON,
  duplicate keys, nonfinite numbers, disallowed control characters, wrong shapes and
  pathological nesting are `malformed_output`. A well-formed answer that breaks
  direction/template, stem line, literal concatenation or the target surface is
  `contract_violation`. Neither reason repairs or retries, and Latin correctness
  and prose are not checked mechanically.
- The pipeline report is now schema 3. Historical 1/2 reports stay readable and
  cannot claim `contract_violation`.
- Acceptance correction (formerly deferred as T251 scope): the stem convention
  and ending role now follow the derivation rule, so PPP-anchored adjective cards
  claim bounded-generator-base/1 with a case ending. Reconciling a well-formed
  contract-violating response records a charged `contract_violation` failure with
  its response retained; malformed retrieved bytes still raise unrecorded. The
  historical-store control now sends the fdb0a27 objective layout; a scratch probe
  against that commit's code matched its objective bytes (only the implementation
  digest in `config_sha256` differs).
- 2026-11-27T20:13:00Z: verification pass
- 2026-11-27T21:32:29Z: verification pass
