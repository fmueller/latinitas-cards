---
id: T-051-establish-claim-level-evidence-and-calibrated
title: Establish claim-level evidence and calibrated review gates
status: completed
priority: high
spec_ref: specs/v0.2.0.md#calibrated-form-parsing
dependencies:
    - T-049-establish-reviewed-source-extraction-fixtures-and
updated_at: "2026-10-05T01:26:00Z"
---

# T-051-establish-claim-level-evidence-and-calibrated Establish claim-level evidence and calibrated review gates

## Description

Define and implement the precision-first claim/review contract independently of extraction and display styling. Establish reviewed domain-relevant evaluation before automatic acceptance.

## Acceptance

- Represent individual grammatical labels, segmentation, and explanation claims with evidence/provenance, alternatives, acceptance or withholding status, and reasons. A whole-analysis confidence score is insufficient.
- Establish a reviewed evaluation sample, independently authored expected claims, measurable precision, and explicit project-policy thresholds; document applicability and limitations. Do not invent threshold values without evaluation.
- Declare development and evaluation material separately or explicitly disclose overlap. Report accepted correct/error claims, accepted coverage, and withheld cases per relevant claim/analyzer category, with sample limits for each threshold decision. Extraction counts and principal-part results do not calibrate later token parsing.
- Evaluate automatic claims by relevant claim/analyzer category. Until evidence supports a threshold, withhold automatic acceptance and require explicit review.
- Expose a review path for accepting or withholding individual claims with recorded decisions bound to the specific candidate/claim and evidence/profile version. Changed evidence or configuration requires revalidation or renewed review, not silent reuse. A setup-proposed fourth-role label is not an explicitly reviewed PPP/supine choice. Knowledge acceptance does not approve destination writes.
- Analyzer output, LLM self-confidence, successful extraction, and unreviewed examples never establish calibration. Optional LLM use remains network-explicit; core operation is offline.
- Tests distinguish accepted, rejected, ambiguous, and unsupported claims and ensure unsupported segmentation/explanations cannot leak into asserted knowledge.

## Verification

Publish the evaluation results and policy decision. For code, use red/green tests and run the mandatory ruff, mypy, pytest -v chain.

## Implementation and workflow evidence

- Scope: offline `claim_review.py` domain API, claim/review and evaluation tests,
  18 independently authored synthetic expected proposals, and
  `docs/claim-review-policy.md`. No analyzer, recipe, renderer, CLI or destination
  behavior was broadened. T-052 owns supported analysis; T-053 owns rendering.
- Baseline: fetched origin/main contained accepted T-048–T-050 commit
  `38788b402c95f181396b78af0f856fdc73dfca51`; clean checkout at that commit.
  Validated v0.2.0 selector/dependency before start and again before finalization.
- RED: claim-review collection failed with missing `latinitas_cards.claim_review`;
  evaluation collection failed with missing `summarize_evaluation`. Candidate-text
  invalidation test then failed with unexpected `candidate_text` keyword.
  GREEN: per-claim binding and summary implemented; final focused claim/evaluation
  and unchanged source-extraction regressions: 43 passed.
- Bindings include exact candidate ID/text, kind/value, alternatives, analyzer,
  raw extraction, evidence reference/text/version and actual profile/version.
  Individual explicit reviews do not accept siblings, alternatives, unsupported
  claims, setup-proposed roles or destination writes. Changed bindings require
  renewed review; asserted knowledge rechecks decisions.
- Policy: all automatic acceptance withheld. Synthetic overlapping evaluation is
  not held out, not Latin-expert approval and not real analyzer accuracy. Each
  claim category has six cases, two accepted-correct manual replays, zero accepted
  errors, coverage 2/6 and four withheld; automatic coverage 0/6 and precision
  undefined. Category-specific withheld strata are published in the policy.
  No numeric threshold is justified; production analyzers have zero observations.
  Extraction/principal-part counts never calibrate later token parsing.
- Initial exact chain: ruff passed; mypy passed for 70 files; pytest 675 passed.
  Earlier import-order and heterogeneous test-parameter typing failures were fixed
  and the full chain rerun from ruff. No failure was suppressed.
- Dedicated code-simplifier Task loaded its skill: no safe simplification; no edits;
  16 claim/evaluation tests and scoped ruff passed.
- Separate parallel independent code-reviewer Tasks: General loaded code-reviewer;
  Python loaded python-reviewer plus python-patterns; Machine Learning loaded
  mle-reviewer plus mle-workflow. Python and ML: "No concrete task-relevant
  findings." General independently adjudicated fixtures/policy and found T051-1.
  Security omitted: no new external ingestion, authentication, network or write
  boundary. Database/framework lanes omitted: no DB/schema/framework changes.
- Fresh candidate-validation Task upheld T051-1 verbatim: "These two deponent
  proposals are contradicted by the stipulated facts, so classify them as
  `rejected` rather than `unsupported`." Evidence: evaluation JSONL rows 6/17.
  Impact verbatim: "The replay’s withheld-category counts misstate how many claims
  are unsupported versus explicitly wrong. Both still remain withheld, so this
  does not affect the zero-auto-acceptance gate or the reported accepted-error
  count." No rejected or duplicated candidates.
- Disposition: fixed both judgments and category-specific test/policy counts.
  Regression RED: evaluation test 1 failed, 1 passed, `unsupported != rejected`.
  GREEN: 43 focused regressions passed. Fresh disposition-verification Task:
  "F=T051-1 — RESOLVED." and "New task-related findings: None." It independently
  checked judgments/table and reran 16 claim/evaluation tests. One review/fix cycle.
- Final exact chain: `uv run ruff check` passed; `uv run mypy` clean for 70 files;
  `uv run pytest -v`: 675 passed in 10.42s. `git diff --check` passed.
  Offline API smoke: automatic withheld, explicit individual review accepted,
  one asserted claim, destination approval absent. No visual UI changed.
- Remaining limits: no numeric calibration threshold, persisted review UI or
  authenticated reviewer identity. The API requires actual current snapshots.
  Existing T-052/T-053 own integration; no additional task implemented here.

## Implementation Notes

- 2026-10-05T01:26:00Z: verification pass
- 2026-10-05T01:26:00Z: Implemented and reviewed offline claim contract and bounded evaluation policy; final ruff/mypy/pytest chain 675 passed. T-052/T-053 remain separate.
