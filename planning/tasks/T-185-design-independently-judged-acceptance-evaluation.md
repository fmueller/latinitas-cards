---
id: T-185-design-independently-judged-acceptance-evaluation
title: Design independently judged acceptance evaluation
status: blocked
priority: high
spec_ref: specs/v0.5.0.md#domain-specific-confidence-calibration
dependencies:
    - T-184-design-v0-5-0-acceptance-and-pipeline-contracts
updated_at: "2026-11-12T22:10:22Z"
---

# T-185-design-independently-judged-acceptance-evaluation Design independently judged acceptance evaluation

## Description

P50-1 evaluation-design draft using the contract draft's claim classes. Own the
proposed protocol and feasibility evidence; do not approve numeric policy or
implement consumers. Follow the spec's execution-isolation and validation rules.

## Acceptance

- Identify an independent Latin reference-judgment owner, permissible sources and
  access. Include classical/biblical and varied supported inputs, claim types,
  recipes, ambiguous cases and counterexamples without a processing whitelist.
- Define development/held-out separation, adjudication, units/denominators for
  accepted errors, precision, coverage, withheld cases and processing failures,
  plus uncertainty reporting. Zero accepted items is not evaluable.
- Propose justified minimum sample sizes and explicit pass/fail criteria by stratum
  with source or measured pilot evidence. Account for correlated library/model
  errors; agreement is not truth. Do not tune against held-out answers.
- Define handling of insufficient evidence and failed/inconclusive strata, narrower
  coverage or disabled automatic paths, including approval and evidence refresh.
- Limit work to one documented feasibility/protocol pass. Missing independent
  judgments, data permission or justified thresholds blocks the maintainer gate,
  rather than prompting invented evidence or indefinite research.

## Verification Notes

Review provenance, rationale, holdout separation and competing wrong answers with
the reference owner. Maintainer approval must cover this exact protocol revision;
neither dependent implementation nor release readiness is established.

## Implementation Notes

- Bounded feasibility pass at base ac6ff36235d92c7f4fc71ce05ee180da92724553:
  `docs/v0.5.0-evaluation-protocol.md`, SHA256
  d6789ff4811bd11e13c4292f14b39556b35e10072be7a2138cb215cae3d6cd8c.
  Contract proposal pinned to SHA256
  aef0af7771c125998b4e3d8e3ae5cdb72d23823ab9a150ed24a51d27fe1bc27b.
- Draft covers reference roles and access inventory, classical/biblical and
  varied input/claim/recipe strata, independent adjudication, clustered
  development/holdout separation, deployment versus challenge panels,
  assertion/task/card/source denominators, zero-acceptance handling,
  conditional statistical sizing, failed/insufficient results and refresh.
- Acceptance remains incomplete: no named independent reference owner,
  confirmed evaluation rights/access, independent judgments/owner review,
  protected holdout or measured pilot. NIST exact-binomial methodology
  supports a conditional sizing formula, not Latin-specific risk limits,
  numerical minima or acquisition feasibility. Those cannot be invented.
  T-186 must resolve exact-revision policy and prerequisites; T-200 owns
  later execution. Existing tasks cover the missing prerequisites, so no
  duplicate or post-release follow-up was created. T-179 is unchanged;
  T-186/T-205 remain blocked. No consumer implementation is authorized.
- Dedicated code-simplifier clarified strata, zero denominators and missing
  evidence without changing the protocol. Independent conventional-review
  used medium-risk coverage with separately dispatched code-review and
  comment-audit lanes, plus a fresh adversarial overall-outcome verifier;
  the coordinator could not spawn nested agents, so evidence was relayed.
  No verified defects or corrective edits remained. Verified challenges
  covered conditional-precision concealment, selection/mixed-card errors,
  correlated pseudo-replication, holdout retries and lifecycle overclaiming.
  All reviewers supported failed/incomplete verification and blocking this
  task; structural validity does not meet its owner/criteria acceptance.
- Documentation-only checks passed: `git diff --check`,
  `mise exec -- taskrail validate`, local document links/anchor and SHA256
  binding, and `uv run pytest -q tests/unit/install_docs_test.py` (3 passed).
  The Python tests cover existing installation documentation, not this
  protocol's linguistic/statistical adequacy. No runtime changes or new
  executable examples; full ruff/mypy/pytest chain is not required for this
  scope and was not run. No linguistic, provider, library, native Anki,
  release or German-card-video acceptance is claimed.
- 2026-11-12T22:10:22Z: verification fail
- 2026-11-12T22:10:22Z: Bounded feasibility pass lacks appointed independent Latin reference owner, confirmed evaluation rights/access and judgments, owner review, protected samples/pilot and justified numeric minima/thresholds; maintainer must resolve prerequisites and exact-revision T-186 policy before consumers.
