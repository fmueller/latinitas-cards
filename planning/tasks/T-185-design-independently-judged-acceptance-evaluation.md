---
id: T-185-design-independently-judged-acceptance-evaluation
title: Design independently judged acceptance evaluation
status: blocked
priority: high
spec_ref: specs/v0.5.0.md#domain-specific-confidence-calibration
dependencies:
    - T-184-design-v0-5-0-acceptance-and-pipeline-contracts
updated_at: "2026-11-13T14:32:46Z"
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
- 2026-11-12T22:35:09Z: Maintainer directs independent agent judges; reopen only the bounded protocol reconciliation, not evidence completion or T-186 approval.
- Agent-judge reconciliation at base
  0bddc4a6037e4eef93c0c2018a885072a84ee96c updates the protocol to SHA256
  f5f8b287b653decdcb857ab5a9c381353eac03f28b100c39392b34a68d018df1.
  The maintainer's direction resolves judge type: independent agent judges
  may own references; human-only reference judging is not required by the
  spec. Earlier missing-owner notes describe the earlier snapshot, not a
  new human-only prerequisite. Task acceptance and spec/contract are unchanged.
- Concrete logical roles are REF-A (reference owner), REF-B (independent
  solve), ADJ (independent solve then evidence-based adjudication), with
  overseer holdout custody. Actual run binding/competence remains pending.
  Agent research records do not become legacy human decisions or curated
  content approvals. Model agreement and self-confidence are not truth.
- Section 7 supplies packet/record fields and judge/adjudication prompts,
  source/target-language competence checks and protected-store requirements.
  Proposed development bounds are one qualification packet of at most eight
  tasks per judge and one feasibility packet of at most twelve source
  clusters, not statistical minima or approved thresholds. The overseer
  owns high-effort judging launches after review; none were launched here.
- Dedicated simplifier clarified the reconciliation. Medium-risk independent
  conventional review used general code-review, comment-audit/prompt and
  custody lanes plus a fresh adversarial verifier, relayed by this agent
  because the coordinator could not launch nested agents. One non-blocking
  finding was fixed: fresh-context dispatch envelopes now explicitly carry
  record fields, run metadata, scope defaulting to none, retained hashes
  and missing-field rejection. All lanes and coordinator verified the fix;
  no remaining findings. No claimed judge behavior was tested.
- Final documentation checks on this hash: `git diff --check`, Taskrail
  validation via `mise exec --`, local links/anchor and contract-hash
  correspondence passed; `uv run pytest -q tests/unit/install_docs_test.py`
  passed (3 tests of existing installation docs, not this protocol).
  No runtime inputs changed; full ruff/mypy/pytest gate not run.
- Acceptance still lacks actual source-cleared packets, evidence-backed
  Latin/German/English agent qualification, independent judgments and owner
  review, provisioned protected custody, measured pilot/acquisition evidence
  and justified numerical criteria. Record a new failed verification and
  block this revision. T-186 exact-revision approval, curated human approvals
  and T-205 publication authority remain unchanged. No duplicate task or
  successor was created; overseer should run the bounded section 7 handoff.
- 2026-11-12T22:44:01Z: verification fail
- 2026-11-12T22:44:01Z: Agent judge type resolved; still need actual run bindings and source-backed Latin/German/English qualification, cleared source packets, independent judgments and owner review, enforced protected custody, measured bounded pilot and justified numerical criteria. Overseer owns section 7 launches; T-186 exact-revision approval remains blocked.
- 2026-11-12T23:43:37Z: Authorized answer-free qualification evidence update only; no new packet, retry, consumer or T-186 approval.
- Qualification evidence update at base
  43bd6ecc52a7f192cb703b972dfd252739198334: protocol SHA256
  6aee6794b870294eb867b385ac792107e06f76b9ada9115c9e6784958dcff3c5.
  Section 8 supersedes earlier no-run statements for this later stage,
  preserving the design/reconciliation history and unchanged contract hash.
- Only two authorized answer-free records were downloaded from ADJ custody:
  qualification report SHA256
  9eb373da3f65006c3c3dbfbd6850cbb03c00ce07b12028505b28e1c466e3f266
  and freeze receipt SHA256
  b2648478cbc65f4d555748c0186b0c50d5592343ada2a828f95210fd40875f05.
  Both matched expected bytes. No answers, source/task mappings or other
  judge-workspace files were transferred. Answer-bearing references remain
  isolated; this update records reported evidence, not a new adjudication.
- Actual three requested-high-effort runs and source-based adjudication
  support bounded tested slices only. Eight purposive probes in three
  related clusters are not independent samples. The report audits 84
  explicit propositions and ancillary prose across 24 task records;
  104 citation definitions match excerpts, not proof of entailment.
  Two explicit corrections and one additional ancillary finding retain
  separate denominators (A 0/B 2/ADJ 1 total bounded findings).
  Four unclosed evidence scopes and four final scoped-withholding records
  remain; native abstention counts are not comparable rates. Same-model/
  common-source dependence and missing provider telemetry bar reliability
  claims. Original acquisition hashes cannot be rechecked from excerpts.
- Full language/claim qualification and the conditional section-7 gate are
  not established. ADJ lacked complete gate/cap text; its supported-slice
  suggestion grants no authority. No pilot, candidate output, accepted-yield
  measurement, protected heldout evaluation or numerical policy was run or
  approved. No new packet, retry, successor or duplicate task was launched.
  Recommend retaining blocked scope pending a concrete maintainer choice;
  any future supported-slice study needs an explicit scope/protocol amendment,
  source/correction dispositions and complete gate/cap context, not a bypass.
- Dedicated simplifier and medium-risk independent conventional review
  completed. General/code-review and comment-audit/evidence lanes plus a
  fresh adversarial verifier found no defects; coordinator reconciled
  their findings. Nested dispatch was unavailable, so lanes were relayed.
  No fixes required. Review validates safe-report correspondence, not
  underlying linguistic correctness or custody implementation.
- Final docs-only checks passed: `git diff --check`, Taskrail validation
  through `mise exec --`, links/anchor/hash correspondence, and
  `uv run pytest -q tests/unit/install_docs_test.py` (3 passed; existing
  installation docs only). Full Python gate not run for non-executable
  prose. Record failed/incomplete verification and block T-185; T-186/T-205
  and curated human-content approvals remain unchanged.
- 2026-11-12T23:50:38Z: verification fail
- 2026-11-12T23:50:38Z: Actual qualification supports tested slices, not complete language/claim cells; four evidence scopes unresolved. Section-7 gate unestablished; ADJ lacked complete gate/cap text. No feasibility/yield or justified numerical policy. Retain blocked pending maintainer decision; any supported-slice study needs explicit amended scope, no packet/retry authorized.
- 2026-11-13T14:18:24Z: Maintainer authorizes one bounded development study capped at twelve source clusters without waiving T-186; reopen for prerequisite protocol amendment only, no study launch.
- Section-9 amendment at accepted base
  c7da7258daa1bc2944b6c6f1f8963a5680802c5b records that authorization as
  an exception to the old complete-qualification prerequisite for one
  development study only. Reviewed protocol SHA256
  6bc9cf35d4e55cf19c6ca03892c91b135aad8abe6038bf097c5883893148f259;
  unchanged contract SHA256
  aef0af7771c125998b4e3d8e3ae5cdb72d23823ab9a150ed24a51d27fe1bc27b.
  Historical qualification findings, full acceptance, spec and human curated
  approval/publication authority remain unchanged. No new task or successor.
- Dispatch is limited to one frozen development packet: at most 12 source
  clusters, two tasks per cluster, 24 tasks total, four requested atomic
  propositions per task, one initial solve per role/task then one ADJ
  comparison per task. At most 72 initial and 24 adjudication records;
  32 KiB per task record and 64 KiB for the answer-free report. Failures
  consume slots; no replacement, repeat qualification or repair campaign.
  These cap study records, not provider requests, tokens, cost or runtime.
- Affirmative reference scope is the demonstrated three-role intersection:
  bounded German labels; bounded English labels, declared-convention
  segmentation and explanations. A-only German explanations and English
  translations remain audit/withheld at final-reference level; unresolved
  cells and the four open scopes are not promoted. Isolated source-backed
  errata must retain B's meaning omission, B/ADJ's support extensions and
  ADJ's insufficient alternative/metadata issue without overwriting solves
  or claiming false Latin, requalification or newly supported synonyms.
- Overseer must review this exact amendment and satisfy the complete gate:
  freeze permitted sources/rights, manifest/caps/scope, supply actual gate
  text and record fields, enforce isolated custody and bind fresh distinct
  requested-high-effort A/B/ADJ runs. Freeze every initial solve before
  comparison and any generator exposure. This thread has prepared no sources
  and launched no judges. The next study may measure reference yield,
  citation/entailment failures, disagreement/abstention, acquisition/custody
  feasibility and exposed elapsed/cost only. Candidate/card/accepted yield
  remains unmeasured. No statistical release inference or numeric approval.
- Dedicated simplifier completed. High-risk independent conventional review
  used general/code-review, comment-audit, custody, measurement and separate
  request-claims lanes, relayed because nested dispatch is unavailable.
  Fine-grained slice descriptions and corrections were checked against the
  permitted hash-matched answer-free report, not answer content. Fresh
  authority/custody and bounds/measurement adversarial verifiers sustained
  the documentary conclusions; no substantive findings or protocol fixes.
  Temporary in-progress versus final blocked wording is resolved by the
  supported final fail/block lifecycle and readback, not policy changes.
  Review does not validate linguistic truth, actual rights or enforcement.
- Documentation checks: `git diff --check`, Taskrail validation, protocol/
  contract hashes and local links/anchor; installation-doc tests remain
  three passing tests, not evaluation-methodology validation. Full Python
  lint/type/test chain is not required for this non-executable prose scope.
  Full T-185 acceptance still lacks study evidence and justified numerical
  acquisition/sample/error/uncertainty criteria; preserve failed verification
  and block. T-186 exact-revision approval and T-205 remain blocked. The
  overseer owns the gated study launch after this amendment handoff.
- 2026-11-13T14:32:46Z: verification fail
- 2026-11-13T14:32:46Z: Authorized section-9 study not yet executed; reference/acquisition/custody feasibility and justified numerical sample/error/uncertainty criteria remain incomplete. Overseer owns gated fresh high-effort study launch; no qualification pass, consumer permission or T-186/T-205 waiver.
