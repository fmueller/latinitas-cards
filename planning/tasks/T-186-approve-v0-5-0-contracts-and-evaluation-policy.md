---
id: T-186-approve-v0-5-0-contracts-and-evaluation-policy
title: Approve v0.5.0 contracts and evaluation policy
status: completed
priority: high
spec_ref: specs/v0.5.0.md#design-approval-and-execution-isolation
dependencies:
    - T-184-design-v0-5-0-acceptance-and-pipeline-contracts
    - T-185-design-independently-judged-acceptance-evaluation
updated_at: "2026-11-17T04:30:33Z"
---

# T-186-approve-v0-5-0-contracts-and-evaluation-policy Approve v0.5.0 contracts and evaluation policy

## Description

Maintainer-only design checkpoint. Only the maintainer may unblock after reviewing
the exact contract/protocol revisions. This is separate from planning approval and
spec activation. No autonomous loop may approve or unblock it.

## Acceptance

- Record explicit maintainer adoption of the sequencing amendment and exact P50-1
  engineering/method revisions: schemas/upgrades, comparison/check contracts, finite
  limits, selection/batching, ownership/migration, curated approvals and handoffs.
  Use docs/v0.5.0-decision-package.md and the forward evaluation plan; retain this
  task's T-185 dependency and require design verification against adopted criteria.
- Resolve engineering choices with no enabled automatic policies. Missing empirical
  qualification, permitted acquisition, custody/output feasibility and justified
  numerical criteria remain explicit T-200 checkpoints, not waived evidence or
  an approval of automatic export. Preserve historical failures and exact revisions.
- Require policy denial at decision/reuse/export/handoff, isolated synthetic positive
  tests and no shipped test-policy authority. Neither stored success nor a model,
  profile or stage command can approve a policy. Existing human/curated gates remain.
- Record future native/library/provider/release prerequisites without treating them
  as satisfied or granting hosted-processing/publication permission.
- Reconcile downstream bodies with the approved contract. Changed policy requires
  refreshed affected review, reuse invalidation and maintainer acceptance before
  dependent consumers proceed.

## Verification Notes

Inspect signed-off revisions and unresolved-decision lists, not status alone.
Every implementation consumer depends transitively on this gate. The maintainer
separately authorized v0.5.0 activation on 2026-11-03; that activation does not
approve these contracts or unblock this gate.

## Implementation Notes

- 2026-11-02T23:10:47Z: maintainer design approval: accept exact contract and independently judged evaluation revisions before consumers
- 2026-11-17T04:21:21Z: Recording explicit maintainer authorization of 2026-11-17T03:49:34.787Z after accepted T-185 design verification at b5d56a04946225d3345c9b85d3b88a31ebc3c404; exact adoption 568f02cf491e6252e795e10474833bd6a137fc21; judges remain paused, no automatic policies enabled

## Separate maintainer engineering-approval record

### Human authority and satisfied condition

The maintainer, Felix Müller, explicitly instructed:

> Approve T-186 and resume the loop, after T-185 design verification; keep judge runs paused

This is the direct human message of 2026-11-17T03:49:34.787Z, message ID
`M-034ywj4MjMI6KZz87p3Eax`, in the
[overseer thread](https://ampcode.com/threads/T-01a1d1a5-1333-7120-9647-233e76eb0a44).
It answers the request for exact-package T-186 engineering sign-off conditional on
T-185 design verification. Earlier human content approval, “Approve the package,
keeping judge runs and implementation paused”, is message ID
`M-034ywSNT57KVtXcsmI4net` of 2026-11-17T03:38:37.046Z in that thread.
The later instruction supersedes the implementation pause, not the judge pause.
This worker records that human decision; it does not approve on agent authority.

The condition is satisfied at commit
`b5d56a04946225d3345c9b85d3b88a31ebc3c404`: T-185's separately appended
design-verification record maps all six amended design criteria, preserves the
failed empirical history and records supported verification/completion at
2026-11-17T04:06:50Z. The overseer's independent delivered-package structural
acceptance passed on that exact head, confirming eleven source identities,
historical preservation, six method criteria, dependency/state/link/bootstrap
consistency and no empirical success or enablement. These are design evidence,
not an inference from task status. T-184 and T-185 are completed dependencies;
the refreshed starting checkout had no active task writer and T-186 was blocked.

The approved sources deliberately retain preparatory “draft”, “proposed” and
“pending” wording. This separate human-approval record supersedes that wording
for adoption/engineering permission only; it does not rewrite the approved bytes,
claim that proposed commands exist, or replace historical results. The overseer
retains loop ownership and selects any later implementation worker. This worker
owns only T-186 and starts no successor.

### Immutable approved sources

Reviewed base: `57bd835de4995bff73486500839eb02754a5429f`.
Exact eleven-file adoption: `568f02cf491e6252e795e10474833bd6a137fc21`.
Reconciled recording base: `b5d56a04946225d3345c9b85d3b88a31ebc3c404`.
Approved source manifest SHA256:
`b636f8192b6a4817ca7517acd2c1c7299da487ffe5eebd7e18fbaa69762de0f8`.
Approved source archive SHA256:
`9c8b51500ffbf600f634b80fa625d155abfddcbfbcd748befbd07533957fb22a`.
The manifest/archive identities are retained from the T-185 adoption record;
each file digest below was independently recomputed from the adoption commit.
Task digests identify the adopted body before later recording/lifecycle deltas.

| Path at adoption commit | SHA256 of exact bytes |
| --- | --- |
| `specs/v0.5.0.md` | `fbbea23eabd998d0dee79c1582a76834da1716c145a4c223334272eef51a1f6a` |
| `docs/v0.5.0-contract-proposal.md` | `c216cc31283eb1a3cfbcd90809f3166b68bdf564f4a081e498a391176d010f08` |
| `docs/v0.5.0-decision-package.md` | `0c49ad4a5bd51066a8e8737d6ec305510028fcfe7a9d43421808b16aaf0562b8` |
| `docs/v0.5.0-evaluation-plan.md` | `9c6d626b385ed031a4322511168d65382d519253e96f134d5e5ec0726ec3a31a` |
| `planning/tasks/T-185-design-independently-judged-acceptance-evaluation.md` | `c6a784e149b5724988ac7f4864110f0f7638bb15d4f722766de91d90e15e4aae` |
| `planning/tasks/T-186-approve-v0-5-0-contracts-and-evaluation-policy.md` | `4704d04eac0717a03b7ea8b79b68bc813ea3934068c7a064b3c5ad54ec3a8f55` |
| `planning/tasks/T-187-implement-versioned-automatic-decisions-and-reuse.md` | `4c9c4f6f77b5a5e8a5aaeeb3e1a83de9084272b44763f1d6c032e2b65db61ce5` |
| `planning/tasks/T-190-gate-complete-card-assertions-on-accepted-evidence.md` | `f7822f3c6ea450117655c2e61b438a6f574c3a487545269a8af6ad49fb41b7af` |
| `planning/tasks/T-199-document-v0-5-0-workflows-and-reconcile-cli.md` | `3570402260f6121f9e3723fa4a8efadd3d0f6909ef77f6bdf8078e3401bc58e4` |
| `planning/tasks/T-200-evaluate-held-out-automatic-acceptance-against.md` | `90417ec8ecf2774f981787b5a228162632a150cc323452e8d435a944633578c1` |
| `planning/tasks/T-204-check-v0-5-0-release-readiness-without-publication.md` | `e8d1e804076826b8d37b5f1f1d7b3a34e9d397abc0d1ba25f48182d4663b5829` |

### Six accepted engineering decisions

The human approval adopts the sequencing amendment and all six recommendations
in the exact decision package, including its finite-support/comparison rules.
No new policy design or amendment is made here.

1. **Commands, storage and compatibility:** accept `pipeline`/`curated`, strict
   contract sections 1–3 shapes, separate `latinitas-acceptance` v1 storage and
   explicit phrase v2 adapters. Preserve historical strict readers, human origins,
   confirmed PPP/supine roles and German goldens. Initial finite support uses the
   existing noun/adjective/verb vocabulary and four controlled families; bounded,
   evidenced target derivation is allowed, not invented source forms. T-187 owns
   bindings, T-188 applicability and T-191 enumeration. Implemented registry
   revisions must be frozen before T-200 strata; representability is not enablement.
2. **Authority and evidence:** accept contract section 4 and the forward method,
   with an empty enabled automatic-policy scope, including library-only paths.
   Morphology compares canonical features, segmentation uses cited conventions
   plus reconstruction, translation uses supported propositions/alternatives and
   explanations require clause-level entailment. Missing support stays withheld.
   Maintainer-reviewed application authority is rechecked at decision, reuse,
   export and handoff; no stored success, stage, model, profile or input registry
   grants it. Bootstrap `v050-disabled` revision `1` remains disabled,
   authorization=null, digest
   `9958ecce67297078d192aff579e352a07ee2c020d9045b60c44c3b29faffd538`.
   All-checks-pass still yields automatic/withheld/`policy_not_enabled`.
   Positive automatic vectors use isolated synthetic application fixtures only,
   never shipped enabled or counted as linguistic evidence. Human/curated gates
   retain their separate bindings and authority.
3. **Operational ceilings:** accept 60 seconds/request, one retry, 200 total
   attempts, 200000 input-plus-reserved-output tokens/run, at most 2048 reserved
   output tokens/attempt and one in-flight call. Conservative bounds, durable
   spending and explicit unknown-call reconciliation remain mandatory. These
   are termination/cost ceilings, not numerical reliability criteria or permission
   to reopen any exhausted allocation. Exhaustion defers work; no blind retry.
4. **Selection and batching:** accept section 5 ordering, versioned alternative
   ranking, CardRef deduplication, semantic suppression, limit precedence,
   receipt-bound replay and run locking. Default one variant/binding and no
   output-card quota; processing ceilings do not silently truncate eligible
   export. Changed snapshots need reconciliation; replay does not advance.
5. **Layouts, ownership and lifecycle:** accept section 7 opt-in `acceptance-1`
   fields (Acceptance Data, Evidence Summary, Acceptance Receipt), answer-only
   evidence, append-only slots, receipt-owned namespaces and explicit adoption.
   Existing identities/ordinals and user fields/tags/history stay preserved.
   Tag-writing updates require fresh destination reconciliation or explicit
   manual preservation. Manual/managed retained-field handoffs do not authorize
   direct integration, template replacement, CSV retirement/suspension or a
   claim that export applied anything.
6. **Curated approvals:** accept section 8 exact rule/mnemonic/rendered-card
   fingerprints, explicit supersession and language-specific approvals. Rule-only
   cards need no mnemonic. Any displayed mnemonic needs its own approved language
   and rule association; automatic checks cannot approve curated content.
   Rule/translation/template changes invalidate affected approvals, not unrelated
   human history.

### Acceptance mapping and consumer reconciliation

| T-186 acceptance | Recorded disposition / affected owners |
| --- | --- |
| 1: explicit exact-revision adoption after design verification | Human provenance, satisfied condition and eleven-file identities above bind all six decisions and the forward method; T-184/T-185 dependency edges remain unchanged. |
| 2: engineering resolution without automatic enablement | Decisions 1–6 resolve the engineering choices only. T-200 retains ordered acquisition/rights, qualified independent references, custody/development-output feasibility, justified numerical-policy approval/freeze, separately authorized held-out execution, independent reporting and separate exact-policy/evidence/registry enablement. Missing prerequisites stop dependent phases. |
| 3: observable denial and isolated positives | Decision 2 and adopted T-187/T-190 require denial on decision/reuse/export/handoff, forged/stale/resumed/batch controls, non-learning would-accept artifacts and isolated synthetic positives with no shipped authority. T-188/T-189 checks cannot bypass that boundary. |
| 4: later real/native/release evidence | T-199 documents runnable disabled defaults; T-201 still owns real permitted libraries/providers and disposable Linux Anki Desktop German/English import/update/invalidation with fields/tags/history/scheduling preserved. T-202/T-203 audits, T-204 exact-candidate adversarial QA/readiness and final German-card Anki video, and T-205 separate publication permission remain unsatisfied here. |
| 5: downstream reconciliation and change control | Adopted T-187/T-190/T-199/T-200/T-204 bodies already express the revised boundary; unchanged T-188–T-198 consumers inherit the exact spec/contract through their dependencies. Selection/forms, receipts/handoff and curated/presentation ownership remain as mapped in the decision package. No downstream rewrite is needed. Any changed policy/contract requires refreshed affected review/evidence, reuse invalidation and maintainer acceptance before affected consumers proceed. |

T-179 and T-205 are unchanged. Historical protocol/capture/comparison contracts,
failed verification and exhausted attempts remain intact. Attempt 2 remains ended:
12/12 valid initial records; three valid comparisons and one complete invalid
comparison (35866 bytes versus 32768), no validated reference commitment for that
slot, all 16 initial triples but only 12 validated comparison-output tuples.
This is output-feasibility failure, not measured linguistic error or success.

No judge/reference/provider run, study, private-answer acquisition,
repair/retry/replacement, empirical-policy approval, automatic enablement, hosted
processing, implementation consumer, publication, release or deployment is
authorized or performed by this recording. T-200 prerequisites are not waived;
zero acceptance remains not evaluable, insufficient evidence cannot pass and
narrowing scope requires a separate maintainer amendment with refreshed evidence.
No engineering-only or human-only fallback is claimed as completed v0.5.0.

The original planning review's two initial gates and one targeted recheck remain
used; its one remaining targeted recheck is reserved. Task-local simplification
and independent workflow review do not reset or expand that planning budget.

### Task-local review and verification evidence

- Dedicated `code-simplifier` pass: no edits or concerns; diff/Taskrail checks passed.
- Independent `conventional-review` coordinator classified the recording as medium
  risk. Separate `code-review` general and `comment-audit` prose/authority passes
  found no discrepancies. A fresh adversarial overall-outcome challenge returned
  HOLDS. Final coordinator verdict: APPROVE for recording-only scope, no findings
  or unresolved dispositions. One task-local challenge round was used; no planning
  recheck was consumed. Reviewed pre-verification diff against the recording base,
  SHA256 `8f61c03cfcf47c7222c6d9776304fdc9c567a99b5b6ba577af97434778519d6b`.
- Independent checks recomputed eleven adoption hashes and the canonical bootstrap
  digest; confirmed preserved adopted bodies, six summaries/five mappings and
  downstream dependencies. Human-message provenance and the overseer's independent
  T-185 acceptance were supplied through the parent thread read; Git does not
  authenticate those messages. Manifest/archive values were corroborated from
  T-185, not reconstructed. These limits are retained, not hidden as runtime proof.
- Structural validation passed: eleven exact source hashes, nine non-recording
  sources byte-identical to adoption, only T-186/STATE changed from the recording
  base, original T-186 body/history preserved, T-185/T-179/T-205 and all other
  historical files unchanged, completed dependencies and exclusive task ownership,
  T-187–T-205 transitive gating, eight approved local links/anchors and canonical
  disabled bootstrap/empty scope. `git diff --check` passed and
  `mise exec -- taskrail validate` returned `state valid`.
- Documentation/approval recording only: no code, test, fixture or runtime input
  changed and no test consumes the changed narrative. No new runtime test or full
  Python gate was run for T-186. T-185's existing exact-adoption evidence remains
  ruff PASS, mypy PASS (202 files), pytest 3344 passed/seven skipped; it is prior
  evidence, not a new T-186 execution or linguistic/native acceptance claim.
- Supported lifecycle uses the pinned Taskrail 0.4.0 via `mise exec -- taskrail`:
  `unblock --reason`, `start`, then `verify --result pass --summary` and `complete
  --note` only after review and final structural checks. Status/generated STATE
  are never hand-edited. Later lifecycle metadata and this evidence appendix are
  recording deltas, not changes to the reviewed engineering decisions.
- 2026-11-17T04:30:33Z: verification pass
- 2026-11-17T04:30:33Z: Recorded explicit maintainer engineering approval of adoption 568f02cf491e6252e795e10474833bd6a137fc21 after accepted T-185 design condition at b5d56a04946225d3345c9b85d3b88a31ebc3c404; verification pass 2026-11-17, task-local review approved and structural checks passed; judges remain paused, empty enabled-policy scope, T200 later gates and T205 publication gate retained; overseer owns successor selection
