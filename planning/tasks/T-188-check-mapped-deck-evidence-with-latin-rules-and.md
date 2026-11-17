---
id: T-188-check-mapped-deck-evidence-with-latin-rules-and
title: Check mapped deck evidence with Latin rules and libraries
status: completed
priority: high
spec_ref: specs/v0.5.0.md#shared-evidence-and-automatic-acceptance
dependencies:
    - T-187-implement-versioned-automatic-decisions-and-reuse
updated_at: "2026-11-17T15:06:53Z"
---

# T-188-check-mapped-deck-evidence-with-latin-rules-and Check mapped deck evidence with Latin rules and libraries

## Description

P50-2 evidence adapters outside cli.py. Reuse confirmed deck mappings and focused
analysis modules. Own rule/library evidence under the approved check matrix;
follow the spec's execution-isolation and final-validation rules.

## Acceptance

- Bind each used form/meaning to its confirmed mapping; reuse plausible source forms
  without inventing absent forms or repairing source identity through acceptance.
- Store rule/library results, versions, scope, alternatives and contradictions.
  Supported corroboration can support library-only automatic acceptance; unsupported
  paradigms and missing corroboration remain explicit outcomes.
- A source form alone does not approve translation, segmentation or explanation.
  Contradictions need recorded resolution or withholding, never blanket source trust.
- Missing optional analyzers/resources is processing-failed/unavailable evidence,
  not linguistic rejection. Default dev tests must not import CLTK.

## Verification Notes

Check asymmetric government, ambiguous form/sense, unsupported and missing-resource
fixtures through public stage outputs. Actual resource-backed smoke evidence belongs
to integrated acceptance. Run the full mandatory validation chain.

## Implementation Notes

Candidate implemented locally from accepted origin/main
16ab3830c6314291fe985409ce3789f93015e3df. No commit or push was made.
The executable diff SHA256 is
6a5e0270a5c9939336bc895bbc4283687f6d9b90055b81867cd9b2485793092b.

Focused adapters outside cli.py bind confirmed readable form/meaning spans,
preserve raw source provenance and identity, and record morphology, cited lexical
sense plausibility and four-family construction checks. Optional local Stanza
records package/resource versions, preserves alternatives and reports unavailable
processing without downloads. It is not exhaustive ambiguity or lexical-sense
proof. Production authority remains v050-disabled revision 1; no automatic scope,
public bypass or source-owned translation/explanation authority was introduced.

TDD reproduced missing/incomplete analysis and first-round resource, Unicode,
HTML and perfect-tense defects before corrections. Final-round regressions
reproduced seven failures for metadata/cross-format source spans, inconsistent
same-key references, context-losing reuse and passive valency; exact duplicate
references failed separately before deduplication. Positive controls pass.
Dedicated simplification and independent high-risk conventional review completed.
Six initial lanes, V1-V3, F1-F3, final D1/D2 and reserved V4 were consumed;
both permitted fix rounds and the targeted-follow-up allowance are exhausted.
Retained code findings are independently closed. The proposed mandatory bundled
exhaustive backend requirement was rejected as unsupported by the contract.

Public API smoke persisted corroborated, contradictory and ambiguous synthetic
results with disabled-policy withholding. Actual local-resource probing reported
unavailable processing and failed processing_unavailable decisions; Stanza,
CLTK and default models are absent, and CLTK was not imported. No successful
resource-backed linguistic result, integrated acceptance or empirical approval
is claimed. T200 and T201 retain their separate gates.

Final mandatory chain on the above unchanged executable state: Ruff passed;
mypy passed for 212 files; pytest failed with 1 failed, 3448 passed, 7 intentional
skips and one existing serializer warning in 440.97 seconds. The only failure is
test_no_translation_string_grading_in_source: its exact reviewed-comparison
inventory lacks deck_evidence._construction_issues' morphology comparison and
LocalStanza.analyze's token/source alignment comparison. V4 reproduced the
failure and confirmed these are legitimate source checks, not translation
grading. Required guard inventory maintenance remains unresolved; the detector
must not be weakened or comparisons rewritten to evade it.

Stopped at the exhausted review/fix budget without editing that inventory,
running Taskrail verify/complete, selecting a successor or pushing. A separately
authorized continuation can review the two exact inventory additions and rerun
the entire Ruff -> mypy -> pytest chain from the start on its new candidate.
The checkout and disposable public-stage evidence remain available for QA.
- 2026-11-17T07:35:57Z: Final mandatory pytest gate fails exact source-comparison inventory: two legitimate new morphology/token-alignment entries need review. Both high-risk fix rounds and reserved review budget exhausted; no verify/complete or delivery. See task implementation notes.
- 2026-11-17T14:58:04Z: Maintainer authorized only the two exact source-comparison inventory additions, narrow readback/disposition and fresh full validation; exhausted general review budgets remain unchanged.

### Maintainer-approved inventory continuation

Revalidated accepted local/remote base, pinned scope and absence of another
implementation owner before supported unblock/start. Added only the two approved
owner/expression tuples to the exact source-comparison inventory. The detector,
assertion and comparison implementations are unchanged. The inventory test
failed before this correction, then passed with all nine detector controls.
Independent narrow disposition verification confirmed both comparisons are
legitimate and closed G1; its ten checks passed. This specific allowance does
not reset the consumed review/fix budgets or authorize another general round.

The prior five-path implementation digest remains unchanged. The six-path
executable/test diff SHA256, now including the inventory correction, is
f79f60505d714ca763849ea6034291ebafb2b86f8250e67679c0998736775ec1.
Fresh final validation on that exact state passed in order: Ruff; mypy for
212 files; pytest with 3449 passed, 7 intentional skips and one existing
serializer warning in 462.66 seconds. All six changed Python files are formatted.
Historical failed-run evidence above remains valid for the prior candidate.

Successful resource-backed smoke remains unexecuted. This uv environment lacks
Stanza, CLTK and torch; default Stanza/CLTK resource directories remain absent.
The existing CPU annotate extra can supply locked packages (Stanza 1.15.0,
CLTK 2.5.1, Linux CPU torch 2.14.1+cpu). Separately authorized setup must also
provide a compatible local Stanza resources.json and la/ model tree for the
tokenize,pos,lemma processors and their dependencies. Pass that tree explicitly
to LocalStanza; pipeline downloads remain disabled. Installing dependencies or
models was not authorized or performed. A successful tagger execution still
does not establish exhaustive ambiguity, lexical-sense proof or enabled policy.
Judge/study work, T200 empirical enablement and T201 full native/library
acceptance remain separate; task-local completion is not integrated acceptance.
- 2026-11-17T15:06:53Z: verification pass
- 2026-11-17T15:06:53Z: Completed task-local engineering after maintainer-approved exact two-entry guard maintenance and independent narrow disposition. Fresh final Ruff/mypy/pytest passed (3449 passed, 7 intentional skips). Prior failed evidence and consumed review budgets preserved; resource-positive/integrated acceptance and empirical policy enablement remain unexecuted. No successor selected or started.
