---
id: T-082-persist-explicitly-selected-no-write
title: Persist explicitly selected no-write reconciliation decisions
status: completed
priority: high
spec_ref: specs/v0.2.0.md#managed-update-plans
dependencies: []
updated_at: "2026-10-07T23:03:10Z"
---

# T-082-persist-explicitly-selected-no-write Persist explicitly selected no-write reconciliation decisions

## Description

Confirmed in new adversarial cycle round 1 on 2026-10-07, testing accepted
revision c608310442d599aa01013fd4812d325c5dc1cd02 against v0.2.0.
Managed Update Plans requires explicit keep-destination decisions and their
resulting baseline to be recorded. Destination Tag Ownership also requires
persisted separate origins, keep overrides and reviewed suppression.

`compose_plan()` omits a field operation when destination equals its resolved
value and no conflict remains. It omits a tags operation when the final tag set
is unchanged and no tag conflict remains. Thus an explicit `keep_destination`
decision resolves the conflict in the displayed plan but is not selectable:
`managed approve ... --operation <id>/field/Lemma` or `<id>/tags` fails with
`Managed error: unknown selected operation` (exit 1). An ownership-only removal
of one overlapping source/configured contribution likewise produces `unchanged`
with zero operations despite different proposed ownership.

This differs from completed T-070/T-071: those correctly preserve unselected
conflicts. Here the operator wants to select the reviewed resolution but cannot.
Existing T-074 through T-081 do not cover this missing approval/persistence path.
No data-loss or native-import claim is made; the supported reconciliation flow
cannot finish without separately re-adopting/reconciling the destination.

Deterministic reproduction, using only the committed disposable native fixture:

```sh
uv run --with anki==26.9.3 python scripts/check-managed-anki.py \
  --output-dir /tmp/no-write-review-native
uv run python - <<'PY'
import json
from copy import deepcopy
from pathlib import Path
from latinitas_cards.destination_state import BoundDestination, read_snapshot, adopt
from latinitas_cards.managed_plans import compose_plan

d = json.loads(Path('/tmp/no-write-review-native/before-snapshot.json').read_text())
b = BoundDestination(d['destination'], d['profile'], d['schema'], d['managed_set'])
s = read_snapshot(d, b)
owners = {n['identity']: {'source_tags': ['source'], 'configured_tags': ['source'],
    'keep_tags': ['manual'], 'keep_fields': []} for n in d['notes']}
state = adopt(s, owners, 'Review both origins and personal tag')
n = s.payload['notes'][0]
identity = n['identity']
changed = deepcopy(d)
changed['notes'][0]['fields']['Lemma'] = 'DESTINATION ONLY'
changed['notes'][0]['tags'] = ['manual']
proposal = {'identity': identity, 'fields': n['fields'],
    'contributions': {'source_tags': ['source'], 'configured_tags': ['source']},
    'decisions': {
        'field:Lemma': {'action': 'keep_destination', 'approval': 'Keep local lemma'},
        'tag:source': {'action': 'keep_destination', 'approval': 'Keep tag deleted'}}}
plan = compose_plan(read_snapshot(changed, b), state, [proposal],
    {'generated_note_type': 'Latinitas Native Managed', 'target_deck': 'Latin::Managed'})
Path('/tmp/no-write-review-plan.json').write_text(json.dumps(plan))
print(identity, plan['notes'][0]['classification'], plan['notes'][0]['operations'])
PY
# Use the identity printed above for each operation:
uv run latinitas-cards managed approve /tmp/no-write-review-plan.json \
  --review 'Select reviewed keep decision' --operation '<identity>/tags'
```

Expected: independently selectable reviewed resolutions with unchanged text/tags,
then persisted decision/baseline provenance only after the supported observation
boundary. Actual: `unchanged []`; both reviewed operations are unknown. For the
origin-only variant use the initial snapshot, no decisions, source_tags `[]`,
configured_tags `['source']`: resolved tags remain exactly `['manual', 'source']`,
but the ownership change also has no selectable operation.

Root paths: `src/latinitas_cards/managed_plans.py` field equality skip and
tag-write/conflict condition; `approve_plan()` only selects emitted operations;
`managed_application.emit_updates()` only persists selected decision/ownership
changes. Do not remove the latter safety boundary or fabricate approval for
unselected operations to fix this.

## Acceptance

- A reviewed keep-destination field decision can be explicitly selected even when its destination text does not change; observation records its decision and managed baseline, and a later unchanged proposal does not reopen that resolved conflict.
- Reviewed suppression of a deleted generated tag and keep-as-user-owned removal overrides can be explicitly selected without a tag-set change; observation persists exact ownership and decision provenance.
- Ownership-only changes between overlapping source/configured contributions are inspectable and independently selectable without removing the still-contributed tag.
- Unselected decisions never advance field baselines or tag ownership; retain T-070/T-071 regression coverage. Unknown/stale approval and failed/unobserved handoffs remain rejected or pending.
- Fresh asymmetric plan/approve/emit/observe/reload/replan tests cover the three no-write cases, exact field/tag sets, personal-field exclusion, and decision receipts. The mandatory ruff/mypy/pytest chain passes.

## Verification Notes

- Confirmed on 2026-10-07 with fresh synthetic branches over closed Anki 26.9.3 fixture snapshots and the installed CLI; both missing operation selections returned exit 1. Domain-level keep/suppress reconciliation itself returns the correct values. No product fix implemented in this filing.

## Implementation Notes

- Pinned to `specs/v0.2.0.md`; accepted remote main contained the requested
  orchestration baseline. Initial `taskrail validate` passed and `next --json`
  selected exactly T-082 before `start`.
- Expose equal-value reviewed field/tag decisions and ownership-only tag changes
  through existing operation IDs. Tag operations show old/new origins explicitly.
  Approval, full-column CSV mapping, backup, and observation boundaries remain.
- RED: four asymmetric field-keep, deleted-tag suppression, keep-as-user-owned,
  and overlapping-origin cases failed with `unknown selected operation`.
  GREEN: lifecycle tests exercise plan/approve/emit/observe/reload/replan,
  exact values/origins, pending unattested observations, personal-field exclusion,
  and selected decision provenance. Unselected origins remain separately pending.
- Dedicated code-simplifier Task loaded its skill and made no changes; its
  focused checks passed (54 tests). Independent read-only General, persistence,
  Python, and Security Tasks each loaded code-reviewer and mapped guidance.
  Persistence was routed through the Database lane; its SQL companions have
  limited applicability to this file journal. Framework/ML/network lanes omitted
  because no corresponding contract changed. All non-persistence lanes returned
  "No concrete task-relevant findings."
- Fresh candidate-validation Task reproduced and validated F1 below. Fixed
  only reviewed-but-unselected convergence, retaining T-070's ordinary
  convergence behavior. RED: reviewed case failed `already converged != old`,
  ordinary case passed. GREEN: both passed. No rejected candidates or deferrals.
- Fresh disposition-verification Task: "F1 — RESOLVED." and "No concrete
  task-relevant findings." It independently passed ruff/mypy/full pytest.
  One review/fix/recheck cycle; no unresolved findings.
- Final exact chain: `uv run ruff check` passed; `uv run mypy` passed (87 source
  files); `uv run pytest -v` passed (851 tests). Focused managed/reconciliation
  tests passed. Synthetic installed-CLI plan/approve/emit/observe selected both
  no-write field and origin operations and persisted exact provenance.
- Optional `check-managed-anki.py` passed against Anki 26.9.3. Additional closed
  disposable native no-write field/origin import observed both selections with
  identical full notes/cards/revlog tables. The first exploratory native probe
  failed due to an unclosed SQLite connection; explicitly closing it corrected
  the fixture setup. No Desktop dialog, Mobile, migration or GUI proof claimed.

### Validated review finding F1 (verbatim)

> FINDING F1 — correctness
> Severity: medium
> Evidence: src/latinitas_cards/managed_plans.py:192 now creates a selectable operation for an equal-value reviewed field decision. In src/latinitas_cards/managed_application.py:119-124, however, baseline_fields advances a field when it is selected or when its destination equals its proposal. If that reviewed field is left unselected while another operation for the same note is approved, observation can still advance its baseline.
> Finding: Do not advance the baseline for an unselected field merely because its destination equals its proposal.
> Failure/impact: A plan can record a reviewed equal-value decision as selectable, but selecting only another operation can still persist the unselected field’s baseline when observation confirms. This violates the operation-level selection boundary and can change later reconciliation.
> Recommended direction: Advance a field baseline only when its exact field operation is selected; add a mixed-selection regression test where an equal-value reviewed field remains unselected while another operation is observed.

Disposition: fixed narrowly for reviewed decisions; automatic ordinary convergence
remains supported. Regression tests distinguish both sides of this boundary.
- 2026-10-07T23:03:10Z: verification pass
