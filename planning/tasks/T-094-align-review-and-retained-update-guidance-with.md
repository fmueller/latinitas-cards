---
id: T-094-align-review-and-retained-update-guidance-with
title: Align review and retained-update guidance with supported CLI
status: completed
priority: medium
spec_ref: specs/v0.2.1.md#review-identity-and-export
dependencies: []
updated_at: "2026-10-11T20:19:15Z"
---

# T-094-align-review-and-retained-update-guidance-with Align review and retained-update guidance with supported CLI

## Description

ROUND2 found three related stale capability statements while following the public
workflow on 45a7cbfb7a73db3aab77ab838f94423829dfdfe6. Severity: low; confusing user
guidance, not a failed retained update. Keep this a focused documentation/CLI
wording correction, not a capability expansion or general documentation review.

- `docs/contextual-form-parsing.md` says there is currently no CLI command to
  author decisions and instructs manual JSON editing. Shared `claims record`
  successfully authored and corrected the fresh parsing fixture.
- `docs/reference-note-type.md` says the current journal rejects changed per-card
  answers as unsupported slot effects, even after CSS setup. Fresh approved
  retained principal/parsing/phrase updates completed emit/observe/reconcile.
- Public principal `preview --profile PROFILE --input SOURCE` prints
  `retained-slot payload application unsupported`. The approved retained path
  is supported; structural creation/retirement and layout migration remain blocked.

Expected: accurately distinguish supported claim recording and approved retained
payload updates from initial manual setup, structural changes and native safety.
Dedup: T080/T081/T089 completed guidance work does not track these remaining
contradictions. T090/T091 fixes do not own these statements.

## Acceptance

- Point parsing authors to the supported shared claim CLI without removing the
  legitimate explicit manually authored JSON route or individual-review requirement.
- Reference setup and principal preview describe the separately approved retained
  update path accurately while retaining strict exact layout/CSS binding and manual
  structural guidance. Do not imply ordinary overwrite CSV safely updates a deck.
- Preserve explicit native acceptance/sync/restore limits; no new transport,
  migration or native verification claim.
- Check changed CLI wording through a fresh supported preview and cross-check the
  named commands against current help and the exercised retained workflow.

## Verification Notes

- Fresh ROUND2 full protocols observed 44 principal/supine/parsing/defective and
  24 phrase operations with zero unresolved operations; closed-storage fixture
  evidence does not prove native CSV import or scheduling/sync/restore safety.
- Captured principal, supine and defective preview logs all contain the stale
  warning. Exact excerpts, commands and outputs are retained in the ROUND2 bundle.
- Native Desktop/Mobile remains maintainer-deferred/pending. No product fixes or
  task verification performed by tester.

## Implementation Notes

- Verified HEAD and fetched origin/main at
  `80c76f4801448a2bcca4dda5c994b095de5d93de`. Pinned v0.2.1 spec SHA256:
  `0b2150fc9778a3d7c0b058a5d0961a819ee283f148569323c32eee07ac3776b4`.
  `mise exec -- taskrail validate` reported `state valid`; `next --json`
  selected only this task, off_spec false, before lifecycle writes.
- Corrected only parsing review guidance, reference retained-update guidance,
  principal preview wording, and the existing public CLI regression test. No
  transport, layout, lifecycle, schema or native capability was added. Embedded
  manual JSON review remains valid; initial CSS/layout setup/import/adoption,
  strict digest binding, separate linguistic/write approvals, unsupported
  structural effects and pending native/sync/restore limits remain explicit.
- Strict CLI RED: `uv run pytest
  tests/unit/principal_part_command_test.py::test_principal_preview_reports_terminology_override
  -q` failed with `AssertionError` because `retained-slot payload application
  unsupported` was present. Minimal wording correction made it pass. Documentation
  prose has no executable behavior; no artificial docs snapshot TDD was claimed.
- Initial checks: `uv run pytest tests/unit/principal_part_command_test.py
  tests/unit/review_commands_test.py -q`: 55 passed. Exact initial ruff/mypy/pytest
  chain: all checks passed, 107 source files clean, 1045 tests passed.
- Manual CLI: copied the parsing sample without embedded decisions; `claims list
  cases.json --json` returned current fingerprints; `claims record cases.json
  --fingerprint <exact-current> --status accepted --reviewer 'Latin reviewer'
  --reason <individual contextual reason> --json` recorded case, lemma and number
  separately (each recorded 1, diagnostics empty). `form-parsing preview cases.json`
  first withheld missing required features, then showed eligible true with
  Lemma: puella, Casus: nominativus, Numerus: pluralis; gender remained withheld.
- Manual principal CLI: synthetic CSV `verb-1,dīcō,dīcō — dīcere — dīxī — dictum`
  with the example profile; `generate --profile docs/examples/profile.json --input
  <temporary-source> --output <temporary-csv> --approve-scope`, then fresh
  `preview --profile docs/examples/profile.json --input <temporary-source> --limit 1`:
  source entries 1, objects 1, notes 1, cards 6, skipped 0; corrected retained-update
  warning and native-unverified/manual-CSS warnings observed. The unmodified example
  noun CSV was also tried: zero eligible objects due to too few configured parts;
  this was fixture suitability, not a code failure.
- Help cross-check: `claims record --help`, `preview --help`, `managed --help`,
  and each `managed {plan,approve,emit,observe,reconcile} --help` succeeded.
  Named argument/operation/request contracts match linked retained workflow.
  `uv run pytest
  tests/unit/managed_plans_test.py::test_retained_payloads_use_bound_layout_and_reject_structural_changes
  tests/unit/managed_application_test.py::test_cli_emit_observe_and_reconcile -q`:
  4 passed, covering principal/parsing/phrase retained emit-observe-reconcile and
  structural rejection. Synthetic evidence is not native import or restore proof.
- Workflow step 4: dedicated Task loaded personal code-simplifier, reviewed all
  four scoped files, made no changes; shortening/consolidation rejected because
  it would obscure boundaries. `git diff --check` passed.
- Step 5: separate parallel read-only General and Python Tasks loaded code-reviewer.
  General used ECC code-reviewer (no mapped companion); Python used python-reviewer,
  python-patterns and common testing/review guidance. Python reran regression: 1 pass.
  General verbatim: "No concrete task-relevant findings."
  Python verbatim: "No concrete task-relevant findings."
  Security/database/framework lanes omitted: no trust, persistence or framework
  behavior changed; existing safety contracts were reviewed in General.
- Fresh candidate-validation Task loaded code-reviewer and inspected the actual
  scoped diff, linked contracts and command help; regression 1 pass. Verbatim:
  "No concrete task-relevant findings." Candidate IDs, rejected IDs and dedup IDs
  were empty. Step 6: no findings, fixes or deferrals to disposition.
- Step 7: fresh read-only disposition-verification Task loaded code-reviewer with
  General/Python references and Python companion; actual diff and linked contracts
  inspected, diff check passed. Verbatim: "No concrete task-relevant findings."
  One review cycle; none unresolved. Final chain and Taskrail verification follow
  this independent verification, not the initial check. Native Desktop/Mobile
  acceptance remains pending under the 2026-10-11 implementation-loop-only deferral.
  ROUND2/5 remains NOT CLEAN; parent owns ROUND3 after acceptance of this fix.
- Final exact chain: `uv run ruff check` -> all checks passed;
  `uv run mypy` -> no issues in 107 source files;
  `uv run pytest -v` -> 1045 passed in 41.91s.
  Taskrail verification pass and completion: 2026-10-11T20:19:15Z.
  Subsequent validate: state valid; next --json: no active-spec eligible task,
  with only off-spec T-038 (v0.3.0 PyPI publishing) skipped. No follow-up,
  spec/gate/dependency/priority edits, native acceptance or release action.
- 2026-10-11T20:19:15Z: verification pass
