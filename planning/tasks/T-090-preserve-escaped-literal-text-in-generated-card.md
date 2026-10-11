---
id: T-090-preserve-escaped-literal-text-in-generated-card
title: Preserve escaped literal text in generated card inspection
status: completed
priority: medium
spec_ref: specs/v0.2.1.md#reviewed-claim-decisions-for-all-generated-cards
dependencies: []
updated_at: "2026-10-11T18:20:05Z"
---

# T-090-preserve-escaped-literal-text-in-generated-card Preserve escaped literal text in generated card inspection

## Description

ROUND1 confirmed a medium-impact inspection-fidelity defect at revision
6eb490102cb167404b4534015e9d5f3d79447763. A reviewed explanation containing
`BEFORE <script> AFTER` exports safely as `BEFORE &lt;script&gt; AFTER`, but
`claims inspect --artifact ... --check --json` reports only `BEFORE` and loses
subsequent comparison/context text. It exits zero with empty diagnostics.
The benign control `BEFORE <datum> AFTER` loses the literal tag but retains
AFTER; `BEFORE AFTER` retains both words. This is not unsafe export or a
native-import defect. Agents cannot faithfully audit displayed answers through
the required plain-text inspection interface.

This violates the rendered-text inspection acceptance under
`specs/v0.2.1.md#reviewed-claim-decisions-for-all-generated-cards`.

## Acceptance

- Generated-card inspection preserves literal angle-bracket text and following
  content for supported workspace/CSV/package inspection, without interpreting
  already escaped prose as active HTML. Cover both literal benign tags and an
  unclosed literal script token through the public CLI.
- Safely escaped exports remain unchanged and check mode still accepts matching
  reviewed artifacts. Text and JSON inspection faithfully represent the answer.
- Preserve source-owned HTML normalization and suppression of actual active
  markup; do not globally weaken the source-HTML safety contract to repair
  inspection of already generated content.
- Existing supported template/layout and source-bound drift checks remain intact.
- Run the mandatory ruff, mypy and pytest chain.

## Verification Notes

- Independently reproduced twice through public CLI on 2026-10-11 in ROUND1.
  Input is one synthetic audiō lexeme and one individually authored explanation.
  Expected: inspected answer includes literal `BEFORE <script> AFTER` and
  subsequent context. Actual: `audiō\nLemma: audiō\nBEFORE`; export retains the
  escaped literal and AFTER. `--check` returns 0, diagnostics `[]`.
- Reproduction from repository root, using a new disposable directory:

  ```sh
  W=$(mktemp -d)
  printf 'ID,Word,Parts,Meaning\nr1,audiō,audiō — audīre — audīvī — auditum,hören\n' > "$W/source.csv"
  uv run latinitas-cards setup --input "$W/source.csv" --profile "$W/profile.json" \
    --lexical-entry-field Word --principal-parts-field Parts --meaning-field Meaning \
    --source-identity-strategy source_id_field --source-id-field ID \
    --role present_1s --role present_infinitive --role perfect_1s --role supine \
    --separator ' — ' --recipe principal_part_completion --recipe principal_part_recognition \
    --non-interactive --confirm --json
  uv run latinitas-cards generate --input "$W/source.csv" --profile "$W/profile.json" \
    --output "$W/bootstrap.csv" --approve-scope --approve-fresh-import
  uv run latinitas-cards claims list "$W/source.csv" --profile "$W/profile.json" --json > "$W/claims.json"
  F=$(jq -r '.claims[] | select(.kind=="explanation" and .form=="audiō") | .fingerprint' "$W/claims.json")
  printf '[{"source":"ROUND1","reference":"literal prose","version":"1","text":"Literal angle brackets must remain visible."}]' > "$W/evidence.json"
  uv run latinitas-cards claims record "$W/source.csv" --profile "$W/profile.json" \
    --fingerprint "$F" --status accepted --reviewer ROUND1 --reason 'Literal text control' \
    --value 'BEFORE <script> AFTER' --evidence "$W/evidence.json" --json
  uv run latinitas-cards generate --input "$W/source.csv" --profile "$W/profile.json" --output "$W/export.csv"
  uv run latinitas-cards claims inspect "$W/source.csv" --profile "$W/profile.json" \
    --artifact "$W/export.csv" --check --json > "$W/inspection.json"
  jq '.cards[] | select(.semantic_key=="principal_part_completion:present_1s") | .answer' "$W/inspection.json"
  grep -o 'BEFORE &lt;script&gt; AFTER' "$W/export.csv"
  ```

## Implementation Notes

- Deduplicated against completed T-064 (inspection), T-026 (source HTML safety),
  T-053 (literal generated markup), and T-087/T-088 (phrase inspection/export).
  None tracks this generated-HTML-to-plaintext loss; do not reopen or duplicate
  their broader work.
- Investigation points to `review_inspection.py` reusing `source_html_to_text`:
  `html_text.py` deliberately reparses decoded source markup. Generated escaped
  prose and source normalization need distinct semantics. Implementation remains
  the repair worker's responsibility; ROUND1 changes no product code.

### Repair Cycle Evidence (2026-10-11)

- Verified fetched origin/main and HEAD at
  `e2a3731f15e5732f4f83b7a16677adae15e25a41`, pinned v0.2.1 spec SHA256
  `0b2150fc9778a3d7c0b058a5d0961a819ee283f148569323c32eee07ac3776b4`.
  Taskrail validate returned `state valid`; next JSON selected T-090 before start.
  Start timestamp: 2026-10-11T18:01:20Z. No task/spec/gate metadata was hand-edited.
- Minimal boundary: generated HTML gets one structural parse with one entity
  decode; source-owned HTML retains its existing stricter second parse.
  Actual script/style/comment suppression and export generation are unchanged.
- Strict public-contract RED:
  `uv run pytest tests/unit/review_commands_test.py -k 'principal_recipes_authored or offline_package' -q`
  returned `3 failed, 3 passed, 23 deselected`: literal script and datum answers
  failed fidelity assertions; supported package truncated at Personal annotation.
  Plain-text and rejected-template/slot controls passed.
- GREEN and post-simplification checks:
  `uv run pytest tests/unit/review_commands_test.py tests/unit/html_text_test.py -q`
  returned `38 passed`. Initial and post-review mandatory chains passed:
  ruff `All checks passed!`, mypy `Success: no issues found in 106 source files`,
  pytest `1024 passed`.
- Dedicated Task loaded code-simplifier; only moved the test html import to module
  scope. Its checks returned `29 passed`, ruff passed, diff check passed.
- Separate parallel read-only General, Security and Python Tasks loaded
  code-reviewer with ECC code-reviewer (no companion), security-reviewer plus
  security-review, and python-reviewer plus python-patterns respectively.
  Each returned verbatim: `No concrete task-relevant findings.`
  General covered correctness/contracts/domain; Security covered input safety
  and source/generated semantics; Python covered the directly affected language.
  Database omitted: no SQL/persistence implementation changed (package fixture
  only). Framework/ML/network lanes omitted: no affected framework or domain.
- Fresh code-reviewer candidate-validation returned verbatim:
  `No concrete task-relevant findings.` Empty candidate set; no accepted,
  rejected, deduplicated or deferred findings and no finding fixes required.
- Fresh code-reviewer disposition-verification returned verbatim:
  `No unresolved or newly introduced task-relevant issues.` It independently
  ran the mandatory chain with `1024 passed`. One review cycle, no unresolved
  dispositions, no task-local follow-up discovered.
- Manual original public setup/generate/claims record/inspect reproduction ran
  against the accepted baseline in a disposable worktree, then the repair.
  Baseline script answer: `audiō\nLemma: audiō\nBEFORE`; repaired answer retains
  `BEFORE <script> AFTER` and subsequent audīre/comparison/meaning context.
  Benign datum and plain-text controls also passed after repair. Checked CSV
  text/JSON returned exit 0 and diagnostics []; same-input baseline/repaired
  CSV exports were byte-identical for all three authored values.
- Synthetic supported principal-part .apkg and .colpkg artifacts for all three
  values passed public source-bound check and standalone inspection; literal
  tokens and subsequent comparison survived and package bytes were unchanged.
  Checked artifacts returned diagnostics []. Actual script/style/comments were
  suppressed, encoded source script normalization stayed strict, and double
  entities decoded only once. Existing drift/layout tests stayed green.
- UI/export appearance unchanged; no native or browser appearance claim made.
  Desktop/AnkiMobile acceptance remains maintainer-deferred on 2026-10-11 and
  pending, not passed or covered. T-091 is separate and untouched. ROUND1/5 was
  already used; only the parent may start ROUND2 after both repairs are accepted.
- 2026-10-11T18:19:58Z: verification pass
- 2026-10-11T18:20:05Z: Verification pass at 2026-10-11T18:19:58Z after RED/GREEN, manual original CLI/package checks, independent review and fresh disposition verification; final ruff/mypy/pytest 1024 pass. Native acceptance remains pending, maintainer-deferred 2026-10-11; T-091 separate.
