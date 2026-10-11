---
id: T-090-preserve-escaped-literal-text-in-generated-card
title: Preserve escaped literal text in generated card inspection
status: todo
priority: medium
spec_ref: specs/v0.2.1.md#reviewed-claim-decisions-for-all-generated-cards
dependencies: []
updated_at: "2026-10-11T17:45:51Z"
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
