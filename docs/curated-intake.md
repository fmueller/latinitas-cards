# Curated candidates and human review

The `curated` commands run offline and never call a model or write to Anki. Without `--plan`, preview/review inspect intake only. With an explicit recipe plan, preview renders learning cards, review records exact rendered decisions, and export writes eligible CSV plus receipts. Existing `authored` vocab/form/QA notes remain unchanged; `include` is not curated approval.

## Format and provenance

The exact `latinitas-curated` v1 shape implements section 8 of the [approved contract](v0.5.0-contract-proposal.md#8-curated-payload-and-approval-proposal). Every listed key is required, with no extras. Use `mnemonics: []` and `examples: []` when absent. Rule evidence must be nonempty. Rules contain text, `de|en` language, applicability, caveats and evidence; examples contain Latin, explanation and evidence. Each provenance object retains document, section, reference and `quoted|agent_proposed|author_composed` wording. Quote only actual source wording. Suggestions are candidates, never quotations or course authority.

Persist the source scope and author-controlled item keys across edits; do not derive keys from text, language, tags or approval. Mnemonic keys are unique within each language and must associate with the containing rule key. Keep German wordplay verbatim. An English alternative is a separate candidate with its own provenance and human review. Neither an English rule nor an approved German mnemonic approves an English mnemonic.

Published JSON Schemas describe the payload, approval and permission records. Runtime validation additionally enforces uniqueness, associations, exact digests and retained-history supersession. Duplicate JSON keys, unknown versions (including boolean versions), empty required strings and malformed evidence reject. Validation establishes structure, not linguistic correctness or legal rights.

## Permission record

Every command requires a separate `--permissions` file: `latinitas-curated-permission` v1, `source_scope` and nonempty `sources`. Each source has exactly `document`, `source_sha256`, `rights`, `extraction_permission`, `processing`, `provider`, `provider_permission`. The document must cover every rule and mnemonic provenance document. The source hash identifies the original bytes, not the edited candidate file. Rights and extraction permission are explicit nonempty operator attestations; intake cannot authenticate them or verify a source file it was not given.

`processing` describes the prior extraction: `manual`, `local` or `hosted`. Manual requires null provider fields; local/hosted require both a named provider and explicit provider permission. Record every upstream processing step honestly in these attestations. Earlier hosted extraction remains hosted even if a later model is local. An orb is not the user's local machine. These records document past permission and never authorize new provider calls; future calls require the separate provider-boundary disclosure and permission checks. Do not process private notes, infer permission from possession, or send anything to a provider without explicit authorization.

## Permitted note extraction

The hand-authored [original synthetic lesson](../tests/fixtures/curated-extraction/lesson.md) grants offline extraction and redistribution permission and contains no private material. Its [candidate file](../tests/fixtures/curated-extraction/candidates.json) is an agent-authored extraction fixture, not evidence of a model run or approved Latin content. The accompanying permissions record identifies the exact lesson bytes. Tests check quoted rule/caveat/example/mnemonic wording against the source, preserve the German aid, accept the rule without an aid and leave section 3's QA note out.

For a permitted extraction, read the source's rights and allowed processing first. Extract precise rules and applicability with exceptions and traceable references into a candidate file. Prefer course-supplied aids. Leave mnemonics empty when none exists; optional proposed aids need `agent_proposed` provenance and separate review. Preserve stable keys when re-extracting. Do not silently convert existing QA notes. Validate and inspect the resulting file before a human decides. This workflow requires no hosted extraction and enables no automatic policy.

```sh
uv run latinitas-cards curated validate tests/fixtures/curated-extraction/candidates.json --permissions tests/fixtures/curated-extraction/permissions.json
uv run latinitas-cards curated preview tests/fixtures/curated-extraction/candidates.json --permissions tests/fixtures/curated-extraction/permissions.json
```

Before: no curated intake commands. After: validation reports `Valid: 2 candidates; not export authorization`; preview emits JSON containing exact wording, evidence, caveats, provenance, fingerprints, independent statuses, permission attestations and effective offline configuration. Unreviewed mnemonic text is visible only as candidate inspection, not as a learning card. JSON escapes control characters; this is not an HTML card renderer. Downstream HTML rendering must escape all source content.

## Explicit decisions and history

`curated review INPUT --permissions PERMISSIONS --decisions DECISIONS --approvals OLD_HISTORY --output NEW_HISTORY` validates an explicit array of complete `latinitas-curated-approval` v1 records. Omit `--approvals` on the first review. There is no approve-all default. The output must be a new file: existing files, source notes, candidate bytes and old history are never overwritten. Keep all history files; omitting history cannot revoke an earlier decision in any downstream store.

Each decision includes scope, item key, `subject`, subject key, language, exact fingerprint, `approved|rejected`, reviewer, reason, UTC RFC3339 `reviewed_at` and explicit `supersedes` IDs. For rules the subject key equals the item key; for mnemonics it equals the mnemonic key. `approval_id` is SHA-256 of the canonical remaining record (UTF-8, sorted keys, compact separators, ordered arrays, no Unicode folding). The Python `make_approval(**fields)` helper computes this digest from an explicitly supplied decision; it does not decide or authenticate the reviewer. An agent cannot substitute its own judgment for human approval. Tests use synthetic reviewer identities only.

For example, a human can prepare the explicit fields after inspecting the preview and seal them offline:

```python
from latinitas_cards.curated import make_approval

decision = make_approval(
    source_scope="original-synthetic-lesson",
    item_key="lesson:imperfect-a",
    subject="rule",
    subject_key="lesson:imperfect-a",
    language="en",
    fingerprint="COPY_THE_EXACT_CURRENT_RULE_FINGERPRINT_FROM_PREVIEW",
    decision="rejected",
    reviewer="YOUR_REVIEWER_IDENTITY",
    reason="YOUR_EXPLICIT_REVIEW_REASON",
    reviewed_at="2026-11-19T00:00:00Z",
    supersedes=(),
)
# Serialize [decision.model_dump(mode="json")] as the decisions JSON array.
```

The uppercase placeholders are required human inputs, not executable example values or approvals. A unique unsuperseded matching decision applies. Multiple matching decisions are `conflict`, even if both say approved. Resolve by explicitly superseding every conflicting head; timestamps never override a rejection. References must point to earlier retained decisions for the same scope/item/subject/key/language; missing, forward, cyclic or cross-subject references reject. Supersession may replace an older fingerprint of the same subject without erasing history. A stale decision cannot be submitted as a current review.

## Invalidation and downstream boundaries

- Rule fingerprint: item key, complete rule (language, applicability, caveats and evidence), examples including explanations/evidence, and item provenance. Changes invalidate rule and dependent mnemonic approvals. Tags do not change linguistic acceptance.
- Mnemonic fingerprint: complete mnemonic, including language, wording, provenance and rule association, plus the rule fingerprint. A mnemonic-only change invalidates that variant, not the rule or other language variants. It must not invalidate unchanged rule-only cards downstream.
- Source scope/item key/language are also exact approval-match boundaries. Unchanged content can reuse an unsuperseded approval; translations require new evidence and approval.
- Rendered review binds the semantic card key, actual front/back HTML, explanations/caveats/provenance, rule fingerprint, every displayed mnemonic fingerprint, template digest and presentation configuration. Theme/template/rendered-language changes renew affected rendered approval, not unchanged rule acceptance. Intake without `--plan` preserves rendered history but cannot record rendered approval.

Every exported curated card requires both approved rule/explanation and approved rendered content. Any mnemonic shown on either side additionally requires its language-specific approval and rule association. Rule-only cards need no placeholder or mnemonic approval. Automatic form acceptance cannot approve curated content. Intake-only preview still reports `export_authorized: false`; the shipped disabled policy and empirical gates are unchanged. Native Anki history/scheduling preservation still requires real-stack acceptance.

## Learning recipes and exact rendered review

Pass a `latinitas-curated-recipes` v1 JSON file with `--plan`. It contains `format`, `version` and a finite `targets` array. Each target names a stable author-controlled `key`, `item_key`, a `formation` description (or null), an exact `example` Latin value from that item's examples (or null), `analysis: supported|unsupported|ambiguous`, `mnemonic_key` (or null) and `mnemonic_side: front|back`. Required `related_production` lists explicit production note identities for provenance links; use `[]` when absent. It does not select, generate or approve those drills. This companion is an exercise selector, not a second free-form QA format: all answers, caveats and evidence come from the intake rule/examples. The published JSON Schema rejects unknown keys; runtime validation additionally rejects duplicate `(item_key, key)` pairs.

`supported` is an author's declared analysis status, not a machine's linguistic verdict. Unsupported/ambiguous targets are withheld; missing or ambiguous examples/mnemonics are reported rather than guessed. Human review still has to assess every declared supported analysis and its actual rendering. Use cited explanations to retain legitimate alternative segmentations, such as `puell- + -ārum` versus `puellā- + -rum`; grading is self-assessment, not exact-string matching.

Without a preset, curated application is the default. `--include-recipe` and `--exclude-recipe` independently select `rule_application`, `formation_rule_recall`, `component_explanation` and `mnemonic_recall`. Application asks learners to apply the rule to the selected Latin example. Formation recall asks how to form the selected grammatical target; the renderer does not insert the reference answer into its prompt. Authors must keep the `formation` description free of tested answers and mnemonic cues, and rendered human review must assess this property. Component explanation asks about the supplied Latin form's base/stem, markers and ending; its answer comes from that example's cited explanation. Mnemonic recall asks for the rule using a uniquely selected original-language aid. No sibling recipes or all-form enumeration are implicit. A mnemonic may be explicitly displayed on either side of other recipes, except a front-side mnemonic is refused for formation recall. Every displayed aid needs separate approval. A null mnemonic key omits aids from non-mnemonic recipes; mnemonic recall can use the sole requested-language variant, but never chooses among ambiguous alternatives.

Preview returns actual `front_html`/`back_html`, approval requirements with exact fingerprints/statuses, withheld reasons and layout installation data. `--html NEW_FILE` writes the same front/back fragments in a static side-by-side review page; it is not an approval action. Source-controlled text is HTML-escaped. `--language de|en` selects output explicitly; absent a profile/flag, the first item's rule language is reported as the effective choice. Rule text, applicability, explanations and caveats must already exist in that language with their own evidence and approval. No wordplay or terminology is automatically translated.

For rendered decisions, use the same explicit approval array workflow with `curated review ... --plan PLAN`; the rendered `subject_key` is the preview's `card_key`. Copy its exact fingerprint and language after inspecting the actual card. Rule and mnemonic subjects retain their existing intake keys. Review records neither implicit approval nor automatic decisions. Omit `--plan` to keep intake-only review behavior.

## Saved grammar-study selection and production

`--profile PIPELINE_PROFILE --preset grammar-study --save-profile NEW_PROFILE` saves the approved pipeline profile shape with the preset, resolved recipes, overrides, finite selection and language. The profile must contain a `kind: curated` input reference whose resolved path and SHA-256 match the input; confirm/rebind it explicitly after source corrections. Saved input/deck paths become absolute so relocation of the saved profile does not silently change inputs. Original profiles are not overwritten. Reuse the saved profile for preview/review/export. Precedence is defaults or preset → saved overrides → explicit CLI overrides; same-level include/exclude conflict is an error. Unselected pipeline profiles keep their original translation-first defaults. A curated invocation without the preset adds application to its effective profile unless explicitly excluded. No preset enables clozes, providers or either approval model.

The preset selects production (verb/noun/adjective), formation recall, component explanation and mnemonic recall. Supply `--production-run` to preview the existing T-191 run with the exact same effective profile. Curated and production cards retain separate approval gates and explicit source/rule links. For export, also supply `--production-output NEW_CSV`; curated output uses `--output NEW_CSV --run CURATED_RECEIPT_STORE`. Production keeps its existing grammar-v1 note type and receipt store; curated uses the explicit new `Latinitas Curated v1` note type reported by preview. This avoids changing existing template ordinals or silently migrating an installed deck.

The two exports are separate receipt transactions, not an atomic multi-file installation. The report includes both outcomes without discarding eligible curated output. An absent production run is withheld; a failed production export reports `recovery_required`, not zero publication, because its CSV may already exist. Do not import or replace that output; retry the unchanged request to recover the receipt. Conversely missing curated content/approval does not approve or prevent eligible production drills. Empty curated intake requires an explicit language or shared profile and still supports receipt reconciliation. Neither CSV applies changes to Anki. `--production-after-receipt`/`--production-reconcile-receipt` address production batching separately from curated `--after-receipt`/`--reconcile-receipt`. Missing material may yield fewer than four types; reports show the omission instead of manufacturing content.

## Receipts, identity and personal ownership

Curated note identity uses persisted source scope and author-controlled item key. Card keys use recipe and explicit target key, not wording, theme or approval. Keep these keys across corrections. Related recipes share one item note; conflicting targets for a single item/template slot are withheld, never picked arbitrarily. Output limits count cards and carry already emitted siblings when writing another slot on the same note.

Export rereads content/permissions/approvals at publication and saves acceptance-bound receipts through the existing recoverable batch writer. Unchanged reruns replay their receipt, without duplicate cards or overwriting another output. Changed selection/content requires explicit receipt reconciliation, not an old cursor. The store retains every approval/rejection it has seen at export; a later history must include those decisions so an older approval file cannot erase a seen withdrawal. Retain both the history and receipt store across reruns.

CSV includes only managed fields and inherited item tags. `Personal Notes` is an unowned note-type field shown on the back but omitted from export and ownership commitments. Receipts own no user/source tags. Existing-note reimport still requires a fresh destination inspection and explicit full-field/tag preservation approval; do not treat a generated CSV as approved for direct reimport. Tests prove offline field exclusion/ownership, not native history or scheduling preservation.

Withdrawal, invalidation or unavailable current coverage compares every earlier receipt and produces manual actions with exact note/card identities and previous decision IDs. A missing CSV row does not retire or suspend an installed card. An affected sibling withholds the whole note row in ordinary regenerated CSV, even when an unchanged rule-only sibling remains linguistically approved. Unrelated notes may still export. Handoffs remain unresolved until a separate supported destination workflow verifies resolution; this command does not apply or attest it. A mnemonic-only change leaves unchanged rule-only approvals valid, while mnemonic-bearing cards require renewed review and installed-content handoff.
