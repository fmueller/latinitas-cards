# Curated candidates and human review

`curated validate`, `curated preview` and `curated review` are offline intake commands. They do not call a model, import authoritative course content, generate cards or authorize CSV/Anki export. Existing `authored` vocab/form/QA notes remain unchanged; `include` is not curated approval. Card recipes, rendered-card review and export belong to T-197.

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
- T-197 must bind actual semantic card key, prompt/answer, visible explanations/caveats/provenance, rule fingerprint, every displayed mnemonic fingerprint, template digest and presentation configuration. Theme/template/rendered-language changes renew affected rendered approval, not unchanged rule acceptance. Intake accepts the versioned rendered approval shape for history preservation but cannot record or assess rendered-card approval.

Every exported curated card will require both approved rule/explanation and approved rendered content. Any mnemonic shown on either side of any recipe additionally requires its language-specific approval and rule association, not just mnemonic-recall cards. Rule-only cards need no placeholder or mnemonic approval. Automatic form acceptance cannot approve curated rules, explanations or mnemonics. This task leaves `export_authorized: false`, the shipped disabled policy and T-200's empirical gates unchanged. Native Anki history/scheduling preservation still requires T-201 real-stack acceptance.
