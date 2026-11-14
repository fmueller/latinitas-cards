# T-185 comparison harness v1

This is the separately versioned comparison prerequisite for the four remaining ADJ records in the already authorized `t185-supported-slice-v1-attempt-2`. It does not authorize another study, judge, replacement, initial-response repair, feature consumer or policy approval. The overseer owns independent checkpoint QA, actual custody binding and ADJ dispatch. All actual answers and source/task dictionaries stay in isolated custody.

## Invocation and unchanged custody

Use the same standard-library Python 3.13+ script in isolated custody:

```sh
python3 capture-evaluation.py --binding comparison-binding.json --task-id TASK_ID --output PREBOUND_NEW_DIRECTORY < directly-emitted-comparison.json
```

The output directory must be new and equal the task's prebound slot. Exact stdin bytes are spooled/flushed to `original.bin` before any comparison binding, initial artifact, JSON/schema or cap checks. Original content is never normalized, reconstructed, compressed or rewritten. `receipt.json` is detached derived custody evidence. Per-task validity remains 32,768 bytes inclusive; the retained transport ceiling remains 1,048,576 bytes plus one observed lookahead byte. Over-limit/malformed bytes remain retrievable/hashable within that ceiling; prefixes are incomplete failures. Storage failure or abrupt kill can prevent a receipt and must be reported as custody failure, never retried to replace an answer. No-overwrite and no-repair behavior is unchanged.

`t185.binding.v1` and `t185.initial.v1` retain their original behavior and hashes in frozen initial custody. They do not accept comparison records. The new comparison binding selects a distinct validator; the old v1 validator is reused only to check unchanged initial records and a derived view of final reference fields. That view is never saved as an initial response or used to repair one. Valid comparisons produce `t185.comparison-capture.v1` receipts; failed records rejected before binding/schema selection may use the generic v1 failure receipt. No failed receipt has a final-reference commitment.

## Binding: `t185.comparison-binding.v1`

Freeze exact UTF-8 binding bytes before comparison. Required fields:

- `schema_version`: `t185.comparison-binding.v1`; `role`: `ADJ`.
- Nonempty `attempt_id`, `thread_id`, `run_id`: actual current attempt and ADJ identity, not invented provider telemetry. The ADJ thread must equal its own frozen initial appointment; all three initial threads must be distinct.
- Lowercase SHA256 commitments: `packet_sha256`, `source_dictionary_sha256`, `protocol_sha256`, `schema_sha256`, `harness_sha256`, `prompt_sha256`, `envelope_sha256`. The harness commitment must equal the executing script bytes. The custodian must separately verify the exact schema/protocol/dispatch envelope and source dictionary, not merely their hash-shaped declarations.
- `initials_before_peer_exposure`: exactly `true`, attesting all independent initial freezes preceded peer exposure. This declaration is not a retrospective proof; actual gate/custody evidence remains mandatory before dispatch.
- `safe_report_path`: nonempty frozen target reference for the later answer-free report; `safe_report_limit`: exactly `65536` bytes. The report does not yet exist at the first comparison. After all four slots, the custodian must preserve its exact bytes, measured count and detached hash, inspect that it contains no answers/maps/passages/errata contents, and reject an oversized or unsafe report. A comparison receipt does not certify this later report.
- `tasks`: exactly four opaque task IDs. Each value has `output_directory` (nonempty local absolute path or binding-directory-relative path) and `initials`. All four resolved output directories must be distinct. One exclusive output directory per task enforces the frozen four-slot allocation under custodian control; changing bindings/directories to obtain another response is prohibited, not a retry facility.

Each task's `initials` has exactly `REF-A`, `REF-B`, `ADJ`. Each role entry preserves:

- `thread_id`, `run_id`, `frozen: true` and `initial_protocol_sha256` for that original initial record. Runtime provider revision/effort/cost remain unavailable unless independently observed; creation configuration is not runtime telemetry.
- `original_path`, `original_sha256`, `receipt_path`, `receipt_sha256`, `binding_path`, `binding_sha256`: exact original, capture receipt and initial binding references. Paths resolve locally from the comparison binding's directory. These files must already be privately frozen; the comparison harness only reads them.

For the selected task, all three originals are bounded-read, hash-checked, decoded strictly and revalidated against their own unchanged v1 binding. Receipt version, validity/completeness, errors, exact byte count/hash, task, binding, limits and declared scope intersection must agree. Original roles, attempts, runs, old protocol commitments and packet/source dictionary must correspond. The three selected-task requested/language/eligibility definitions and citation allowlists must agree. Missing/unfrozen/incomplete/invalid originals block comparison; there is no reconstruction. Initial bindings may contain only the selected task or a larger original allocation; no new field is imposed retroactively on v1.

The bound envelope/dictionaries retain actual appointments/configuration, manifest/register/contract/correction commitments, source snapshot/edition/rights/access limits, language/stratum scope, normalized definitions and product-state unavailability. Exact citation entries resolve original/excerpt hashes, locators/spans/quotes. Neither this structural validator nor matching bytes establishes source authenticity, rights, entailment, linguistic competence or real isolation.

## Record: `t185.comparison.v1`

Exactly these top-level keys are required:

- `schema_version: t185.comparison.v1`, `phase: comparison`, `binding_sha256` (exact new binding hash), `attempt_id`, `task_id`, `role: ADJ`, `thread_id` matching the binding.
- `decision_basis`: exactly `source_based`, never majority vote or model confidence. This is a declared basis, not proof the analysis followed it.
- `answerability`, `limitations`, `run_observations`: unchanged v1 types. Actual tools/network, exposure, shared dependencies, timestamps, hash scope and attempt budget observations retain explicit scope/reason and unavailable/not-applicable states. Record comparison exposure honestly; do not relabel it as an independent initial solve.
- `results`: exactly one result per immutable requested ID, with no missing/duplicate/substitute ID, at most four.
- `counts`: the typed accounting object below.

Each result has exactly:

- `id`: immutable requested ID.
- `reference`: all v1 proposition fields, including separate judge-local assertion IDs, answer/support/citation references, alternatives, caveats, applicability, segmentation, competing answer/rejection, missing evidence, native abstention and all eleven scoped audit observations. The enclosing requested ID must match. Source correspondence, entailment and applicability remain separate; supported references require the v1 observed/pass checks and nonempty support. Final support additionally requires eligibility for **all three** initial roles, not only ADJ.
- `positions`: exactly three role keys, each the exact initial judge-local assertion-ID array for this request. Together with the binding's original/hash references this retains each complete initial answer and its provenance without duplicating it. Do not replace these with reconstructed summaries.
- `disputed_assertion_ids`: exactly three role keys, each a unique string array drawn from that role's initial assertion IDs. Empty arrays explicitly record no named dispute; disagreements can remain unresolved, not voted away.
- `agreement_audit`: nonempty scoped assessment of agreements and possible shared errors, required even where there is a disagreement. Inspect every initial position, including ADJ and apparent agreement, against the exact inspected sources; matching responses are not truth.
- `source_checks`: exactly three role keys; each has exactly `correspondence`, `entailment`, `applicability` (`pass|fail|unavailable`) and nonempty `reason`. These assess the referenced initial position separately from the final reference's checks. Cite the inspected source entries in `reference.citation_ids` and the scoped support/corroboration observations; distinguish independently acquired corroboration from common-source agreement.
- `changes`: exactly three role keys, each with exactly `disposition` (`changed|unchanged|unresolved`), nonempty `reason`, and `citation_ids` (known reference strings). Changed positions require at least one citation. Preserve cited reasons for every change, especially changes from ADJ's own initial position. Semantic equivalence rather than identical wording determines an unchanged position; the validator cannot prove that assessment.
- `unresolved`: scoped string array retaining open issues. It must be empty for a supported reference and nonempty for a withheld/insufficient reference. Unresolved change dispositions, material objections or missing evidence must not be hidden behind a supported status; substantive consistency remains source audit work.

Scopes, legitimate alternatives, counterexample assessments, corrections consulted/affected/dispositions, source and target-language checks, native abstention and normalized observations retain their v1 meanings. Do not remove them to fit the cap. Structural validity is not reference acceptance or qualification.

## Accounting and verifiable reference commitment

`counts` has exactly `unit: proposition`, `role: ADJ`, `phase: comparison`, `language` (the bound initial language), `requested`, `attempted`, `adjudicable`, `withheld`, `failed`. The harness recomputes and checks these values without pooling units:

- `requested`: exact immutable request count.
- `attempted`: number whose reference processing observation is observed and not `not_attempted`; null if any processing observation is unavailable/not-applicable. It is not inferred from a nonempty answer.
- `adjudicable`: number of request tuples with all three structurally valid frozen initial positions present. This is not the number source-supported, eligible or accepted; audit-only/withheld positions remain inspectable tuples.
- `withheld`: number of references whose status is not supported, including insufficient.
- `failed`: number whose observed reference processing outcome starts `failed_`; null if any processing observation is unavailable/not-applicable. Withholding and failure may overlap, so never sum them into an error total.

Counts must be exact integers (not boolean aliases), or the specified null unknown. The receipt also derives separate per-role initial requested/attempted/withheld/failed counts with unit, phase and language; eligibility remains per-request in all three immutable bindings. The per-request scoped normalized observations remain separate from these summaries. Source-support errors, disputes and native abstention are not interchangeable with processing failures. Failed/missing comparison slots stay failed in the allocation ledger, never disappear from denominators.

For a valid comparison the harness derives `final_reference_sha256`, never accepts an arbitrary supplied hash. Its preimage is UTF-8 JSON of exactly `{schema_version: "t185.reference.v1", binding_sha256: NEW_BINDING_HASH, task_id: TASK_ID, results: RESULTS}`, encoded by Python `json.dumps(..., sort_keys=True, ensure_ascii=True, separators=(",", ":"))`. No floats are needed for this reference data; all decoded numbers must be finite. The detached receipt identifies `final_reference_schema: t185.reference.v1`. Recompute this commitment from the frozen comparison to verify its entire reference/disposition projection and links to all three originals. It is a **derived canonical commitment**, not the raw comparison hash; `raw_sha256` separately commits to original received bytes. No self-referential digest is put inside its own hashed response.

## All eleven comparison requirements remain represented

1. Packet/prompt/run/manifest/register and actual ADJ metadata: comparison binding, exact envelope/dictionaries and output-specific run observations.
2. Three original paths/hashes and preserved text: checked role-keyed original/receipt/binding references, read-only source files.
3. Disputes, agreements, inspected sources/corroboration: per-request disputed IDs, mandatory agreement audit, source checks and reference support/audit fields.
4. A/B/ADJ positions and dependencies: exact role-position references, initial checks and scoped run limitations.
5. Source resolutions or unresolved/withheld status, alternatives/abstention/scope: full reference fields and explicit unresolved array.
6. Changes, including ADJ's: all-role cited dispositions; source-based declared decision basis, no voting/confidence field.
7. Final-reference artifact/hash: verifiable derived projection commitment in the detached receipt; unresolved results remain in the projection.
8. Request/status/eligibility/count/correction/processing/exposure: full reference observations, unchanged initial definitions and independently derived scoped summaries.
9. Three freezes, inspection including agreement, one bounded record/task: checked metadata/artifacts, mandatory audits and exclusive prebound four-slot paths; 32 KiB validity and 1 MiB retained-prefix limit.
10. Final eligibility: exact all-three-role intersection; no promotion of audit-only scopes.
11. Answer-free report: frozen report target/64 KiB limit; after comparison, exact measured bytes/hash and content-safety audit remain required before accepting the study report. This later report is not an extra linguistic solve or a reason to repair responses.

## Synthetic verification limits

Tests use only authored structural placeholders and opaque IDs. They exercise asymmetric positions, invalid role/attempt/task/initial hashes and metadata, missing audits/dispositions/unresolved fields, duplicate IDs, forbidden vote basis, count/support intersections, strict JSON, byte boundaries, exact retention and no overwrite. The four-request fixture is exactly 23,998 bytes and includes every required reference and comparison field, six 128-byte text values and four 128-byte entries in each alternatives/caveats/missing-evidence array per request. Some audit observations explicitly record unavailable evidence. This finite capacity example is not a worst-case guarantee: meaningful richer analyses can overflow and must remain failed/insufficient, not shortened or regenerated. Synthetic captures consume no study slots.
