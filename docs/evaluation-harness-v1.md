# T-185 initial-output harness v1

`scripts/capture-evaluation.py` captures one task's original received bytes before
JSON, binding, field, ID, reference or byte-cap validation. It uses Python 3.13+
and the standard library only. It is research infrastructure, not a feature
consumer, linguistic checker, judge launcher or response generator.

## Invocation and custody

Run in the isolated judge/custodian workspace, never the product checkout with
real answers. The parent directory must exist; the output directory must not.

```sh
python3 capture-evaluation.py --binding binding.json --task-id TASK_ID --output NEW_PRIVATE_DIRECTORY < original-task.json
```

Prefer streaming the agent's directly emitted UTF-8 JSON into the command's stdin
at emission time. For an agent file-writing interface, write the raw task response
directly to a new private file first, then capture that exact file unchanged. Do
not construct a verbose answer-bearing Python serializer or hold the only output
in memory pending size/parse assertions. If the interface cannot expose/preserve
the emitted task bytes, record capture failure and stop; do not reconstruct them.
No model/provider runtime or transport integration is supplied by this script.

The new directory is reserved exclusively (0700); `original.bin` and `receipt.json`
are new private files (0600), never overwritten. Input is spooled/flushed in up to
8-KiB chunks, before validation. The original bytes are never decoded/re-encoded,
normalized, pretty-printed, compressed or repaired in storage. A receipt records
their SHA256 and byte count, binding hash when available, capture completeness,
limits, errors and structural validity. Exit 0 means structurally valid; exit 1
means captured but failed; exit 2 means the capture directory could not be reserved.
An existing directory is an error even if its previous receipt failed.

The task cap remains **32,768 bytes**, inclusive. The finite transport ceiling is
**1,048,576 retained bytes per slot**, plus at most one read-only lookahead byte
to distinguish exact ceiling EOF from overflow. Overflow retains only a prefix,
sets `capture_complete=false`, reports an observed lower bound and fails; it never
labels a truncated prefix a complete original. The bound binding file is also at
most 1 MiB. The receipt hash always describes retained bytes, not omitted bytes.
I/O failures fail capture; if retained bytes cannot be read, count/hash are null.
An abrupt process kill/disk exhaustion may prevent a receipt: retained raw alone is
not a valid record. Custodian must record that failure, never rerun or reconstruct
the answer to replace it. This tool is not a tamper-proof or disaster-proof store;
custodian must freeze outputs and prevent later direct filesystem mutation.

At the new attempt's maximum allocation, at most 72 initial captures retain 72 MiB
of raw prefixes plus small receipts. No extra response slots arise from transport
failures. A valid receipt proves transport/structure only, not source authenticity,
entailment, competence, policy or reference acceptance. Original task response bytes
are neither agent serializer artifacts nor complete provider transcripts. Preserve
the separately bound metadata/dictionary and exact envelopes in isolated custody.

## Binding schema: `t185.binding.v1`

The custodian supplies this UTF-8 JSON object **before solving** and freezes its
exact bytes. No answers belong in it. Every task record references the SHA256 of
this file; changing metadata changes that hash. The compact task schema version is
`t185.initial.v1`; older study records are not migrated or reserialized.

Required binding fields:

- `schema_version`: `t185.binding.v1`.
- `attempt_id`, `run_id`: nonempty actual new attempt/run identifiers.
- `role`: `REF-A`, `REF-B` or `ADJ`.
- `packet_sha256`, `protocol_sha256`, `prompt_sha256`, `envelope_sha256`,
  `source_dictionary_sha256`: lowercase 64-hex SHA256 commitments.
- `citation_ids`: unique nonempty strings naming exact citations in the preserved
  source dictionary. This list may be empty if all requested claims lack support.
- `tasks`: map from opaque task ID to `{ "language": "de"|"en", "requested":
  { "PROPOSITION_ID": { "scope": "nonempty scope or immutable entry reference",
  "eligibility_by_role": { "REF-A": "eligible"|"audit", "REF-B": "eligible"|"audit",
  "ADJ": "eligible"|"audit" } } } }`, one to four atomic requests per task, including presuppositions. All three role entries are mandatory. The receipt's `final_scope_eligible_ids` is their intersection, not an accepted reference.

The source dictionary referenced by hash preserves source/excerpt identities,
edition/locator/line/quote/byte commitments, permissions/access provenance and
applicability limitations. Its citation IDs resolve exact locators, not merely
whole books. The bound envelope preserves packet/cluster mapping, language/claim
scope, field definitions, correction dispositions, actual appointments of all three
fresh peers, configured effort/model and unavailable runtime telemetry, tool/network
scope, hashes, exposure log and normalized counting definitions. Do not duplicate
these per proposition. Extra binding metadata is permitted and hash-bound; it is
not dropped or rewritten by the harness. A hash-shaped string alone does not prove
the referenced file exists or corresponds: the custodian verifies actual bytes,
resolution and complete gate before launch. This harness checks the declared
allowlist, not source-dictionary authenticity or launch authorization.

## Task schema: `t185.initial.v1`

Exactly these top-level keys (extra/missing keys fail):

- `schema_version`: `t185.initial.v1`.
- `phase`: `initial`; adjudication is rejected by this version.
- `binding_sha256`: hash of the supplied exact binding bytes.
- `task_id`: the requested ID also passed on the command line.
- `answerability`: `answerable`, `partial` or `insufficient`.
- `limitations`: array of nonempty strings; `[]` explicitly means none recorded.
- `propositions`: one record per requested ID, in any order, no duplicates or extras,
  at most four. IDs/eligibility are inherited from the frozen binding, not chosen
  after seeing outcomes. There is no requirement to assert support.
- `run_observations`: exactly `tools_network_scope`, `prior_exposure`, `conflicts_shared_dependencies`, `timestamps`, `hash_scope`, `attempt_budget`, `exposure_log`, each a scoped observation as defined below. These are actual output-specific observations, not assumptions copied from a frozen plan.

Each proposition has exactly these keys:

- `id`: requested proposition ID.
- `assertion_ids`: array of judge-local assertion IDs linked to that requested `id`, unique across this role's task record. Supported propositions require at least one nonempty ID. `[]` explicitly means no assertion was made for a withheld/insufficient request, not that the request disappeared. These IDs are distinct from the immutable requested IDs.
- `status`: `supported`, `withheld` or `insufficient`. Audit-only IDs cannot claim
  support. `supported` is an initial judge assertion, not adjudicated acceptance.
- String fields: `answer`, `support`, `applicability`, `segmentation_convention`,
  `wrong_answer`, `rejection`, `abstention_reason`.
- String-array fields: `citation_ids`, `alternatives`, `caveats`, `missing_evidence`.
  Every element is nonempty; empty arrays explicitly record none.
- `audit`: exactly `corroboration`, `answerability`, `segmentation`, `target_language_validation`, `counterexample_handling`, `native_abstention`, `normalized_counting`, `correction_links`, `reference_status`, `processing_outcome`, `source_support_disposition`, each a scoped observation.

A scoped observation has exactly `{ "state": "observed"|"unavailable"|"not_applicable", "scope": "nonempty", "reason": "nonempty", "value": ... }`. Observed values must be non-null finite JSON; unavailable/not-applicable values must be null with an explicit reason. No missing check is inferred as passed. These observations preserve actual checks separately from invariant source/envelope declarations.

Observed segmentation values have exactly `convention_ref`, `units`, `recombination`, `component_functions`, `support`; observed target-language validation has exactly `wording`, `Latin_feature_or_sense_support`, `meaning_check`, `limitations`. Values may be concise scoped text or structured JSON, never null; units/components must remain distinguishable. Observed corroboration has exactly `citation_ids` (known-reference string array), `claim_scope`, `disposition`; record independently acquired source agreement/conflict/gap, not merely a second copy of the same source. Source-support disposition has exactly `correspondence`, `entailment`, `applicability`, each `pass`, `fail` or `unavailable`, with scoped reasons in the observation and material objections in `caveats`/`missing_evidence`. Supported assertions require all three checks to be observed/pass; this remains self-report, not proof of truth.

Observed `processing_outcome` is exactly one of `not_attempted`, `attempted_complete`, `failed_partial`, `failed_malformed`, `failed_over_limit`, `failed_processing`, `failed_custody_gate`. Observed `normalized_counting` is a nonempty object keyed by measurement name (for example `attempted`, `withheld`, `failed`); each value has exactly `{ "unit": "task"|"proposition", "scope": "nonempty scope/reference", "value": nonnegative_integer|boolean }`. Booleans denote identified binary flags, not counts inferred from native prose. Scope must identify or resolve phase, role, language, eligibility and task/request under the bound definitions. Several overlapping observations remain separate named entries; never add them into a pooled error total. Unobservable values use the explicit unavailable wrapper, not a fabricated zero.

For other observation values use scoped text or structured JSON that preserves counterexample assessment and exact affected request/support, native abstention text/scope/availability, correction IDs consulted/affected/disposition, and research-only reference status/limitations. Semantic correctness of these contents is a separate audit, not established by structural validity. Unobserved checks retain their explicit state/scope/reason.

Supported propositions require a nonempty answer, concise source-entailment/
applicability explanation and at least one known citation reference. Withheld or
insufficient propositions require a nonempty abstention reason. Fields must remain
present even where empty/not applicable; use a reasoned not-applicable statement
for convention or wrong-answer/rejection where relevant. Never remove legitimate
alternatives, caveats, missing evidence or contrary analysis to satisfy size.
Ambiguity and alternative translations/analyses remain in `alternatives`/`caveats`;
dictionary references replace repeated source bytes, not proposition-specific
entailment. Language and run/prompt provenance resolve through the binding.

Capture outcome is in `t185.capture.v1` receipts, not an untrusted self-declared
"complete" flag in the answer. Native reasons remain in original fields; normalized
requested/withheld/failed counts follow the bound definitions and receipt, retaining
overlap rather than pooling failures and linguistic errors. JSON duplicate keys,
non-UTF-8, nonstandard numeric constants, malformed structure and unknown references
fail; the raw bytes remain unchanged. Validation does not add inferred missing fields.

## Required-field correspondence and comparison boundary

The answer-free custodian definitions (`field-definitions.safe.json`, SHA256 `ca7a57bbdd423f9222b88821d929c43b6bac276a81d109dc13a2d1e020d36ed2`) identify 20 run fields, 25 task fields and 11 later comparison requirements. This v1 is an explicit projection, not permission to omit those requirements.

The 20 run fields resolve as follows: `role`, actual `thread_id`, `run_id`, `provider_model`, `immutable_provider_revision`, configured `effort`, `prompt_sha256`, `dispatch_envelope_sha256`, `packet_sha256`, `source_register_sha256`, `manifest_sha256` and `language_scope` are preserved in the hash-bound envelope/dictionaries (actual missing metadata has an explicit scope/reason). The seven run observations above retain actual tools, exposure, dependencies, timestamps, hash coverage and budget observations in each output. `output_sha256` is detached in the capture receipt. No unavailable runtime telemetry is inferred from configuration.

The 25 task fields resolve as follows:

- `task_cluster_split_strata_ids`, `binding_versions`: immutable task/envelope entries, including explicit not-supplied/not-applicable reasons for product binding/decision/card state because no candidate exists.
- `assertion_ids`, `requested_claims`, `scope_eligibility`: separate judge-local assertion arrays linked through each immutable requested `id`, exact bound requested map, all-role eligibility and scoped receipt intersection. Absent/failed requests remain in the allocation ledger even if no valid record exists.
- `propositions`, `support_locators`: `answer`, `support`, known `citation_ids` resolving exact dictionary edition/revision/original/excerpt hashes, section/line/span/quote and rights/access evidence. Actual entailment/applicability is not replaced by the dictionary.
- `corroboration`, `answerability`, `segmentation`, `target_language_validation`, `counterexample_handling`, `native_abstention`, `normalized_counting`, `correction_links`, `reference_status`, `processing_outcome`, `source_support_disposition`: the named output-specific audit observations; answerability also has a task-level summary.
- `alternatives`, `applicability_caveats`, `known_wrong_answers`: scoped `alternatives`, `applicability`/`caveats`, `wrong_answer`/`rejection` and their citation support. State explicit unavailability/not-applicability reasons rather than silently emptying a relevant check.
- `missing_evidence`, `abstention`, `missing_claims`: scoped evidence-needed strings, status, native abstention audit and `abstention_reason` on each exact requested ID; failed/missing-valid originals additionally remain failed in receipts/ledger. Withholding is not automatically a processing error.
- `record_byte_size`: detached receipt's exact retained count and completeness, never a serializer-source or reserialized size.

All 11 comparison requirements remain mandatory if later comparison is separately authorized: actual metadata/hashes; three raw initial paths/hashes/freezes; disputed IDs and audited agreements; A/B/ADJ positions/dependencies/citation versus entailment; source-justified resolutions/unresolved/alternatives/abstentions/scope; cited changes and reasons without voting; final-reference file/hash; normalized/native/request/correction/processing/exposure accounting; inspection of all three unchanged initials including agreements with a 32,768-byte comparison cap; all-role final eligibility intersection; and a 65,536-byte answer-free final report without source/task maps or solutions. This initial-only harness implements no comparison record or qualification decision. A complete comparison schema/gate must be frozen before any later comparison; an initial receipt cannot satisfy or bypass it.

## Demonstrated capacity, not a promise that all answers fit

`tests/integration/evaluation_capture_test.py` uses only synthetic strings and opaque
IDs. Its capacity fixture exercises all four propositions, each with six 224-byte
text fields, four 256-byte alternatives, four 256-byte caveats, four 256-byte missing-
evidence entries, two citation references, separate local assertion identities, two typed overlapping count/flag entries, all audit/run observations populated, and nine 64-byte segmentation/meaning-check detail values per proposition. The fixture is 32,563 bytes with the tested encoding, below 32 KiB without discarding fields. The earlier untyped 32,623-byte fixture did not demonstrate the corrected contract's capacity. This is a measured finite
fixture capacity, not a maximum on truthful analysis or proof every response fits.
Exact 32,768/32,769-byte tests include untouched whitespace; malformed and oversize
records are retained byte-identically with their hashes. If honest content cannot
fit, retain the failed original and report insufficiency; do not shorten/retry it.
