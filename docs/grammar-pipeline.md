# Bounded grammar cards

For a runnable disposable setup, stage/run/resume commands and the distinction
between processing limits and output batches, see [v0.5.0 workflows](v0.5.0-workflows.md).
The [native kit](native-v050-acceptance.md) is planned acceptance, not an executed result.

`pipeline` is an opt-in consumer of the shared evidence store. Preview/export and
reporting are offline; orchestration requires explicitly configured providers and
qualified token accounting before dispatch. No production deployment is qualified
yet. It does not install analyzer resources, approve a policy, install templates,
or modify Anki. The installed `v050-disabled` revision 1 policy remains disabled:
all-pass automatic analyses alone cannot enable ordinary learning CSV output.
Bound, complete human review is separate authority. Generic curated claims are
not supported by this consumer.

## Inputs and selection

Use the approved `latinitas-pipeline-profile` version 1 shape in
[the contract](v0.5.0-contract-proposal.md#2-proposed-public-commands-and-profile). It commits to
the source deck profile and input files by SHA-256. The phrase input is an explicit
version 2 upgrade of `phrase_inputs`: entries and patterns retain their original
shapes, forms add `source_role` and `role_evidence`, and the input adds its current
`binding` and `registry_version: "v050-support/1"`. A confirmed source identity,
CSV scope and current mapped source cells are required. No positional fourth-part
guess upgrades a form to PPP. A profile must explicitly map the `ppp` role and the
form must carry corroborating role evidence and perfect/passive nominal analysis.

The default recipes are `form_translation`, `ppp_translation` and
`phrase_translation`. An unambiguous single-sense source form gets one translation
objective; an explicit objective replaces that default for the same form/recipe.
Translations and their alternatives are reference guidance for self-assessment,
not universal grading strings. Built-in form-sensitive gloss proposals cover the
`love/lieben`, `girl/Mädchen` examples and confirmed `loved/geliebt` PPP meanings.
Other inflected meanings need an explicit form-translation reference, not a bare
lemma gloss. Recognition is independently selectable; it retains
isolated readings or withholds an unanswerable requested distinction.

`grammar-study` explicitly selects `verb_production`, `noun_declension`,
`adjective_declension`, `formation_rule_recall`, `component_explanation` and
`mnemonic_recall`. Missing curated content is reported, not fabricated. Curated
rehearsal integration is separate work. Saved overrides precede CLI overrides;
same-level include/exclude conflicts and unknown recipes fail validation. Neither
production nor the preset enables `form_cloze` or `phrase_form_cloze`.

## One-objective proposals

`--objectives` names a strict `latinitas-grammar-objectives` version 1 document
with `forms` and `phrases` arrays. These are local proposals, not acceptance
records or an orchestration/provider protocol. Each form item has exactly:

```json
{
  "object_key": "present-contrast",
  "entry": "confirmed-source-id",
  "sense": "love",
  "form": null,
  "anchors": ["present", "infinitive"],
  "recipe": "verb_production",
  "target": {"person": "2", "number": "singular", "tense": "present", "mood": "indicative", "voice": "active"},
  "distinction": "analysis",
  "context": "",
  "translations": [],
  "explanation": "",
  "caveats": [],
  "evidence": []
}
```

Entries, senses and anchors name the bound input, not free-text lemmas. The
consumer corroborates every anchor against current source cells. An evidence-free
objective can use confirmed entry/sense/role evidence, but that evidence is not an
automatic semantic approval. Every rendered assertion still needs current authority.

Finite derivation rules are regular first-conjugation present active indicative
from confirmed first-person/infinitive anchors, first-declension case/number from
nominative/genitive singular, and regular `-us/-a/-um` adjective agreement from
three gender anchors. Confirmed masculine nominative singular PPP can use the
same regular adjective rule. The references are Allen and Greenough (DCC/Ayer
2014), first conjugation, first declension and first/second-declension adjectives.
These rules propose individual requested cells, not exhaustive paradigms. They
require applicable evidence at acceptance; an ending alone is not authority.
Unsupported or contradictory anchors/cells are reported. Targets need not already
occur in the source: `amāre → amās/amant`, `puella → puellam/puellārum`,
`bonus → bonam/bona` are contrasting examples. Verb prompts specify all five
features; noun prompts specify case/number, adjectives also gender. Answers and
stems are kept off production fronts.

For translation, recognition or cloze, `form` names an explicit source form;
`anchors` can be empty. Clozes require exactly one `____`, an unambiguous complete
target, a non-leaking lexical cue, and explicit recipe opt-in. PPP recognition and
PPP translation use their distinct recipes; PPP production uses opted-in
`adjective_declension`, and PPP clozes use opted-in `form_cloze`.

Each phrase item has `object_key`, an existing `Selection` under `selection`,
`recipe`, `target`, `distinction`, and an existing contextual `Reference` under
`reference`. The four supported families reuse existing government, agreement,
sense and valency checks and the shared phrase renderer: preposition–noun,
noun–genitive, adjective–noun, and active verb–noun mini-clauses. Translation has
`target: null, distinction: "meaning"`; recognition/cloze names a constituent role
and a non-meaning distinction. References enumerate complete contextual readings,
translations, alternatives, role contributions and explanations. No corpus syntax
or new vocabulary is discovered.

## Preview and manual CSV

```bash
uv run latinitas-cards pipeline validate --profile pipeline.json
uv run latinitas-cards pipeline migration-plan --json
uv run latinitas-cards pipeline adapt --kind phrase-inputs --profile pipeline.json \
  --objectives objectives.json --output run --preset grammar-study --save-profile study.json
uv run latinitas-cards pipeline preview --run run --json
uv run latinitas-cards pipeline preview --run run --include-recipe ppp_recognition --json
uv run latinitas-cards pipeline export --run run --output new-cards.csv --json
```

Preview-only overrides never mutate an existing run. To export different recipe
choices, supply the saved profile with `export --profile`; existing acceptance
is reusable only where its linguistic binding is unchanged. Explicit object keys and
recipe/target/distinction identify exercises; changing wording or authority does
not allocate another semantic identity. Legacy v1 inputs, layouts, identities and
German characterization outputs are unchanged.

Each contrasting target is a distinct exercise note. Compatible recipe slots on
one exercise note are merged into one CSV row; CardKey never substitutes for the
first-field note identity. A continuation row preserves currently eligible,
previously emitted sibling slots on each touched note; `exported` counts only new
CardRefs and `carried_siblings` lists those retained fields. Processing ceilings
count selected objectives, including automatically enumerated source-form
objectives, after filters. Supported phrase bindings are explicit, never a Cartesian product.
The versioned `selection-v1` ranking visits source identities and patterns in lexical
order, then role/form keys. `variants_per_binding` bounds supported alternatives
within a pattern's entry/sense role tuple, not the number of standalone exercises.
Duplicate CardRef pairs are suppressed; conflicting representations are withheld.
After acceptance, equivalent assertion sets are suppressed only within the same
source/object, recipe, target, language and tested distinction. Visible text alone
never merges different learning objectives or source notes.

### Full output and optional batches

With sufficient processing limits, the default `selection.output_limit: null`
exports all eligible selected cards, including more than 20. A positive
`--output-limit` overrides the profile's output limit, after filtering, processing,
acceptance and suppression. Legacy phrase `max_exercises` is not an output quota.
Preview/export expose `suppressed`, `accepted_unselected` and
`deferred_source_work` separately; deferred objectives are not linguistic rejections
or counts of undiscovered candidates. Provider ceilings remain separate; these
offline consumers never send requests or renew provider budgets.

```bash
uv run latinitas-cards pipeline export --run run --output batch-1.csv --output-limit 10 --json
# Use the returned receipt_id, not an offset, to request the next batch:
uv run latinitas-cards pipeline export --run run --output batch-2.csv --output-limit 10 \
  --after-receipt RECEIPT_ID --json
```

Repeating either request replays its receipt and original files, even if a new
output path is supplied. It never silently advances. Receipts bind ordered CardRef
pairs, effective options, snapshot and output hashes; identical slot keys on
different notes remain distinct. Every committed ancestor output is hash-checked
on continuation, reconciliation and replay; retain those CSV files. Tampered output
and foreign/stale predecessors are rejected. Files from another batch are never overwritten. Run-local locking
rejects concurrent export/decision writers with `run_busy`; a durable staged
publication recovers before another batch. Recovery rechecks current eligibility,
and does not publish a staged batch after its snapshot becomes stale. A retained
private stage establishes output ownership by file identity, not equal bytes;
an unrelated destination is preserved and reported as a conflict.

For renewed processing limits or confirmed source changes, pass an updated hashed
`--profile` to preview, `stage decide` and export. Changed entries invalidate their
dependent assertions, while unchanged entries retain valid decisions. A changed
selection/eligibility snapshot requires `--reconcile-receipt RECEIPT_ID`, not
`--after-receipt`. Reconciliation carries forward previously emitted pairs,
separates retained from now-ineligible pairs, and selects only the remainder.
This is local selection history, not an installed-card retirement or destination
handoff. No source edit or batch can enable the disabled automatic policy.

Complete application analyses can be placed in `run/analyses/` and projected
through `pipeline stage decide --run run`. This consumes strict current bindings,
versions, assertions, check envelopes and persisted stages; it cannot establish
semantic authority from a matching model answer. The profile's requested policy
cannot enable an installed policy. No public bypass or policy-registration command
exists. Synthetic policy fixtures belong only in tests.

Successful counter stages also retain a `latinitas-source-comparison` engineering
proof. It reconstructs supported claims from the current mapped source and
installed references, independently of both answers. Each assertion has a
separate comparison result and commitment to the actual proposal/counter output
bytes. The blind envelope commits to the saved proposal-free task/request and
counter response. Decision/replay readers reconstruct the proof and verify the
actual request bodies, response artifacts, scope and current versions offline.

Supported claims include the narrow context-free active -ō `present_1s` source-role
label, [T234's finite attested target cells](#attested-morphology-evidence-t234), and
[T235's named segmentations and exact citation contexts](#source-bound-components-and-contexts-t235).
T237 adds bounded bilingual meaning, explanatory/role clauses and complete rendered
projections for its finite form and phrase examples. These produce equivalent,
contradictory, compatible-alternative, ambiguous or unresolved diagnostics without
qualifying arbitrary prose or naturalness.
With both answers and no open objections or unqualified alternatives, wrong
features, components or cited contexts yield `contradiction`. Missing assertions,
open objections, unqualified alternatives and unsupported classes/contexts remain
`unresolved`. No additional segmentation convention is installed; compatible
meaning alternatives are not evidence for such a convention. Matching unqualified
translation, segmentation, lexical/sense or explanatory prose supplies no evidence.
This does not verify arbitrary regular-paradigm applicability (including dō/dāre →
dāmus), arbitrary prose, lexical truth or arbitrary complete rendered answers.

The proof is diagnostic, with `authorizes_acceptance: false`; it is not a generic
model-acceptance envelope. The shared acceptance reader now reconstructs model
checks for the complete original card from current mapped sources, actual saved
requests/outputs and the verified stage graph. It retains the original whole-task
blind check and every semantic partner, including unresolved siblings. Blind-only
checks, trimmed assertion sets and edited/detached proofs cannot replace that set.
Verified proof structure is not eligibility: bounded complete-card positives remain
research diagnostics, and every unsupported lexical, translation or rendered sibling
still withholds the whole card. Policy authority is checked separately and ordinary
automatic learning export stays disabled. Other producer classes still need
class-specific evidence contracts and separately lawful, qualified reference data.
Offline synthetic transport tests are engineering checks, not product evaluation
or completion of the v0.5 semantic prerequisite.

Automatic decisions are immutable revisions keyed by the exact analysis bytes
and, for model-assisted evidence, the counter-stage commitment. Unchanged evidence reuses its
decision; malformed/stale analyses are reported
per card and do not prevent independent siblings from being decided.

## Whole-card producer inventory (T233)

This inventory is grounded in source revision
[`cb28344`](https://github.com/fmueller/latinitas-cards/commit/cb283442cd6b84adff6ffc34f4834a6e2d0ac48d).
It implements the revision55 P3 inventory prerequisite, not producers, evaluation,
policy enablement or a reduction of v0.5.0. The governing requirements remain
[shared evidence](../specs/v0.5.0.md#shared-evidence-and-automatic-acceptance),
[the assertion/check matrix](v0.5.0-contract-proposal.md#4-evidence-comparison-and-reuse),
[T228](../planning/tasks/T-228-verify-model-assisted-checks-in-acceptance-readers.md)
and [T200](../planning/tasks/T-200-evaluate-held-out-automatic-acceptance-against.md).
All citations below describe that source, not a claim of linguistic qualification.

### Recipes, assertions and applicability

The finite [SUPPORT_REGISTRY](../src/latinitas_cards/acceptance_registry.py)
represents 15 recipes, two output languages and exact materials/roles/targets.
Representability does not license a Cartesian product or establish evidence.
`pipeline_consumers.current_cards` calls `_render_form`/`_render_phrase`, then adds
the `rendered` label assertion for the complete prompt/answer. Form cards also
carry `source` and `target` labels, translation when requested, segmentation for
derived stems, and every explanation/caveat. Phrase cards carry `construction`,
each reading's translation, explanation and role contributions. A passing target
never covers those siblings. See [the renderer](../src/latinitas_cards/pipeline_consumers.py).

| Registry recipes / material | Actual proposal boundary | Evidence still required for whole-card automatic acceptance |
| --- | --- | --- |
| `form_translation`, `form_recognition` / `source-form/1` | Confirmed selected source form; recognition excludes the infinitive role in the registry. Only active -ō citation `present_1s`, with no anchors, has the narrow rule target check. | Lexical/sense truth and rendered clauses; translation additionally needs sense, government and meaning equivalence. Other roles/features need independent morphology/applicability. |
| `ppp_translation`, `ppp_recognition` / `confirmed-ppp/1` | Explicit confirmed PPP role plus role evidence, never supine/fourth position alone. Recognition retains isolated ambiguity. | Independently supported participial features, contextual number/gender/case/voice where asserted; translated participial meaning, not lemma gloss; all lexical/rendered siblings. |
| `verb_production`, `noun_declension` / `source-form/1`; `adjective_declension` / `source-form/1` or `confirmed-ppp/1` | `form_recipes.derive_form` proposes finite regular cells with full feature tuples and anchor-shape checks. | Independent paradigm applicability, exceptions, generated surface, stem convention/sound changes and complete source/rendered coverage. A rule citation is not proof its paradigm applies to this lemma. |
| `form_cloze`, `phrase_form_cloze` / `target-cloze/1` | Explicit supported non-meaning target/distinction and context; cue must not disclose answer. Neither is enabled by the default or grammar-study preset. | Contextual applicability, supported alternatives and answerability, constituent evidence for phrases, plus all translation/explanation/rendered siblings. |
| `phrase_translation`, `phrase_recognition` / `preposition-noun/1`, `noun-genitive/1`, `verb-noun/1`, `adjective-noun/1` | Explicit controlled selections and reviewed reference readings; not original-corpus role discovery. | Each constituent's bound morphology/sense, agreement/government/valency, contextual meaning and every reading's clauses. Pattern validation alone is insufficient. |
| `rule_application`, `formation_rule_recall`, `component_explanation`, `mnemonic_recall` / `curated/1` | Separate curated consumer implements these recipes and shared-profile/production-run linkage; generic curated assertions are not produced by pipeline `_analyze`. | Genuine current rule, rendered-card and mnemonic approvals as applicable; checks can flag errors but cannot replace human approval or grant linked production cards authority. |

`acceptance_records.Assertion` supports label, segmentation, translation and
explanation; translation alone requires `selected_sense`. Its `Check` methods are
`library`, `rule`, `blind_model`, `semantic_comparison`, `human`. These types validate
shape, evidence and objections, not truth. Exact method applicability is:

| Method / source owner | Real production and immutable reading | Limit / discriminating evidence |
| --- | --- | --- |
| Rule: `pipeline_consumers.source_role_target`, `form_rule_checks`; caller `pipeline_orchestration._analyze` | Reconstructs source-bound active -ō `present_1s` target and exact feature tuple from confirmed mapping; produces per-assertion envelopes/stages. | Derived cells return missing independent applicability; source/sense, translation, segmentation and clauses remain unsupported. `grammar_pipeline_test.test_source_bound_rule_producer_keeps_production_policy_disabled` has target pass, false regularity explanation unsupported and empty export. `test_source_bound_rule_rejects_candidate_claims`, `test_source_bound_rule_does_not_infer_exception_applicability` and `test_source_bound_rule_requires_confirmed_citation_role` discriminate wrong features/forms/alternatives, dō/dāre and role/citation failures. |
| Library: `deck_evidence.analyze_deck`, `_corroborate`, `_sense` | Source spans/identity, lemma/POS/full features and sense are separate checks. Exhaustive supported readings can corroborate morphology; fixed prepositional government can filter ambiguity. Sense pass needs one evidenced matching gloss/key and consistent references. | `latin_library.LocalStanza` is non-exhaustive, returns no senses, never downloads; absent/unreadable resources are processing-unavailable, not linguistic rejection. Library integration positives use supplied test libraries, not a qualified installed lexicon. In `_analyze`, only explicit selected form objectives use this optional adapter; evidence is attached, unavailable downgrades checks, contradiction vetoes a rule pass. It does not promote broader library checks into pipeline sibling passes, and phrases/derived objectives do not acquire a library producer by registry membership. |
| Construction rule: `deck_evidence.analyze_construction`, `_construction_issues`; `phrase_inputs.construction_diagnostics` | Rechecks persisted constituent stage closure, source bindings, compatible senses/context, required features and four bounded families. | Fixed-government prepositions only; dual government is unresolved, passive valency is unsupported. Unresolved constituents withhold composite evidence. This API is not called by pipeline `_analyze` for phrase objectives and does not prove translated/explanatory prose. |
| Blind model: `providers._request`, `ProviderRun`; `pipeline_orchestration._task`, `execute`, `_model_envelope` | Both roles receive the source/task/schema, not each other's answer. `_task` selects mapped spans and currently supplies empty `reference_constraints`. Saved actual request/response bytes and successful invocation/stage links are verified. | A complete transport/coverage proof says neither that assertions are correct nor that adequate independent reference evidence exists. Generic `providers.compare_answers` is structural identity only. |
| Semantic comparison: `semantic_comparison.compare_source_answers`; readers `read_source_comparison`, `read_acceptance_model_checks` | Compares each original card assertion after both immutable invocations; binds outputs, retains whole-task blind coverage, objections and alternatives. `AcceptanceReader.model_checks`/`assess` regenerate the complete set, not caller-selected target-only checks. | Only the supported source-role label gets `equivalent`; wrong source surface/features get `contradiction`. Unsupported siblings, missing coverage, context, objections and unqualified alternatives remain `unresolved`. `compatible_alternatives` is representable but no producer branch emits it. |
| Human: `acceptance_adapters.adapt_deck`, `adapt_corpus`, `verify_projection`; `curated.record_reviews`, `approval_status`; `curated_cards.render_cards` | Current explicit review projects only the bound reviewed assertions. Curated rule/rendered/mnemonic histories remain separate approvals. | Added factual clauses, stale bindings, missing language-specific/rendered/mnemonic reviews cannot inherit another approval. Human authority is not automatic policy authority. |

Source owners: [deck evidence](../src/latinitas_cards/deck_evidence.py),
[local library](../src/latinitas_cards/latin_library.py),
[form recipes](../src/latinitas_cards/form_recipes.py),
[phrase recipes](../src/latinitas_cards/phrase_recipes.py),
[providers](../src/latinitas_cards/providers.py),
[orchestration](../src/latinitas_cards/pipeline_orchestration.py),
[comparison](../src/latinitas_cards/semantic_comparison.py),
[acceptance](../src/latinitas_cards/acceptance.py),
[human adapters](../src/latinitas_cards/acceptance_adapters.py),
[curated approvals](../src/latinitas_cards/curated.py).

### Public evidence and synthetic controls

These focused tests pin different boundaries; they are not interchangeable proof:

- [semantic_comparison_test](../tests/integration/semantic_comparison_test.py):
  `test_source_comparison_proof_through_offline_consumers` and
  `test_acceptance_reader_regenerates_complete_model_checks` exercise saved synthetic
  HTTP invocations through actual pipeline/reader paths without enabling policy.
  The target passes, source/translation/rendered partners do not; receipt cards are
  empty. `test_agreement_does_not_supply_source_evidence` contrasts agreed wrong
  person/surface, open objections, missing assertions, alternatives, changed sense
  and contextual applicability. Tampering and changed-execution tests reject
  leakage, edited/detached proofs and stale commitments. This is engineering proof,
  not MiniMax quality evaluation or broad semantic success.
- [deck_evidence_test](../tests/integration/deck_evidence_test.py):
  `test_government_disambiguates_without_blanket_source_authority`,
  `test_asymmetric_government_and_ambiguity`, `test_four_family_construction_evidence`
  and `test_verb_person_and_number_are_not_interchangeable` test finite adapters
  using supplied libraries. Negative controls include incomplete analyzers,
  missing actual local resources, incompatible same-key senses, changed context
  and passive constructions (`test_incomplete_analyzer_hypothesis_cannot_reject_or_disambiguate`,
  `test_library_only_control_and_actual_missing_local_resources`,
  `test_inconsistent_same_key_references_never_pass_in_either_order`,
  `test_context_dependent_constituent_cannot_be_reused_without_its_context`,
  `test_active_valency_cannot_license_passive_constructions`). Those fixtures do
  not establish that a production exhaustive library/sense source is installed.
- [form_recipes_test](../tests/unit/form_recipes_test.py) contrasts finite regular
  cells and rejected incomplete/inapplicable targets. This tests proposals, not
  independently established paradigm applicability.
  [phrase_recipes_test](../tests/integration/phrase_recipes_test.py) preserves
  ambiguous readings and withholds a single-case guess; English requires its own
  accepted reference and isolated possibilities are not contextual evidence.
  Those are legitimate reference-guided alternatives, not an installed automatic
  `compatible_alternatives` semantic producer.
- [grammar_pipeline_test](../tests/integration/grammar_pipeline_test.py)
  `test_public_controlled_families` uses `install_synthetic`/`prove_cards`.
  Its passing cards, production positives and sibling/installed-checker fixtures
  prove rendering/gate/invalidation behavior under injected authority, not real
  automatic producers. `test_source_bound_rule_keeps_library_objections` injects
  analyzer states only: contradiction fails, unavailable stays unresolved,
  unsupported does not veto the narrow source rule. Do not cite that as an actual
  installed-library contradiction. The actual missing-resource test above is
  separate. `test_bound_human_authority_is_distinct_from_automatic_policy` and
  [acceptance_adapters_test](../tests/integration/acceptance_adapters_test.py)
  cover bound human authority and added-clause/missing-review refusals.
- [curated_cards_test](../tests/integration/curated_cards_test.py)
  `test_saved_grammar_study_integrates_production_without_approval_substitution`
  exercises shared profiles and linked curated rehearsal while production stays
  withheld without its own authority. Its later injected production human decision
  is an engineering fixture, not a native approved baseline.
  `test_export_requires_both_exact_approvals`,
  `test_every_displayed_mnemonic_needs_language_specific_approval` and
  `test_formation_target_edit_requires_new_rendered_review` pin the separate rule,
  rendered-card and mnemonic requirements and changed-target refusal.

### Isolated evaluator: development-only research contract

`latinitas_cards.research_classification` provides a public Python boundary over
an existing pipeline run. `classify(root)` reconstructs every current card from
the saved profile and source, then reloads analysis, both blind invocations and
the recursive source-comparison proof. It never dispatches inference or changes
the installed registry. `save(path, record)` writes an exclusive research file;
`read(root, path)` reconstructs and compares the complete record, not its flag.

```python
from pathlib import Path
from latinitas_cards.research_classification import classify, read, save

root = Path("workflow/run")
for index, record in enumerate(classify(root)):
    path = Path("research") / f"card-{index}.json"
    save(path, record)
    assert read(root, path) == record
```

The schema is `latinitas-research-classification`, integer version 1. Its fixed
`development-complete-evidence/1` criteria are development checks, not approved
statistical study thresholds. `would_accept` means every assertion has source
evidence, complete blind and semantic checks, no open alternatives or objections,
and passes shared evidence validation. `would_withhold` retains missing evidence,
contradictions, ambiguous/compatible interpretations and unsupported siblings.
`unscorable` retains known assertion slots when saved evidence cannot be processed
or validated. An invalid source/run inventory raises rather than inventing slots;
the study caller must retain that failed input separately. Unscorable files can
be saved for diagnosis but cannot be reused by `read`.

Each record retains the full assertions, per-assertion scope cells and obligations,
all relevant individual checks/applicability outcomes, reasons, alternatives,
objections and processing failures. It binds the implementation, support registry,
versions, policy identity, exact scope and criteria, source/input/config binding
and recursive stage commitments to actual saved requests and outputs. Source rights
and reference commitments remain in those checked source and evidence records.
The evidence validator is shared with `acceptance.assess`; only production
`assess` applies policy authority and stored decision-state requirements.

No research file is an acceptance decision or human approval. The ordinary
preview/decision/reuse/export boundaries still require the installed production
policy; the stock disabled policy emits zero learning rows even for research
positives. The finite T237 reference and producer limits above remain in force:
six present-active amō cells and four exact phrase families in German/English,
not general Latin semantics, naturalness, real hosted inference or native Anki.

T238's independent installed checkpoint matched 22 development oracle cases over
120 assertion slots and passed 15 named integrity/restoration controls. Its ten
bilingual positives cover **amō first singular only** and the four phrase families,
not six complete form-cell journeys. Eleven overlapping positive-store denial
checks produced zero learning CSV rows. Contradictory or unresolved non-rendered
siblings also withhold the aggregate rendered projection; unchanged rendered text
alone is insufficient. The original text-only oracle mismatch and subsequent
prospective correction remain recorded rather than retroactively counted as a pass.

The checkpoint reused an identical isolated wheel and offline dependencies, not a
fresh dependency solve. Fresh synthetic local HTTP used an explicitly injected
length-based input-token bound; no semantic, policy or saved-reader patch was used.
Stock dispatch still denies missing token bounds. Explicit `resume --offline`
still denies export but stales the previous research record; exact original-store
restoration permits saved reuse again. These are development limits, not study or
hosted/native qualification.

The allocated implementation of proposal E below introduces the research record
(`latinitas-research-classification/1`). Freeze
candidate code/data, exact policy scope/criteria, support registry, comparison,
bindings, versions, source rights and input manifest. Retain the complete card's
assertions, every check/result/applicability reason, required coverage, actual blind
requests/output/stage commitments, alternatives, objections and failures. Classify
`would_accept`, `would_withhold`, or `unscorable`; distinguish missing evidence,
processing failure and contradiction. No stored accepted decision, human origin
conversion, ordinary learning CSV or import-ready handoff may result. Research
classification separates candidate semantic sufficiency from production authority;
it must not discard checks, treat unknown applicability as success, or reinterpret
matching model prose as truth. All original unsupported siblings still withhold.

Reuse the underlying check-envelope, coverage, source-comparison and recursive
proof-validation semantics of `acceptance.assess`/`AcceptanceReader` and the
orchestration readers; isolate authorization rather than copying a weakened gate.
The extraction is `acceptance.assess_evidence`; production policy authorization
remains in `acceptance.assess`. Independent fidelity
validation must compare per-assertion reasons and whole-card classifications on
lawful source-grounded positive, incomplete, contradictory and ambiguous development
cases, including altered saved requests/graphs. Ordinary disabled-policy preview,
decision, replay and export must still deny the same cards. Synthetic enabled
registry fixtures cannot qualify fidelity. This validation precedes any T200 study
freeze; classification alone does not establish reference truth or enablement.

### Bounded follow-up proposals (unallocated)

These proposals are separately verifiable, not new Taskrail IDs or assignments.
All are anchored to [shared evidence](../specs/v0.5.0.md#shared-evidence-and-automatic-acceptance);
E also depends on [confidence calibration](../specs/v0.5.0.md#domain-specific-confidence-calibration).
All depend on this inventory and preserve disabled policy, full sibling coverage,
curated human approvals and explicit unsupported outcomes. No reference acquisition,
protected study or hosted call is authorized by these proposals. Before using a new
reference, establish lawful rights, exact edition/locator/digest, applicability and
independent expected outcomes; currently no qualified broad reference is established.

- **A — Morphology/applicability and derived cells.** Owners: `deck_evidence`,
  `latin_library`, `form_recipes`, `pipeline_consumers`, orchestration integration.
  Supply bound paradigm/exception evidence and independent generated-cell checking
  for the actual supported verb/noun/adjective cells, not just regular anchor shapes.
  Positive: independently supported amāre → amāmus and asymmetric noun/adjective
  case/number/gender cells. Negative: dō/dāre must not acquire dāmus, missing voice
  or incomplete/non-exhaustive readings cannot pass. Checkpoint: public pipeline
  persists/reconstructs applicability and surface checks, invalidates changed
  reference/checker bytes and withholds every still-unsupported sibling.
- **B — Segmentation and contextual PPP/cloze.** Owners: `form_recipes`,
  `phrase_inputs`, `deck_evidence`, renderers and source comparison. Depends on A
  where generated cells are used. Supply named segmentation conventions, component
  roles, reconstruction/sound changes and explicit source-bound PPP/context targets.
  Positive: independently supported alternative puell- + -ārum / puellā- + -rum
  conventions, confirmed PPP in a disambiguating context. Negative: unsupported
  segmentation, supine-as-PPP, isolated case guesses or answer-revealing cues.
  Checkpoint: retain qualified alternatives without inventing equivalence; contextual
  recognition/cloze and rendered stem clauses have complete evidence or withhold.
- **C — Lexical sense, government and controlled construction.** Owners:
  `latin_library`, `deck_evidence.analyze_deck`/`analyze_construction`, `phrase_inputs`,
  pipeline analysis/comparison integration. Depends on A/B for affected constituents.
  Supply rights-qualified lexical/sense/voice-specific valency evidence and connect
  actual constituent proofs to pipeline phrase checks, not run-supplied truth.
  Positive: supported fixed-government and all four construction families with exact
  senses. Negative: dual government without context, inconsistent same-key sources,
  passive licensed only by active valency, missing constituent or stale source.
  Checkpoint: public phrase journey reconstructs each constituent plus construction;
  lexical/semantic ambiguity is retained and all dependent cards withhold on a gap.
- **D — Translation, explanation and complete-card semantic coverage.** Owners:
  `form_recipes`, `phrase_recipes`, `semantic_comparison`, providers/orchestration
  and shared reader. Depends on A–C for asserted facts. Decompose rendered factual
  clauses and supported meaning propositions by selected sense, features, language
  and context. Compare legitimate supported alternatives independently of wording;
  preserve proposal-free input and source/reference citations. Positive: independently
  grounded German/English paraphrases with identical meaning and complete factual
  clause coverage. Negative: agreeing wrong models, changed person/number/voice,
  gloss overlap with different senses, true label plus false causal explanation,
  omitted reading or unsupported sibling. Checkpoint: real public complete-card
  evidence classifies discriminatingly while ordinary automatic CSV remains empty;
  one positive cannot stand for untested recipes/materials/languages. Curated rule,
  rendered-card and mnemonic approval remains human, not part of this automation.
- **E — Isolated research classification and fidelity.** Owners: shared acceptance
  semantics, orchestration proof readers and the `research_classification` public
  caller/versioned records implemented in T238. Depends on A–D's scoped producers
  for source-grounded whole-card positives; record/denial plumbing can be prepared
  earlier but cannot claim qualification. T238 implements the versioned contract
  above. Positive: qualified development evidence produces would-accept
  classification without accepted output.
  Negative: incomplete, stale, contradictory, ambiguous or tampered evidence,
  disabled-policy learning export and fabricated research records fail closed.
  Checkpoint: independent fixed-candidate public-boundary fidelity validation,
  frozen code/config identities and denial controls before T200's separately
  authorized reference/custody/feasibility/criteria/freeze gates. No study runs here.

### Checker revisions and native baseline authority

A genuine implementation pair already exists:
[`068162e` (T227)](https://github.com/fmueller/latinitas-cards/commit/068162ed4dad216a9d82629fa2c99cacc1ec1a5d)
→ direct child
[`831e2ed` (T228)](https://github.com/fmueller/latinitas-cards/commit/831e2ede04de7e86cb6c39a079ffdf18b4107d07).
T228 changed `acceptance.assess`, `AcceptanceReader` and the orchestration readers
to reconstruct complete model proofs. Its task records independent review and
checkpoint evidence. This is real checker/validator code, not a version-only bump.
Historical source-byte reconstruction gives implementation digests
`c06df46710a5e9427a5669f79303b9996027e6cbea30c6191d0a9a97a10a874c`
and `6700c6620383723eeda71726d08ffd7bd2bcef2439d532c146c847f418842847`
(the latter matches the T228 checkpoint). At this inventory's base the digest is
`b0e433930e3ff1c5c6629c38d904bacc5f0b8a456460544eaaa2f08dc2359f97`.
`acceptance_policy.implementation_digest` hashes all package Python except authority
records; later executable changes also alter it. Digests alone do not prove a
checker change, and these reconstructions are not installed-package attestations.

The [v050 native kit](native-v050-acceptance.md#application-checker-revision-against-all-installed-batches-plannedblocked)
remains planned/blocked: no genuinely reviewed/imported v050 accepted baseline,
closed-database identity/history inventory or native replay on this pair is
established here. T201 remains unexecuted. Historical
[v040 native evidence](release-v0.4.0.md) exists, but is not v050 automatic authority
or proof of populated review history. The installed-checker regression in
`grammar_pipeline_test.test_installed_checker_change_covers_all_batches_and_corrupt_history_blocks`
injects synthetic enabled authority, passing checks and a patched digest; it proves
receipt/invalidation behavior, not native installation or genuine accepted truth.

A genuinely reviewed human grammar baseline could test real checker-bound
invalidation without automatic enablement: `assess` still binds human decisions
to checker versions. That installed baseline and native execution are missing.
An automatic baseline additionally requires supporting T200 evidence and separate
maintainer enablement. A–D could supply genuine future checker changes, but none
may invent a revision/digest just to exercise invalidation. No producer proposal
substitutes for T201's actual native baseline/import/update/omission and history
checks or for required curated approvals.

### Execution limits retained

This documentation task performs no hosted calls, campaign restoration, reference
acquisition, protected-data access or judging. Revision55 remains the same campaign:
four development calls remain charged, including the false silvae blind counter;
no new budget or reliability claim follows. Guaranteed mode, rights/custody,
disabled authority, T200/T201 lifecycle and the T205 publication stop are unchanged.
No local Ollama or manual maintainer test is a prerequisite to this inventory.

## Attested morphology evidence (T234)

The T233 inventory above describes its pinned base. T234 added a bounded
source-backed producer for **derived target labels**. It reads installed
paradigm tables independently of `derive_form`, matches the confirmed source lemma,
POS, exact anchors and complete target features, and compares the proposed surface
against the attested cell. Input declarations, anchor suffixes and model agreement
cannot extend its coverage. This reusable boundary does not qualify a general Latin
analyzer or license all lexemes in a productive class.

The retained reference is Charles E. Bennett, *New Latin Grammar*, 1918 edition,
Gutenberg 15665 (2005-04-20), quantity-preserving GITenberg HTML. Original bytes and
full license are installed in `references/bennett-1918.html`; SHA256 is
`13f80dd1202e958854117c9c9cb2914630e92e37de2967791a45237f571327e6`.
The source is public domain in the US; the attached license retains its outside-US
jurisdiction caveat. See [rights and acquisition ledger](t234-reference-acquisition.json)
and [third-party notices](../THIRD_PARTY_NOTICES.md). No modern translation is used
as linguistic authority. These are exposed development sources, never holdouts.

| Generator proposal scope | Independently attested coverage | Still unsupported |
| --- | --- | --- |
| Six first-conjugation present indicative active cells | §101: amō, anchors amō/amāre, all six cells, including amāmus | Arbitrary first-conjugation lexemes, other voices/tenses/moods, inferred classes and exceptions |
| Six cases × two numbers, masculine/feminine first-declension nouns | §20: porta/portae, feminine only, twelve cells | Masculine lexical gender, other nouns including puella, inferred gender/class and exceptional forms |
| Six cases × two numbers × three genders, regular us/a/um adjectives | §63: bonus/bona/bonum, all 36 cells | Other adjectives, pronominal classes, PPP/contextual applicability and unattested exceptions |

The mechanism reads all 54 attested cells. Public production rendering retains its
existing answer-disclosure guard: porta nominative/vocative singular and bonus
masculine nominative singular disclose their answers in their cues, so they remain
withheld rather than relaxing the guard. The other 51 cells receive target-only
diagnostics. §127 explicitly gives dō/dăre and dămus; invented regular dō/dāre
anchors do not match an attested regular paradigm and cannot license dāmus.
Breves are not silently equated to macrons. Unknown/missing lemma, POS, voice,
class, incomplete features, context or resource evidence remain withheld. These
tables are generation evidence, not exhaustive ambiguity analysis of an isolated
surface. Non-exhaustive LocalStanza readings still cannot disambiguate it.

Evidence retains normalized source-section text, edition, locator and the original
file's digest in immutable analysis envelopes. The complete original HTML remains
installed byte-for-byte. Reference bytes join the implementation commitment,
so reference/checker changes invalidate old evidence. The existing proposal-free
blind and complete-card semantic proof readers reconstruct this same target
boundary; they do not inherit truth from agreeing models. Unsupported source/sense,
translation, segmentation, explanation, rendered and contextual siblings remain
unresolved. Human/curated approval and the disabled automatic registry are unchanged;
no whole-card automatic export, evaluator qualification or release readiness follows.

Acquisition is closed at three charged requests, 2,061,202 unique retained source
bytes and a conservative fifteen minutes acquisition work, separate from 1.094
seconds HTTP execution. One HTTP 503 and the quantity-omitting ASCII edition remain
recorded. No hosted inference, campaign restoration or additional allocation occurred.

## Source-bound components and contexts (T235)

`form_evidence` reads the same digest-pinned Bennett HTML, without a new fetch,
analyzer fixture or model inference. `pipeline_consumers.source_form_claims` binds
that evidence to the current mapped lemma, POS, anchors, selected form and objective.
Rule envelopes and complete-card semantic comparisons reconstruct the same claims.
Neither a generated stem nor two agreeing models supplies independent evidence.

The renderer's `stem` assertion now records a named convention, ordered components
with roles, reconstruction operation and surface. Its visible stem/base line is
unchanged. The installed conventions have different source obligations:

- `bennett-declensional-stem/1` uses the explicit **portā-** stem in §20 and
  the stem/case-ending distinction of §17. Only the two attested nonempty
  concatenations **portā + rum → portārum** and **portā + s → portās** pass.
  The other ten porta cells require weakening, coalescence or zero-ending
  evidence not implemented here. Mere successful string reconstruction does not
  establish that a different split is a valid grammatical convention.
- `bennett-present-stem/1` uses the amō/amāre identity and cells in §101,
  present-stem description in §§97/117 and personal endings in §96. Three
  concatenations pass: **amā + s → amās**, **amā + mus → amāmus**,
  **amā + tis → amātis**. The other three present-active cells require changes
  not checked here. This is Bennett's pedagogical convention, not an exclusive
  historical analysis: §117 footnote 39 distinguishes underlying thematic-vowel
  stems such as amā-ĕ-/amā-ŏ-. No inferred sound change is accepted.
- `bounded-generator-base/1` identifies the adjective generator's proposal,
  not a corroborated convention. All 36 bonus component analyses and all
  PPP-derived component analyses remain unsupported. T234 target diagnostics
  remain separate and retain their original finite coverage and cue refusals.

The proposed **puell- + -ārum** termination analysis and **puellā- + -rum**
stem/case-ending analysis are both unqualified: this edition's §17 distinguishes
termination from case-ending and §20 attests porta, not puella applicability.
Neither proposal is silently equated to the installed porta convention.
Unsupported conventions, incorrect quantities/components, invented operations and
same-spelling alternative splits cannot receive an equivalence pass. Material
alternatives and objections are retained; no `compatible_alternatives` producer
is claimed.

Context support is a finite **grammatical paradigm citation**, not free-text
disambiguation or a corpus/lexical resource. §§99/101 independently identify the
amō principal parts; §102 explicitly labels the perfect-passive compounds and
their participles; §100 supplies the auxiliaries. Nominal facts also retain the
applicable chain: §95 classifies participles as adjective forms; §233.3 gives
them adjective construction; §234 supplies gender/number/case agreement; §166
places the finite subject (including an implied pronoun) in the nominative;
§254 supplies finite person/number and compound participle gender agreement.
The attested -us/-a/-um and -ī/-ae/-a variants are read with §63's labelled
nominal columns, not treated as a productive ending-only proof. The masculine
rows give exactly six supported contexts: **amātus sum / es / est**,
**amātī sumus / estis / sunt**.
The first row's parenthesized feminine/neuter alternatives and footnote markers
are removed only to read its explicit masculine variant. No other gender,
tense, lemma, arbitrary sentence or context-free participle gains evidence.

For `ppp_recognition`, `context` must equal one of those complete citation strings.
The prompt shows that exact context. For explicitly selected `form_cloze`, replace
only its participle with one `____`, for example `____ es`, selecting the confirmed
source `amātus` with nominal nominative/masculine/singular/perfect/passive features.
The existing complete-target and nonleaking lexical-cue guards still apply.
This is a constrained inflection completion with specified gender, not a claim
that the blank alone excludes every feminine/neuter completion. The evidence
independently supplies the participial features; run-provided features do not
prove them. The target assertion commits to the exact displayed context, and
reference evidence records the restored citation, target span, auxiliary
person/number, compound role and nominal agreement basis. Both model roles receive
the exact recognition context and selected source-form key in a proposal-free
structured objective; cloze context remains in its registered target input.
Participle features do not inherit the finite compound's person/mood. This
confirms the attested compound use, not an exhaustive
adjectival/participial reading of arbitrary prose. Supines, guessed isolated
cases, incompatible auxiliary/participle numbers and unknown contexts withhold.

The original whole-card `source`, translation, explanation/caveat, rendered-card
and curated obligations are not trimmed or granted authority by these diagnostics.
All remain unsupported here. Changed source, context, reference or checker bytes
invalidate prior commitments; blind tasks remain proposal-free, and ordinary
automatic learning exports remain denied under the disabled installed policy.
No held-out study, native acceptance, enablement or release-scope reduction follows.

The installed checkpoint's controlled loopback dispatch uses an explicit request-byte
accounting bound; an unconfigured initial provider dispatch safely defers with
`missing_token_bound`. Normal resume can reuse saved calls with the server stopped.
Explicit `resume --offline` instead installs LLM-off active wrappers: historical
proofs remain immutable, but active model checks are not preserved by that mode.
Neither behavior grants export authority or is a T235 provider/accounting change.

## Source-bound lexical and construction diagnostics (T236)

Phrase recipes assert a structured `construction`, one translation, explanation
and role-contribution assertion per reference reading, and the complete rendered
front/back. Their supplied pattern, gloss, features and reference prose are input
claims, not independent corroboration. `phrase_evidence.phrase_claims` now produces
constituent `analyze_deck` reports and a composite `analyze_construction` report
for the actual pipeline analysis. Source comparison and its immutable reader
reconstruct the same proofs from current inputs and installed reference bytes.

The installed `BennettLexicon` reads the already-retained 1918 *New Latin Grammar*
HTML, including its original Project Gutenberg terms. Its exact byte digest and
edition are recorded by `bennett_sections`; no additional reference was acquired.
The lexical adapter's stable keys and exact English glosses are **gate** (`porta`,
§20), **garden** (`hortus`, §23), **boy** (`puer`, §23), **good** (`bonus`, §63),
and **love** (`amō`, §101). They are not general dictionary coverage. Other keys,
glosses, languages and unreferenced valencies remain unsupported. The active love
reading has nominative subject and accusative object, grounded by §§170, 172,
174–175, including *parentēs amāmus*. It is not a passive valency or an assertion
that every use of *amō* requires an overt object; absolute use remains outside
this constructed-clause scope.

Exactly 78 qualified morphology tuples are available: the prior twelve `porta`
cells, 36 `bonus` cells and six present-active-indicative `amō` cells, plus the
twelve masculine cells each of `hortus` and `puer` read from §23. No suffix rule
extends those tables to another lexeme, tense or voice. Alternate analyses of
syncretic forms are retained. A constructed constituent proves that its selected
tuple exists in the independent table, **not** that an isolated surface or an
arbitrary sentence has only that reading. Recognition is still non-exhaustive.

The four construction checks are grammatical compatibility, not translated
meaning, naturalness or the truth of a phrase. Fixed-preposition government uses
the finite existing `GOVERNMENT` map with §§141–143; `in`/`sub` do not gain a case
from a supplied pattern. Noun-genitive attachment uses §195 without selecting a
possessive/material/etc. interpretation. Attributive adjective agreement uses
§234; the selected active transitive love clause uses its independently retained
valency and §254 person/number agreement. Public positive controls are
*cum portā*, *porta hortī*, *hortus bonus* and *puer portam amat*.

Each constituent retains its exact identity, mapped form/meaning spans, selected
sense, table alternatives, checker version and source binding. Composite proof
commitments include source reports and immutable stage identities, but exclude
stage wall-clock timestamps so unchanged independent reconstruction is stable.
Actual source, context, constituent artifact, reference or checker changes reject
reuse. Comparison preserves complete assertion coverage and proposal-free calls;
neither agreeing responses nor an input pattern supplies missing truth.
Both calls receive the original supplied entry metadata, identities and pattern
as explicitly unverified input, plus the required assertion IDs and kinds. They
do not receive independent expected proofs or proposal output. HTML and padded
source lemmas use the same normalization as constituent analysis, and reconciled
manifest identities are forwarded unchanged.

Fixed-preposition checks establish case compatibility only: for example, §142's
`ē`/`ex` allomorph restriction is not a rendered-spelling check. The retained verb
table has no passive cells; passive source claims therefore lack constituent
support before any active-only valency can be reused.

Translation, explanatory/role prose, rendered answers and curated approval remain
unsupported siblings. No assertion is removed, automatic policy remains disabled,
and these diagnostic positives cannot export a whole card. Proposal D and the
isolated evaluator remain separate prerequisites. Controlled loopback response
fixtures test transport and saved-proof custody, not hosted model competence,
campaign qualification, native Anki acceptance or policy enablement. Normal
server-stopped saved-call reuse is distinct from the explicit `--offline` active
projection limitation recorded above.

## Trial error regressions and ten-entry replay

The reported trial mistakes are development regressions, not protected evaluation
material. Disposable source-bound comparisons exercise **viās**, not **vīās**;
**laudant** without the unsupported **laudānt** alternative; and isolated **puellae**
with genitive/dative singular and nominative/vocative plural readings. Recognition
keeps all supplied same-surface readings rather than treating plural readings as
exhaustive. These cases do not add Latin tables or lemma-specific exceptions.

Both synthetic model roles can agree on a wrong quantity, an unsupported alternative
or an incomplete rendered answer and still return locally valid JSON. They remain
withheld, as do the correct controls: these source forms have no qualified
independent whole-card evidence. This is safe withholding, not a claim that the
checker diagnoses each linguistic error. Source and rendered semantics can remain
unsupported even where another card's target or stem diagnostic passes. Existing
missing-sibling checks and strict local parsing also reject incomplete coverage
and malformed assertion keys. Schema adherence improves structure, not Latin truth.

For card feedback, give the maintainer the run/candidate identity, recipe, front,
back, disputed text and suggested correction with a source. Use `pipeline preview`
and `pipeline report` to identify the card; do not send private collection files or
credentials. Feedback is not acceptance authority. Correct confirmed input/objective
errors or fix the responsible implementation defect, then regenerate with current
bindings; do not edit sound inputs to match a model error or edit receipts. Optional
[whole-card human review](v0.5.0-workflows.md#library-only-stages-optional-review-and-resources)
is separate from automatic safeguards and is not required for every generated card.

The coordinating maintainer/session owns the private ten-entry inventory, hosted
execution and campaign ledger. After independent high-effort integrated QA, it can reproduce a fresh
same-ten-entry replay as follows:

1. Pin the delivered T-243 commit SHA and `specs/v0.5.0.md`; record the clean code
   state and QA result. Retain the original ten identities and objectives, source
   hashes, recipe/language selections and historical trial receipts. Record any
   intentional profile changes instead of silently changing the sample.
2. Prepare a new run/output identity on that exact candidate. Use the T-241 complete
   answer-free task inventory and presentation/language contracts. Select the T-242
   documented capability for each exact endpoint/model: explicit strict schema
   where supported, no undocumented format parameter for known MiniMax Chat hosts.
   Keep the blind call proposal-free and retain every whole-card assertion.
3. Inspect the existing allocation and authoritative ledger without initializing,
   replacing or resetting either. Reconcile unknown calls before new dispatch;
   failed/unknown calls remain charged. Compute the exact outbound disclosures and
   full wire/accounting reservations. Obtain fresh prospective wire-bound permission
   with current expiry, pricing and scope within the existing phase/campaign caps.
   An expired trial permission or an insufficient remaining budget blocks replay;
   it does not authorize another allocation, hidden retry or model fallback.
4. Dispatch only after those checks. Save new request/response commitments, actual
   outcomes/usage and retained charges alongside the unchanged historical receipts.
   Regenerate the ten-entry report with front/back previews, structural failures,
   per-assertion source comparisons, alternatives and whole-card decisions. Compare
   each original problem against the new observation, including unresolved cases;
   do not call saved receipt reuse a fresh hosted result.

No hosted replay was performed by this regression task. Offline tests do not prove
live linguistic correctness, protected-study qualification or release readiness.
Automatic policy remains `v050-disabled`; protected evaluation, native Anki
acceptance, lifecycle safety and the maintainer's publication gate remain separate.
Neither feedback nor a replay authorizes collection writes, publication or deployment.

## Outcome reports

After `pipeline run` or `resume`, inspect the saved snapshot without provider calls:

```bash
uv run latinitas-cards pipeline report --run workflow/run
uv run latinitas-cards pipeline report --run workflow/run --json
```

The readable report groups reasons with representative source links, examples,
origin, policy revision, alternatives, objections and evidence limitations. JSON
preserves the unversioned execution diagnostics and adds `outcome_report`, with
`schema_version: 3`, `record` and `details`. Schema 3 may report the
`contract_violation` processing reason; historical reports with schema 1 or 2 (equal
to their profile version) stay readable, and claiming that reason under them is
rejected. The strict `record` follows the approved `latinitas-pipeline-report` shape;
`details` holds counts, source links, examples, findings, processing problems and
actual analysis/provider reuse for this invocation.
The envelope schema is available as
`latinitas_cards.pipeline_reporting.Reporting.model_json_schema()`; the approved
record schema is `PipelineReport.model_json_schema()` in the same module. Legacy
execution snapshots remain readable as JSON without fabricated v1 outcomes.
Snapshot profiles use the exact shared profile shape; reading historical policy
identifiers does not install or authorize those policies for execution.
Reports are snapshots, not acceptance authority; a later source/policy change still
requires current validation. Running selection/rendering alone retains applicable
receipt-backed exports rather than treating missing invocation-local export data
as a new deferral.

Outcomes distinguish `not_generated`, `deferred`, `withheld`, `processing_failed`,
`accepted_unselected` and `exported` (accepted and selected into a CSV receipt, not
imported into Anki). Missing token accounting is a deferred prerequisite, not a
linguistic rejection; missing analyzer resources and uncertain/failed provider calls
are processing problems. `malformed_output` covers malformed answers and nested values;
`contract_violation` marks a well-formed form answer that breaks its fixed direction,
presentation or literal stem-concatenation contract. Disputed but well-formed Latin
stays `withheld`. Valid independent authority can still export a card when an
optional stage fails; `outcome_report.details.processing_outcomes` retains that
failure separately. `complete` is false while work, assessments or stages remain
deferred, failed, unknown or cancelled, even if an export operation completed.

`source_units` counts filtered recipe/objective work, including automatic objectives,
not lexical entries. Stages, distinct linguistic claims, generated candidates and
CardRef pairs have separate counts. A shared assertion with different supporting
checks counts once; one failed claim can affect multiple cards. Unattempted work
has null candidate/card IDs; missing recipe content does not invent exercises.
Source links identify confirmed input entries and controlled-phrase patterns by
file URI and JSON pointer. Limits separate call/token reservations and permission
caps from output selection. Renewed processing uses the existing explicit receipt
reconciliation; reporting does not reset budgets or create a second journal.
Untrusted exception details are omitted on publication and read; runtime credentials
are redacted. If redaction collides with a structural identity, reporting fails
without replacing the previous snapshot rather than publishing corrupted hashes.
Readable output escapes terminal controls and aggregates distinct findings across
each reason group; only its exercise example and source links are representative.
Structured findings suffice; private reasoning traces are neither needed nor shown.

Preview reconstructs all source/target/translation/stem/explanation/role/caveat
and rendered-front/back assertions. CSV repeats current-source reconstruction and
the full-card gate immediately before writing. Ineligible siblings stay out while
eligible siblings remain exportable. Existing output files and evidence-store
paths are protected.

`grammar-v1` is an explicit append-only new-note-type proposal. `migration-plan`
prints its complete fields, templates and CSS. Install it manually under a new
name matching the deck profile before importing CSV; no old template ordinal is
repurposed. CSV does not install templates or retire existing cards. Export reports
withheld objectives; it is not a managed handoff, native Anki acceptance, empirical
qualification, release or publication approval.

## Finite bilingual meaning and complete-card diagnostics (T237)

`meaning_evidence` checks source applicability, selected sense, whole utterances,
explanatory/role clauses and the actual front/back projection separately. It never
calls the renderer to establish truth. `meaning_language` contains newly composed
German/English utterances linked to bounded propositions from the retained Bennett
edition; [initial claims and independent dispositions](meaning-reference-review.md)
retain rejected candidates as well as accepted ones. This is an explicitly finite
development language, not a free-prose evaluator or a native linguistic study.

The actual form inventory is `source`, `target`, optional `translation`, derived
`stem`, optional `explanation`, every `caveat-N`, and `rendered`. `source` serializes
the entire entry and selected sense, so qualification checks unselected form/sense
metadata too. The phrase inventory is `construction`, then `translation-N`,
`explanation-N`, `roles-N` for **every** supplied reading, and `rendered`. A passed
construction does not silently qualify its prose. Projection checks verify exact
Latin prompts, instructional framing, each translated reading, every explanation,
all caveats and the absence of additional answer lines. Role clauses remain checked
even when the translation recipe does not display them separately.

Complete **diagnostic** coverage is available in German and English for:

- `form_translation`: the six independently tabulated present indicative active
  cells of *amō* (`amō`, `amās`, `amat`, `amāmus`, `amātis`, `amant`) in the selected
  `love` sense, without extra context/anchors. Every serialized source form must
  have an attested table/anchor reading and every selected feature must agree.
  Primary translations preserve person/number; only the listed present-active
  explanations and bounded-interpretation caveats are qualified.
- `phrase_translation`: the exact singular constructed readings *cum portā*
  (accompaniment), *porta hortī* (unspecified garden relation), *hortus bonus*
  (evaluative good), and *puer portam amat* (boy as lover, gate as loved entity).
  The selected senses, cases, gender, agreement, government and active third-person
  singular verb must independently pass constituent/construction checks. Listed
  different-wording phrase renderings preserve these propositions in both languages.

Each qualified phrase also requires the exact constructed exercise premise in
`reference.evidence`, tagged `constructed-exercise-context`. The renderer displays
that premise on the front and both blind requests receive it as explicitly
unverified context, not a reference answer. The genitive and verb examples stipulate
definite noun reference; the adjective example stipulates indefinite reference.
The *cum* example supplies a hypothetical *profectus est* motion setting and asks
for indefinite noun reference. This supplies the contextual premise of Bennett
§222 without claiming that every bare *cum* fragment expresses accompaniment.
Missing, changed or additional contexts prevent complete meaning evidence.

Qualification of a listed reading does not select every possible contextual
interpretation. Unknown reading alternatives, extra causal claims,
unverified naturalness or an unsupported source sibling leave the card incomplete.
The rejected `feel love` / `Liebe empfinden` proposals are compatible experiential
claims, not established equivalents of the isolated Latin finite predicate.
Finite diagnostics distinguish these compatible claims, recognized conflicting
person/number or lover/loved-role mappings, mixed ambiguous alternatives and
uninterpreted wording. A source-relative contradiction denotes a conflicting
translation mapping, not an assertion that both events are logically impossible.
All unlisted wordings, cells, contexts, senses, phrase variants and languages remain
unsupported. Recognition, cloze, production, PPP, segmentation and curated recipes
retain their prior partial coverage; a translation positive does not qualify them.
Curated rules, rendered curated cards and mnemonics still need actual human approval.

Analysis and semantic comparison use the same independent producer; immutable
readers reconstruct it from current source, reference and checker bytes. Both
provider roles receive the same readable, explicitly unverified source metadata
and requested assertion inventory, but no proposal output, supplied translation or
reference explanation. Controlled transport responses reconstruct only that captured
input. They are not independent linguistic evidence. Source-backed predicates and
independent language review supply the bounded semantic authority, not agreement.

A complete diagnostic requires all assertions and methods; ordinary installed
`v050-disabled` decisions and exports still withhold every automatic card. Exact
request/response commitments, all objections and alternatives remain in the saved
proof. Successful stopped-server reuse is tested before dependency/tamper denial.
No hosted inference, campaign restoration, held-out reference work, policy enablement,
evaluator qualification, release readiness or publication approval follows.
