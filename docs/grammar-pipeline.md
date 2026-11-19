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

Automatic decisions are immutable revisions keyed by the exact analysis bytes.
An unchanged analysis reuses its decision; malformed/stale analyses are reported
per card and do not prevent independent siblings from being decided.

## Outcome reports

After `pipeline run` or `resume`, inspect the saved snapshot without provider calls:

```bash
uv run latinitas-cards pipeline report --run workflow/run
uv run latinitas-cards pipeline report --run workflow/run --json
```

The readable report groups reasons with representative source links, examples,
origin, policy revision, alternatives, objections and evidence limitations. JSON
preserves the unversioned execution diagnostics and adds `outcome_report`, with
`schema_version: 1`, `record` and `details`. The strict `record` follows the approved
`latinitas-pipeline-report` v1 shape; `details` holds counts, source links, examples,
findings, processing problems and actual analysis/provider reuse for this invocation.
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
are processing problems. Valid independent authority can still export a card when
an optional stage fails; `outcome_report.details.processing_outcomes` retains that
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
