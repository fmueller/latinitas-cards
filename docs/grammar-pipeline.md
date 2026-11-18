# Bounded grammar cards

`pipeline` is an offline, opt-in consumer of the shared evidence store. It does
not run providers, install analyzer resources, approve a policy, install templates,
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
choices, adapt a new run using the saved profile. Explicit object keys and
recipe/target/distinction identify exercises; changing wording or authority does
not allocate another semantic identity. Legacy v1 inputs, layouts, identities and
German characterization outputs are unchanged.

Each contrasting target is a distinct exercise note. Compatible recipe slots on
one exercise note are merged into one CSV row; CardKey never substitutes for the
first-field note identity. Processing ceilings count selected explicit objectives
after filters. This consumer creates at most one surface per exact explicit
binding; `variants_per_binding` never suppresses a different learning object.

Complete application analyses can be placed in `run/analyses/` and projected
through `pipeline stage decide --run run`. This consumes strict current bindings,
versions, assertions, check envelopes and persisted stages; it cannot establish
semantic authority from a matching model answer. The profile's requested policy
cannot enable an installed policy. No public bypass or policy-registration command
exists. Synthetic policy fixtures belong only in tests.

Automatic decisions are immutable revisions keyed by the exact analysis bytes.
An unchanged analysis reuses its decision; malformed/stale analyses are reported
per card and do not prevent independent siblings from being decided.

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
