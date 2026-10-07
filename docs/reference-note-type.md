# Reference note type and safe import

This is the manual reference setup for the Latinitas multi-card learning-object model:
the exact versioned fields, the stable template-slot registry, copyable guarded
front/back templates and CSS for both initial recipes, and the safe first/repeat
import guidance. The setup is consumed from — and checked against — the same
authoritative note-schema and slot contract the exporter uses
(`latinitas_cards.notes` / `latinitas_cards.cards`); there is no second field or
slot definition here. The exporter side of that contract, including the
deterministic CSV headers and the prior-export checkpoint, is documented in
[deterministic-csv-export.md](deterministic-csv-export.md).

This is a **manual reference setup**: you create the note type and templates in
Anki yourself. General automated note-type/template provisioning is v0.6.0
roadmap work, and broad cross-platform/Anki-version compatibility certification
is v0.9.0 roadmap work (see `specs/v0.1.0.md`).

## Morphology presentation contract

Profile schema **2** owns `morphology`: `version: 1`, `theme: "muted"` or
`"monochrome"`, `appearance: "light"` or `"dark"`, and `comparison: "static"`
or `"disclosure"`. Defaults are muted/light/static. Effective profile overrides
merge individual settings; setup's JSON report and principal-part preview report
the effective values. Unknown versions, names and modes fail validation. Choose a
theme by editing the saved profile's `morphology` section; `setup` has no theme flags.
Legacy schema 1 profiles **without** morphology settings are explicitly upgraded
in memory to schema 2 with those defaults and serialize as schema 2. Schema 1
with morphology settings is rejected; edit the version to 2 deliberately.
Upgrading the linguistic profile schema requires renewed old claim reviews.

Generation and preview share the same managed answer markup, `morphology-v1`.
Only accepted segmentation/explanation is asserted. Two reviewed pipe components
mean stem | ending; three mean stem | formation marker | ending. Other reviewed
notation stays literal, not inferred. Typography and separators work without color.
Absent and withheld roles are explicit and their proposed explanations are not facts.
The core explanation and compact comparison remain on the revealed answer.
`Stammformen vergleichen` contains the complete four-role handoff and accepted
further explanation, either as a readable static section or native details/summary.
There is no script dependency. **Static is the unverified-client default.**

CSV installs neither CSS nor templates. Paste the versioned reference CSS below
only as a separately reviewed manual setup step. It supports both themes and
light/dark classes on the managed answer, without changing field order, recipe
keys, slots or scheduled template bindings. Existing destination CSS/template
digest mismatches block acquisition pending manual setup/review; no scheduled
template rebinding, provisioning or migration is performed. Presentation changes
alter generated fields and therefore require renewed destination payload approval,
not reuse of an old observation plan. The current destination journal rejects
changed per-card answer fields as unsupported slot effects, even after CSS setup;
this is a migration/application block, not permission to import into a scheduled
layout. Retain the collection and seek separately reviewed manual setup or the
explicit backed-up fresh-start path. No application capability is added here.
Linguistic claim review ignores only
`morphology`; all evidence and linguistic configuration remain bound.
Personal Notes are neither sanitized nor written by generation/CSV.

Browser renders are **not native Anki proof**. The maintainer reported native acceptance
passed on 2026-10-06 for Anki Desktop 26.09.2 and AnkiMobile 25.09 with the synthetic
four-verb kit on the listed devices
(see [morphology-native-verification.md](morphology-native-verification.md));
disclosure is verified only within that recorded scope and static remains the
default/unverified-client fallback. The 2026-10-07 waiver of additional representative-deck
acceptance is not broader compatibility or existing-scheduled-CSS proof.

## How the note type is organized

One generated CSV row is one coherent learning object rendered as **one note**
with zero or more eligible **cards** — never one row per card. The note's fields
split by ownership:

- **Managed regular fields** (`LatinitasID`, `Lemma`, `Principal Parts`,
  `Meaning`, provenance, generation metadata, and each card's
  `Enabled`/`Prompt`/`Answer` triple) are owned by regeneration and may be
  updated by a repeat import.
- **`Personal Notes`** is user-owned, note-level, and shared by every card of the
  note. It is deliberately absent from the generated CSV columns: generated
  import data never offers a personal value, so an import cannot accidentally
  overwrite it.
- **Tags** are note-level metadata shared by every card of the note. In the CSV,
  `Tags` is the special transport column (column 5, directed by
  `#tags column:5`), **not** a regular note field, and it is not a field of this
  note type at all.

All cards of a note are genuine siblings: they share the note's fields, Personal
Notes, and tags, but each keeps its own Anki card ID, review history, and FSRS
scheduling state. Latinitas does not implement or change scheduling.

Because of that split, keep the counts separate too. The CLI preview reports
`Source entries`, `Objects`, `Notes`, `Cards`, `Zero-eligible notes`, `Skipped`,
and `Ambiguous` as distinct numbers; do not read a CSV row count as a card
count.

## The reference setup

The block below is generated from the authoritative contract (note schema and
frozen template registry, with the registry digest) and is verified verbatim by
a test; it changes only when that contract changes.

<!-- latinitas-reference-setup begin -->
Reference setup for note schema `3` and template registry `v1` (digest `card-registry-sha256:b1e7709e3db8ac17645c0254d9cecfa2d558a105b9b40e87bfcc329ef0adb96d`).
This block is generated from the authoritative contract; a test fails if this
published copy drifts from it. Do not hand-edit inside the markers.
Reference styling `v2` supports morphology markup `v1`.
CSV carries managed markup, never installs CSS or provisions/rebinds templates.
Existing destinations with a different CSS/template digest are blocked for
separately reviewed manual setup; do not rewrite scheduled templates.

### Reference fields (43, exact order)

Create exactly these fields, in this order, on the dedicated note type.
`LatinitasID` must stay the first field: Anki matches notes for update by the
first field. `Personal Notes` stays last. `Tags` is deliberately absent: it is
the special CSV transport column, never a note-type field.

```text
LatinitasID
Lemma
Principal Parts
Meaning
Source ID
Source Scope
Source Kind
Source Location
Source Path
Note Schema
Generator
Profile
CompletionPresentEnabled
CompletionPresentPrompt
CompletionPresentAnswer
CompletionInfinitiveEnabled
CompletionInfinitivePrompt
CompletionInfinitiveAnswer
CompletionPerfectEnabled
CompletionPerfectPrompt
CompletionPerfectAnswer
CompletionPPPEnabled
CompletionPPPPrompt
CompletionPPPAnswer
CompletionSupineEnabled
CompletionSupinePrompt
CompletionSupineAnswer
RecognitionPresentEnabled
RecognitionPresentPrompt
RecognitionPresentAnswer
RecognitionInfinitiveEnabled
RecognitionInfinitivePrompt
RecognitionInfinitiveAnswer
RecognitionPerfectEnabled
RecognitionPerfectPrompt
RecognitionPerfectAnswer
RecognitionPPPEnabled
RecognitionPPPPrompt
RecognitionPPPAnswer
RecognitionSupineEnabled
RecognitionSupinePrompt
RecognitionSupineAnswer
Personal Notes
```

### Card templates (10, frozen registry order)

Create one card template per entry below with exactly this name, front, and
back. Every front is wholly guarded by its per-card `Enabled` field, so a form
the exporter marks ineligible renders an empty front and creates no card
instead of a blank or label-only card. Every back is guarded the same way and
renders the managed answer, shared managed context (`Lemma`, `Principal Parts`,
optional `Meaning`), and the note-level, user-owned `Personal Notes`.

#### `Completion Present` &#8212; ordinal 0 &#8212; `principal_part_completion:present_1s`

Front:

```html
{{#CompletionPresentEnabled}}{{CompletionPresentPrompt}}{{/CompletionPresentEnabled}}
```

Back:

```html
{{#CompletionPresentEnabled}}
<div class="latinitas-answer">{{CompletionPresentAnswer}}</div>
<div class="latinitas-context">{{Lemma}} &#183; {{Principal Parts}}{{#Meaning}} &#183; {{Meaning}}{{/Meaning}}</div>
{{#Personal Notes}}<div class="latinitas-personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}
{{/CompletionPresentEnabled}}
```

#### `Completion Infinitive` &#8212; ordinal 1 &#8212; `principal_part_completion:present_infinitive`

Front:

```html
{{#CompletionInfinitiveEnabled}}{{CompletionInfinitivePrompt}}{{/CompletionInfinitiveEnabled}}
```

Back:

```html
{{#CompletionInfinitiveEnabled}}
<div class="latinitas-answer">{{CompletionInfinitiveAnswer}}</div>
<div class="latinitas-context">{{Lemma}} &#183; {{Principal Parts}}{{#Meaning}} &#183; {{Meaning}}{{/Meaning}}</div>
{{#Personal Notes}}<div class="latinitas-personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}
{{/CompletionInfinitiveEnabled}}
```

#### `Completion Perfect` &#8212; ordinal 2 &#8212; `principal_part_completion:perfect_1s`

Front:

```html
{{#CompletionPerfectEnabled}}{{CompletionPerfectPrompt}}{{/CompletionPerfectEnabled}}
```

Back:

```html
{{#CompletionPerfectEnabled}}
<div class="latinitas-answer">{{CompletionPerfectAnswer}}</div>
<div class="latinitas-context">{{Lemma}} &#183; {{Principal Parts}}{{#Meaning}} &#183; {{Meaning}}{{/Meaning}}</div>
{{#Personal Notes}}<div class="latinitas-personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}
{{/CompletionPerfectEnabled}}
```

#### `Completion PPP` &#8212; ordinal 3 &#8212; `principal_part_completion:perfect_passive_participle`

Front:

```html
{{#CompletionPPPEnabled}}{{CompletionPPPPrompt}}{{/CompletionPPPEnabled}}
```

Back:

```html
{{#CompletionPPPEnabled}}
<div class="latinitas-answer">{{CompletionPPPAnswer}}</div>
<div class="latinitas-context">{{Lemma}} &#183; {{Principal Parts}}{{#Meaning}} &#183; {{Meaning}}{{/Meaning}}</div>
{{#Personal Notes}}<div class="latinitas-personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}
{{/CompletionPPPEnabled}}
```

#### `Completion Supine` &#8212; ordinal 4 &#8212; `principal_part_completion:supine`

Front:

```html
{{#CompletionSupineEnabled}}{{CompletionSupinePrompt}}{{/CompletionSupineEnabled}}
```

Back:

```html
{{#CompletionSupineEnabled}}
<div class="latinitas-answer">{{CompletionSupineAnswer}}</div>
<div class="latinitas-context">{{Lemma}} &#183; {{Principal Parts}}{{#Meaning}} &#183; {{Meaning}}{{/Meaning}}</div>
{{#Personal Notes}}<div class="latinitas-personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}
{{/CompletionSupineEnabled}}
```

#### `Recognition Present` &#8212; ordinal 5 &#8212; `principal_part_recognition:present_1s`

Front:

```html
{{#RecognitionPresentEnabled}}{{RecognitionPresentPrompt}}{{/RecognitionPresentEnabled}}
```

Back:

```html
{{#RecognitionPresentEnabled}}
<div class="latinitas-answer">{{RecognitionPresentAnswer}}</div>
<div class="latinitas-context">{{Lemma}} &#183; {{Principal Parts}}{{#Meaning}} &#183; {{Meaning}}{{/Meaning}}</div>
{{#Personal Notes}}<div class="latinitas-personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}
{{/RecognitionPresentEnabled}}
```

#### `Recognition Infinitive` &#8212; ordinal 6 &#8212; `principal_part_recognition:present_infinitive`

Front:

```html
{{#RecognitionInfinitiveEnabled}}{{RecognitionInfinitivePrompt}}{{/RecognitionInfinitiveEnabled}}
```

Back:

```html
{{#RecognitionInfinitiveEnabled}}
<div class="latinitas-answer">{{RecognitionInfinitiveAnswer}}</div>
<div class="latinitas-context">{{Lemma}} &#183; {{Principal Parts}}{{#Meaning}} &#183; {{Meaning}}{{/Meaning}}</div>
{{#Personal Notes}}<div class="latinitas-personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}
{{/RecognitionInfinitiveEnabled}}
```

#### `Recognition Perfect` &#8212; ordinal 7 &#8212; `principal_part_recognition:perfect_1s`

Front:

```html
{{#RecognitionPerfectEnabled}}{{RecognitionPerfectPrompt}}{{/RecognitionPerfectEnabled}}
```

Back:

```html
{{#RecognitionPerfectEnabled}}
<div class="latinitas-answer">{{RecognitionPerfectAnswer}}</div>
<div class="latinitas-context">{{Lemma}} &#183; {{Principal Parts}}{{#Meaning}} &#183; {{Meaning}}{{/Meaning}}</div>
{{#Personal Notes}}<div class="latinitas-personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}
{{/RecognitionPerfectEnabled}}
```

#### `Recognition PPP` &#8212; ordinal 8 &#8212; `principal_part_recognition:perfect_passive_participle`

Front:

```html
{{#RecognitionPPPEnabled}}{{RecognitionPPPPrompt}}{{/RecognitionPPPEnabled}}
```

Back:

```html
{{#RecognitionPPPEnabled}}
<div class="latinitas-answer">{{RecognitionPPPAnswer}}</div>
<div class="latinitas-context">{{Lemma}} &#183; {{Principal Parts}}{{#Meaning}} &#183; {{Meaning}}{{/Meaning}}</div>
{{#Personal Notes}}<div class="latinitas-personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}
{{/RecognitionPPPEnabled}}
```

#### `Recognition Supine` &#8212; ordinal 9 &#8212; `principal_part_recognition:supine`

Front:

```html
{{#RecognitionSupineEnabled}}{{RecognitionSupinePrompt}}{{/RecognitionSupineEnabled}}
```

Back:

```html
{{#RecognitionSupineEnabled}}
<div class="latinitas-answer">{{RecognitionSupineAnswer}}</div>
<div class="latinitas-context">{{Lemma}} &#183; {{Principal Parts}}{{#Meaning}} &#183; {{Meaning}}{{/Meaning}}</div>
{{#Personal Notes}}<div class="latinitas-personal-notes">{{Personal Notes}}</div>{{/Personal Notes}}
{{/RecognitionSupineEnabled}}
```

### Shared styling (paste once into the note type's Styling)

```css
/* Latinitas reference styling v2; morphology markup v1. Manual setup only. */
.card {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 22px;
  text-align: center;
  color: #1a1a1a;
  background-color: #ffffff;
}

.latinitas-answer {
  font-size: 30px;
  font-weight: bold;
}

.latinitas-context {
  margin-top: 0.6em;
  font-size: 18px;
  color: #444444;
}

.latinitas-personal-notes {
  margin-top: 1em;
  padding-top: 0.5em;
  border-top: 1px solid #cccccc;
  font-size: 16px;
  font-style: italic;
  color: #666666;
  text-align: left;
}

.morphology-v1 {
  --stem: #356357;
  --marker: #785523;
  --ending: #5d5079;
  margin: 0.8em auto 0;
  padding: 1em;
  max-width: 42em;
  font-size: 18px;
  line-height: 1.5;
  font-weight: normal;
  text-align: left;
  color: #222222;
  background: #f7f7f4;
  border: 1px solid #bdbdb7;
  border-radius: 0.3em;
  overflow-wrap: anywhere;
}
.morphology-v1.morphology-dark {
  --stem: #9bc9b9;
  --marker: #e1bf85;
  --ending: #c4b5e1;
  color: #eeeeee;
  background: #222526;
  border-color: #666666;
}
.morphology-v1.morphology-monochrome {
  --stem: currentColor;
  --marker: currentColor;
  --ending: currentColor;
}
.morphology-lemma { font-style: italic; }
.morphology-stem { color: var(--stem); font-weight: bold; }
.morphology-marker { color: var(--marker); border-bottom: 1px dotted; }
.morphology-ending { color: var(--ending); text-decoration: underline; }
.tested-form-explanation, .related-stems { margin-top: 0.5em; }
.morphology-v1 summary, .morphology-static h3 {
  margin: 0.8em 0 0.3em;
  font-size: 1em;
  font-weight: bold;
}
.morphology-v1 summary { cursor: pointer; }
.morphology-role { margin: 0.4em 0; }
```
<!-- latinitas-reference-setup end -->

Both recipes follow the same shape: the **completion** front shows the
principal-part series with the target blanked (`____`) and asks for the missing
form; the **recognition** front shows one form and asks which role it is. The
backs render the managed answer plus the shared context and Personal Notes.

## Synthetic missing-form examples

These synthetic examples show how missing forms behave without fourth-role claim
review. A proposed PPP/supine profile label alone does not approve a fourth-form
card; explicitly review its contextual role through the offline claim API.
Each row is one note;
"eligible templates" lists exactly the templates whose `Enabled` field is `1`
(every other slot renders an empty front and creates no card):

| Forms value | Confirmed roles | Eligible templates |
| --- | --- | --- |
| `ferō — ferre — tulī — lātum` | present_1s, present_infinitive, perfect_1s, supine | Completion Present, Completion Infinitive, Completion Perfect, Recognition Present, Recognition Infinitive, Recognition Perfect |
| `ferō — ferre —  — ` | present_1s, present_infinitive, perfect_1s, perfect_passive_participle | Completion Present, Completion Infinitive, Recognition Present, Recognition Infinitive |
| `amō — amāre — <b></b> — amātum` | present_1s, present_infinitive, perfect_1s, supine | Completion Present, Completion Infinitive, Recognition Present, Recognition Infinitive |

- An absent form guards its own cards off without shifting any positional
  meaning: the remaining prompts keep every confirmed position and show the
  omitted position as an em dash (`—`).
- A markup-only value such as `<b></b>` normalizes once to empty before
  eligibility (the T-032 fix); it produces no card and is not treated as a
  present answer. Source HTML elsewhere is rendered under the existing safe-HTML
  contract: display text is escaped once at export, never re-decoded.
- An object whose roles yield zero eligible cards stays a valid object but is
  omitted from the CSV (native import would otherwise create a blank note) and
  reported as a distinct `Zero-eligible notes` count, separate from parser
  skips.

## Eligibility loss at the export checkpoint

Regeneration compares current card eligibility against the retained prior-export
checkpoint beside the source (`source.csv.latinitas-cards.json`). When a
previously exported card key is no longer eligible — data loss, recipe
deselection, or a shared-field change that removes required data — the exporter
**withholds the whole affected note row** and reports it under
`Card eligibility reviews`. It never exports emptied `Enabled` fields, deletes
cards, or resets schedules: clearing a guard on import would let Anki drop the
existing card, so the row is withheld for review instead. Missing, corrupt, or
incompatible checkpoint state is a review gate requiring recovery/review or an
explicit `--approve-fresh-import` confirmation; a read-only preview never
advances the checkpoint. v0.2.0 managed plans show retirement as an unsupported,
plan-only effect; until a supported transport applies it, retain the existing
notes or explicitly suspend affected cards in a backed-up manual workflow. Details: [deterministic-csv-export.md](deterministic-csv-export.md).

## Read this before the first import and before every reimport

> **Warning: a mapped `Tags` column can replace destination-only manual tags.**
> When `Tags` is mapped — and it must be mapped to Anki's tags column, not to a
> field — a native Anki CSV Update re-asserts the exported tag set on every row
> it updates: it removes a manually added tag and restores a removed exported
> tag. Native verification on this multi-card model (Anki 26.09.3) reproduced
> that removal when the reimported row also changed a managed field, while a
> tag-only reimport whose row is unchanged in every managed field is reported by
> Anki as **Skipped** and retains the manual tag. Do not rely on either
> direction: treat every reimport as tag-destructive. Tags inherited from the
> source parent are part of the exported set — inheritance is not preservation
> of destination edits. Before the first import and before every reimport:
> **back up the collection** (and export the destination tags you care about),
> or **defer the import**. v0.1.0 performs no destination-aware tag merging; a
> destination-aware update workflow is planned for v0.2.0 and does not exist
> today.

The tag-replacement mechanism was first observed with a native Anki Desktop 26.09.3
collection for the earlier per-exercise model through Anki's import backend (see
the "Native tag-import verification" notes in
[deterministic-csv-export.md](deterministic-csv-export.md)); the release-gate
verification then probed the boundary above against this multi-card model
through Anki's native import dialog on 26.09.3 with synthetic disposable
collections. The historical backend observation did not isolate a tag-only
change, and the dialog run observed the tag-only row reported Skipped with the
manual tag retained; the cause of that difference is not established, so both
outcomes are recorded as Anki CSV Update behavior rather than a property of a
specific Latinitas model, and the backup-or-defer warning above stays the
contract.

## First and repeat import checklist

1. **Back up the collection**, and use a disposable profile to verify the
   reference schema first.
2. Create the **dedicated Latinitas note type** (named by the profile, normally
   `Latinitas Principal Parts`) from the exact reference fields and templates
   above; keep `LatinitasID` first. Do not modify source note types and do not
   reuse legacy exercise types.
3. Choose the configured **target deck**; enable **Allow HTML in fields**; match
   on the **first field** with **Note Type** match scope; for repeat imports
   choose **Update existing notes when first field matches**. Verify the mapping
   in the native import dialog before applying.
4. Map managed fields and `Tags` deliberately (`Tags` to the tags column); leave
   **`Personal Notes` unmapped** — the generated CSV has no such column, so the
   dialog offers nothing to map to it; confirm that stays true. Accept the
   manual-tag replacement warning above first.
5. Check the preview **counts separately**: source entries, objects/notes,
   eligible cards, skipped/review cases. Do not confuse one CSV row with one
   card.
6. Inspect **both card sides** and the sibling relationships; repeat an
   unchanged import and a controlled managed-field update. For updates compare
   full card/review-log tables, and for no-op imports full
   note/card/review-log tables — not a scheduling-column subset.
7. Enable the desired Anki **sibling-burying** settings manually (see below);
   verify the cards schedule independently and observe burying with synthetic
   siblings, without claiming to validate Anki's scheduler.

## Sibling burying and independent schedules

Anki owns FSRS scheduling and burying; Latinitas neither implements scheduling
nor changes your burying settings. If you want the ten potential siblings of one
note not to surface together, enable the deck options' bury-siblings controls
(new/review/interday-learning siblings, per Anki version) yourself on the target
deck. All cards of a note remain independent cards with separate review
histories and scheduling states; enabling or disabling a recipe changes only
which cards are eligible, never note identity, slot order, or ordinals.

## Ownership and drift boundaries

Source-specific field inference, profile preparation, and raw provenance stay
with the source adapters; CSV metadata, headers, and escaping belong to the
transport serialization in the exporter. This document consumes the frozen
contract rather than redefining it, so a schema or registry change must be
republished here through `reference_setup_markdown()` (a test fails on drift).
Legacy APKG mutation and corpus commands remain experimental and are not part
of this workflow.
