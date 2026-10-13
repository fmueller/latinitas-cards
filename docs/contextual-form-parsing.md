# Reviewed contextual form parsing (experimental)

`form-parsing preview cases.json` exposes individual claim fingerprints, evidence,
alternatives, review reasons and eligibility skips. `form-parsing export cases.json
parsing.csv --approve-fresh-import` exports eligible cases only. Core operation is
offline: no CLTK, Stanza, LLM, or network service is needed. This is explicit reviewed
generation, not an automatic token analyzer. There is no approved automatic policy;
no precision threshold or measured accuracy is claimed here. Future automatic
acceptance needs domain-relevant measured precision and approved policy **for each
feature category**, not a principal-part comparison or analyzer self-confidence.

## Complete runnable example

The [complete cases.json sample](examples/form-parsing-cases.json) includes the
profile, evidence, alternatives and three explicit accepted decisions for the
intended reading of **Puellae rosas portant.** (The girls carry roses.) These are
independently authored sample judgments, not parser-generated expectations,
expert validation, an automatic acceptance policy or a calibration corpus.
The optional gender proposal is deliberately unreviewed and withheld.

With `latinitas-cards` installed and on PATH, run this from the repository root
(or download the linked sample and copy its absolute path instead):

```sh
sample="$PWD/docs/examples/form-parsing-cases.json"
work=$(mktemp -d)
cp "$sample" "$work/cases.json"
cd "$work"
latinitas-cards form-parsing preview cases.json
latinitas-cards form-parsing export cases.json parsing.csv --approve-fresh-import
```

Preview must contain exactly one eligible exercise, semantic key
`form_parsing:contextual_analysis`, and ID
`latinitas-v2-ea8b6ea917204b230894b6b0f5ca2edc91201a958b1734092aea51b1d796e9ef`.
Accepted facts are `Lemma: puella`, `Casus: nominativus`, `Numerus: pluralis`;
`femininum` remains withheld and is absent from the answer. Export prints:

```text
Fresh contextual CSV only; managed application and template provisioning unsupported.
```

The CSV contains one data row for **Latinitas Contextual Form Parsing v1**, not
the principal-part note type named in the profile. It has no Personal Notes field.
Use the [separate manual setup below](#stable-bindings-and-manual-reference-setup)
before importing. `--approve-fresh-import` acknowledges new scheduling and separate
manual setup; it does not approve claims or preserve old card histories. Without
it, export refuses to write. Repeating export to `parsing.csv` refuses overwrite.

The supplied decisions reproduce only with the exact sample form, context,
source assignments, feature lists, proposals, evidence and linguistic profile.
For example, editing only context to `Rosam puellae dat.` retains the note ID
but makes the three decisions stale: preview is ineligible with an empty answer,
and export to a new file contains no data rows. Re-review the changed claims;
do not transplant the sample fingerprints or reasons to your own material.

## Input and individual review

The version-1 JSON envelope has `schema_version: 1`, a schema-2 `profile` (including
presentation-v1 `morphology`), and `cases`. The existing profile's recipes and
principal-part layout remain unchanged; they do not prove token features and are
not used to infer parsing. Its deck/tags/presentation are used for fresh export.
Each case supplies these immutable identity and contextual evidence fields:

```json
{
  "source_identity": "sentence-17",
  "source_scope": "my-reviewed-course/v1",
  "object_key": "token-2",
  "form": "puellae",
  "context": "Puellae rosas portant.",
  "applicable_features": ["lemma", "case", "number", "gender"],
  "required_features": ["lemma", "case", "number"],
  "proposals": [
    {
      "feature": "case",
      "value": "nominativus",
      "analyzer": "individual-review/v1",
      "evidence": [{"source": "reviewed Latin", "reference": "sentence 17",
                    "version": "v1", "text": "Subject of plural portant"}],
      "alternatives": ["genetivus singularis", "dativus singularis"],
      "supported": true
    }
  ]
}
```

This partial example deliberately generates **no card**: lemma and number are
missing and the case proposal is unreviewed. Add separate proposals/evidence for
each relevant category (`lemma`, `person`, `number`, `tense`, `mood`, `voice`,
`case`, `gender`). The author/reviewer declares applicability; the generator does
not invent it from endings. Require lemma and at least one other applicable
feature. Missing optional features are omitted, not filled from a paradigm.
Extraction candidates/hints may be cited as evidence, but extraction success and
principal-part roles are never token-level proof. Unknown formats stay review work.

After linguistic review of **that proposal in that context**, use the shared
[claim CLI](reviewed-claim-cli.md#list-and-record) to record one explicit decision:

```sh
uv run latinitas-cards claims list cases.json --json
uv run latinitas-cards claims record cases.json \
  --fingerprint '<one exact current fingerprint from list>' --status accepted \
  --reviewer 'Latin reviewer' --reason 'Individually checked this feature in context'
uv run latinitas-cards form-parsing preview cases.json
```

Use `withheld` to withhold a claim; `claims record --value` can author a corrected
value with its own evidence-bound review. Parsing sources contain their profile,
so omit `--profile`. The CLI writes a shared decision sidecar, read automatically
by preview/export. Alternatively, manually add an explicit `decision` object to
that proposal with `claim_fingerprint` copied from preview, `status` equal to
`accepted` or `withheld`, nonempty `reviewer`, and an explanatory `reason`; no hash
reconstruction is needed. The complete sample shows this embedded decision shape.
Use your own reviewer identity and reason, then preview again to check currency.
A review is a user-authored attestation, not authentication, export approval or
managed write approval. Neither route permits mechanical approval of all proposals.

Changing the form, context, category, evidence, alternatives, applicable/required
features or linguistic profile invalidates the review. Presentation changes do
not. Unsupported proposals cannot be accepted. Conflicting accepted alternatives
block the entire card, even for an optional feature. Preview retains each proposal
and its status, but only uncontested accepted claims can appear on study cards.
The complete sample demonstrates one intended contextual reading and optional
withholding, not universal grammatical coverage. A plausible ending alone does
not resolve all contextual alternatives.

## Stable bindings and manual reference setup

Define a persistent source scope, source entry identity and object key **before**
generation. The object key identifies an encountered token/context, not the lemma
string or generated wording. Distinct contexts/tokens need distinct keys, even
when visible forms/lemmas coincide. Related features of that one contextual object
share one note. Store and reuse these assignments; this command does not allocate
or reconcile keys. Do not reuse a key to replace an unrelated context. Correcting
text retains the note identity but requires renewed reviews.

The note ID uses existing v2 identity derivation with the object namespace
`contextual-form:<object_key>`. This is intentionally separate from lexeme
principal-part notes and does not consolidate them. The dedicated note type is
**Latinitas Contextual Form Parsing v1** with these fields, in this order:

1. LatinitasID
2. Form
3. Context
4. ParsingEnabled
5. ParsingPrompt
6. ParsingAnswer
7. Personal Notes

Its sole frozen template is **Contextual Analysis**, ordinal **0**, semantic key
`form_parsing:contextual_analysis`. Existing principal-part registry version 1's
ten slots are unchanged and must not be reused for parsing. Copy this front:

```html
{{#ParsingEnabled}}{{ParsingPrompt}}{{/ParsingEnabled}}
```

And this back:

```html
{{#ParsingEnabled}}{{FrontSide}}<hr id=answer>{{ParsingAnswer}}{{#Personal Notes}}<div>{{Personal Notes}}</div>{{/Personal Notes}}{{/ParsingEnabled}}
```

Copy the published [shared reference CSS](reference-note-type.md#shared-styling-paste-once-into-the-note-types-styling)
(reference style v4, morphology markup v1), but use the contextual fields and
single template above, not the principal-part fields or templates. The
parsing answer is static and Latin-first; no disclosure or custom script is added.
Use a backed-up, explicitly separate new destination for this experimental manual
setup. Import CSV as HTML with LatinitasID first-field matching, the exported Tags
column as special tags, and Personal Notes **unmapped** (not present in the CSV).
New cards have new scheduling, not inherited principal-part histories. Export
refuses an existing output file. Plain CSV cannot preserve later destination tags.

No automatic provisioning is implemented. The separate `parsing-v1` managed binding
now permits individually approved prompt/answer updates on observed retained parsing
cards through `managed plan request.json` and the
[capture/adoption and retained-update handoff](reviewed-phrase-inputs.md#approved-retained-slot-changes-and-blocked-withdrawal).
Fresh `form-parsing export` is still creation-only; its warning does not describe that
separate managed path. Structural additions, slot rebinding, retirement/reactivation
and migrations remain unsupported. Export/linguistic approval is not managed-plan
approval. Never relabel parsing CSV/schema evidence or import it over scheduled
principal-part notes. Old CSS/layout digests require reviewed manual setup and fail closed.

## Verification boundary

Run the existing `scripts/check-managed-anki.py` for affected conservative managed
backend regressions. `scripts/check-form-parsing-anki.py --input cases.json
--output-dir <new-disposable-directory>` imports eligible reviewed parsing into a
disposable native Anki backend and verifies exact render content and no-op tables.
It is an isolated test harness, **not general user-collection provisioning**.

Backend and browser evidence do not prove AnkiMobile or Desktop GUI presentation.
The native retests ran on 2026-10-06. Fresh CSV import through the Desktop dialog,
light/dark reveal and readability passed on Anki Desktop 26.09.2 and AnkiMobile 25.09
(iPhone 16 Pro Max, iPad Air M4), with the note type delivered by a carrier package
per maintainer decision. See
[the native acceptance run](morphology-native-verification.md#native-acceptance-run--2026-10-06).
This covers the tested synthetic cases and clients only; no release readiness is
claimed.
