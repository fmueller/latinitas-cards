# Offline reviewed claims and card checks

`claims` reviews individual linguistic claims, not reviewer authentication,
source identity, export approval, scheduling migration, or managed destination
approval. No command accepts every open claim automatically. The shared contract
covers principal-part recipes, contextual form parsing, and versioned
[phrase lexical/pattern inputs](reviewed-phrase-inputs.md) via `--phrase-input` on
`claims list` and `claims record`. Generated phrase exercises use `--selections`
and `--recipes` on `list`, `record` and read-only `inspect --check`; their combination
and rendered-answer attestations live beside the recipe plan, not the lexical input.
Phrase CSV export and managed destination handoffs are documented separately;
linguistic review does not approve destination writes.

## List and record

For principal parts, use an already confirmed profile and source identity scope:

```sh
uv run latinitas-cards claims list source.csv --profile profile.json --json
uv run latinitas-cards claims record source.csv --profile profile.json \
  --fingerprint '<one fingerprint from list>' --status accepted \
  --reviewer 'Latin reviewer' --reason 'Individually checked this role'
```

For contextual parsing, omit `--profile`: the JSON source contains its profile.
The same commands accept `withheld`. Text is the default; `--json` emits JSON
including item-indexed diagnostics and uses a nonzero exit for rejection.
`list` includes accepted, withheld and open claims, not just accepted content.
`review_status` distinguishes `open`, `accepted` and explicit `withheld` decisions;
`status` is the knowledge gate, which remains withheld until a fresh acceptance.
Principal-part claims are shared by both enabled recipes; their underlying
identity and role are independent of recipe slots.

An agent may submit an explicit JSON array using `--batch decisions.json`:

```json
[
  {
    "fingerprint": "<exact current fingerprint>",
    "status": "accepted",
    "reviewer": "Latin reviewer",
    "reason": "Checked this particular claim against the context"
  }
]
```

Every item is validated before a single atomic sidecar replacement. Invalid,
stale or conflicting items reject the entire batch and leave the sidecar bytes
unchanged. Opposed decisions on the same claim and conflicting accepted values
for a feature are rejected. Repeating the same decision is idempotent. File locks
serialize writers; inspection and listing acquire no locks and write nothing.

Decisions live in `<source>.claims.json`, a version-1 human-readable sidecar.
Principal-part preview/export and form-parsing preview/export load it automatically.
Do not treat it as destination state or put it in a managed approval file.
Stale/unmatched decisions are reported and ignored; resolve stale sidecar entries
before recording further decisions. Retain old evidence separately if needed.

## Authored corrections

An item can add `value` and an `evidence` array containing `source`, `reference`,
`version`, and `text`. These create a new reviewer-authored claim/fingerprint and
review it individually; they do not mutate the proposal's provenance. A reviewed
new claim may replace a previously reviewed proposal. Single-item recording uses
`--value '...' --evidence evidence.json`. Unsupported placeholders for missing
splits, explanations or contextual features cannot themselves be accepted;
author a value with evidence instead.

Canonical role labels must match the confirmed profile. Change a wrong PPP/supine
profile explicitly rather than changing slot identity through free text. Missing
forms stay missing, and unresolved source alternatives require separate source
correction; authoring an explanation cannot supply or select a source form.

Fingerprints bind form, context, alternatives, evidence, extraction and the
linguistic profile. The existing versioned morphology presentation settings are
excluded. Note identities and frozen slot ordinals do not change with decisions.
Withheld/missing linguistic values never become asserted card content. Existing
source-role and export/checkpoint gates remain in force.

## Principal-part translations

`claims list source.csv --profile profile.json --json` includes an open
`translation` claim for each present form/role. There is no automatic translation
or acceptance. Supply a German form-appropriate value, its evidence, and an
explicit selected sense using the same individual review route:

```sh
uv run latinitas-cards claims record source.csv --profile profile.json \
  --fingerprint '<translation fingerprint>' --status accepted \
  --value 'ich habe geliebt' --selected-sense 'love' \
  --evidence evidence.json --reviewer 'Latin reviewer' \
  --reason 'Checked amāvī, perfect first singular, selected love sense'
```

Batch items use `"selected_sense": "love"` alongside `value` and `evidence`.
Use a stable, explicit sense name and describe the selected meaning in the
evidence, especially for a multi-sense meaning field. Each role is reviewed
separately; a lemma gloss is not a translation of every form. Deponents need
active-meaning translations (e.g. `locūtus sum` → `ich habe gesprochen`), not a
passive translation inferred from spelling. Defective/irregular forms receive no
automatic support. Ambiguous source alternatives cannot be resolved by authoring
a translation. Missing roles stay absent. An accepted fourth-form translation
does not confirm PPP versus supine: its canonical role label still needs review.

Accepted translations appear for the tested form and in the four-role comparison
on answers in both principal-part recipes. They are never added to fronts.
The existing optional source meaning/gloss on prompts is unchanged. Withheld or
open translations show no value. Latin/German grammatical terminology does not
translate this content: it follows the profile's user language (German only now;
other user languages reject authored form translations).

Form, role/profile, source meaning, selected sense, translation and evidence bind
review. Source meaning edits invalidate translation decisions, not independent
label/split/explanation decisions. Correcting a translation or sense requires a
new individual decision against its current fingerprint; editing sidecar content
without renewing review fails check. Inspection exposes selected sense, language,
evidence and review status. Exported answers retain that provenance.

Identities and slots do not change. Changed answers require new managed payload
review, independently of linguistic acceptance. Current-schema retained principal-part,
parsing and phrase prompt/answer changes can use separately approved managed CSV
handoffs. Initial creation, retirement/reactivation and template/CSS installation
remain manual; old layout digests fail closed. Follow the
[retained-update workflow](reviewed-phrase-inputs.md#approved-retained-slot-changes-and-blocked-withdrawal),
including closed capture, explicit adoption, fresh approval and observation/reconciliation.
A browser preview proves neither native Anki import nor scheduling preservation.

## Inspect and fail-closed check

```sh
uv run latinitas-cards claims inspect source.csv --profile profile.json --json
uv run latinitas-cards claims inspect source.csv --profile profile.json \
  --artifact exported.csv --check --json
uv run latinitas-cards claims inspect cases.json --artifact parsing.apkg --check
uv run latinitas-cards claims inspect exported.csv --json
```

Reports show each note/card identity, recipe, frozen slot, eligibility, plain-text
prompt/answer, backing claims and withholding/absence reasons. Standalone artifacts
show their retained evidence as **unverified**; check requires the current source,
profile and sidecar and fails without them. Parsing CSV answers retain accepted
claim evidence and withholding reasons in a hidden `parsing-review` span; no
withheld value is asserted. Principal-part exports retain their existing hidden
comparison evidence. Hidden provenance is omitted from rendered plain text.

Checks compare managed fields and eligible note/card identities with the current
source, reporting drift, unreviewed asserted content, stale decisions and slot
mismatches. Personal Notes and transport Tags are not linguistic assertions.
No files, destination state, reviews, network, LLM or Anki are needed or written.

### Supported package layouts

Offline `.apkg`/`.colpkg` inspection supports legacy JSON `col.models` metadata in
SQLite `.anki2`/`.anki21` collections (and zstd `.anki21b` members), with exact:

- principal-part note schema 3, frozen template registry 1 and the documented
  `REFERENCE_CARD_TEMPLATES`/reference field order; or
- `Latinitas Contextual Form Parsing v1` fields and its contextual template pair.

The known guard fields and prompt/answer fields determine eligibility and text.
Stored card ordinals must match eligible frozen slots. The documented Personal
Notes/context sections and parsing FrontSide are included in plain text.
Unknown fields, modified templates, mixed recipe layouts, normalized/protobuf
model metadata, arbitrary JavaScript, cloze and arbitrary conditional templates
are unsupported and fail check mode. This is not a general Anki template engine
and does not generate packages or provision note types.
