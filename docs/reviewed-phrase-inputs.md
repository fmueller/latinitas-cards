# Reviewed lexical inputs and bounded patterns (version 1)

`phrases` imports a JSON file, inspects its proposals, and saves explicit reviewed
selections. `phrases generate` previews bounded phrase proposals; `phrases recipes`
saves explicit recipe choices and `phrases preview` renders their reference proposals.
These commands do **not** approve generated answers. `phrases export` exports only
individually approved exercises; `phrases plan` connects them to managed review.
No corpus, network, LLM, CLTK or source-note edits are
needed. Use the existing confirmed deck profile and source identities, then supply
missing lexical knowledge in a separate file rather than altering the deck.

## Explicit recipe plan (version 1)

After `phrases generate`, copy the desired `candidate_key` into a recipe request.
Choose `phrase_translation` (target `null`, distinction `meaning`) or
`phrase_recognition` (explicit constituent role target, distinction `analysis`
or one of `case`, `number`, `gender`, `person`, `tense`, `mood`, `voice`). There is
no `both`/`all` shortcut. Each plan requires `version: 1`, `max_exercises` (1–1000)
and an `exercises` list no longer than that limit. Each exercise has an authored
German `reference` with `readings` and `evidence`; nothing guesses a translation.

For a `cum amico` candidate whose sole constituent is `object`, an exercise is:

```json
{
  "candidate_key": "COPY-THE-CURRENT-CANDIDATE-KEY",
  "recipe": "phrase_recognition",
  "target": "object",
  "distinction": "case",
  "reference": {
    "language": "de",
    "readings": [{
      "translations": ["mit einem Freund", "mit dem Freund"],
      "features": {"object": {"case": "ablative", "number": "singular", "gender": "masculine"}},
      "contributions": {"object": "Begleitung; cum regiert den Ablativ."},
      "explanation": "Die Präposition cum bestimmt hier den Kasus."
    }],
    "evidence": [{"source": "your reference", "reference": "page/section", "version": "1", "text": "contextual evidence"}]
  }
}
```

```bash
uv run latinitas-cards phrases recipes inputs.json selected.json recipe-request.json recipes.json --json
uv run latinitas-cards phrases preview inputs.json selected.json recipes.json --json
uv run latinitas-cards phrases preview inputs.json selected.json recipes.json
```

Every reading describes exactly the candidate's constituent roles, with applicable
canonical features and a German contribution/explanation. Alternative translations
and contextual readings stay visible. Isolated `amico` permits dative/ablative;
`cum amico` supplies ablative government. Isolated `amici` nominative plural does
**not** establish that reading in `liber amici`: a noun `liber` is singular.
An authored vocative-head reading (addressing the book) may accompany the
nominative-head reading, both with genitive singular `amici`. For `dona bona`,
neuter plural nominative/accusative/vocative remain possible without more context.
A single-case exercise is withheld when the supplied readings differ in case;
`analysis` may teach explicitly reviewed alternatives rather than invent context.

Preview labels unreviewed content **Proposed reference — not asserted** and shows
withholding reasons. Preview automatically consumes the recipe plan's shared review
sidecar. Eligibility requires reviewed lexical/pattern inputs, explicit combination
correctness/naturalness approval, and a separately approved rendered reference.
Latin (default) or German terminology comes from the profile; German prose and
translations do not change. Translation is Latin-first self-assessment, never
exact-string grading or German-to-Latin production. Targets and tested distinctions
are persisted and hashed into semantic exercise identity; reference wording and
terminology are not. Reference/content changes invalidate an existing review.

`phrase-v1` / layout v1 extends the old reference contract append-only: the ten
existing templates and all fields stay unchanged; phrase Translation/Recognition
use ordinals 10/11 and append their Enabled/Prompt/Answer field triples. This is
a **reviewed manual setup contract**, not an installation or change to current principal-part
exports. Capture binds exact fields, templates and CSS v3; source
notes, scheduled templates and Personal Notes are untouched. Browser previews are
not native Desktop/AnkiMobile acceptance.

## Import and review

Copy `docs/examples/{phrase-inputs.json,deck.csv,profile.json,deck.csv.latinitas.json}`
to a disposable directory to try all four synthetic pattern families. The example
scope is test-only: never copy it onto a real deck. For real CSV decks, reuse the
existing `.latinitas.json` identity sidecar. If no scope has been confirmed, first
use the [CSV identity workflow](deterministic-csv-export.md), including explicit
`generate --approve-scope --approve-fresh-import` where appropriate; inspect its
identity decisions before proceeding. APKG/COLPKG use the confirmed profile's
native note GUIDs. Manifest-based CSV identities must reconcile without pending
source review. This command never allocates identities or scopes.

```sh
uv run latinitas-cards phrases inspect inputs.json --json
uv run latinitas-cards claims list inputs.json --phrase-input --json
uv run latinitas-cards claims record inputs.json --phrase-input \
  --fingerprint '<one current fingerprint>' --status accepted \
  --reviewer 'Latin reviewer' --reason 'Checked this form/sense/pattern and evidence'
uv run latinitas-cards phrases inspect inputs.json --check
uv run latinitas-cards phrases select inputs.json request.json selected.json --json
uv run latinitas-cards phrases generate inputs.json selected.json --json
```

`claims record --batch decisions.json` accepts the same explicit per-fingerprint
decision array as [T-064's shared CLI](reviewed-claim-cli.md), not an approve-all
operation. Decisions remain in `inputs.json.claims.json`, using its locking,
atomic validation, stale/conflict detection and individual attestations. Inspect
and list are read-only. Text is default; `--json` provides machine-readable reports.
`inspect --check` and failed `select` exit nonzero. Selection errors leave any
existing output untouched; input/source/profile/identity/review files and the
shared review lock cannot be selection outputs, including symlink/hard-link aliases.

Edit the imported JSON to supply a missing paradigm, gender, sense/valency, or
pattern with its evidence, then individually review the resulting fingerprints.
Structured phrase claims cannot be replaced by free-text `claims record --value`:
the imported schema is authoritative. Missing evidence cannot be accepted. A
changed entry, source fields, scope, profile or pattern invalidates its previous
claim decisions. As with other claim adapters, archive and remove stale sidecar
entries before recording renewed decisions; do not silently carry approval over.

## File contract

The root requires `version: 1`, `source`, `profile`, `entries`, and `patterns`.
Paths are relative to the input JSON (absolute paths also work). Optional
`manifest` overrides the CSV identity-sidecar path. Optional `example_selection`
contains a request, not pre-approved selections. Unknown fields/versions, duplicate
entry/sense/form/pattern keys, and unknown compatibility references are rejected.

- An entry's `identity` is an exact resolved deck source ID/GUID/manifest assignment,
  not its word spelling, gloss, row number or local Anki note number. Its `pos` is
  `noun`, `adjective` or `verb`. Nouns need reviewed `gender`.
- `forms` explicitly enumerate bounded supported forms: stable `key`, actual `text`,
  and `features`. A gloss or principal parts alone never supplies a paradigm. This
  is a partial, reviewed paradigm, not a promise that every Latin form is supported.
  Nominal forms need case/number/gender; finite verbs need person/number/tense/mood/voice.
- `senses` contain a stable `key`, German `meaning`, `valency` and `evidence`.
  Verb valency maps every supported argument role (`subject`, optional `object`) to
  its case; non-verbal senses use an empty map. No implicit sense choice is made.
- Entry `evidence` supports individual gender/form proposals; sense and pattern
  `evidence` support their own proposals. Each evidence item requires nonempty
  `source`, `reference`, `version`, and `text`. Imported assertions are never
  automatically reviewed, even when they look plausible.
- Keys name underlying senses, patterns and tested forms. Keep their meanings stable;
  a genuinely different sense/construction/form gets a new key. Wording, spelling
  presentation and gloss corrections keep keys and note identity but require review.

Features use English canonical values: cases `nominative`, `accusative`, `genitive`,
`dative`, `ablative`, `vocative`, `locative`; numbers `singular`, `plural`; genders
`masculine`, `feminine`, `neuter`; persons `1`, `2`, `3`; tenses `present`,
`imperfect`, `future`, `perfect`, `pluperfect`, `future-perfect`; moods `indicative`,
`subjunctive`, `imperative`; voices `active`, `passive`. Unsupported/incomplete
forms are diagnosed, not inferred from suffixes.

## Bounded construction contract

Every pattern has stable `key`, `family`, explicit `function_words` (including an
empty array when none), `roles`, and its own evidence. Each role declares `pos`, a
finite allowlist `senses` of `entry-identity/sense-key`, and `requires` features.
Review of a pattern attests compatibility for **all** combinations of its listed
senses, not just one example. Narrow the allowlists or split patterns when this
Cartesian set would include unnatural meanings. There is no semantic inference.

| Family | Canonical roles | Enforced constraints |
|---|---|---|
| `preposition-noun` | `object` (noun) | Explicit function word(s), required governed case |
| `noun-genitive` | `head`, `dependent` (nouns) | Dependent requires genitive |
| `verb-noun` | `subject` (noun), `verb`, optional `object` (noun) | Nominative subject, third-person verb, number agreement, exact reviewed sense argument roles and cases |
| `adjective-noun` | `head` (noun), `modifier` (adjective) | Case, number and gender agreement |

Role requirements and lexical gender are checked independently. Function words
are visible in inspection and owned by the pattern, never silently imported as
content vocabulary. The fixture contrasts `cum amico`/accusative, `liber amici`/
nominative dependent, masculine noun/feminine adjective, singular subject/plural
verb, and accusative-object/dative-valency mismatches. Review attests grammatical
and semantic evidence; this bounded checker is not a general Latin syntax engine.

## Requests, persistence and identity

Extract `example_selection` from the inspection JSON or author this structure:

```json
{
  "version": 1,
  "limits": {"max_exercises": 20, "variants_per_binding": 2},
  "selections": [{
    "pattern": "with",
    "bindings": [{"role": "object", "entry": "friend", "sense": "person", "form": "abl-sg"}]
  }]
}
```

No random or implicit vocabulary selection is saved. Select validates all named
entries, senses, forms, roles, reviewed compatibility and current individual
decisions before atomically replacing output. `max_exercises` is an integer 1–1000;
`variants_per_binding` is an integer 1–32. The
number of unique explicit selections must fit `max_exercises`. Neither limit
truncates silently. Requests are retained for reruns; saved output is a separate
version-1 bound-selection contract (not an input request).

Output persists limits, pattern key and sorted role-labelled constituent bindings
including each exact source identity/scope, sense and form key. Reordering request
selections or bindings yields byte-identical output; duplicates collapse. The
composite `phrase-source-v1-…` identity hashes **all canonical role/source bindings**,
never picks a contributing word as owner. Even a one-content-word prepositional
pattern uses this composite namespace. Swapping subject/object changes it.
`source_scope: phrase-input-v1` is the domain scope for the existing
`derive_latinitas_id(source_identity, object_key, source_scope=…)` contract.
`phrase-object-v1-…` hashes the pattern and all bound sources/senses/form keys;
distinct tested forms stay distinct while wording/gloss/evidence does not feed
identity. Recipes and their semantic card keys remain separate later work.

## Proposal generation (bounded subset)

`generate` consumes the saved contract, not the input request. It reloads current
lexical/pattern claims and checks government, agreement, selected-sense allowlists
and valency again. Saved source scopes and composite keys must match the current
bindings. Missing forms, stale/unreviewed inputs and incompatible combinations
appear in `skipped` with named diagnostics and repair actions, never guessed data.
Malformed contracts exit nonzero; a valid report with skipped proposals exits zero.
Saved files are limited to 4 MiB before parsing. Persisted entry/source identities
and scopes are limited to 2048 characters, and role/sense/form/pattern/composite keys
to 256 characters, so a rejected selection cannot echo arbitrarily large strings.

Only explicitly saved form choices are visited. No lexical, sense or form Cartesian
product is constructed, and unselected variants are never auto-selected. Canonical
binding order makes reruns deterministic. `max_exercises` caps **attempts**, including
rejections and duplicates, as well as output; `visited`, `unvisited` and `limit_reason`
expose truncation. Sorting/parsing is bounded by the saved contract's 1000 selections.
`variants_per_binding` caps distinct emitted form combinations for the same pattern
and role-labelled sources/senses. Lower limits can be saved for a later rerun; narrow
selections rather than relying on a particular variant winning canonical order.

Preview includes `candidate_key` (the persisted object key), source identities/scopes,
sense keys/meanings, pattern key/version (version-1 contract), complete chosen forms,
explicit function words, evidence and contributing claim assessments. All generated
combinations have `review_state: proposal`, even when all inputs are reviewed.
Function words precede content; content order is object for prepositions, head then
dependent/modifier for nominal patterns, and subject–object–verb for clauses.
Patterns must be reviewed for this rendering order. No translations or answers are
inferred, and no naturalness/combination approval is carried over from input review.

The synthetic fixture's independently specified expected finite-verb analyses are:

| Explicit choices | Expected analysis | Control |
|---|---|---|
| `puella amicum videt` | `videt`: third-person singular present active indicative | `puella … vident` is rejected for number disagreement |
| `puellae amicum vident` | `vident`: third-person plural present active indicative | No isolated-form ambiguity is erased; no recognition recipe is rendered |

To request the plural clause, explicitly select subject `girl/person/nom-pl`, object
`friend/person/acc-sg`, verb `see/perceive/present-pl` with pattern `sees`. For `ad amicum`,
select `friend/person/acc-sg` with `towards`; `with` requires `abl-sg` (`cum amico`).
The example request deliberately selects only four choices; it does not select every
available form or pattern. This subset supports nominative noun subjects and reviewed
third-person finite forms only, not exhaustive conjugation or general syntax analysis.

Selection/generation is not approval of a generated phrase or rendered answer.
Initial CSV import and destination setup are manual; managed creation remains unsupported.

## Review generated exercises and inspect eligibility

Use the shared commands with both saved selections and recipes:

```bash
uv run latinitas-cards claims list inputs.json --selections selected.json --recipes recipes.json --json
uv run latinitas-cards claims record inputs.json --selections selected.json --recipes recipes.json \
  --fingerprint <fingerprint> --status accepted --reviewer teacher --reason "Individually checked"
uv run latinitas-cards claims record inputs.json --selections selected.json --recipes recipes.json \
  --batch decisions.json --json
uv run latinitas-cards claims inspect inputs.json --selections selected.json --recipes recipes.json --check --json
```

`recipes.json.claims.json` uses the same T064 locking, atomic batches, stale/unmatched
diagnostics and conflicting-decision rejection as other recipes. Combination claims
attest correctness **and** naturalness, and cannot be rewritten as free text. Answer
claims contain a complete JSON `Reference`; an authored correction supplies that JSON
as `value` plus its own `evidence` in an individual or batch decision. Invalid references
or unresolved tested distinctions remain ineligible. Withholding either gate disables
the exercise. These linguistic attestations are not export or managed-plan approval.

The saved role-labelled phrase object links every contributing source identity/sense,
pattern and form. Each target/distinction question has a separate coherent note ID
derived from the composite source and semantic exercise key, plus fixed recipe slot
10 (translation) or 11 (recognition); no card is attached to one lexical word and source
notes are never merged or changed. Gloss/reference wording and presentation edits keep
object/note/card IDs and slots; new sense/form keys or questions distinguish exercises.
Changed linguistic context, evidence, analysis or translation invalidates the relevant
approval; unchanged reruns and terminology/appearance changes preserve it.

Read-only inspection shows identities, exercise keys, slots, contributing constituents,
claims/fingerprints, evidence, alternatives and withholding reasons without Anki. `--check`
fails on open/withheld/stale/unmatched or unavailable exercises. Phrase CSV/package
checking supports the documented phrase-v1 layout (supplied packages only; no package
generation). Tampered answers, stale reviews, wrong identities/slots, missing payload
columns and incompatible template/style layouts fail nonzero. Native automatic appearance
acceptance on Desktop/Mobile remains pending; browser rendering does not substitute.

## Initial CSV, manual setup and explicit adoption

After the individual decisions above, use new output paths:

```bash
uv run latinitas-cards phrases preview inputs.json selected.json recipes.json --json
uv run latinitas-cards phrases export inputs.json selected.json recipes.json initial.csv --json
uv run latinitas-cards phrases export inputs.json selected.json recipes.json rerun.csv --json
cmp initial.csv rerun.csv
uv run latinitas-cards claims inspect inputs.json --selections selected.json --recipes recipes.json \
  --artifact initial.csv --check --json
```

Export omits unapproved exercises and reports diagnostics. This is **initial creation
only**, not a withdrawal/update strategy. It refuses existing output/source/review paths.
The same source, profile, selections and sidecars produce byte-identical CSV; no lexical
note, source manifest or review decision is written. Read-only package checking uses
the same command with `--artifact supplied.apkg` or `supplied.colpkg`.

Obtain the exact manual setup contract without opening Anki:

```bash
uv run python -c 'import json; from dataclasses import asdict; from latinitas_cards.managed_layouts import layout_version; print(json.dumps(asdict(layout_version("phrase-v1")), ensure_ascii=False, indent=2))' > phrase-layout.json
```

Review all fields, ordered templates and CSS before applying them in a disposable
test collection. Prefer a **separate dedicated note type/destination** for new phrase
notes, named exactly as the profile's `generated_note_type`; preserve `LatinitasID`
as the first field and do not map Personal Notes. The original fields/templates are
unchanged, with phrase fields appended after them and slots 10/11. Install the exact
CSS v3 text manually; CSV installs neither CSS nor templates. Never renumber or
repurpose scheduled templates. Old CSS/template digests fail closed: do not edit a
snapshot/journal to match or silently rebind it. Review manual setup, acquire a fresh
closed backup, and perform explicit ownership adoption in a new journal. Existing
scheduled collections need reviewed manual resolution, not an automatic migration.

Preview import mapping carefully, import **initial.csv manually** into the dedicated
note type, then export a closed complete backup. Follow the acquisition, immutable
membership and ownership instructions in
[destination snapshots and file transport](destination-snapshots-and-file-transport.md).
Phrase membership is `[LatinitasID, Source Scope, Source ID, exercise_key]`; use the
semantic `exercise_key` from workspace inspection **without `--artifact`**, not the
shared phrase `object_key` or the template's `semantic_key`:

```bash
uv run latinitas-cards claims inspect inputs.json --selections selected.json --recipes recipes.json \
  --check --json > workspace-inspection.json
jq '.cards[] | {identity, exercise_key, recipe, slot, constituents}' workspace-inspection.json
```

The exported row supplies `Source Scope` and `Source ID`; the workspace card supplies
the final membership key. Artifact inspection instead reports every layout slot and
checks exported fields against the workspace. Do not infer membership from displayed text.
Capture validates actual fields/templates/CSS; linguistic decisions are never write
approval or proof of a successful import.

```bash
uv run --with anki==26.9.3 latinitas-cards managed capture closed.anki2 \
  --selection membership.json --closed-backup --interval-confirmed > snapshot.json
uv run latinitas-cards managed adopt snapshot.json --ownership ownership.json \
  --review "Manually inspected initial import and every ownership origin" --state state.json
```

## Approved retained-slot changes and blocked withdrawal

Correct an answer through individual shared decisions, or change presentation in the
profile. Linguistic/content changes invalidate review; presentation retains claims but
changes the payload and requires **new managed write approval**. Use fresh observed
evidence and inspect the exact changed field operations:

```bash
uv run latinitas-cards phrases plan inputs.json selected.json recipes.json snapshot.json state.json \
  --json > plan.json
jq '.notes[] | {identity, classification, card_effects, blocked, operations}' plan.json
uv run latinitas-cards managed approve plan.json --operation 'EXACT-ID/field/PhraseRecognitionAnswer' \
  --review "Reviewed this exact retained answer and full import footprint" > approval.json
```

For principal-part and parsing payloads, the existing `managed plan request.json`
interface uses the same retained-slot guard. Parsing has its own `parsing-v1` binding,
not phrase slots. Only prompt/answer fields of observed retained cards can change;
identity, source provenance, eligibility guards, slot ordinals, personal fields and
unselected destination fields/tags remain unchanged. Creation, retirement/reactivation,
blank eligible payloads, stale receipts, conflicts and changed card/history evidence
remain blocked. Inspect both selected operations and full-column rows before import.

Build the existing handoff JSON from the fresh snapshot, binding, plan and receipt;
set `backup`, reviewed `recovery`, `note_type` and `deck` exactly as approved. Then:

```bash
uv run latinitas-cards managed emit handoff.json --state state.json --output updates.csv
# Manual Anki import: update existing within this note type; never map Personal Notes.
# Acquire a fresh closed result and prepare observation.json using its actual snapshot.
uv run latinitas-cards managed observe observation.json --state state.json --interval-confirmed
# If interrupted/partial or destination drifted: review ownership against fresh evidence.
uv run latinitas-cards managed reconcile reconciliation.json --state state.json
```

Emission is pending, not success. Anchors advance only after observed fields, tags,
native note/card identity, scheduling/history and Personal Notes digest match. Do not
review/edit/sync on any device between capture, import and observation; recoverability
and this interval are operator attestations, not a lock, age limit or restore proof.
Unknown outcomes require fresh observation/reconciliation, never blind replay.

Removing an exercise from the explicit recipe plan (including an empty `exercises`
list) makes `phrases plan` produce an explicit **unsupported retirement** for its
adopted phrase note. Withheld/unavailable references produce blocked uncertain plans.
CSV omission does not remove a note, and no tag alone retires a card. Review a manual
withdrawal/suspension resolution and acquire fresh evidence, then reconcile ownership;
the blocked plan itself does not perform or certify retirement. A later update needs
a new plan and receipt. Native auto acceptance remains pending under the scoped
2026-10-11 implementation-loop deferral; preserve the exact checklist and limits in
[morphology native verification](morphology-native-verification.md#automatic-appearance-native-checklist-t067).
