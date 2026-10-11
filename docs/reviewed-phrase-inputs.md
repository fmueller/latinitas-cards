# Reviewed lexical inputs and bounded patterns (version 1)

`phrases` imports a JSON file, inspects its proposals, and saves explicit reviewed
selections. `phrases generate` previews bounded phrase proposals. It does **not**
render recipes, approve generated answers, or export cards. No corpus, network, LLM, CLTK or source-note edits are
needed. Use the existing confirmed deck profile and source identities, then supply
missing lexical knowledge in a separate file rather than altering the deck.

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
Recipes, explicit combination/answer review and CSV export remain later work.
