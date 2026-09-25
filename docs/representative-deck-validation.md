# Sanitized representative-deck validation

This note documents only the committed sanitized fixture
`tests/fixtures/representative-university-latin.apkg`. It is a small synthetic APKG
created to exercise the source-adapter, profile, and parser boundaries. Reproducing the
smoke test does not require any external deck or source metadata.

## Provenance and sanitization

The fixture's structural layout and variants were selected from actual user-supplied deck
observations during T-011. Those observations informed the schema shape and variant
coverage, but this file is a synthetic replay of the reviewed evidence, not the original
source.

For sanitization, fixture schema labels, lexical, gloss, and form text, identities, and
metadata were replaced with neutral/public synthetic examples. No original deck,
collection database, or media is distributed. No original/private source filename, URL,
hash, raw note, or source-derived aggregate is included in the fixture or this note.

The representative-deck smoke tests run only against the committed sanitized artifact;
they do not access or revalidate the original source. The smoke tests therefore
demonstrate adapter, profile, and parser behavior for selected observed boundaries, not
source-wide counts or a replay of the original evidence.

## Fixture observations

The fixture contains one modern APKG note type, five synthetic notes, seven named fields,
and deterministic `fixture-guid-*` note identities. The role-relevant fields are:

| Fixture field | Profile role | Fixture coverage |
| --- | --- | --- |
| `Entry` | lexical entry | single token, pipe alternative, HTML, and slash alternative |
| `German gloss` | optional German meaning | German values plus one explicit blank |
| `Construction hints` | principal-part field | comma layout, pipe alternatives, blank, and semicolon variant |
| `Reference A/B/C`, `Flag` | unmapped source fields | sanitized placeholders only |

The fixture values are public synthetic examples:

```text
dīcere           -> dīcere, dīcō, dīxī, dictum       -> sagen
amāre|amare      -> amāre|amare, amō|amo, ...         -> lieben
<b>sequor</b>    -> [blank]                           -> <i>folgen</i>
vidēre / videre  -> vidēre; videō; vīdī; vīsum        -> sehen
ferre            -> ferre, ferō, tulī, lātum          -> [optional blank]
```

These examples exercise adapter and parser boundaries; a structural parse does not
establish eligible verb coverage. Pipe alternatives and multiword or formatted values
remain source data to review, not evidence of universal morphology or automatic semantic
eligibility.

The blank German gloss and concrete fixture values are synthetic optional-field tests, not
claims about source behavior.

## Approved profile boundary

The confirmed representative profile maps `Entry`, `German gloss`, and `Construction
hints` explicitly and uses the literal comma separator `(",",)`. Its semantic role order is:

```text
present_infinitive, present_1s, perfect_1s, perfect_passive_participle
```

The fourth role is explicitly `perfect_passive_participle`, called `Partizip Perfekt
Passiv (PPP)`. The role comes from the confirmed profile, not from an isolated `-um`
ending. PPP is distinct from the generic legacy `supine` profile default. Role order may
be changed only through explicit profile confirmation; the parser returns structured
success, incomplete, unsupported, or ambiguous outcomes rather than guessing.

The two confirmed recipe identities are `principal_part_completion` and
`principal_part_recognition`. Generated study content uses the explicit German language
tag (`de` by default), while CLI controls and diagnostics remain English. German-to-Latin
production, word-component explanations, coloring, and full morphological parsing are
not part of this v0.1.0 workflow.

## Verification boundary

The unit suite verifies that the committed fixture is readable without mutation, retains
its synthetic note GUIDs, preserves named fields and layout variants, and reports the
expected structured parser outcomes. The fixture is the only deck artifact used for
repository smoke tests; the profile and export workflow never writes it.

The broader parser support matrix is in
[principal-part-parsing.md](principal-part-parsing.md), and the repeat-import contract is
in [deterministic-csv-export.md](deterministic-csv-export.md).
