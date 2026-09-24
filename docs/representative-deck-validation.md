# Representative deck validation

This note records the local validation of the user-provided university Latin APKG for
T-011. The original APKG, extracted collection database, media, note values, GUIDs,
tags, and raw samples are private temporary inputs and are not repository artifacts.

## Source observations

The canonical APKG adapter read the private source without modifying it:

| Observation | Result |
| --- | --- |
| Collection format | Modern APKG with a compressed `collection.anki21b` member and a legacy compatibility member |
| Note type count | One source vocabulary note type |
| Note count | 677 |
| Named fields | Seven; the relevant roles are lexical entry, German gloss, and construction/form data |
| Native identity | 677 non-empty, unique note GUIDs retained by the adapter |
| Note template | One template references the Latin entry, construction/hint field, German field, and lesson/reference fields; there is no separate template field named as principal parts |

The semantic mapping is therefore not based on field position alone:

- The source lexical-entry field is non-empty for all 677 notes, but it is
  mixed: 559 single-token values, 79 pipe alternatives, 9 slash alternatives, 12 HTML
  values, 8 multiword values, and smaller dash/semicolon forms.
- The source German gloss/meaning field is non-empty for all 677 notes and
  includes both plain and formatted/multi-gloss values.
- The source construction/form field is blank in 52 notes,
  and the populated values include formatted text, alternatives, and comma/semicolon
  lists. It is not treated as principal parts merely because it contains separators.

### Principal-form evidence

Private structural inspection of the source construction/form field, after removing HTML tags and
decoding entities only for analysis, found a bounded candidate shape:

```text
role 1: one form ending like -re
role 2: one form ending like -o/-ō
role 3: one form ending like -i/-ī
role 4: one supplied form with an observed shape like -um
```

For the confirmed university profile, role 4 is modeled as
`perfect_passive_participle`, called `Partizip Perfekt Passiv (PPP)`. A bare `-um`
ending is only a structural observation: it does not establish the semantic role,
case, or gender. The sanitized fixture places `lātum` in the explicitly configured
fourth slot as the public fixture value for this PPP role; the
role assignment comes from the approved profile, not from the isolated form.

The four roles occur in that order in 152 of 677 records. This is evidence for a
principal-form layout in this field, not a claim that all 152 are eligible verbs:

- 104 candidates have one simple lexical-entry token matching the first role after
  display-only normalization.
- 47 candidates have pipe alternatives in the lexical-entry and/or form field.
- 1 candidate has a multiword lexical-entry value.
- 85 candidates use plain comma spacing, 65 use HTML/entity formatting with mixed
  spacing, and 2 use HTML with comma spacing.
- 12 other four-segment values have the first three role-like shapes but no fourth
  form matching the confirmed PPP candidate shape. They remain exceptional/incomplete
  candidates.

The adapter/parser boundary is measurable but must not be confused with semantic
eligibility:

- A literal comma profile (`(",",)`) structurally splits 189 records into four
  segments. Only 152 of those have the observed role-order shape; the other 37 are
  not counted as principal-form candidates. The 12 incomplete fourth-role records
  described above are a subset of these 37, not an additional group.
- A literal comma-space profile (`(", ",)`) parses only 118 records because source
  HTML/entity formatting removes the literal space in some separators.
- The 118/677 comma-space result is a separator-profile observation, not a successful
  eligible-verb count.
- The confirmed profile uses `(",",)` and relies on the parser's existing boundary
  trimming. It preserves display values and does not split pipe alternatives into
  additional semantic roles.
- Records with other separator counts, blank form fields, HTML-only uncertainty, or
  role-order mismatch remain incomplete, unsupported, or ambiguous. They are not
  silently assigned to a recipe.

The existing source template displays the construction/form field alongside the German
gloss and reference fields. That template relationship, the four-form role order, and
the normalized lexical-entry matches together support the mapping to the form field;
separator counts alone would not.

## Sanitized fixture

`tests/fixtures/representative-university-latin.apkg` is a newly generated, minimal
modern APKG fixture. It is not a copy or extract of the private deck.

Sanitization and provenance method:

1. Retain only the observed one-note-type/seven-field shape and the mixed lexical,
   gloss, blank, alternative, formatted, comma, and semicolon structures needed by
   the tests.
2. Replace the note type and all field names with role-neutral names. Reference fields
   contain only synthetic placeholders.
3. Replace every note ID and GUID with deterministic `fixture-guid-*` values; remove
   tags, decks, media, and collection metadata.
4. Replace all lexical, German, and form text with short public Latin/German examples
   while preserving the observed primary order and variant boundaries. The blank gloss
   is an explicit optional-field test boundary, not a claim that the source had a blank
   source gloss value.

The fixture maps the source roles as follows:

| Sanitized field | Source role | Fixture evidence |
| --- | --- | --- |
| `Entry` | source lexical-entry field | single token, pipe alternative, HTML, and slash alternative |
| `German gloss` | source German gloss field | German values plus one explicit optional blank |
| `Construction hints` | source construction/form field | primary comma form order, pipe alternatives, blank, and semicolon variant |
| `Reference A/B/C`, `Flag` | source reference/flag fields | sanitized placeholders only |

Masked source shapes used during inspection included:

```text
lexical entry:          A        A|A        A / A        <A>A</A>
construction/form:      A, A, A, A        A|A, A|A, A|A, A|A
                        [blank]            A; A; A; A
German gloss:           A        A, A        <A>A</A>
```

The committed fixture examples are sanitized and traceable to those shapes, not raw
note samples:

```text
dīcere           -> dīcere, dīcō, dīxī, dictum       -> sagen
amāre|amare      -> amāre|amare, amō|amo, ...         -> lieben
<b>sequor</b>    -> [blank]                           -> <i>folgen</i>
vidēre / videre  -> vidēre; videō; vīdī; vīsum        -> sehen
ferre              -> ferre, ferō, tulī, lātum          -> [optional blank]
```

These examples demonstrate adapter and parser boundaries; they do not validate the
user's exact wording or establish that every candidate is a verb. The `ferre` row
specifically exercises the approved PPP role while retaining the optional blank gloss.

## Review proposal and approval

The following narrow choices were reviewed and approved before assisted setup or recipe
work proceeds.

### Mapping and eligibility

1. Map the source lexical-entry field to the lexical-entry role and the source German
   gloss field to the optional German gloss.
2. Map the source construction/form field to the principal-form role only under the confirmed
   four-role profile, with this role order:

   ```text
   present_infinitive, present_1s, perfect_1s, perfect_passive_participle
   ```

   Use the literal comma separator `(",",)` so both comma-spacing variants are
   structurally readable after trimming.
3. The primary layout boundary is a four-slot comma-separated value in the mapped
   construction/form field whose slots match the confirmed role order. Treat the 104
   single-token candidates that also match the first role as structural/lexical-shape
   candidates for review, not as verified verb eligibility. Separator counts and
   role-like endings do not establish that an entry is a verb or that the fourth slot
   is a PPP.
   Keep the 47 pipe alternatives and one multiword value in a review queue; do not
   infer role identity by splitting their alternatives or automatically include any
   candidate in a recipe without explicit eligibility confirmation.
4. The exceptional/review-only boundary includes the 37 four-segment but
   role-order-mismatched values, the 12 incomplete fourth-role candidates, blanks,
   non-comma/role-order variants, and ambiguous variants such as pipe alternatives
   and multiword entries; keep them out of the automatic recipe set. HTML/entity
   formatting is not itself a role variant: the comma profile can preserve it, but
   those displays still need rendering review.
   No three-role/deponent layout is claimed from this deck yet; it would require an
   explicit confirmed profile rather than position guessing.

   Neither exceptional nor ambiguous variants are silently treated as eligible verbs.

This boundary distinguishes whole-deck adapter fit (677/677), four-slot structural
parses under the comma profile (189), observed principal-form candidates (152),
structural/lexical-shape candidates with a simple lexical entry (104), genuinely
unsupported or incomplete layouts (including the 12 exceptional fourth-role cases),
and unresolved variants. None of these counts alone is a count of eligible verbs.

### German terminology and generated wording

Approved terminology and exact wording:

| Concept | Approved German |
| --- | --- |
| Principal parts | `Stammformen` |
| Gloss/meaning label | `Bedeutung` (source mapping remains the German gloss field) |
| `present_infinitive` | `Infinitiv` |
| `present_1s` | `Präsens, 1. Person Singular` |
| `perfect_1s` | `Perfekt, 1. Person Singular` |
| `perfect_passive_participle` | `Partizip Perfekt Passiv (PPP)` |
| Completion prompt | `Ergänze die fehlende Stammform.` |
| Recognition prompt | `Welche Stammform ist „{form}“?` |

Recipe proposal is limited to the two already defined recipe identities:
`principal_part_completion` and `principal_part_recognition`. German-to-Latin
production is not proposed. CLI diagnostics remain English as required by the spec;
the German wording applies to generated study content only.

The user approved the field mapping, conservative eligibility boundary, German terms
and prompts above, including `PPP` / `Partizip Perfekt Passiv` for the fourth role.
The private deck upload was treated as input evidence, not as approval. Word-component
explanations, coloring, and morphological segmentation remain planned v0.2 work and
are not part of this task.
