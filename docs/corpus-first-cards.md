# Corpus-first cards

The active v0.4.0 implementation generates cards from normalized Latin corpora,
not from a vocabulary deck. It is not a release-readiness declaration. Native
acceptance and the final release review still have separate gates; publication
requires the maintainer. The released v0.3.0 package does not have `corpus-cards`.
Run these examples from a current checkout after `uv sync --locked --dev`.

## Profiles and recipes

A `latinitas-corpus-profile` v1 names its corpora, target language, dedicated
note type, deck and recipe budgets. See the [complete example](examples/corpus-profile.json)
and [schema](schemas/latinitas-corpus-profile.v1.schema.json). Validation reads only
the profile; corpus IDs and reading bounds are checked when consuming corpora.
Supply one `--corpus FILE` for every listed corpus, with no extra corpora.

| Recipe | Context and target | Required accepted facts |
| --- | --- | --- |
| `corpus_lexical_recognition` | Sentence, one word | Token morphology and token translation. |
| `corpus_guided_analysis` | Sentence, one word | Token morphology with at least one feature and sentence translation. |
| `corpus_phrase_translation` | Sentence, 2–8 consecutive word tokens forming a proper sub-span, from the first word's start to the last word's end | Translation anchored to that exact phrase span. |
| `corpus_sentence_translation` | Sentence, or a multi-sentence passage within the token limit | Translation of that unit. |

Sentence facts use the exact current sentence span. For a one-sentence passage,
a span-less translation also serves as the sentence translation. A multi-sentence
passage translation does not supply its individual sentence translations.
Spans are half-open Unicode code-point offsets, not bytes, and spanned facts bind
the current segmentation ID. Morphology must use `latinitas-morph/1`, a lemma,
supported part of speech and controlled canonical features; arbitrary analyzer
tagsets are not automatically converted. See [fact rules](../specs/v0.4.0.md#fixed-contracts).

Import approval permits an annotation write; it does not accept a linguistic claim.
Confidence and producer kind never grant acceptance. Reference translations are
guidance for study, not grading keys or a test of a learner's wording. Review each
translation and morphological assertion in its Latin context before accepting it.
Exact word-token glosses serve lexical recognition and are not phrase candidates.
Other malformed phrase spans remain in the report as `invalid_span` entries.

## Runnable English walkthrough

This disposable Bash example creates its own tiny Latin TEI source and aligned
English text. It needs Python and `jq` as well as the checkout. Keep the shell
variables for subsequent blocks. Only the synthetic assertions below are suitable
for this scripted review; real editions need individual linguistic decisions.

```bash
work=$(mktemp -d)
export work
cat > "$work/latin.xml" <<'XML'
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader><fileDesc><titleStmt><title>Study example</title></titleStmt></fileDesc></teiHeader>
  <text><body><div type="edition" xml:lang="lat">
    <div type="textpart" subtype="section" n="1"><p>Marcus amicum videt.</p></div>
  </div></body></text>
</TEI>
XML
printf '1\tMarcus sees a friend.\n' > "$work/english.txt"
uv run latinitas-cards corpus normalize "$work/latin.xml" --adapter tei \
  --corpus-id study-latin --output "$work/latin.jsonl" --json
uv run latinitas-cards corpus normalize "$work/english.txt" --adapter text \
  --corpus-id study-english --language en --citation-template study:section \
  --output "$work/english.jsonl" --json
uv run latinitas-cards annotations segment "$work/latin.jsonl" \
  --timestamp 2026-01-01T00:00:00Z --json
uv run latinitas-cards annotations import-translations "$work/latin.jsonl" \
  --parallel "$work/english.jsonl" --language en \
  --producer-kind human --producer-name study --producer-version 1 \
  --output "$work/parallel.patch.json" --timestamp 2026-01-01T00:00:00Z --json
```

Alignment uses `native_ref`, not passage order or similar-looking text. Unmatched
Latin and parallel refs warn separately. The parallel header must match the
requested language's primary subtag and must not be Latin. The source CORPUS
header must declare primary language `la` (case-insensitive); otherwise import
fails with `unsupported_language` (exit 3). This checks metadata, not the passage
text's language, so success alone does not establish a valid Latin study source.

Add explicitly anchored facts to the imported passage translation. The Python
block authors a patch, not a sidecar or review decision. It reads generated IDs
and hashes rather than inventing them.

```bash
uv run python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["work"])
passage = next(r for r in map(json.loads, (root / "latin.jsonl").read_text().splitlines()) if r["record"] == "passage")
segment = next(r for r in map(json.loads, (root / "latin.annotations.jsonl").read_text().splitlines()) if r.get("layer") == "segmentation")
patch = json.loads((root / "parallel.patch.json").read_text())
anchor = {"passage_id": passage["passage_id"], "passage_hash": passage["passage_hash"], "segmentation_id": segment["annotation_id"]}
for layer, span, value in [
    ("morphology", [7, 13], {"tagset": "latinitas-morph/1", "lemma": "amicus", "pos": "noun", "features": {"case": "accusativus"}}),
    ("translation", [7, 13], {"language": "en", "text": "friend"}),
    ("translation", [7, 19], {"language": "en", "text": "sees a friend"}),
]:
    patch["operations"].append({"operation": "add", "layer": layer, "anchor": anchor | {"span": span}, "value": value,
                                "producer": {"kind": "human", "name": "study", "version": "1"}})
(root / "facts.patch.json").write_text(json.dumps(patch))
profile = {
    "format": "latinitas-corpus-profile", "version": 1, "profile_id": "study-en",
    "corpora": ["study-latin"], "target_language": "en", "note_type": "Latinitas Corpus Context v1", "deck": "Latin::Study",
    "recipes": [{"recipe": recipe, "budget": 10} for recipe in ["corpus_lexical_recognition", "corpus_guided_analysis", "corpus_phrase_translation", "corpus_sentence_translation"]],
}
(root / "profile.json").write_text(json.dumps(profile))
PY
uv run latinitas-cards annotations plan "$work/facts.patch.json" \
  --corpus "$work/latin.jsonl" --sidecar "$work/latin.annotations.jsonl" \
  --output "$work/facts.plan.json" --json
jq '.operations[] | {operation_id, route, reasons, records}' "$work/facts.plan.json"
```

Inspect those four operations, including the phrase `amicum videt` ("sees a
friend"). Copying an approval command is an assertion that you reviewed
the exact operations; it is not evidence of authentication or linguistic truth.

```bash
mapfile -t operations < <(jq -r '.operations[].operation_id' "$work/facts.plan.json")
uv run latinitas-cards annotations approve "$work/facts.plan.json" \
  --operation "${operations[0]}" --operation "${operations[1]}" \
  --operation "${operations[2]}" --operation "${operations[3]}" \
  --actor human:study-reviewer --review 'Checked these four synthetic proposals' \
  --output "$work/facts.receipt.json" --json
uv run latinitas-cards annotations apply "$work/facts.plan.json" \
  --corpus "$work/latin.jsonl" --sidecar "$work/latin.annotations.jsonl" \
  --receipt "$work/facts.receipt.json" --timestamp 2026-01-01T00:00:00Z --json
uv run latinitas-cards claims list "$work/latin.jsonl" \
  --annotations "$work/latin.annotations.jsonl" --json > "$work/claims.json"
jq . "$work/claims.json"
```

After checking each claim in the output, record each decision explicitly. This
loop is only for the known synthetic fixture, including its segmentation claim;
it is not an approve-all facility for real data.

```bash
while IFS= read -r fingerprint; do
  uv run latinitas-cards claims record "$work/latin.jsonl" \
    --annotations "$work/latin.annotations.jsonl" --fingerprint "$fingerprint" \
    --status accepted --reviewer 'Study reviewer' --reason 'Checked this synthetic claim'
done < <(jq -r '.claims[].fingerprint' "$work/claims.json")
uv run latinitas-cards claims inspect "$work/latin.jsonl" \
  --annotations "$work/latin.annotations.jsonl" --check --json
uv run latinitas-cards corpus-cards profile validate "$work/profile.json" --json
uv run latinitas-cards corpus-cards select "$work/profile.json" \
  --corpus "$work/latin.jsonl" --selection "$work/selection.json" --report "$work/candidates.json" --json
uv run latinitas-cards corpus-cards preview "$work/profile.json" \
  --corpus "$work/latin.jsonl" --selection "$work/selection.json" --json
exercise=$(jq -r '.selected[0].latinitas_id' "$work/selection.json")
uv run latinitas-cards corpus-cards deselect "$work/selection.json" \
  --exercise "$exercise" --reason 'Pause this exercise' --json
uv run latinitas-cards corpus-cards deselect "$work/selection.json" \
  --exercise "$exercise" --reason 'Reviewed restoration' --restore --json
uv run latinitas-cards corpus-cards export "$work/profile.json" \
  --corpus "$work/latin.jsonl" --selection "$work/selection.json" --output "$work/initial.csv" --json
uv run latinitas-cards corpus-cards inspect "$work/profile.json" \
  --corpus "$work/latin.jsonl" --selection "$work/selection.json" --check "$work/initial.csv" --json
```

Expect four eligible exercises, one per recipe, with no extra phrase candidate
for the one-word gloss. An existing export path is rejected, even for identical
bytes. Choose a new path for each initial export.

## Ranking and sticky selection

Selection is sticky: existing selections stay, even when ineligible or over a
new budget, and occupy recipe/work slots. Each run fills only free budgets;
`per_run` limits newly selected exercises, not kept ones. Deselect creates a
tombstone with your reason. Later selection never restores it automatically;
`--restore` retains its previously reviewed context digest.

`latin-rank/1` scores accepted facts with integer weights for reading-range membership,
target lemmas, novelty, length, unknown words and morphology coverage. Length and
unknown factors are non-positive, so positive weights penalize them. Filters run
before ranking; near-duplicate multiset overlap and per-lemma exclusions then
limit repetition. Supply both `--known-lemmas PROFILE` and
`--known-lemmas-source SOURCE` to select/preview if using a confirmed deck as a
read-only known-lemma source. Without that pair every word is unknown, so a low
`max_unknown` can filter everything. Export/inspect have no known-lemma input;
preview without that pair is the export eligibility view.
Candidate reports and previews include `deck_links` for every matching deck
source identity and its scope on an accepted target lemma. Nonmatching targets
and phrase/sentence candidates have no links. These links never enter card fields
or selection state, and no deck files are changed.

The [agent contract](agent-cli-contract.md#ranked-corpus-selection) lists every
outcome/withholding reason and their precedence. Reports retain ineligible
candidates; `kept` is not an eligibility promise. Missing targets remain selected
but withheld. Changed context requires review followed by
`select --confirm-context ID`; restoring a tombstone alone does not confirm new context. Identity never
migrates to another target. Unchanged selection bytes and run counts are preserved;
the candidate report is rewritten and is never read as selection state.

## Initial import, adoption and managed updates

An initial CSV import **can create new notes**. Prepare the dedicated `corpus-v1`
note type with its four guarded card slots and stylesheet before manual import;
CSV does not install templates or CSS. The fields, templates and CSS are defined
by `corpus_notes.py` and `managed_layouts.py`, not the principal-part note type.
Import with LatinitasID first and the exact Note Type mapping. Exactly one slot
is enabled per exported exercise; Personal Notes is never exported.

After importing, obtain a complete closed-backup snapshot of the whole managed
set, review per-note ownership and run `managed adopt`. See the
[capture/adoption prerequisites and executable sanitized recipe](managed-capture-and-adoption.md).
Never manufacture a real destination snapshot from the generated CSV: it cannot
prove native card identities, schedules, history, ownership or collection binding.

Correcting an accepted annotation is not an uninterrupted CLI-only flow today.
Superseding it leaves an obsolete stored claim decision; `claims record` refuses
new decisions while any stale/unmatched decision remains. There is no CLI command
to prune or migrate those decisions. Stop, preserve the original claims sidecar
as separate evidence, and resolve only the obsolete entries in a reviewed v2
copy before accepting the replacement. Preserve all still-current accepted and
withheld decisions unchanged; do not delete the sidecar to bypass review. See
[corpus claim review](reviewed-claim-cli.md#corpus-annotation-review-version-2).
The new annotation still needs its own explicit linguistic decision before a
managed content update can be eligible.

For an adopted set, `corpus-cards plan` creates a request, not an import:

```bash
uv run latinitas-cards corpus-cards plan "$work/profile.json" \
  --corpus "$work/latin.jsonl" --selection "$work/selection.json" \
  snapshot.json baseline.json --output corpus-request.json --json
uv run latinitas-cards managed plan corpus-request.json > managed-plan.json
```

Here `snapshot.json` and `baseline.json` must be real outputs of the documented
capture/adoption flow. Review operations, then use the exact request and approval
fields in the [managed handoff contract](destination-snapshots-and-file-transport.md#managed-csv-handoff-and-observation):
`managed approve` selects operation IDs; `managed emit` requires that approval,
snapshot, backup and recovery instructions; `managed observe` checks the actual
post-import snapshot/report before advancing the baseline. Use `managed reconcile`
after partial import or restored backup, then replan and renew approval. Neither
the corpus command nor emission changes Anki. Keep reviews, edits and sync stopped
on every device through observation; backup recoverability is an operator assertion.

**Managed CSV updates cannot create or retire notes**, add/remove sibling cards,
change templates or suspend/unsuspend cards. Compatible retained-slot content/tag
changes may proceed only through their separate approval. Withheld anchored notes
retain safe anchored fields; unanchored out-of-set exercises are `not_imported`.
Only explicitly deselected anchored exercises propose retirement, which remains
unsupported. Omission, an empty Enabled field or a retirement tag is not deletion
or suspension. Manual initial import followed by adoption is not managed create.

## Citations and source provenance

The existing `Citation` field separates a readable author/work/passage label
from an expandable source-provenance section. German examples are
`Julius Caesar, Der Gallische Krieg, Buch 1, Kapitel 1, Abschnitt 1` and
`Vergil, Aeneis, Buch 1, Vers 11`; English uses `The Gallic War`, `Virgil`,
`Aeneid`, `Book`, `Chapter`, `Section` and `Line`. Names are localized only from
verified work metadata; arbitrary supplied names remain literal. Missing authors
are identified as missing, and missing work titles fall back to the local work ID.
Normalized citation parts and identity fields are not translated.

Open **Source provenance and limitations** (German: **Quellennachweis und Grenzen**)
on an Anki answer to inspect HTML-escaped canonical JSON independently of the label.
DEL/C1 characters use JSON Unicode escapes for safe field transport; decoding the
JSON recovers the exact supplied values.
It retains the supported header/work metadata, local passage/work IDs, native ref,
raw citation parts and passage hash. The header includes the raw XML filename,
byte count and SHA256, edition, publisher, source URL, supplied revision and source
licence. Optional missing metadata stays absent/null, with explicit limitations
and `incomplete_provenance` CLI warnings. Preview shows only the readable citation;
the full provenance travels through CSV in `Citation`, without extra note fields.

Exact online links are currently verified only for these pinned source readings:

| Reading | Edition | Raw XML SHA256 |
| --- | --- | --- |
| Caesar BG 1.1.1 | `phi0448.phi001.perseus-lat2` | `15ec3ef2a36352742ddb75d4f7da23a143899b6b60ee3a7ca3192d4917ca1568` |
| Aeneid 1.11 | `phi0690.phi003.perseus-lat2` | `e639e6a0e27427ef28031850c4d73aa2b63b9f2cf8a5cf584b37685c52fb874c` |

The sources are from `PerseusDL/canonical-latinLit` revision
`8b536bdf3ed1f43494055efe112433b4a58eab67`. Scaife reader URLs for these full CTS
edition/passage references were verified in a browser on 2026-10-31, including
Holmes's Oxford 1914 and Greenough's Boston 1881 edition descriptions. The online
resolver uses its own revision: `revision_matches_source: false` explicitly avoids
claiming byte/revision equality with the pinned XML. Link identity is identical
in German and English. Generation makes no network request; future remote
availability is not guaranteed. Other references, unknown editions or mismatched
source bindings remain nonclickable, with a limitation instead of a guessed URL.
The link also requires the current passage hash to match the independently measured
pinned reading. Corrected text remains supported but loses this link when its text
changes. Source-level verified names can still be localized for other passages.
An arbitrary supplied source URL is retained as data, never injected into a link.

Supply the pinned raw URL and revision via a descriptor when normalizing; neither
is invented from the filename. A source-attested edition URN permits a separately
marked derived CTS reference, not a claim of verified online resolution.
Source licence/provenance does not cover translations: their annotation producers,
review and rights must be considered separately. These records do not grant
permission to redistribute corpora or translations.

The note type, CSV columns, LatinitasIDs and sticky selections are unchanged.
Existing normalized records remain readable; renormalize TEI to obtain new
author/edition metadata. Older strict corpus readers reject enriched records.
Metadata-only changes keep passage hashes and existing annotation claims; actual
header changes require pending header-bound plans to be recreated/reapproved.
Use the managed retained-content workflow to update an imported `Citation`.

## English and language changes

Corpus profiles accept primary target languages `de` and `en`, including regional
tags. Fixed prompts use English for `en`; terminology uses English in `user` mode
and Latin in `latin` mode. Reference translations must match the target primary
subtag. The tool trusts the declared language; it cannot detect mistranslated text.
See [deck language behavior](../README.md#card-language) for deck-first profiles.

A **deck profile language switch stales claims**, because claim fingerprints bind
the whole profile except morphology. Re-review all claims, not just translations;
recipe overrides do not exempt claims from that fingerprint contract.

Corpus annotation claims bind annotation content, anchor and provenance, not the
corpus profile. Switching a corpus target language requires matching translation facts
and a new selection: the old selection fails `binding_drift`. Corpus exercise IDs
include the primary target language, so new-language cards have different IDs;
that does not retire old-language notes. No language switch bypasses managed
structural restrictions or authorizes deletion of old claims or destination notes.

For source limits, the complete recognized TEI elements, and exclusions see
[normalized corpora](normalized-corpus.md#structural-tei). The fixed contracts
remain authoritative; blocked maintainer questions do not grant extra behavior.
