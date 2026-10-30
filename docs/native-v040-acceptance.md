# v0.4.0 Linux Anki Desktop acceptance kit

## Scope and evidence boundary

This is offline preparation for [Native Acceptance](../specs/v0.4.0.md#native-acceptance),
not a native result or release approval. All native rows below are **pending**.
Browser previews and backend fixtures do not substitute for actual Linux Anki
Desktop evidence. AnkiMobile is outside the gate and remains **unverified**;
this is not a compatibility claim. Do not use a real profile, account, sync,
device, scheduled collection or user data for this kit.

The structure follows the v0.2.1 [Desktop checklist](morphology-native-verification.md#desktop-checklist-recorded-2026-10-13-now-superseded-by-replay-3):
pin source/setup, isolate synthetic data, import through the native dialog,
inspect the actual reviewer, record updates and state limitations. The old
carrier waiver does not apply: v0.4.0 requires manual corpus-v1 setup.

## Regeneration

Run from this checkout with the locked development environment. The destination
must not exist; the generator refuses reuse rather than opening an existing kit.

```bash
uv sync --locked --dev
kit=$(mktemp -d)/v040-kit
uv run python scripts/build-v040-native-kit.py "$kit"
```

The generator runs the public CLI in-process, using relative paths inside
`reproduction/`. It normalizes the committed synthetic `Marcus amicum videt.`
fixture, segments at fixed timestamp `2026-01-01T00:00:00Z`, plans seven explicit
annotation additions, records a fixture-only approval receipt, applies, records
the eight individual claims (including segmentation), validates the read-only
profile, selects, exports and inspects. It performs no Anki or network operations.
The exact proposals, receipts, decisions and selection files remain in the kit.
This is the [current runnable corpus flow](corpus-first-cards.md#runnable-english-walkthrough),
not direct construction of a destination snapshot or accepted-fact bypass.

`tests/fixtures/native-v040/facts.json` stipulates only the visible synthetic
morphology and German/English reference translations. It is not an analyzer's
authority to accept real claims. `amicum` is accusative singular of `amicus`;
`amicum videt` means "sees a friend" / "sieht einen Freund" in this sentence.
References are guidance, not unique grading keys.

English deck samples reuse the committed T-154 fixtures and their immutable
source identities/scopes and object keys. The generator copies only the inputs
and identity/review sidecars needed by these exporters, not unrelated golden
variants or historical decision batches. It archives original claim sidecars
before removing only fingerprints made obsolete by the English profile/reference.
Still-current decisions remain unchanged. It records fresh,
stipulated fixture decisions through `claims record`; original withheld claims
stay withheld, and principal-part translation claims stay withheld rather than
asserting German translations as English. The phrase reference explicitly gives
"with a friend / with the friend" for `cum amico`. Parsing retains the reviewed
lemma/case/number/gender assertions, all four accepted in the original fixture.
The inherited principal-part fixture supplies the withheld-analysis examples. This bounded
fixture migration is not a prune facility for real data; see
[the stale-decision limitation](corpus-first-cards.md#initial-import-adoption-and-managed-updates).

`manifest.json` binds every other file with SHA-256, relative names only, and no
generation timestamp. Compare manifests and file bytes from two fresh directories
to check regeneration. Source revision and actual native run timestamps belong
in the operator's evidence record, not generated card bytes.

## Inventory and manual setup

| Files | Expected initial content |
| --- | --- |
| `corpus-{de,en}-light.csv` | Four notes and four cards each, one per recipe |
| `corpus-{de,en}-{dark,auto}.csv` | Same four identities and slots, different appearance |
| `corpus-{de,en}-update.csv` | Same four identities and slots; fixed dark instead of fixed light panels |
| `english-principal-{light,dark,auto}.csv` | One source note, principal-part completion/recognition slots |
| `english-phrase-{light,dark,auto}.csv` | Two cum-amico exercises, recognition and translation |
| `english-parsing-{light,dark,auto}.csv` | One contextual puellae parsing card |
| `setup-{corpus-v1,3,phrase-v1,parsing-v1}/` | Exact current field order, templates and CSS |
| `reproduction/`, `fixture.json`, `manifest.json` | Offline provenance; never import as native evidence |

CSV installs neither templates nor CSS. In a **new isolated synthetic Anki
profile**, manually create dedicated note types with these exact CSV names:

| Setup directory | Note type name |
| --- | --- |
| `setup-corpus-v1` | `Latinitas Native Corpus` |
| `setup-3` | `Latinitas Principal Parts` |
| `setup-phrase-v1` | `Latinitas Native Phrase` |
| `setup-parsing-v1` | `Latinitas Contextual Form Parsing v1` |

Create fields in exact `fields.txt` order: **LatinitasID first** in every layout.
Personal Notes stays in its listed position: last except in `phrase-v1`, where
six phrase fields follow it. Do not reorder that established layout.
`Tags` is a special import column, never a note field. Create card types
in the ordinal order and with names in `templates.json`; paste each matching
`N-front.html`/`N-back.html` verbatim, preserving enable guards. Paste `style.css`
verbatim into Styling. Corpus CSS is `CORPUS_CARD_CSS`, not the reference CSS;
the other layouts use their current registered stylesheet. Delete any default
Basic card type not in the registry. Record field/template/CSS hashes from the
manifest and manually compare installed content. Do not rebind existing cards.

## Pending native checklist

Record source commit, manifest digest, date/time, Linux distribution, actual
Anki Desktop version/build, installation method and evidence paths. Record
expected versus observed counts, note/card IDs, slot ordinals and any reload
needed. Do not mark a row passed from browser screenshots or automated tests.

1. **Manual corpus-v1 setup:** follow the exact fields, four guarded templates
   and CORPUS_CARD_CSS above in the isolated synthetic profile. Verify names,
   order and hashes in the actual native editor. All setup evidence is pending.
2. **Initial German and English import:** import both light CSVs through the
   native dialog with comma delimiter, HTML enabled, the exact dedicated note
   type and deck directives, LatinitasID mapped to the first field, all managed
   fields by name, Tags to tags, and Personal Notes **unmapped**. Expect four
   notes/four cards per language, eight distinct notes/cards total. Confirm
   all four recipes, Language/Source/Note Schema metadata and no duplicate slots.
3. **Update and preservation:** enter different nonempty Personal Notes on at
   least one German and one English note; record every note/card ID and ordinal.
   Re-import each matching `update.csv`, select update existing notes, match
   by **first field (LatinitasID)** within the same note type (not GUID), HTML
   enabled and Personal Notes **unmapped** again. Expect four updated/zero new
   per language; same IDs, retained enabled slots, unchanged Personal Notes,
   and visible dark rather than light panels. Re-import once more and
   confirm zero new notes/cards and unchanged identities. This is a manual
   isolated CSV re-import exercise, not a managed-update authorization or proof
   about real schedules/history.
4. **Front/answer matrix:** for each language and each of lexical recognition,
   guided analysis, phrase translation and sentence translation, import the
   matching `light`, `dark`, then `auto` CSV as updates with the same mapping.
   In the actual reviewer inspect **both front and revealed answer**. Check
   Latin-first prompts, target highlight, accepted meaning/features, reference
   guidance, citation/passage, Personal Notes, contrast, wrapping and no clipping.
   Record every language × recipe × appearance × side row, not just one example.
5. **Light → night → light:** for each appearance on those same cards, change
   native Desktop appearance light → night → light without regeneration or
   reimport between toggles. Auto should recolor light → dark → light; fixed
   light/dark should retain their panel palette and stay readable. Record any
   required reviewer reload/reveal. Keep each toggle step's front/answer evidence.
6. **English deck-first output:** manually set up the three separate layouts,
   import each English family and inspect at least one principal-part, one
   phrase and one form-parsing card front/answer in light/dark/auto and across
   the same native light → night → light toggle. Principal completion and
   recognition and both phrase exercises are provided for fuller coverage.
   Check English fixed prose and user terminology, Latin forms and explicitly
   withheld analysis; do not interpret withheld claims as asserted facts.
7. **Disposition:** keep failures and missing rows explicit; attach native
   screenshots/logs and preservation observations to the maintainer gate.
   This kit closes no native row, starts no successor, changes no release gate
   and authorizes no publication. AnkiMobile remains unverified.
