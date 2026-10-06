"""Build the T-062/T-060 native acceptance kit: carrier .apkg + exporter CSVs.

uv run --with anki==26.9.3 python scripts/build-native-acceptance-kit.py <new-output-dir>

Claims are stipulated fixture reviews (reviewer "native-test fixture"), not
calibrated analysis. The carrier only transports the reference note types.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from anki.collection import Collection, ExportAnkiPackageOptions

from latinitas_cards.claim_review import Claim, Evidence, assess_claim, review_claim
from latinitas_cards.form_parsing import (
    PARSING_BACK,
    PARSING_FIELDS,
    PARSING_FRONT,
    PARSING_NOTE_TYPE,
    PARSING_SLOT,
    ParsingCase,
    ParsingInput,
    ParsingProposal,
)
from latinitas_cards.generation import SINGLE_LEXEME_OBJECT_KEY
from latinitas_cards.identity import derive_latinitas_id
from latinitas_cards.preview_export import prepare_principal_part_export, write_principal_part_csv
from latinitas_cards.principal_parts import PrincipalPartParseSuccess, parse_principal_parts
from latinitas_cards.profile import DeckProfile, SourceIdentityConfig
from latinitas_cards.reference_templates import REFERENCE_CARD_CSS, REFERENCE_CARD_TEMPLATES, REFERENCE_NOTE_TYPE_FIELDS
from latinitas_cards.sources import read_source_records

ROOT_DECK = "Latinitas Test"
PP_NOTE_TYPE = "Latinitas Principal Parts Test"
TAG = "latinitas-native-test"
REVIEWER = "native-test fixture"

SOURCE_ROWS = (
    ("amo", "amō", "amō — amāre — amāvī — amātum", "lieben"),
    ("moneo", "moneō", "moneō — monēre — monuī — monitum", "mahnen"),
    ("fero", "ferō", "ferō — ferre — tulī — lātum", "tragen"),
    ("dico", "dīcō", "dīcō — dīcere — dīxī — ", "sagen"),
)

# (source id, role, candidate form, kind, value, status)
CLAIMS = (
    ("amo", "present_1s", "amō", "segmentation", "am- | -ō", "accepted"),
    (
        "amo",
        "present_1s",
        "amō",
        "explanation",
        "Präsens 1. Sg.: Stamm am(ā)- mit Endung -ō; ā + ō verschmilzt zu ō.",
        "accepted",
    ),
    ("amo", "present_infinitive", "amāre", "segmentation", "amā- | -re", "accepted"),
    (
        "amo",
        "present_infinitive",
        "amāre",
        "explanation",
        "Infinitiv Präsens Aktiv: Präsensstamm amā- + Endung -re.",
        "accepted",
    ),
    ("amo", "perfect_1s", "amāvī", "segmentation", "amā- | -v- | -ī", "accepted"),
    (
        "amo",
        "perfect_1s",
        "amāvī",
        "explanation",
        "v-Perfekt: An den Präsensstamm amā- tritt das Perfektzeichen -v-, dann die Endung -ī.",
        "accepted",
    ),
    ("amo", "supine", "amātum", "label", "supine", "accepted"),
    ("amo", "supine", "amātum", "segmentation", "amā- | -t- | -um", "accepted"),
    ("amo", "supine", "amātum", "explanation", "Supinum: Präsensstamm amā- + -t- + Endung -um.", "accepted"),
    ("moneo", "present_1s", "moneō", "segmentation", "mone- | -ō", "withheld"),
    ("moneo", "perfect_1s", "monuī", "segmentation", "mon- | -u- | -ī", "accepted"),
    (
        "moneo",
        "perfect_1s",
        "monuī",
        "explanation",
        "u-Perfekt: Verbalstamm mon- + Perfektzeichen -u- + Endung -ī.",
        "accepted",
    ),
    (
        "fero",
        "perfect_1s",
        "tulī",
        "explanation",
        "Suppletiv: tulī gehört zu einem anderen Stamm (tul-) als das Präsens fer-; keine lautliche Ableitung.",
        "accepted",
    ),
    ("fero", "supine", "lātum", "label", "supine", "accepted"),
    (
        "fero",
        "supine",
        "lātum",
        "explanation",
        "Suppletiv: lātum bildet einen eigenen Stamm lāt-, nicht von fer- abgeleitet.",
        "accepted",
    ),
    ("dico", "perfect_1s", "dīxī", "segmentation", "dīx- | -ī", "accepted"),
    (
        "dico",
        "perfect_1s",
        "dīxī",
        "explanation",
        "s-Perfekt: dīc- + -s- ergibt dīx- (x = c + s); hier nur grob als Stamm + Endung geteilt.",
        "accepted",
    ),
)

# key, deck label, theme, appearance, comparison, lexemes
VARIANTS = (
    ("A", "A muted light static", "muted", "light", "static", ("amo", "moneo", "fero", "dico")),
    ("B", "B muted dark static", "muted", "dark", "static", ("amo",)),
    ("C", "C monochrome light static", "monochrome", "light", "static", ("amo",)),
    ("D", "D monochrome dark static", "monochrome", "dark", "static", ("amo",)),
    ("E", "E muted light disclosure", "muted", "light", "disclosure", ("amo",)),
    ("F", "F monochrome dark disclosure", "monochrome", "dark", "disclosure", ("amo",)),
)
RETHEME = ("monochrome", "dark", "disclosure")


def profile_for(deck: str, theme: str, appearance: str, comparison: str) -> DeckProfile:
    base = DeckProfile.default(
        note_type="Latin",
        lexical_entry_field="Lemma",
        principal_parts_field="Forms",
        meaning_field="Meaning",
        source_identity=SourceIdentityConfig(strategy="source_id_field", field="ID"),
        generated_note_type=PP_NOTE_TYPE,
        target_deck=deck,
        tags=(TAG,),
    )
    return base.apply_overrides({"morphology": {"theme": theme, "appearance": appearance, "comparison": comparison}})


def reviewed_claims(source: Path, profile: DeckProfile, scope: str, lexemes: tuple[str, ...]) -> list:
    records = {r.source_identity: r for r in read_source_records(source, source_id_field="ID")}
    assessments = []
    for source_id, role, form, kind, value, status in CLAIMS:
        if source_id not in lexemes:
            continue
        parsed = parse_principal_parts(records[source_id], profile)
        assert isinstance(parsed, PrincipalPartParseSuccess), parsed
        identity = derive_latinitas_id(source_id, SINGLE_LEXEME_OBJECT_KEY, source_scope=scope)
        claim = Claim(
            f"{identity}:{role}",
            form,
            kind,
            value,
            "native-test-fixture/v1",
            (Evidence("native test fixture", f"{source_id} {role}", "v1", "Stipulated for native presentation only"),),
            profile,
            "v1",
            extraction=parsed.value.evidence,
        )
        decision = review_claim(claim, status=status, reviewer=REVIEWER, reason="Stipulated native-test review")
        assessments.append(assess_claim(claim, decision))
    return assessments


def build_principal_parts(out: Path) -> dict[str, str]:
    csvs: dict[str, str] = {}
    for key, label, theme, appearance, comparison, lexemes in VARIANTS:
        work = out / "work" / key
        work.mkdir(parents=True)
        source = work / "source.csv"
        rows = [r for r in SOURCE_ROWS if r[0] in lexemes]
        source.write_text(
            "ID,Lemma,Forms,Meaning\n"
            + "".join(f'{i},{lemma},"{forms}",{meaning}\n' for i, lemma, forms, meaning in rows),
            encoding="utf-8",
        )
        profile = profile_for(f"{ROOT_DECK}::{label}", theme, appearance, comparison)
        initial = prepare_principal_part_export(source, profile, approve_new_scope=True, approve_fresh_import=True)
        write_principal_part_csv(initial, work / "initial.csv")
        claims = reviewed_claims(source, profile, initial.source_scope, lexemes)
        reviewed = prepare_principal_part_export(source, profile, claim_assessments=claims)
        assert not reviewed.generation.skips or key == "A", reviewed.generation.skips
        target = out / f"{key}-principal-parts.csv"
        write_principal_part_csv(reviewed, target)
        csvs[target.name] = summary(reviewed)
        if key == "A":
            retheme = profile_for(f"{ROOT_DECK}::{label}", *RETHEME)
            rethemed = prepare_principal_part_export(source, retheme, claim_assessments=claims)
            assert [n.latinitas_id for n in rethemed.generation.notes] == [
                n.latinitas_id for n in reviewed.generation.notes
            ]
            assert [n.card_keys for n in rethemed.generation.notes] == [n.card_keys for n in reviewed.generation.notes]
            target = out / "A2-retheme-update.csv"
            write_principal_part_csv(rethemed, target)
            csvs[target.name] = summary(rethemed)
    return csvs


def summary(result) -> str:  # type: ignore[no-untyped-def]
    notes = result.generation.notes
    return f"{len(notes)} notes, {sum(len(n.card_keys) for n in notes)} cards"


def proposal(feature: str, value: str, evidence: str, alternatives: tuple[str, ...] = ()) -> ParsingProposal:
    return ParsingProposal(
        feature=feature,  # type: ignore[arg-type]
        value=value,
        analyzer="individual-review/v1",
        evidence=(Evidence("native test fixture", "sanitized sentence", "v1", evidence),),
        alternatives=alternatives,
    )


def parsing_input(deck: str, theme: str, appearance: str, scope: str) -> ParsingInput:
    profile = profile_for(deck, theme, appearance, "static")
    cases = (
        ParsingCase(
            source_identity="sentence-1",
            source_scope=scope,
            object_key="token-1",
            form="puellae",
            context="Puellae rosas portant.",
            applicable_features=("lemma", "case", "number", "gender"),
            required_features=("lemma", "case", "number"),
            proposals=(
                proposal("lemma", "puella", "Nominal form of puella"),
                proposal(
                    "case", "nominativus", "Subject of plural portant", ("genetivus singularis", "dativus singularis")
                ),
                proposal("number", "pluralis", "Agrees with plural portant", ("singularis",)),
                proposal("gender", "femininum", "Optional; deliberately left unreviewed"),
            ),
        ),
        ParsingCase(
            source_identity="sentence-2",
            source_scope=scope,
            object_key="token-3",
            form="amāvit",
            context="Marcus patriam amāvit.",
            applicable_features=("lemma", "person", "number", "tense", "mood", "voice"),
            required_features=("lemma", "person", "number", "tense"),
            proposals=(
                proposal("lemma", "amō", "v-perfect of amō"),
                proposal("person", "3. Person", "Subject Marcus"),
                proposal("number", "singularis", "Subject Marcus"),
                proposal("tense", "perfectum", "Perfect stem amāv-"),
                proposal("mood", "indicativus", "Main clause statement"),
                proposal("voice", "activum", "Active ending -it"),
            ),
        ),
        ParsingCase(
            source_identity="sentence-3",
            source_scope=scope,
            object_key="token-2",
            form="puellae",
            context="Rosam puellae dat.",
            applicable_features=("lemma", "case", "number"),
            required_features=("lemma", "case", "number"),
            proposals=(
                proposal("lemma", "puella", "Nominal form of puella"),
                proposal("case", "dativus", "Recipient of dat", ("genetivus",)),
                proposal("number", "singularis", "No plural verb agreement", ("pluralis",)),
            ),
        ),
    )
    data = ParsingInput(profile=profile, cases=cases)
    withhold = {("sentence-1", "gender"): None, ("sentence-3", "case"): "withheld"}
    reviewed = []
    for case in data.cases:
        proposals = []
        for p in case.proposals:
            status = withhold.get((case.source_identity, p.feature), "accepted")
            if status is None:
                proposals.append(p)
                continue
            decision = review_claim(
                p.to_claim(case, profile), status=status, reviewer=REVIEWER, reason="Stipulated native-test review"
            )
            proposals.append(p.model_copy(update={"decision": decision}))
        reviewed.append(case.model_copy(update={"proposals": tuple(proposals)}))
    return data.model_copy(update={"cases": tuple(reviewed)})


def build_parsing(out: Path) -> dict[str, str]:
    csvs = {}
    for key, label, theme, appearance in (
        ("G", "G parsing muted light", "muted", "light"),
        ("H", "H parsing monochrome dark", "monochrome", "dark"),
    ):
        data = parsing_input(f"{ROOT_DECK}::{label}", theme, appearance, f"latinitas-native-test/{key}")
        cases = out / "work" / f"{key}-parsing-cases.json"
        cases.write_text(data.model_dump_json(indent=2), encoding="utf-8")
        target = out / f"{key}-form-parsing.csv"
        subprocess.run(
            ["latinitas-cards", "form-parsing", "export", str(cases), str(target), "--approve-fresh-import"],
            check=True,
        )
        csvs[target.name] = "2 notes, 2 cards expected (sentence-3 withheld)"
    return csvs


def build_carrier(out: Path) -> None:
    path = out / "work" / "carrier.anki2"
    col = Collection(str(path))
    try:
        pp = col.models.new(PP_NOTE_TYPE)
        for field in REFERENCE_NOTE_TYPE_FIELDS:
            col.models.add_field(pp, col.models.new_field(field))
        for ref in REFERENCE_CARD_TEMPLATES:
            template = col.models.new_template(ref.slot.template_name)
            template.update(qfmt=ref.front, afmt=ref.back)
            col.models.add_template(pp, template)
        pp["css"] = REFERENCE_CARD_CSS
        col.models.add(pp)
        parsing = col.models.new(PARSING_NOTE_TYPE)
        for field in PARSING_FIELDS:
            col.models.add_field(parsing, col.models.new_field(field))
        template = col.models.new_template(PARSING_SLOT.template_name)
        template.update(qfmt=PARSING_FRONT, afmt=PARSING_BACK)
        col.models.add_template(parsing, template)
        parsing["css"] = REFERENCE_CARD_CSS
        col.models.add(parsing)
        deck = col.decks.id(f"{ROOT_DECK}::0 setup placeholder")
        for model, enabled, prompt in (
            (col.models.by_name(PP_NOTE_TYPE), "CompletionPresentEnabled", "CompletionPresentPrompt"),
            (col.models.by_name(PARSING_NOTE_TYPE), "ParsingEnabled", "ParsingPrompt"),
        ):
            note = col.new_note(model)
            note["LatinitasID"] = "SETUP-PLACEHOLDER-" + model["name"]
            note[enabled] = "1"
            note[prompt] = "Setup placeholder - delete this note after importing the CSVs"
            col.add_note(note, deck)
        col.export_anki_package(
            out_path=str(out / "0-note-types-carrier.apkg"),
            options=ExportAnkiPackageOptions(
                with_scheduling=False, with_deck_configs=False, with_media=False, legacy=False
            ),
            limit=None,
        )
    finally:
        col.close()


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    (out / "work").mkdir()
    build_carrier(out)
    summaries = {**build_principal_parts(out), **build_parsing(out)}
    manifest = {
        "summaries": summaries,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()},
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
