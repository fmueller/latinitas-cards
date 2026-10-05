"""Disposable native backend parsing smoke/no-op gate; never a user collection.

uv run --with anki==26.9.3 python scripts/check-form-parsing-anki.py \
  --input reviewed-cases.json --output-dir /new/disposable/evidence
"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from anki.collection import Collection
from anki.import_export_pb2 import CsvMetadata, ImportCsvRequest

from latinitas_cards.form_parsing import (
    PARSING_BACK,
    PARSING_FIELDS,
    PARSING_FRONT,
    PARSING_NOTE_TYPE,
    PARSING_SLOT,
    ParsingInput,
    generate_parsing,
    parsing_csv,
)
from latinitas_cards.reference_templates import REFERENCE_CARD_CSS


def capture(path: Path) -> dict[str, list[tuple]]:
    with sqlite3.connect(path) as db:
        return {
            table: db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
            for table in ("notes", "cards", "revlog", "notetypes", "fields", "templates", "decks", "deck_config")
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = ParsingInput.model_validate_json(args.input.read_text(encoding="utf-8"))
    expected = [r for r in generate_parsing(data) if r["eligible"]]
    assert expected, "At least one explicitly reviewed eligible case is required"
    args.output_dir.mkdir(parents=True, exist_ok=False)
    csv = args.output_dir / "parsing.csv"
    csv.write_bytes(parsing_csv(data))
    path = args.output_dir / "collection.anki2"
    collection = Collection(str(path))
    try:
        model = collection.models.new(PARSING_NOTE_TYPE)
        for field in PARSING_FIELDS:
            collection.models.add_field(model, collection.models.new_field(field))
        template = collection.models.new_template(PARSING_SLOT.template_name)
        template.update(qfmt=PARSING_FRONT, afmt=PARSING_BACK)
        collection.models.add_template(model, template)
        model["css"] = REFERENCE_CARD_CSS
        collection.models.add(model)
        collection.decks.id(data.profile.target_deck)
    finally:
        collection.close()
    renders = []
    for iteration in range(2):
        collection = Collection(str(path))
        try:
            metadata = collection.get_csv_metadata(str(csv), None)
            metadata.dupe_resolution = CsvMetadata.UPDATE
            metadata.match_scope = CsvMetadata.NOTETYPE
            assert metadata.is_html and metadata.tags_column == 7
            assert metadata.global_notetype.field_columns[-1] == 0
            collection.import_csv(ImportCsvRequest(path=str(csv), metadata=metadata))
            ids = collection.find_notes("")
            assert len(ids) == len(expected)
            expected_by_id = {r["latinitas_id"]: r for r in expected}
            for nid in ids:
                note = collection.get_note(nid)
                result = expected_by_id[note["LatinitasID"]]
                assert note["ParsingPrompt"] == result["prompt"] and note["ParsingAnswer"] == result["answer"]
                assert note["Personal Notes"] == ""
                cards = note.cards()
                assert len(cards) == 1 and cards[0].ord == 0
                assert result["prompt"] in cards[0].question()
                assert result["answer"] in cards[0].answer()
                if iteration == 0:
                    renders.append({"question": cards[0].question(), "answer": cards[0].answer()})
        finally:
            collection.close()
        observed = capture(path)
        if iteration == 0:
            before = observed
        else:
            assert observed == before, "No-op import changed native tables"
    report = {
        "scope": "Anki 26.9.3 native backend, fresh parsing and no-op only; devices deferred",
        "notes": len(expected),
        "cards": len(expected),
        "renders": renders,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(csv.read_bytes()).hexdigest(),
        "no_op_tables_equal": True,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PASS: {len(expected)} contextual notes/cards, exact native render content, identical no-op tables")


if __name__ == "__main__":
    main()
