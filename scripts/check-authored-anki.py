"""Native Anki gate in a disposable collection, never a user's live collection.

Run: uv run --with anki==26.9.3 python scripts/check-authored-anki.py
Anki is deliberately optional and absent from the normal pytest environment.
"""

import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from anki.collection import Collection
from anki.import_export_pb2 import CsvMetadata, ImportCsvRequest

from latinitas_cards.authored_notes import AUTHORED_NOTE_TYPES


def main() -> None:
    print(f"Native Anki {version('anki')} on {sys.platform}")
    with TemporaryDirectory(prefix="authored-anki-") as temporary:
        root = Path(temporary)
        source = root / "notes.jsonl"
        output = root / "csv"
        output.mkdir()
        rows = [
            {"kind": "vocab", "lemma": "amō", "meaning": "lieben"},
            {
                "kind": "form",
                "text_form": "amāvērunt",
                "base_form": "amō",
                "analysis": "Perfekt, 3. Pl.",
                "translation": "sie liebten",
                "context": "Multī amāvērunt.",
            },
            {"kind": "qa", "question": "Was ist ein Ablativ?", "answer": "Ein Kasus."},
        ]
        for row in rows:
            row.update(
                schema_version=1,
                key=f"lesson:{row['kind']}:1",
                status="include",
                language_tag="de",
                provenance={"document": "Synthetic ä", "section": "S", "reference": "R"},
                tags=["lesson::1", "ä"],
            )

        def export() -> None:
            source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "latinitas_cards",
                    "authored",
                    "export",
                    str(source),
                    "--namespace",
                    "native-synthetic",
                    "--output-dir",
                    str(output),
                    "--deck",
                    "Latin::Authored",
                ],
                check=True,
            )

        export()
        original_bytes = {kind: (output / f"{kind}.csv").read_bytes() for kind in AUTHORED_NOTE_TYPES}
        export()
        assert original_bytes == {kind: (output / f"{kind}.csv").read_bytes() for kind in AUTHORED_NOTE_TYPES}
        collection = Collection(str(root / "collection.anki2"))
        try:
            deck_id = collection.decks.id("Latin::Authored")
            for schema in AUTHORED_NOTE_TYPES.values():
                model = collection.models.new(schema.name)
                for field in schema.field_names:
                    collection.models.add_field(model, collection.models.new_field(field))
                template = collection.models.new_template("Recognition")
                template["qfmt"] = schema.front_template
                template["afmt"] = schema.back_template
                collection.models.add_template(model, template)
                collection.models.add(model)

            def import_kind(kind: str, *, updating: bool) -> int:
                path = output / f"{kind}.csv"
                metadata = collection.get_csv_metadata(str(path), None)
                assert metadata.is_html and metadata.tags_column == len(metadata.column_labels)
                assert "Personal Notes" not in metadata.column_labels
                # Native metadata leaves the personal field unmapped, even on a fresh import.
                assert metadata.global_notetype.field_columns[-1] == 0
                metadata.dupe_resolution = CsvMetadata.UPDATE
                metadata.match_scope = CsvMetadata.NOTETYPE
                log = collection.import_csv(ImportCsvRequest(path=str(path), metadata=metadata)).log
                # CSV updates matched by first field use first_field_match, not updated (GUID matches).
                assert log.found_notes == 1 and len(log.first_field_match if updating else log.new) == 1, log
                note_ids = collection.find_notes(f'note:"{AUTHORED_NOTE_TYPES[kind].name}"')
                assert len(note_ids) == 1
                note = collection.get_note(note_ids[0])
                cards = note.cards()
                assert len(cards) == 1 and cards[0].ord == 0 and cards[0].did == deck_id
                assert "lesson::1" in note.tags and "ä" in note.tags
                return note_ids[0]

            identities = {}
            for kind in AUTHORED_NOTE_TYPES:
                nid = import_kind(kind, updating=False)
                note = collection.get_note(nid)
                identities[kind] = (nid, note["LatinitasID"], note.cards()[0].id)
                note["Personal Notes"] = f"My personal {kind} ä"
                collection.update_note(note)
                print(f"{kind}: first import: 1 note, 1 Recognition card; Personal Notes seeded")
            for row, field in zip(rows, ("meaning", "translation", "answer"), strict=True):
                row[field] = f"Edited {row['kind']} ä, <safe>"
            export()
            for kind, field in (("vocab", "Meaning"), ("form", "Translation"), ("qa", "Answer")):
                nid = import_kind(kind, updating=True)
                note = collection.get_note(nid)
                assert (nid, note["LatinitasID"], note.cards()[0].id) == identities[kind]
                assert note["Personal Notes"] == f"My personal {kind} ä"
                assert note[field] == f"Edited {kind} ä, &lt;safe&gt;"
                assert "{{" not in note.cards()[0].question()
                assert note[field] in note.cards()[0].answer()
                assert note["Personal Notes"] in note.cards()[0].answer()
                print(f"{kind}: re-import: same note/LatinitasID/card; edited {field}; Personal Notes preserved")
            assert collection.note_count() == 3 and collection.card_count() == 3
            print("PASS: all three kinds, byte stability, native first import and edited re-import")
        finally:
            collection.close()


if __name__ == "__main__":
    main()
