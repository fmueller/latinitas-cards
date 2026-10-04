"""Native Anki gate in a disposable collection, never a user's live collection.

Run: uv run --with anki==26.9.3 python scripts/check-authored-anki.py
Anki is deliberately optional and absent from the normal pytest environment.
"""

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, help="Save native card answers for visual inspection.")
    render_dir = parser.parse_args().render_dir
    if render_dir:
        render_dir.mkdir(parents=True, exist_ok=True)
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
        content = "left\rright\r\n<safe>\n& ä𐌀\r\n\rend"
        expected = "left<br>right<br>&lt;safe&gt;<br>&amp; ä𐌀<br><br>end"
        for row in rows:
            for _, attribute in AUTHORED_NOTE_TYPES[row["kind"]].content_fields:
                row[attribute] = content
            row.update(
                schema_version=1,
                key=f"lesson:{row['kind']}:1",
                status="include",
                language_tag="de",
                provenance={"document": "Synthetic\r\nä", "section": "S\nT", "reference": "part\rtwo"},
                tags=["lesson::1", "ä"],
            )

        def write_source() -> None:
            source.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")

        def export(*filters: str, destination: Path = output) -> None:
            before = source.read_bytes()
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
                    str(destination),
                    "--deck",
                    "Latin::Authored",
                    *filters,
                ],
                check=True,
            )
            assert source.read_bytes() == before

        write_source()
        for command in ("validate", "preview"):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "latinitas_cards",
                    "authored",
                    command,
                    str(source),
                    "--namespace",
                    "native-synthetic",
                ],
                check=True,
            )
        export()
        original_bytes = {kind: (output / f"{kind}.csv").read_bytes() for kind in AUTHORED_NOTE_TYPES}
        export()
        assert original_bytes == {kind: (output / f"{kind}.csv").read_bytes() for kind in AUTHORED_NOTE_TYPES}
        for index, (reference, matches) in enumerate((("part\rtwo", True), ("part\ntwo", False), ("parttwo", False))):
            filtered = root / f"filter-{index}"
            filtered.mkdir()
            export("--reference", reference, destination=filtered)
            assert {path.name: path.read_bytes() for path in filtered.iterdir()} == (
                {f"{kind}.csv": payload for kind, payload in original_bytes.items()} if matches else {}
            )
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

            def check_reopened(phase: str, prefix: str = "") -> None:
                nonlocal collection
                collection.close()
                collection = Collection(str(root / "collection.anki2"))
                assert collection.note_count() == 3 and collection.card_count() == 3
                for kind, (nid, latinitas_id, card_id) in identities.items():
                    note = collection.get_note(nid)
                    assert (note.id, note["LatinitasID"], note.cards()[0].id) == (nid, latinitas_id, card_id)
                    assert note["Personal Notes"] == f"My personal {kind} ä"
                    for field, _ in AUTHORED_NOTE_TYPES[kind].content_fields:
                        assert note[field] == prefix + expected, (phase, kind, field, note[field])
                    assert note["Document"] == "Synthetic<br>ä"
                    assert note["Section"] == "S<br>T"
                    assert note["Reference"] == "part<br>two"
                    answer = note.cards()[0].answer()
                    assert prefix + expected in answer and "part<br>two" in answer
                    assert note["Personal Notes"] in answer
                    assert "{{" not in note.cards()[0].question()
                    if render_dir:
                        (render_dir / f"{phase}-{kind}.html").write_text(
                            '<!doctype html><meta charset="utf-8"><title>'
                            + phase
                            + " "
                            + kind
                            + "</title>"
                            + "<style>body{font:24px sans-serif;margin:40px;max-width:700px}"
                            + ".reference{color:#555;margin-top:24px}.personal-notes{margin-top:24px}</style>"
                            + answer,
                            encoding="utf-8",
                        )
                    print(f"{kind}: {phase} reopen: exact newline HTML; stable note/ID/card; Personal Notes intact")

            check_reopened("first")
            for row in rows:
                for _, attribute in AUTHORED_NOTE_TYPES[row["kind"]].content_fields:
                    row[attribute] = "Edited " + content
            write_source()
            export()
            for kind, field in (("vocab", "Meaning"), ("form", "Translation"), ("qa", "Answer")):
                nid = import_kind(kind, updating=True)
                note = collection.get_note(nid)
                assert (nid, note["LatinitasID"], note.cards()[0].id) == identities[kind]
                assert note["Personal Notes"] == f"My personal {kind} ä"
                assert note[field] == "Edited " + expected
                assert "{{" not in note.cards()[0].question()
                assert note[field] in note.cards()[0].answer()
                assert note["Personal Notes"] in note.cards()[0].answer()
                print(f"{kind}: re-import: same note/LatinitasID/card; edited {field}; Personal Notes preserved")
            check_reopened("edited", "Edited ")
            assert collection.note_count() == 3 and collection.card_count() == 3
            print("PASS: all kinds, exact filters, source/byte stability, native first/edited import and reopen")
        finally:
            collection.close()


if __name__ == "__main__":
    main()
