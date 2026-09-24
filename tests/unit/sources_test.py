import hashlib
import json
import sqlite3
import zipfile
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

import latinitas_cards.sources as sources_module
from latinitas_cards.sources import (
    CanonicalSourceError,
    inspect_source,
    read_source_records,
)


def _write_legacy_database(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE col (id INTEGER PRIMARY KEY, models TEXT)")
    con.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, guid TEXT, mid INTEGER, flds TEXT)")
    models = {
        "20": {"name": "Zeta", "flds": [{"name": "Front"}, {"name": "Back"}]},
        "10": {"name": "Alpha", "flds": [{"name": "Lemma"}, {"name": "Meaning"}]},
    }
    con.execute("INSERT INTO col (id, models) VALUES (1, ?)", (json.dumps(models),))
    con.executemany(
        "INSERT INTO notes (id, guid, mid, flds) VALUES (?, ?, ?, ?)",
        (
            (200, "zeta-guid", 20, "zeta front\x1fzeta back"),
            (100, "alpha-guid", 10, "amo\x1flove"),
        ),
    )
    con.commit()
    con.close()


def _write_modern_database(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, guid TEXT, mid INTEGER, flds TEXT)")
    con.execute("CREATE TABLE notetypes (id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("CREATE TABLE fields (ntid INTEGER, ord INTEGER, name TEXT)")
    con.executemany(
        "INSERT INTO notetypes (id, name) VALUES (?, ?)",
        ((300, "Vocabulary"), (100, "Grammar")),
    )
    con.executemany(
        "INSERT INTO fields (ntid, ord, name) VALUES (?, ?, ?)",
        (
            (300, 0, "Entry"),
            (300, 1, "German gloss"),
            (100, 0, "Form"),
            (100, 1, "Morphology / Notes"),
        ),
    )
    con.executemany(
        "INSERT INTO notes (id, guid, mid, flds) VALUES (?, ?, ?, ?)",
        (
            (2, "vocab-b", 300, "video\x1fsehen"),
            (1, "grammar-a", 100, "amat\x1f3sg present"),
            (3, "vocab-a", 300, "amo\x1flieben"),
        ),
    )
    con.commit()
    con.close()


def _write_database_without_guid(path: Path) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE notes (id INTEGER PRIMARY KEY, mid INTEGER, flds TEXT)")
    con.execute("CREATE TABLE notetypes (id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("CREATE TABLE fields (ntid INTEGER, ord INTEGER, name TEXT)")
    con.execute("INSERT INTO notetypes (id, name) VALUES (1, 'Vocabulary')")
    con.execute("INSERT INTO fields (ntid, ord, name) VALUES (1, 0, 'Entry')")
    con.execute("INSERT INTO notes (id, mid, flds) VALUES (1, 1, 'private entry')")
    con.commit()
    con.close()


def _package(package_path: Path, database_path: Path, member_name: str) -> None:
    with zipfile.ZipFile(package_path, "w") as archive:
        archive.write(database_path, member_name)


def test_csv_records_decode_utf16_keep_named_fields_and_configured_identity(tmp_path: Path) -> None:
    source = tmp_path / "latin.csv"
    original_bytes = (
        "Stable ID;Lēmma;German gloss\nentry-2;dīcō;".encode("utf-16") + "\nentry-1;amō;lieben\n".encode("utf-16")[2:]
    )
    source.write_bytes(original_bytes)
    original_digest = hashlib.sha256(original_bytes).digest()

    records = read_source_records(source, source_id_field="Stable ID")

    assert [record.source_identity for record in records] == ["entry-2", "entry-1"]
    assert records[0].source_kind == "csv"
    assert records[0].note_type is None
    assert dict(records[0].fields) == {"Stable ID": "entry-2", "Lēmma": "dīcō", "German gloss": ""}
    assert records[0].provenance.source_path == source
    assert records[0].provenance.row_number == 2
    assert records[0].provenance.encoding == "utf-16"
    assert records[0].provenance.content_hash
    reordered = replace(records[0], fields=dict(reversed(tuple(records[0].fields.items()))))
    assert reordered == records[0]
    assert hash(reordered) == hash(records[0])
    assert source.read_bytes() == original_bytes
    assert hashlib.sha256(source.read_bytes()).digest() == original_digest

    with pytest.raises(TypeError):
        records[0].fields["Lēmma"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        records[0].source_identity = "changed"  # type: ignore[misc]


def test_csv_without_stable_id_exposes_manifest_reconciliation_data(tmp_path: Path) -> None:
    source = tmp_path / "arbitrary-fields.csv"
    source.write_text("Field Ω,Notes / context\namo,first row\n", encoding="utf-8-sig")

    records = read_source_records(source)

    assert records[0].source_identity is None
    assert records[0].fields["Field Ω"] == "amo"
    expected_hash = hashlib.sha256(
        json.dumps(
            [["Field Ω", "amo"], ["Notes / context", "first row"]],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert records[0].provenance.content_hash == expected_hash
    assert records[0].provenance.row_number == 2
    assert records[0].provenance.encoding == "utf-8-sig"


def test_csv_duplicate_stable_ids_are_rejected(tmp_path: Path) -> None:
    source = tmp_path / "duplicate-ids.csv"
    source.write_text("Stable ID,Lemma\nentry-1,amo\nentry-1,dico\n", encoding="utf-8")

    with pytest.raises(CanonicalSourceError, match="duplicate"):
        read_source_records(source, source_id_field="Stable ID")


def test_legacy_colpkg_preserves_guids_and_sorts_multiple_note_types(tmp_path: Path) -> None:
    database = tmp_path / "collection.anki2"
    package = tmp_path / "legacy.colpkg"
    _write_legacy_database(database)
    _package(package, database, "collection.anki2")
    original_digest = hashlib.sha256(package.read_bytes()).digest()

    records = read_source_records(package)

    assert [(record.note_type, record.source_identity) for record in records] == [
        ("Alpha", "alpha-guid"),
        ("Zeta", "zeta-guid"),
    ]
    assert records[0].note_guid == "alpha-guid"
    assert records[0].fields == {"Lemma": "amo", "Meaning": "love"}
    assert records[0].provenance.note_id == 100
    assert hashlib.sha256(package.read_bytes()).digest() == original_digest


def test_modern_zstd_apkg_inspection_is_deterministic_and_retains_field_names(tmp_path: Path) -> None:
    import zstandard

    database = tmp_path / "modern.sqlite"
    package = tmp_path / "modern.apkg"
    _write_modern_database(database)
    compressed = zstandard.ZstdCompressor().compress(database.read_bytes())
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("collection.anki21b", compressed)

    inspection = inspect_source(package)
    records_again = read_source_records(package)

    assert inspection.note_types == ("Grammar", "Vocabulary")
    assert inspection.fields_by_note_type == {
        "Grammar": ("Form", "Morphology / Notes"),
        "Vocabulary": ("Entry", "German gloss"),
    }
    assert [(record.note_type, record.source_identity) for record in inspection.records] == [
        ("Grammar", "grammar-a"),
        ("Vocabulary", "vocab-a"),
        ("Vocabulary", "vocab-b"),
    ]
    assert inspection.records == records_again
    assert inspection.records[1].fields["German gloss"] == "lieben"


def test_anki_duplicate_native_guids_are_rejected(tmp_path: Path) -> None:
    database = tmp_path / "duplicate-guids.sqlite"
    package = tmp_path / "duplicate-guids.apkg"
    _write_modern_database(database)
    con = sqlite3.connect(database)
    con.execute("UPDATE notes SET guid = 'same-guid'")
    con.commit()
    con.close()
    _package(package, database, "collection.anki2")

    with pytest.raises(CanonicalSourceError, match="duplicate"):
        read_source_records(package)


def test_malformed_package_error_names_source_and_assumption_without_contents(tmp_path: Path) -> None:
    source = tmp_path / "secret-deck.apkg"
    source.write_bytes(b"not an anki package; private deck contents")

    with pytest.raises(CanonicalSourceError) as raised:
        read_source_records(source)

    message = str(raised.value)
    assert "secret-deck.apkg" in message
    assert "ZIP" in message
    assert "private deck contents" not in message


def test_missing_csv_identity_column_reports_structure_without_row_values(tmp_path: Path) -> None:
    source = tmp_path / "missing-id.csv"
    source.write_text("Lemma,Meaning\nprivate-lemma,secret meaning\n", encoding="utf-8")

    with pytest.raises(CanonicalSourceError) as raised:
        read_source_records(source, source_id_field="Stable ID")

    message = str(raised.value)
    assert "missing-id.csv" in message
    assert "Stable ID" in message
    assert "private-lemma" not in message
    assert "secret meaning" not in message


def test_malformed_csv_quotes_are_rejected_without_row_values(tmp_path: Path) -> None:
    source = tmp_path / "malformed.csv"
    source.write_text('ID,Text\n1,"unterminated\n', encoding="utf-8")

    with pytest.raises(CanonicalSourceError) as raised:
        read_source_records(source)

    message = str(raised.value)
    assert "malformed.csv" in message
    assert "malformed" in message.lower()
    assert "unterminated" not in message


def test_anki_without_guid_column_is_rejected_without_note_contents(tmp_path: Path) -> None:
    database = tmp_path / "missing-guid.sqlite"
    package = tmp_path / "missing-guid.apkg"
    _write_database_without_guid(database)
    _package(package, database, "collection.anki2")

    with pytest.raises(CanonicalSourceError) as raised:
        read_source_records(package)

    message = str(raised.value)
    assert "missing-guid.apkg" in message
    assert "guid" in message.lower()
    assert "private entry" not in message


def test_anki_empty_guid_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "empty-guid.sqlite"
    package = tmp_path / "empty-guid.apkg"
    _write_modern_database(database)
    con = sqlite3.connect(database)
    con.execute("UPDATE notes SET guid = '' WHERE id = 1")
    con.commit()
    con.close()
    _package(package, database, "collection.anki2")

    with pytest.raises(CanonicalSourceError, match="GUID"):
        read_source_records(package)


def test_anki_partial_metadata_is_rejected_without_note_contents(tmp_path: Path) -> None:
    database = tmp_path / "partial-metadata.sqlite"
    package = tmp_path / "partial-metadata.apkg"
    _write_modern_database(database)
    con = sqlite3.connect(database)
    con.execute("DELETE FROM notetypes WHERE id = 300")
    con.commit()
    con.close()
    _package(package, database, "collection.anki2")

    with pytest.raises(CanonicalSourceError) as raised:
        read_source_records(package)

    message = str(raised.value)
    assert "partial-metadata.apkg" in message
    assert "metadata" in message.lower()
    assert "video" not in message


def test_anki_extra_values_without_named_fields_are_rejected(tmp_path: Path) -> None:
    database = tmp_path / "extra-values.sqlite"
    package = tmp_path / "extra-values.apkg"
    _write_modern_database(database)
    con = sqlite3.connect(database)
    con.execute("DELETE FROM fields WHERE ntid = 300 AND ord = 1")
    con.commit()
    con.close()
    _package(package, database, "collection.anki2")

    with pytest.raises(CanonicalSourceError) as raised:
        read_source_records(package)

    message = str(raised.value)
    assert "extra-values.apkg" in message
    assert "field" in message.lower()
    assert "lieben" not in message


def test_oversized_collection_member_is_rejected_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "collection.anki2"
    package = tmp_path / "oversized.apkg"
    _write_legacy_database(database)
    _package(package, database, "collection.anki2")
    monkeypatch.setattr(sources_module, "_MAX_COLLECTION_MEMBER_BYTES", 1)

    with pytest.raises(CanonicalSourceError, match="too large"):
        read_source_records(package)
