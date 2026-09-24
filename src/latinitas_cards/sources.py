"""Immutable canonical records for CSV and Anki deck sources.

The legacy CLI contains readers for its existing commands.  This module is the
adapter boundary for profile setup and generation: it reads a source without
writing it and turns each row or note into the same typed record shape.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

import zstandard

SourceKind = Literal["csv", "apkg", "colpkg"]

_CSV_FALLBACK_ENCODINGS = ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1")
_ANKI_DATABASE_SUFFIXES = (".anki21b", ".anki21", ".anki2")
_MAX_COLLECTION_MEMBER_BYTES = 512 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    """Safe source location and reconciliation metadata for one record."""

    source_path: Path
    location: str
    row_number: int | None = None
    note_id: int | None = None
    encoding: str | None = None
    delimiter: str | None = None
    content_hash: str = ""


@dataclass(frozen=True, slots=True)
class CanonicalSourceRecord:
    """One immutable source row or Anki note with named fields."""

    source_kind: SourceKind
    note_type: str | None
    fields: Mapping[str, str]
    provenance: SourceProvenance
    source_identity: str | None
    note_guid: str | None = None

    def __post_init__(self) -> None:
        copied_fields = MappingProxyType(dict(self.fields))
        object.__setattr__(self, "fields", cast(Mapping[str, str], copied_fields))

    @property
    def available_note_type(self) -> str | None:
        """Return the note type when the source exposes one."""

        return self.note_type

    @property
    def source_id(self) -> str | None:
        """Return the configured or native immutable source identity."""

        return self.source_identity

    def __hash__(self) -> int:
        return hash(
            (
                self.source_kind,
                self.note_type,
                tuple(sorted(self.fields.items())),
                self.provenance,
                self.source_identity,
                self.note_guid,
            )
        )


@dataclass(frozen=True, slots=True)
class SourceInspection:
    """Deterministic schema summary and records for an inspected source."""

    records: tuple[CanonicalSourceRecord, ...]
    note_types: tuple[str, ...]
    fields_by_note_type: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        frozen_fields = {name: tuple(fields) for name, fields in self.fields_by_note_type.items()}
        object.__setattr__(self, "fields_by_note_type", MappingProxyType(frozen_fields))


class CanonicalSourceError(ValueError):
    """Raised when a source violates an adapter's structural assumptions."""

    def __init__(self, source_path: Path, assumption: str) -> None:
        self.source_path = source_path
        self.assumption = assumption
        super().__init__(f"Cannot inspect source '{source_path.name}': {assumption}")


def read_source_records(
    source_path: str | Path,
    *,
    source_id_field: str | None = None,
    encoding: str | None = None,
    note_type: str | None = None,
) -> tuple[CanonicalSourceRecord, ...]:
    """Read CSV, APKG, or COLPKG input into deterministic canonical records."""

    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_records(path, source_id_field=source_id_field, encoding=encoding, note_type=note_type)
    if suffix == ".apkg":
        return read_apkg_records(path, note_type=note_type)
    if suffix == ".colpkg":
        return read_colpkg_records(path, note_type=note_type)
    raise CanonicalSourceError(path, "expected a .csv, .apkg, or .colpkg source")


def read_csv_records(
    source_path: str | Path,
    *,
    source_id_field: str | None = None,
    encoding: str | None = None,
    note_type: str | None = None,
) -> tuple[CanonicalSourceRecord, ...]:
    """Read a headered CSV while preserving arbitrary named fields and row order."""

    path = Path(source_path)
    raw = _read_source_bytes(path)
    text, selected_encoding = _decode_csv(raw, path, encoding)
    rows, headers, delimiter = _parse_csv(text, path)
    if source_id_field is not None and source_id_field not in headers:
        raise CanonicalSourceError(path, f"configured source-ID column '{source_id_field}' is missing")

    records: list[CanonicalSourceRecord] = []
    for row_number, row in rows:
        fields = dict(zip(headers, row, strict=True))
        source_identity = None
        if source_id_field is not None and fields[source_id_field] != "":
            source_identity = fields[source_id_field]
        records.append(
            CanonicalSourceRecord(
                source_kind="csv",
                note_type=note_type,
                fields=fields,
                provenance=SourceProvenance(
                    source_path=path,
                    location=f"row {row_number}",
                    row_number=row_number,
                    encoding=selected_encoding,
                    delimiter=delimiter,
                    content_hash=_hash_fields(fields),
                ),
                source_identity=source_identity,
            )
        )
    return tuple(records)


def read_apkg_records(
    source_path: str | Path,
    *,
    note_type: str | None = None,
) -> tuple[CanonicalSourceRecord, ...]:
    """Read an Anki package while retaining native note GUIDs."""

    return _read_anki_records(Path(source_path), "apkg", note_type)


def read_colpkg_records(
    source_path: str | Path,
    *,
    note_type: str | None = None,
) -> tuple[CanonicalSourceRecord, ...]:
    """Read an Anki collection package while retaining native note GUIDs."""

    return _read_anki_records(Path(source_path), "colpkg", note_type)


def inspect_source(
    source_path: str | Path,
    *,
    source_id_field: str | None = None,
    encoding: str | None = None,
) -> SourceInspection:
    """Return records and a sorted note-type/field schema summary."""

    records = read_source_records(source_path, source_id_field=source_id_field, encoding=encoding)
    note_types = tuple(sorted({record.note_type for record in records if record.note_type is not None}))
    fields_by_note_type: dict[str, set[str]] = {name: set() for name in note_types}
    for record in records:
        if record.note_type is not None:
            fields_by_note_type[record.note_type].update(record.fields)
    sorted_fields = {name: tuple(sorted(fields)) for name, fields in sorted(fields_by_note_type.items())}
    return SourceInspection(records=records, note_types=note_types, fields_by_note_type=sorted_fields)


def _read_source_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise CanonicalSourceError(path, "source bytes could not be read") from error


def _decode_csv(raw: bytes, path: Path, requested_encoding: str | None) -> tuple[str, str]:
    encodings = (requested_encoding,) if requested_encoding is not None else _candidate_encodings(raw)
    for selected_encoding in encodings:
        try:
            return raw.decode(selected_encoding), selected_encoding
        except (LookupError, UnicodeDecodeError):
            continue
    raise CanonicalSourceError(path, "CSV bytes do not match a supported text encoding")


def _candidate_encodings(raw: bytes) -> tuple[str, ...]:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return ("utf-16",) + tuple(encoding for encoding in _CSV_FALLBACK_ENCODINGS if encoding != "utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return ("utf-8-sig",) + tuple(encoding for encoding in _CSV_FALLBACK_ENCODINGS if encoding != "utf-8-sig")
    if len(raw) >= 4:
        sample = raw[:4096]
        odd_nulls = sum(byte == 0 for byte in sample[1::2])
        even_nulls = sum(byte == 0 for byte in sample[::2])
        half = max(1, len(sample) // 4)
        if odd_nulls >= half:
            return ("utf-16-le",) + tuple(encoding for encoding in _CSV_FALLBACK_ENCODINGS if encoding != "utf-16-le")
        if even_nulls >= half:
            return ("utf-16-be",) + tuple(encoding for encoding in _CSV_FALLBACK_ENCODINGS if encoding != "utf-16-be")
    return _CSV_FALLBACK_ENCODINGS


def _parse_csv(text: str, path: Path) -> tuple[list[tuple[int, list[str]]], list[str], str]:
    sample = text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","

    try:
        parsed_rows = list(csv.reader(io.StringIO(text), delimiter=delimiter, strict=True))
    except csv.Error as error:
        raise CanonicalSourceError(path, "CSV structure is malformed") from error
    if not parsed_rows or not any(parsed_rows[0]):
        raise CanonicalSourceError(path, "CSV must contain a non-empty header row")

    headers = parsed_rows[0]
    if any(not header for header in headers):
        raise CanonicalSourceError(path, "CSV header contains an empty field name")
    if len(set(headers)) != len(headers):
        raise CanonicalSourceError(path, "CSV header contains duplicate field names")

    rows: list[tuple[int, list[str]]] = []
    for row_number, row in enumerate(parsed_rows[1:], start=2):
        if not row:
            continue
        if len(row) != len(headers):
            raise CanonicalSourceError(path, "CSV row has a different number of fields than its header")
        rows.append((row_number, row))
    return rows, headers, delimiter


def _hash_fields(fields: Mapping[str, str]) -> str:
    canonical = json.dumps(list(fields.items()), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_anki_records(
    path: Path,
    source_kind: Literal["apkg", "colpkg"],
    requested_note_type: str | None,
) -> tuple[CanonicalSourceRecord, ...]:
    try:
        with zipfile.ZipFile(path, "r") as archive, tempfile.TemporaryDirectory() as temporary_directory:
            database_name, compressed = _pick_collection_database_name(archive.namelist(), path)
            database_bytes = _read_zip_member(archive, database_name, path)
            if compressed:
                try:
                    database_bytes = zstandard.ZstdDecompressor().decompress(
                        database_bytes,
                        max_output_size=512 * 1024 * 1024,
                    )
                except zstandard.ZstdError as error:
                    raise CanonicalSourceError(path, "collection database is not valid zstd data") from error
            database_path = Path(temporary_directory) / "collection.anki2"
            database_path.write_bytes(database_bytes)
            return _read_anki_database(path, database_path, source_kind, requested_note_type)
    except CanonicalSourceError:
        raise
    except zipfile.BadZipFile as error:
        raise CanonicalSourceError(path, "source is not a valid ZIP archive") from error
    except (OSError, KeyError) as error:
        raise CanonicalSourceError(path, "package contents could not be read") from error


def _pick_collection_database_name(names: Sequence[str], path: Path) -> tuple[str, bool]:
    for suffix in _ANKI_DATABASE_SUFFIXES:
        matches = sorted(name for name in names if name.endswith(suffix))
        if matches:
            return matches[0], suffix == ".anki21b"
    raise CanonicalSourceError(path, "package does not contain a collection database")


def _read_zip_member(archive: zipfile.ZipFile, member_name: str, path: Path) -> bytes:
    try:
        member_info = archive.getinfo(member_name)
        if member_info.file_size > _MAX_COLLECTION_MEMBER_BYTES:
            raise CanonicalSourceError(path, "collection database member is too large to inspect safely")
        contents = bytearray()
        with archive.open(member_info, "r") as member:
            while chunk := member.read(64 * 1024):
                contents.extend(chunk)
                if len(contents) > _MAX_COLLECTION_MEMBER_BYTES:
                    raise CanonicalSourceError(path, "collection database member is too large to inspect safely")
        return bytes(contents)
    except CanonicalSourceError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, ValueError) as error:
        raise CanonicalSourceError(path, "collection database member could not be read") from error


def _read_anki_database(
    source_path: Path,
    database_path: Path,
    source_kind: Literal["apkg", "colpkg"],
    requested_note_type: str | None,
) -> tuple[CanonicalSourceRecord, ...]:
    try:
        con = sqlite3.connect(database_path)
        con.row_factory = sqlite3.Row
        try:
            note_columns = _table_columns(con, "notes")
            required_columns = {"id", "mid", "flds"}
            if not required_columns <= note_columns:
                raise CanonicalSourceError(
                    source_path,
                    "collection notes table is missing a required structural column",
                )
            if "guid" not in note_columns:
                raise CanonicalSourceError(source_path, "collection notes table is missing the required GUID column")
            note_types, field_names = _load_anki_metadata(con, source_path)
            rows = con.execute("SELECT id, mid, flds, guid FROM notes").fetchall()
            records: list[CanonicalSourceRecord] = []
            for row in rows:
                record = _build_anki_record(
                    source_path,
                    source_kind,
                    row,
                    note_types,
                    field_names,
                )
                if requested_note_type is None or record.note_type == requested_note_type:
                    records.append(record)
            records.sort(
                key=lambda record: (
                    record.note_type or "",
                    record.source_identity or "",
                    record.provenance.note_id if record.provenance.note_id is not None else -1,
                )
            )
            return tuple(records)
        finally:
            con.close()
    except CanonicalSourceError:
        raise
    except sqlite3.DatabaseError as error:
        raise CanonicalSourceError(source_path, "collection database is not a valid SQLite database") from error
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise CanonicalSourceError(source_path, "collection metadata has an invalid structure") from error


def _table_columns(con: sqlite3.Connection, table_name: str) -> set[str]:
    rows = con.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {str(row[1]) for row in rows}


def _load_anki_metadata(
    con: sqlite3.Connection,
    source_path: Path,
) -> tuple[dict[int, str | None], dict[int, list[str]]]:
    note_types: dict[int, str | None] = {}
    field_names: dict[int, list[str]] = {}

    if _table_columns(con, "notetypes") >= {"id", "name"}:
        for row in con.execute("SELECT id, name FROM notetypes ORDER BY id"):
            note_types[int(row[0])] = str(row[1]) if row[1] is not None else None
    if _table_columns(con, "fields") >= {"ntid", "ord", "name"}:
        for row in con.execute("SELECT ntid, ord, name FROM fields ORDER BY ntid, ord"):
            field_name = str(row[2]) if row[2] is not None else ""
            field_names.setdefault(int(row[0]), []).append(field_name)

    if not note_types or not field_names:
        legacy_models = _load_legacy_models(con, source_path)
        for mid, model in legacy_models.items():
            note_types.setdefault(mid, model[0])
            field_names.setdefault(mid, model[1])

    for _mid, names in field_names.items():
        if len(names) != len(set(names)):
            raise CanonicalSourceError(source_path, "note type metadata contains duplicate field names")
        if any(not name for name in names):
            raise CanonicalSourceError(source_path, "note type metadata contains an unnamed field")
    return note_types, field_names


def _load_legacy_models(
    con: sqlite3.Connection,
    source_path: Path,
) -> dict[int, tuple[str | None, list[str]]]:
    if not _table_columns(con, "col"):
        return {}
    row = con.execute("SELECT models FROM col ORDER BY id LIMIT 1").fetchone()
    if row is None or not row[0]:
        return {}
    try:
        decoded = json.loads(str(row[0]))
    except json.JSONDecodeError as error:
        raise CanonicalSourceError(source_path, "legacy note type metadata is not valid JSON") from error
    if not isinstance(decoded, dict):
        raise CanonicalSourceError(source_path, "legacy note type metadata is not an object")

    models: dict[int, tuple[str | None, list[str]]] = {}
    for raw_mid, raw_model in decoded.items():
        if not isinstance(raw_model, dict):
            raise CanonicalSourceError(source_path, "legacy note type metadata has an invalid model")
        try:
            mid = int(raw_mid)
        except (TypeError, ValueError) as error:
            raise CanonicalSourceError(source_path, "legacy note type metadata has an invalid model ID") from error
        raw_fields = raw_model.get("flds", [])
        if not isinstance(raw_fields, list):
            raise CanonicalSourceError(source_path, "legacy note type metadata has invalid fields")
        names: list[str] = []
        for raw_field in raw_fields:
            if not isinstance(raw_field, dict):
                raise CanonicalSourceError(source_path, "legacy note type metadata has an invalid field")
            names.append(str(raw_field.get("name", "")))
        raw_name = raw_model.get("name")
        models[mid] = (str(raw_name) if raw_name is not None else None, names)
    return models


def _build_anki_record(
    source_path: Path,
    source_kind: Literal["apkg", "colpkg"],
    row: sqlite3.Row,
    note_types: Mapping[int, str | None],
    field_names: Mapping[int, Sequence[str]],
) -> CanonicalSourceRecord:
    try:
        note_id = int(row["id"])
        mid = int(row["mid"])
    except (TypeError, ValueError) as error:
        raise CanonicalSourceError(source_path, "notes table contains an invalid note identifier") from error

    raw_fields = row["flds"]
    if isinstance(raw_fields, bytes):
        try:
            field_text = raw_fields.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CanonicalSourceError(source_path, "notes table contains invalid field text") from error
    else:
        field_text = str(raw_fields)
    values = field_text.split("\x1f")
    note_type = note_types.get(mid)
    if note_type is None:
        raise CanonicalSourceError(source_path, "notes table refers to missing note type metadata")
    names = list(field_names.get(mid, ()))
    if not names:
        raise CanonicalSourceError(source_path, "notes table refers to missing field metadata")
    if len(values) > len(names):
        raise CanonicalSourceError(source_path, "notes table contains values without named fields")
    field_count = len(names)
    fields = {
        names[index] if index < len(names) else f"field_{index}": values[index] if index < len(values) else ""
        for index in range(field_count)
    }
    raw_guid = row["guid"]
    if isinstance(raw_guid, bytes):
        try:
            note_guid = raw_guid.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CanonicalSourceError(source_path, "notes table contains an invalid GUID") from error
    else:
        note_guid = str(raw_guid) if raw_guid is not None else ""
    if not note_guid.strip():
        raise CanonicalSourceError(source_path, "notes table contains a missing or empty GUID")
    return CanonicalSourceRecord(
        source_kind=source_kind,
        note_type=note_type,
        fields=fields,
        provenance=SourceProvenance(
            source_path=source_path,
            location=f"note {note_id}",
            note_id=note_id,
            content_hash=_hash_fields(fields),
        ),
        source_identity=note_guid,
        note_guid=note_guid,
    )


load_source_records = read_source_records
load_csv_records = read_csv_records
load_apkg_records = read_apkg_records
load_colpkg_records = read_colpkg_records

__all__ = [
    "CanonicalSourceError",
    "CanonicalSourceRecord",
    "SourceInspection",
    "SourceKind",
    "SourceProvenance",
    "inspect_source",
    "load_apkg_records",
    "load_colpkg_records",
    "load_csv_records",
    "load_source_records",
    "read_apkg_records",
    "read_colpkg_records",
    "read_csv_records",
    "read_source_records",
]
