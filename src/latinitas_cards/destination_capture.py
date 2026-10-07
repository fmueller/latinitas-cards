"""Read an attested, closed schema-18 SQLite backup, never an Anki Collection.

The optional pinned Anki package supplies protobuf definitions only. Acquisition
does not load its backend, migrate a database, infer ownership, or approve apply.
"""

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from uuid import uuid4

from .cards import TEMPLATE_REGISTRY
from .destination_state import (
    MANAGED_FIELDS,
    BoundDestination,
    DestinationSnapshot,
    ReconciliationRequired,
    _state,
    bound_card_rows,
    read_snapshot,
    schema_contract,
)
from .reference_templates import REFERENCE_NOTE_TYPE_FIELDS

# Schema-18 storage columns, checked even for empty tables. Do not call missing
# schedule/history/config columns a complete collection capture.
COLUMNS = {
    "col": "id crt mod scm ver dty usn ls conf models decks dconf tags",
    "config": "key usn mtime_secs val",
    "notes": "id guid mid mod usn tags flds sfld csum flags data",
    "cards": "id nid did ord mod usn type queue due ivl factor reps lapses left odue odid flags data",
    "revlog": "id cid usn ease ivl lastivl factor time type",
    "notetypes": "id name mtime_secs usn config",
    "fields": "ntid ord name config",
    "templates": "ntid ord name mtime_secs usn config",
    "decks": "id name mtime_secs usn common kind",
    "deck_config": "id name mtime_secs usn config",
}
DECODER_VERSION = "26.9.3"


def _decode(module: str, message: str, value: bytes) -> Any:
    try:
        if version("anki") != DECODER_VERSION:
            raise ReconciliationRequired(f"capture requires optional anki=={DECODER_VERSION} protobuf definitions")
        cls: Any = import_module(f"anki.{module}_pb2")
        for name in message.split("."):
            cls = getattr(cls, name)
        decoded = cls()
        decoded.ParseFromString(value)
        return decoded
    except (ImportError, PackageNotFoundError) as exc:
        raise ReconciliationRequired(
            f"schema-18 capture needs optional protobuf definitions: uv run --with anki=={DECODER_VERSION} "
            "latinitas-cards managed capture ...; no Anki backend is opened"
        ) from exc
    except Exception as exc:
        # Protobuf DecodeError is optional too; preserve our own actionable errors.
        if isinstance(exc, ReconciliationRequired):
            raise
        raise ReconciliationRequired("invalid native protobuf config") from exc


def _schema(rows: dict[str, list[dict[str, Any]]], note_type_id: str) -> dict[str, Any]:
    models = [row for row in rows["notetypes"] if str(row["id"]) == note_type_id]
    if len(models) != 1:
        raise ReconciliationRequired("missing or ambiguous selected note type")
    model = _decode("notetypes", "Notetype.Config", models[0]["config"])
    if model.kind != 0:
        raise ReconciliationRequired("unsupported cloze note type")
    fields = sorted((row for row in rows["fields"] if str(row["ntid"]) == note_type_id), key=lambda row: row["ord"])
    if [(row["ord"], row["name"]) for row in fields] != list(enumerate(REFERENCE_NOTE_TYPE_FIELDS)):
        raise ReconciliationRequired("unknown field layout; separately reviewed manual setup required")
    for field in fields:
        _decode("notetypes", "Notetype.Field.Config", field["config"])
    schema = schema_contract(note_type_id)
    schema["css_digest"] = hashlib.sha256(model.css.encode()).hexdigest()
    templates = sorted(
        (row for row in rows["templates"] if str(row["ntid"]) == note_type_id), key=lambda row: row["ord"]
    )
    if [row["ord"] for row in templates] != [slot.ordinal for slot in TEMPLATE_REGISTRY]:
        raise ReconciliationRequired("incomplete or ambiguous template ordinals")
    schema["templates"] = []
    for row, slot in zip(templates, TEMPLATE_REGISTRY, strict=True):
        config = _decode("notetypes", "Notetype.Template.Config", row["config"])
        schema["templates"].append(
            {
                "name": row["name"],
                "ordinal": row["ord"],
                "semantic_key": slot.semantic_key,
                "front_digest": hashlib.sha256(config.q_format.encode()).hexdigest(),
                "back_digest": hashlib.sha256(config.a_format.encode()).hexdigest(),
            }
        )
    return schema


def _file_signature(stat: os.stat_result) -> tuple[int, ...]:
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def capture_backup(
    path: Path, selection: dict[str, Any], *, closed_backup: bool, interval_confirmed: bool
) -> DestinationSnapshot:
    """Acquire complete observed rows with an explicit whole-set operator selection."""
    if not closed_backup:
        raise ReconciliationRequired("attest a closed backup copy, never a live user collection")
    if not interval_confirmed:
        raise ReconciliationRequired(
            "attest a fresh backup and no-edit interval on every device; stale backup requires recapture"
        )
    path = path.resolve(strict=True)
    if any(Path(f"{path}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal")):
        raise ReconciliationRequired("backup has SQLite sidecars; obtain a complete closed checkpointed copy")
    with path.open("rb") as source, tempfile.TemporaryDirectory(prefix="latinitas-closed-capture-") as directory:
        signature = _file_signature(os.fstat(source.fileno()))
        contents = source.read()
        before = hashlib.sha256(contents).hexdigest()
        copy = Path(directory) / "backup.anki2"
        copy.write_bytes(contents)
        # Bind SQL reads and the digest to these exact privately acquired bytes,
        # not a pathname that can be replaced while the connection is open.
        with closing(sqlite3.connect(f"{copy.as_uri()}?mode=ro&immutable=1", uri=True)) as db:
            db.row_factory = sqlite3.Row
            for table, columns in COLUMNS.items():
                if {row["name"].lower() for row in db.execute(f"PRAGMA table_info({table})")} != set(columns.split()):
                    raise ReconciliationRequired(
                        f"unknown or missing native {table} columns; complete schema 18 required"
                    )
            # Native indexes use Anki's unicase collation; don't guess it or load
            # a backend. Capture complete table rows with binary ordering.
            rows = {
                table: [
                    dict(row) for row in db.execute(f"SELECT * FROM {table} ORDER BY 1 COLLATE BINARY,2 COLLATE BINARY")
                ]
                for table in COLUMNS
            }
        if signature != _file_signature(os.fstat(source.fileno())) or signature != _file_signature(path.stat()):
            raise ReconciliationRequired("backup changed during capture; recapture")
        if any(Path(f"{path}{suffix}").exists() for suffix in ("-wal", "-shm", "-journal")):
            raise ReconciliationRequired("SQLite sidecar appeared during capture; recapture a closed complete backup")
    if len(rows["col"]) != 1 or rows["col"][0]["ver"] != 18:
        raise ReconciliationRequired("capture supports only native SQLite schema 18; no migration attempted")
    note_type_id = selection["note_type_id"]
    binding = BoundDestination(
        selection["destination"], selection["profile"], _schema(rows, note_type_id), selection["managed_set"]
    )
    checked = binding.payload()
    members = {member[0]: member[1:] for member in checked["managed_set"]["members"]}
    configs = {row["id"] for row in rows["deck_config"]}
    for row in rows["deck_config"]:
        _decode("deck_config", "DeckConfig.Config", row["config"])
    for row in rows["decks"]:
        _decode("decks", "Deck.Common", row["common"])
        kind = _decode("decks", "Deck.KindContainer", row["kind"])
        if kind.WhichOneof("kind") != "normal" or kind.normal.config_id not in configs:
            raise ReconciliationRequired("unsupported filtered deck or missing deck config binding")
    note_ids = {row["id"] for row in rows["notes"]}
    card_ids = {row["id"] for row in rows["cards"]}
    deck_ids = {row["id"] for row in rows["decks"]}
    if any(row["nid"] not in note_ids or row["did"] not in deck_ids for row in rows["cards"]):
        raise ReconciliationRequired("orphan card/note/deck binding")
    if any(row["cid"] not in card_ids for row in rows["revlog"]):
        raise ReconciliationRequired("orphan review history binding; incomplete backup")
    notes = []
    seen: set[str] = set()
    for row in rows["notes"]:
        if str(row["mid"]) != note_type_id:
            # Be conservative even if a foreign model reordered/renamed the ID
            # field: finding this exact portable token never proves absence.
            if any(value in members for value in row["flds"].split("\x1f")):
                raise ReconciliationRequired("selected identity belongs to a different note type")
            continue
        values = row["flds"].split("\x1f")
        if len(values) != len(REFERENCE_NOTE_TYPE_FIELDS):
            raise ReconciliationRequired("incomplete native field values")
        fields = dict(zip(REFERENCE_NOTE_TYPE_FIELDS, values, strict=True))
        identity = fields["LatinitasID"]
        if not identity or identity not in members:
            raise ReconciliationRequired("missing or unknown portable identity; whole dedicated managed set required")
        if [fields["Source Scope"], fields["Source ID"]] != members[identity][:2]:
            raise ReconciliationRequired("stored source provenance differs from immutable membership")
        if identity in seen:
            raise ReconciliationRequired("duplicate portable identity")
        seen.add(identity)
        if not row["guid"] or not row["id"]:
            raise ReconciliationRequired("missing native note identity")
        cards = []
        for card in rows["cards"]:
            if card["nid"] != row["id"]:
                continue
            ordinal = card["ord"]
            if type(ordinal) is not int or not 0 <= ordinal < len(TEMPLATE_REGISTRY):
                raise ReconciliationRequired("unknown card template ordinal")
            slot = TEMPLATE_REGISTRY[ordinal]
            cards.append(
                {
                    **card,
                    "id": str(card["id"]),
                    "semantic_key": slot.semantic_key,
                    "ordinal": slot.ordinal,
                    "template_name": slot.template_name,
                    "suspended": card["queue"] == -1,
                }
            )
        ids = {int(card["id"]) for card in cards}
        note = {
            "identity": identity,
            "source": members[identity],
            "note_type_id": note_type_id,
            "guid": row["guid"],
            "local_id": str(row["id"]),
            "fields": {name: fields[name] for name in MANAGED_FIELDS},
            "tags": sorted(row["tags"].split()),
            "personal_digest": hashlib.sha256(fields["Personal Notes"].encode()).hexdigest(),
            "cards": {"complete": True, "rows": cards},
            "history": {
                "complete": True,
                "rows": [{**log, "id": str(log["id"])} for log in rows["revlog"] if log["cid"] in ids],
            },
        }
        bound_card_rows(note)
        notes.append(note)
    # Preserve full table bytes/settings locally, not Personal Notes as owned values.
    tables = {
        table: [
            {key: value.hex() if isinstance(value, bytes) else value for key, value in row.items()} for row in records
        ]
        for table, records in rows.items()
    }
    payload = {
        **checked,
        "version": 1,
        "snapshot_id": str(uuid4()),
        "captured_at": datetime.now(UTC).isoformat(),
        "artifact_sha256": before,
        "client": f"offline SQLite / Anki {DECODER_VERSION} protobuf definitions",
        "export_method": "attested closed SQLite backup copy",
        "export_options": {"tables": tables},
        "selection": "whole dedicated note type and explicit immutable membership",
        "exclusions": [],
        "expected_count": len(notes),
        "complete": True,
        "fresh": True,
        "level": "collection",
        "deck_options": {name: tables[name] for name in ("decks", "deck_config")},
        "notes": notes,
    }
    return read_snapshot(payload, binding)


def save_adoption(path: Path, state: dict[str, Any]) -> None:
    """Create a validated first journal exclusively; never resume or replace one."""
    checked = _state(state)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(checked, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Atomic publication, fails even for a dangling symlink.
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        os.unlink(temporary)
