"""Closed synthetic SQLite acquisition; native decoder evidence is a separate recipe."""

import hashlib
import json
import sqlite3
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from card_lifecycle_test import generated
from typer.testing import CliRunner

from latinitas_cards.cli import app
from latinitas_cards.destination_state import load_state, schema_contract
from latinitas_cards.reference_templates import REFERENCE_CARD_CSS, REFERENCE_CARD_TEMPLATES, REFERENCE_NOTE_TYPE_FIELDS


@pytest.fixture
def backup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    from latinitas_cards import destination_capture

    def decode(module: str, message: str, value: bytes) -> Any:
        try:
            data = json.loads(value)
        except ValueError as exc:
            raise ValueError("invalid native protobuf config") from exc
        if message == "Deck.KindContainer":
            return SimpleNamespace(
                WhichOneof=lambda name: data["kind"], normal=SimpleNamespace(config_id=data["config_id"])
            )
        return SimpleNamespace(**data)

    monkeypatch.setattr(destination_capture, "_decode", decode)
    path = tmp_path / "closed backup ? ü.anki2"
    note = generated()
    fields = dict(note.to_anki_fields())
    fields["Personal Notes"] = "private ü"
    with sqlite3.connect(path) as db:
        db.executescript(
            "CREATE TABLE col (id INTEGER PRIMARY KEY, ver INTEGER); INSERT INTO col VALUES (1,18);"
            "CREATE TABLE notetypes (id INTEGER PRIMARY KEY,name TEXT,config BLOB);"
            "CREATE TABLE fields (ntid INTEGER,ord INTEGER,name TEXT,config BLOB);"
            "CREATE TABLE templates (ntid INTEGER,ord INTEGER,name TEXT,config BLOB);"
            "CREATE TABLE notes (id INTEGER PRIMARY KEY,guid TEXT,mid INTEGER,flds TEXT,tags TEXT);"
            "CREATE TABLE cards (id INTEGER PRIMARY KEY,nid INTEGER,did INTEGER,ord INTEGER,queue INTEGER,due INTEGER);"
            "CREATE TABLE revlog (id INTEGER PRIMARY KEY,cid INTEGER,ivl INTEGER);"
            "CREATE TABLE decks (id INTEGER PRIMARY KEY,name TEXT,common BLOB,kind BLOB);"
            "CREATE TABLE deck_config (id INTEGER PRIMARY KEY,name TEXT,config BLOB);"
            "CREATE TABLE config (key TEXT,val BLOB);"
        )
        db.execute(
            "INSERT INTO notetypes VALUES (10,'Latinitas',?)",
            (json.dumps({"css": REFERENCE_CARD_CSS, "kind": 0}).encode(),),
        )
        db.executemany(
            "INSERT INTO fields VALUES (10,?,?,?)",
            [(i, name, b"{}") for i, name in enumerate(REFERENCE_NOTE_TYPE_FIELDS)],
        )
        db.executemany(
            "INSERT INTO templates VALUES (10,?,?,?)",
            [
                (t.slot.ordinal, t.slot.template_name, json.dumps({"q_format": t.front, "a_format": t.back}).encode())
                for t in REFERENCE_CARD_TEMPLATES
            ],
        )
        db.execute(
            "INSERT INTO notes VALUES (20,'native-guid',10,?,' source manual ')",
            ("\x1f".join(fields[n] for n in REFERENCE_NOTE_TYPE_FIELDS),),
        )
        db.execute("INSERT INTO cards VALUES (30,20,1,0,-1,987)")
        db.execute("INSERT INTO revlog VALUES (40,30,23)")
        db.execute("INSERT INTO decks VALUES (1,'Latin',?,?)", (b"{}", b'{"kind":"normal","config_id":1}'))
        db.execute("INSERT INTO deck_config VALUES (1,'Default',?)", (b"{}",))
        for table, names in {
            "col": "crt mod scm dty usn ls conf models decks dconf tags",
            "config": "usn mtime_secs",
            "notes": "mod usn sfld csum flags data",
            "cards": "mod usn type ivl factor reps lapses left odue odid flags data",
            "revlog": "usn ease lastIvl factor time type",
            "notetypes": "mtime_secs usn",
            "templates": "mtime_secs usn",
            "decks": "mtime_secs usn",
            "deck_config": "mtime_secs usn",
        }.items():
            for name in names.split():
                db.execute(f'ALTER TABLE {table} ADD COLUMN "{name}" DEFAULT 0')
    request = tmp_path / "selection.json"
    request.write_text(
        json.dumps(
            {
                "destination": "attested-original",
                "profile": "fixture-profile",
                "note_type_id": "10",
                "managed_set": {
                    "scope": "scope",
                    "members": [[note.latinitas_id, "scope", "entry", note.object_key]],
                },
            }
        )
    )
    return path, request, note.latinitas_id


def capture(backup: tuple[Path, Path, str], *extra: str) -> Any:
    path, request, _ = backup
    return CliRunner().invoke(app, ["managed", "capture", str(path), "--selection", str(request), *extra])


def test_capture_requires_closed_fresh_attestations(backup: tuple[Path, Path, str]) -> None:
    result = capture(backup)
    assert result.exit_code == 1
    assert "closed backup" in result.output
    result = capture(backup, "--closed-backup")
    assert result.exit_code == 1
    assert "interval" in result.output


def test_capture_and_adopt_actual_rows(backup: tuple[Path, Path, str], tmp_path: Path) -> None:
    path, _, identity = backup
    before = path.read_bytes()
    result = capture(backup, "--closed-backup", "--interval-confirmed")
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["schema"] == schema_contract("10")
    assert data["artifact_sha256"] == hashlib.sha256(before).hexdigest()
    note = data["notes"][0]
    assert note["identity"] == identity
    assert note["guid"] == "native-guid" and note["local_id"] == "20"
    assert note["cards"]["rows"] == [
        {
            "id": "30",
            "nid": 20,
            "did": 1,
            "ord": 0,
            "queue": -1,
            "due": 987,
            "ordinal": 0,
            "template_name": "Completion Present",
            "semantic_key": "principal_part_completion:present_1s",
            "suspended": True,
            **dict.fromkeys(
                ["mod", "usn", "type", "ivl", "factor", "reps", "lapses", "left", "odue", "odid", "flags", "data"], 0
            ),
        }
    ]
    assert note["history"]["rows"] == [
        {"id": "40", "cid": 30, "ivl": 23, **dict.fromkeys(["usn", "ease", "lastIvl", "factor", "time", "type"], 0)}
    ]
    assert "Personal Notes" not in note["fields"]
    assert note["personal_digest"] == hashlib.sha256("private ü".encode()).hexdigest()
    assert path.read_bytes() == before
    assert set(data["export_options"]["tables"]) == {
        "col",
        "config",
        "notes",
        "cards",
        "revlog",
        "notetypes",
        "fields",
        "templates",
        "decks",
        "deck_config",
    }
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(result.output)
    ownership = tmp_path / "ownership.json"
    ownership.write_text(
        json.dumps(
            {identity: {"source_tags": ["source"], "configured_tags": [], "keep_tags": ["manual"], "keep_fields": []}}
        )
    )
    state = tmp_path / "baseline.json"
    runner = CliRunner()
    args = [
        "managed",
        "adopt",
        str(snapshot),
        "--ownership",
        str(ownership),
        "--review",
        "per-note origins reviewed",
        "--state",
        str(state),
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    baseline = load_state(state)
    assert baseline["anchors"][identity]["source_tags"] == ["source"]
    assert baseline["anchors"][identity]["keep_tags"] == ["manual"]
    assert baseline["plans"] == {}
    original = state.read_bytes()
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and "exists" in result.output
    assert state.read_bytes() == original


@pytest.mark.parametrize(
    "sql, error",
    [
        ("UPDATE col SET ver=11", "schema 18"),
        ("DROP TABLE revlog", "revlog"),
        ("ALTER TABLE cards DROP COLUMN factor", "columns"),
        ("DELETE FROM deck_config", "config"),
        ("UPDATE deck_config SET config='invalid'", "config"),
        ("DELETE FROM fields WHERE ord=0", "field"),
        ("UPDATE templates SET name='repurposed' WHERE ord=0", "schema"),
        ("UPDATE notetypes SET config='{}'", "kind"),
        ("UPDATE notes SET flds='visible amo'", "field"),
        ("UPDATE notes SET guid=''", "identity"),
        ("INSERT INTO notes(id,guid,mid,flds,tags) SELECT 21,guid,mid,flds,tags FROM notes", "duplicate"),
        ("UPDATE cards SET ord=999", "template"),
        ("INSERT INTO cards(id,nid,did,ord,queue,due) SELECT 31,nid,did,ord,queue,due FROM cards", "ambiguous"),
        ("UPDATE cards SET nid=999", "orphan"),
        ("UPDATE revlog SET cid=999", "orphan"),
    ],
)
def test_capture_rejects_unsafe_backup(backup: tuple[Path, Path, str], sql: str, error: str) -> None:
    with sqlite3.connect(backup[0]) as db:
        db.execute(sql)
    result = capture(backup, "--closed-backup", "--interval-confirmed")
    assert result.exit_code == 1, result.output
    assert error in result.output.lower()


def test_capture_absence_and_unknown_identity(backup: tuple[Path, Path, str]) -> None:
    with sqlite3.connect(backup[0]) as db:
        db.execute("DELETE FROM revlog")
        db.execute("DELETE FROM cards")
        db.execute("DELETE FROM notes")
    result = capture(backup, "--closed-backup", "--interval-confirmed")
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["notes"] == []
    assert len(json.loads(result.output)["managed_set"]["members"]) == 1


@pytest.mark.parametrize("review, ownership", [("", "valid"), ("reviewed", "missing"), ("reviewed", "conflict")])
def test_adoption_rejects_unreviewed_ownership(
    backup: tuple[Path, Path, str], tmp_path: Path, review: str, ownership: str
) -> None:
    result = capture(backup, "--closed-backup", "--interval-confirmed")
    assert result.exit_code == 0, result.output
    path = tmp_path / "snapshot.json"
    path.write_text(result.output)
    claims = tmp_path / "ownership.json"
    claims.write_text(
        json.dumps(
            {}
            if ownership == "missing"
            else {
                backup[2]: {
                    "source_tags": ["source"],
                    "configured_tags": [],
                    "keep_tags": [] if ownership == "conflict" else ["manual"],
                    "keep_fields": [],
                }
            }
        )
    )
    state = tmp_path / "new.json"
    result = CliRunner().invoke(
        app, ["managed", "adopt", str(path), "--ownership", str(claims), "--review", review, "--state", str(state)]
    )
    assert result.exit_code == 1
    assert not state.exists()


def test_capture_rejects_mismatched_stored_source(backup: tuple[Path, Path, str]) -> None:
    with sqlite3.connect(backup[0]) as db:
        fields = db.execute("SELECT flds FROM notes").fetchone()[0].split("\x1f")
        fields[4] = "different-source"
        db.execute("UPDATE notes SET flds=?", ("\x1f".join(fields),))
    result = capture(backup, "--closed-backup", "--interval-confirmed")
    assert result.exit_code == 1
    assert "source" in result.output


def test_first_journal_write_failure_does_not_publish_partial_state(
    backup: tuple[Path, Path, str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from latinitas_cards.destination_capture import save_adoption
    from latinitas_cards.destination_state import DestinationSnapshot, adopt

    result = capture(backup, "--closed-backup", "--interval-confirmed")
    snapshot = DestinationSnapshot(result.output)
    state = adopt(
        snapshot,
        {backup[2]: {"source_tags": ["source"], "configured_tags": [], "keep_tags": ["manual"], "keep_fields": []}},
        "reviewed",
    )
    path = tmp_path / "journal.json"

    def fail(descriptor: int) -> None:
        raise OSError("disk failure")

    monkeypatch.setattr("latinitas_cards.destination_capture.os.fsync", fail)
    with pytest.raises(OSError, match="disk failure"):
        save_adoption(path, state)
    assert not path.exists()
    assert not list(tmp_path.glob(".journal.json.*"))


@pytest.mark.parametrize("change", ["missing", "unknown", "wrong-model", "stale", "duplicate-membership"])
def test_capture_identity_selection_and_freshness_rejections(backup: tuple[Path, Path, str], change: str) -> None:
    if change == "stale":
        Path(f"{backup[0]}-wal").write_bytes(b"uncheckpointed")
    elif change == "duplicate-membership":
        data = json.loads(backup[1].read_text())
        data["managed_set"]["members"] *= 2
        backup[1].write_text(json.dumps(data))
    else:
        with sqlite3.connect(backup[0]) as db:
            fields = db.execute("SELECT flds FROM notes").fetchone()[0].split("\x1f")
            if change == "wrong-model":
                db.execute("UPDATE notes SET mid=99")
            else:
                fields[0] = "" if change == "missing" else "unknown-identity"
                db.execute("UPDATE notes SET flds=?", ("\x1f".join(fields),))
    result = capture(backup, "--closed-backup", "--interval-confirmed")
    assert result.exit_code == 1
    assert "Managed error:" in result.output


def test_optional_decoder_absence_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    from importlib.metadata import PackageNotFoundError

    from latinitas_cards.destination_capture import _decode
    from latinitas_cards.destination_state import ReconciliationRequired

    def missing(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr("latinitas_cards.destination_capture.version", missing)
    with pytest.raises(ReconciliationRequired, match="uv run --with anki==26.9.3"):
        _decode("notetypes", "Notetype.Config", b"")


def test_foreign_model_identity_in_later_field_is_not_absent(backup: tuple[Path, Path, str]) -> None:
    with sqlite3.connect(backup[0]) as db:
        db.execute("UPDATE notes SET mid=99,flds=?", (f"visible amo\x1f{backup[2]}",))
        db.execute("INSERT INTO fields VALUES (99,0,'Lemma',?)", (b"{}",))
        db.execute("INSERT INTO fields VALUES (99,1,'LatinitasID',?)", (b"{}",))
    result = capture(backup, "--closed-backup", "--interval-confirmed")
    assert result.exit_code == 1
    assert "different note type" in result.output


@pytest.mark.parametrize("change", ["late-sidecar", "replaced-and-restored"])
def test_capture_is_bound_to_stable_sidecar_free_file(
    backup: tuple[Path, Path, str], monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    import shutil

    path = backup[0]
    connect: Callable[..., sqlite3.Connection] = sqlite3.connect
    replacement = path.with_name("different.anki2")
    retained = path.with_name("retained.anki2")
    shutil.copyfile(path, replacement)
    with connect(replacement) as db:
        db.execute("UPDATE notes SET tags='different'")

    def changing_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        if change == "late-sidecar":
            Path(f"{path}-wal").write_bytes(b"late WAL state")
            return connect(*args, **kwargs)
        path.rename(retained)
        replacement.rename(path)
        connection = connect(*args, **kwargs)
        path.rename(replacement)
        retained.rename(path)
        return connection

    monkeypatch.setattr("latinitas_cards.destination_capture.sqlite3.connect", changing_connect)
    result = capture(backup, "--closed-backup", "--interval-confirmed")
    assert result.exit_code == 1
    assert "recapture" in result.output or "sidecar" in result.output
