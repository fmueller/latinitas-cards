"""Optional native managed CSV evidence, using only disposable closed collections.

Run: uv run --with anki==26.9.3 python scripts/check-managed-anki.py --output-dir DIR
DIR must not exist. Evidence is bound to this fixture, not arbitrary user collections.
"""

import argparse
import hashlib
import json
import platform
import shutil
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from contextlib import closing
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch

from latinitas_cards.cards import TEMPLATE_REGISTRY, render_cards
from latinitas_cards.destination_state import (
    MANAGED_FIELDS,
    BoundDestination,
    DestinationSnapshot,
    ReconciliationRequired,
    adopt,
    load_state,
    review_reconciliation,
    save_state,
    schema_contract,
)
from latinitas_cards.legacy_transition import DestinationNoteEvidence, FreshStartApproval, plan_destination_fresh_start
from latinitas_cards.managed_application import emit_updates, observe_updates
from latinitas_cards.managed_plans import approve_plan, compose_plan, verify_approval
from latinitas_cards.notes import (
    CSV_EXPORT_FIELD_NAMES,
    GeneratedNote,
    GeneratedNoteProvenance,
    GenerationMetadata,
    ManagedNoteContent,
)
from latinitas_cards.preview_export import serialize_anki_csv
from latinitas_cards.principal_parts import ParsedPrincipalParts, PrincipalPartValue
from latinitas_cards.profile import DeckProfile
from latinitas_cards.reference_templates import (
    REFERENCE_CARD_CSS,
    REFERENCE_CARD_TEMPLATES,
    REFERENCE_NOTE_TYPE_FIELDS,
)

MODEL = "Latinitas Native Managed"
DECK = "Latin::Managed"
PROFILE = {"generated_note_type": MODEL, "target_deck": DECK}
TABLES = ("notes", "cards", "revlog", "notetypes", "fields", "templates", "decks", "deck_config")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tables(path: Path) -> dict[str, list[dict[str, Any]]]:
    # Call only after Collection.close(); never inspect a live collection.
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        return {
            table: [
                {key: value.hex() if isinstance(value, bytes) else value for key, value in dict(row).items()}
                for row in db.execute(f"select * from {table} order by 1 collate binary,2 collate binary")
            ]
            for table in TABLES
        }


def generated(key: str) -> GeneratedNote:
    parsed = ParsedPrincipalParts(
        "amo",
        "amo",
        (PrincipalPartValue("present_1s", "amo", "amo"), PrincipalPartValue("perfect_1s", "amavi", "amavi")),
    )
    return GeneratedNote.create(
        source_identity="synthetic-entry",
        source_scope="native-scope",
        object_key=key,
        provenance=GeneratedNoteProvenance("csv", "synthetic row", source_scope="native-scope"),
        metadata=GenerationMetadata("native-profile"),
        content=ManagedNoteContent("amo", "amo, amavi", f"old {key}", ("source",)),
        cards=render_cards(parsed, selected_recipes=("principal_part_completion", "principal_part_recognition")),
    )


def create(
    path: Path,
    notes: list[GeneratedNote],
    csv_path: Path,
    before_import: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    from anki.collection import Collection

    collection = Collection(str(path))
    try:
        collection.decks.id(DECK)
        config = collection.decks.get_config(1)
        config["new"]["bury"] = True
        config["rev"]["bury"] = True
        config["buryInterdayLearning"] = True
        collection.decks.update_config(config)
        model = collection.models.new(MODEL)
        for field in REFERENCE_NOTE_TYPE_FIELDS:
            collection.models.add_field(model, collection.models.new_field(field))
        for reference in REFERENCE_CARD_TEMPLATES:
            template = collection.models.new_template(reference.slot.template_name)
            template.update(qfmt=reference.front, afmt=reference.back)
            collection.models.add_template(model, template)
        model["css"] = REFERENCE_CARD_CSS
        collection.models.add(model)
        csv_path.write_bytes(
            serialize_anki_csv(
                MODEL,
                DECK,
                CSV_EXPORT_FIELD_NAMES,
                (tuple(dict(n.to_anki_fields())[field] for field in CSV_EXPORT_FIELD_NAMES) for n in notes),
            )
        )
    finally:
        collection.close()
    if before_import:
        before_import(str(model["id"]))
    return native_import(path, csv_path)


def native_import(path: Path, csv_path: Path, *, omit_tags: bool = False) -> dict[str, Any]:
    from anki.collection import Collection
    from anki.import_export_pb2 import CsvMetadata, ImportCsvRequest

    collection = Collection(str(path))
    try:
        metadata = collection.get_csv_metadata(str(csv_path), None)
        metadata.dupe_resolution = CsvMetadata.UPDATE
        metadata.match_scope = CsvMetadata.NOTETYPE
        assert metadata.is_html
        assert "Personal Notes" not in metadata.column_labels
        assert metadata.global_notetype.field_columns[-1] == 0
        if omit_tags:
            metadata.tags_column = 0
        from google.protobuf.json_format import MessageToDict

        result = collection.import_csv(ImportCsvRequest(path=str(csv_path), metadata=metadata))
        return {"settings": MessageToDict(metadata), "result": MessageToDict(result)}
    finally:
        collection.close()


def snapshot(path: Path, notes: list[GeneratedNote], label: str, output: Path) -> DestinationSnapshot:
    from anki.collection import Collection

    # This dedicated backend owns the disposable collection; no GUI/client is running.
    collection = Collection(str(path))
    try:
        model = collection.models.by_name(MODEL)
        schema = schema_contract(str(model["id"]))
        schema["fields"] = [
            [field["name"], expected[1]] for field, expected in zip(model["flds"], schema["fields"], strict=True)
        ]
        schema["css_digest"] = hashlib.sha256(model["css"].encode()).hexdigest()
        schema["templates"] = [
            {
                "name": template["name"],
                "ordinal": template["ord"],
                "semantic_key": TEMPLATE_REGISTRY[template["ord"]].semantic_key,
                "front_digest": hashlib.sha256(template["qfmt"].encode()).hexdigest(),
                "back_digest": hashlib.sha256(template["afmt"].encode()).hexdigest(),
            }
            for template in model["tmpls"]
        ]
        config = collection.decks.get_config(1)
        assert config["new"]["bury"] and config["rev"]["bury"] and config["buryInterdayLearning"]
    finally:
        collection.close()
    rows = tables(path)
    model_id = next(row["id"] for row in rows["notetypes"] if row["name"] == MODEL)
    by_identity = {row["flds"].split("\x1f")[0]: row for row in rows["notes"] if row["mid"] == model_id}
    captured = []
    for note in notes:
        if note.latinitas_id not in by_identity:
            continue
        row = by_identity[note.latinitas_id]
        fields = dict(zip(REFERENCE_NOTE_TYPE_FIELDS, row["flds"].split("\x1f"), strict=True))
        cards = []
        for card in rows["cards"]:
            if card["nid"] != row["id"]:
                continue
            slot = TEMPLATE_REGISTRY[card["ord"]]
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
        card_ids = {int(card["id"]) for card in cards}
        captured.append(
            {
                "identity": note.latinitas_id,
                "source": ["native-scope", "synthetic-entry", note.object_key],
                "note_type_id": str(model_id),
                "guid": row["guid"],
                "local_id": str(row["id"]),
                "fields": {key: fields[key] for key in MANAGED_FIELDS},
                "tags": sorted(row["tags"].split()),
                "personal_digest": hashlib.sha256(fields["Personal Notes"].encode()).hexdigest(),
                "cards": {"complete": True, "rows": cards},
                "history": {
                    "complete": True,
                    "rows": [dict(log, id=str(log["id"])) for log in rows["revlog"] if log["cid"] in card_ids],
                },
            }
        )
    evidence = DestinationSnapshot(
        json.dumps(
            {
                "version": 1,
                "snapshot_id": label,
                "captured_at": datetime.now(UTC).isoformat(),
                "artifact_sha256": digest(path),
                "client": f"Anki native backend {version('anki')}",
                "export_method": "closed SQLite copy",
                "export_options": {"compared_tables": list(TABLES)},
                "destination": "disposable-native-original",
                "profile": "native-profile",
                "schema": schema,
                "managed_set": {
                    "scope": "native-scope",
                    "members": [[n.latinitas_id, "native-scope", "synthetic-entry", n.object_key] for n in notes],
                },
                "selection": "all",
                "exclusions": [],
                "expected_count": len(captured),
                "complete": True,
                "fresh": True,
                "level": "collection",
                "deck_options": {
                    "decoded_config": config,
                    "deck_config": rows["deck_config"],
                    "decks": rows["decks"],
                },
                "notes": captured,
            }
        )
    )
    write(output / f"{label}-tables.json", rows)
    write(output / f"{label}-snapshot.json", evidence.payload)
    return evidence


def ownership(snap: DestinationSnapshot) -> dict[str, Any]:
    return {
        n["identity"]: {
            "source_tags": ["source"] if "source" in n["tags"] else [],
            "configured_tags": [],
            "keep_tags": [t for t in n["tags"] if t != "source"],
            "keep_fields": ["Lemma"],
        }
        for n in snap.payload["notes"]
    }


def propose(snap: DestinationSnapshot, index: int = 0, **changes: str) -> dict[str, Any]:
    note = snap.payload["notes"][index]
    return {
        "identity": note["identity"],
        "fields": dict(note["fields"], **changes),
        "contributions": {"source_tags": ["source"]},
    }


def reject(action: Callable[[], Any]) -> str:
    try:
        action()
    except ReconciliationRequired as error:
        return str(error)
    raise AssertionError("unsafe operation was accepted")


def check_absent_note(absent: DestinationSnapshot, state: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    anchored = compose_plan(absent, state, [proposal], PROFILE)
    entry = anchored["notes"][0]
    assert entry["classification"] == "conflict"
    assert (
        entry["blocked"] == entry["reasons"] == ["previously anchored note absent; destination reconciliation required"]
    )
    assert entry["operations"] == entry["card_effects"] == []
    receipt = approve_plan(anchored, [], "review anchored absence; no writes authorized")
    assert receipt["targets"] == {} and receipt["selected_operations"] == receipt["import_rows"] == []
    verify_approval(anchored, receipt, absent, state)

    # A fresh baseline adopts only actual members: the missing member was never anchored.
    fresh_state = adopt(absent, ownership(absent), "review actual remaining destination members")
    create_plan = compose_plan(absent, fresh_state, [proposal], PROFILE)
    entry = create_plan["notes"][0]
    assert entry["classification"] == "create"
    assert len(entry["operations"]) == 1 and entry["operations"][0]["kind"] == "create"
    ops = [entry["operations"][0]["id"]]
    refusal = reject(lambda: approve_plan(create_plan, ops, "must refuse actual creation"))
    assert refusal == "unsupported or unresolved selected operation"
    return {
        "anchored": {"plan": anchored, "approval": receipt},
        "never_anchored": {"plan": create_plan, "selected_operations": ops, "rejection": refusal},
    }


def compare(before: dict[str, Any], after: dict[str, Any], expected: dict[str, Any]) -> list[dict[str, Any]]:
    # Check all columns, not a convenient scheduling subset; all sibling rows included.
    for table in TABLES[1:]:
        assert before[table] == after[table], table
    assert [n["id"] for n in before["notes"]] == [n["id"] for n in after["notes"]]
    deltas = []
    for old, new in zip(before["notes"], after["notes"], strict=True):
        identity = old["flds"].split("\x1f")[0]
        allowed = expected.get(identity, {})
        prior_fields = dict(zip(REFERENCE_NOTE_TYPE_FIELDS, old["flds"].split("\x1f"), strict=True))
        actual_fields = dict(zip(REFERENCE_NOTE_TYPE_FIELDS, new["flds"].split("\x1f"), strict=True))
        changed_fields = {
            field: actual_fields[field] for field in prior_fields if prior_fields[field] != actual_fields[field]
        }
        assert changed_fields == allowed.get("fields", {})
        if "tags" in allowed:
            assert sorted(new["tags"].split()) == allowed["tags"]
        for column in old:
            if old[column] == new[column]:
                continue
            if column == "flds":
                assert changed_fields == allowed.get("fields", {}), changed_fields
                deltas.append({"identity": identity, "column": column, "changes": changed_fields})
            elif column == "tags":
                assert sorted(new[column].split()) == allowed["tags"]
                deltas.append({"identity": identity, "column": column, "before": old[column], "after": new[column]})
            else:
                assert identity in expected and column in ("mod", "usn"), (identity, column)
                deltas.append({"identity": identity, "column": column, "before": old[column], "after": new[column]})
    return deltas


def main() -> None:
    from anki.collection import Collection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    root = parser.parse_args().output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    notes = [generated("sense-A"), generated("sense-B")]
    path = root / "collection.anki2"
    first_report = create(path, notes, root / "first.csv")
    # Seed deliberately unequal synthetic schedules/history offline. No scheduler emulation claim.
    with closing(sqlite3.connect(path)) as db, db:
        for index, (nid,) in enumerate(db.execute("select id from notes order by id").fetchall()):
            flds = db.execute("select flds from notes where id=?", (nid,)).fetchone()[0].split("\x1f")
            flds[REFERENCE_NOTE_TYPE_FIELDS.index("Personal Notes")] = f"Personal {index} ä"
            flds[REFERENCE_NOTE_TYPE_FIELDS.index("Lemma")] = "kept destination lemma"
            db.execute("update notes set flds=?,tags=? where id=?", ("\x1f".join(flds), " manual source ", nid))
        for index, (cid,) in enumerate(db.execute("select id from cards order by id").fetchall()):
            db.execute(
                "update cards set type=2,queue=?,due=?,ivl=?,factor=?,reps=?,lapses=?,left=?,flags=? where id=?",
                (
                    -1 if index == 1 else 2,
                    17 + 3 * index,
                    6 + index,
                    2200 + 10 * index,
                    4 + index,
                    index % 3,
                    index,
                    index % 4,
                    cid,
                ),
            )
            for review in range(1 + index % 3):
                db.execute(
                    "insert into revlog values (?,?,?,?,?,?,?,?,?)",
                    (
                        1700000000000 + index * 100 + review,
                        cid,
                        -1,
                        2 + review % 3,
                        6 + index,
                        3 + index,
                        2200 + index,
                        1100 + index * 71,
                        1,
                    ),
                )
    backup = root / "backup.anki2"
    shutil.copyfile(path, backup)
    initial = snapshot(path, notes, "before", root)
    initial_tables = tables(path)
    assert len(initial_tables["notes"]) == 2 and len(initial_tables["cards"]) == 8
    assert len(initial_tables["revlog"]) == 15
    assert sum(card["queue"] == -1 for card in initial_tables["cards"]) == 1
    for note in initial.payload["notes"]:
        assert {card["ordinal"] for card in note["cards"]["rows"]} == {0, 2, 5, 7}
    state_path = root / "state.json"
    state = adopt(initial, ownership(initial), "review synthetic ownership")
    save_state(state_path, state)
    identity = initial.payload["notes"][0]["identity"]
    reports = {
        "version": version("anki"),
        "first_import": first_report,
        "backup_sha256": digest(backup),
        "fixture": {
            "note_count": 2,
            "card_count": 8,
            "recipes": ["completion", "recognition"],
            "bury_new_review_interday": True,
            "suspended_card_index": 1,
        },
        "cases": {},
    }

    def handoff(
        snap: DestinationSnapshot,
        proposals: list[dict[str, Any]],
        operations: Callable[[dict[str, Any]], bool],
        label: str,
    ) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, Any]]:
        plan = compose_plan(snap, load_state(state_path), proposals, PROFILE)
        selected = [op["id"] for entry in plan["notes"] for op in entry["operations"] if operations(op)]
        approval = approve_plan(plan, selected, f"explicit native {label} subset review")
        csv_path = root / f"{label}.csv"
        emitted = emit_updates(
            plan, approval, snap, state_path, backup, "close and restore backup then reconcile", csv_path, MODEL, DECK
        )
        write(root / f"{label}-plan.json", plan)
        write(root / f"{label}-approval.json", approval)
        return plan, approval, csv_path, emitted

    a = propose(initial, Meaning="approved A", Lemma="unapproved generated lemma")
    a["contributions"] = {"source_tags": ["source", "added"]}
    b = propose(initial, 1, Meaning="UNAPPROVED B")
    plan, approval, csv_path, emitted = handoff(
        initial,
        [a, b],
        lambda op: op["id"].startswith(identity + "/") and (op.get("field") == "Meaning" or op["kind"] == "tags"),
        "content-tags",
    )
    assert len(approval["import_rows"]) == 1
    reports["cases"]["content-tags"] = {"emitted": emitted, "native": native_import(path, csv_path)}
    after = snapshot(path, notes, "content-after", root)
    reports["cases"]["content-tags"]["deltas"] = compare(
        tables(backup),
        tables(path),
        {identity: {"fields": {"Meaning": "approved A"}, "tags": ["added", "manual", "source"]}},
    )
    # Import succeeded; inject an actual local persistence error before replacement.
    assert load_state(state_path)["anchors"] == state["anchors"]
    pending_bytes = state_path.read_bytes()
    with patch(
        "latinitas_cards.managed_application.save_state", side_effect=OSError("injected result persistence failure")
    ):
        try:
            observe_updates(
                state_path, plan["plan_id"], after, "native succeeded, persistence fails", interval_confirmed=True
            )
        except OSError as error:
            reports["cases"]["persistence-failure"] = str(error)
        else:
            raise AssertionError("failure injection did not reach persistence")
    assert state_path.read_bytes() == pending_bytes
    after = snapshot(path, notes, "persistence-reacquired", root)
    recovered = observe_updates(
        state_path, plan["plan_id"], after, "native success before result recording", interval_confirmed=True
    )
    assert recovered["observed"] == approval["selected_operations"]
    reports["cases"]["import-before-recording"] = recovered
    before_noop = tables(path)
    reports["cases"]["noop"] = native_import(path, csv_path)
    assert before_noop == tables(path), "no-op changed full tables"
    no_op = snapshot(path, notes, "noop-after", root)
    recorded = load_state(state_path)
    observe_updates(state_path, plan["plan_id"], no_op, "native no-op reapplication", interval_confirmed=True)
    assert load_state(state_path) == recorded
    reports["cases"]["noop"]["full_tables_identical"] = True
    # Actual backup restoration invalidates the recorded-success anchors.
    shutil.copyfile(backup, path)
    restored = snapshot(path, notes, "restored", root)
    assert tables(path) == tables(backup)
    reports["cases"]["restoration"] = reject(
        lambda: observe_updates(
            state_path, plan["plan_id"], restored, "restored offline backup", interval_confirmed=True
        )
    )
    save_state(
        state_path,
        review_reconciliation(load_state(state_path), restored, ownership(restored), "review backup restore"),
    )
    retry_plan, retry_approval, retry_csv, _ = handoff(
        restored, [a, b], lambda op: op["id"] in approval["selected_operations"], "restored-retry"
    )
    native_import(path, retry_csv)
    retried = snapshot(path, notes, "retried", root)
    report = observe_updates(state_path, retry_plan["plan_id"], retried, "native retry", interval_confirmed=True)
    assert report["observed"] == retry_approval["selected_operations"]
    compare(
        tables(backup),
        tables(path),
        {identity: {"fields": {"Meaning": "approved A"}, "tags": ["added", "manual", "source"]}},
    )
    reports["cases"]["restored-retry"] = report

    # An emitted two-note plan imported only one row, then explicitly reconciled/reapproved.
    shutil.copyfile(backup, path)
    save_state(state_path, state)
    p, receipt, csv_file, _ = handoff(
        initial,
        [propose(initial, Meaning="partial A"), propose(initial, 1, Meaning="partial B")],
        lambda op: op.get("field") == "Meaning",
        "partial",
    )
    lines = csv_file.read_text().splitlines()
    partial_csv = root / "partial-native.csv"
    partial_csv.write_text(
        "\n".join(
            [line for line in lines if line.startswith("#")]
            + [next(line for line in lines if line.startswith(identity))]
        )
        + "\n"
    )
    native = native_import(path, partial_csv)
    partial = snapshot(path, notes, "partial-after", root)
    partial_report = observe_updates(state_path, p["plan_id"], partial, "one row imported", interval_confirmed=True)
    assert partial_report["observed"] == [f"{identity}/field/Meaning"]
    assert len(partial_report["unresolved"]) == 1
    other_identity = initial.payload["notes"][1]["identity"]
    assert load_state(state_path)["anchors"][other_identity] == state["anchors"][other_identity]
    deltas = compare(tables(backup), tables(path), {identity: {"fields": {"Meaning": "partial A"}}})
    reports["cases"]["partial"] = {"native": native, "observation": partial_report, "deltas": deltas}
    reconciled = review_reconciliation(
        load_state(state_path), partial, ownership(partial), "review partial destination"
    )
    save_state(state_path, reconciled)
    p, receipt, csv_file, _ = handoff(
        partial,
        [propose(partial, Meaning="partial A"), propose(partial, 1, Meaning="partial B")],
        lambda op: op.get("field") == "Meaning",
        "partial-retry",
    )
    assert len(receipt["import_rows"]) == 1 and receipt["selected_operations"] == [f"{other_identity}/field/Meaning"]
    native = native_import(path, csv_file)
    final = snapshot(path, notes, "partial-retry-after", root)
    result = observe_updates(state_path, p["plan_id"], final, "remaining row imported", interval_confirmed=True)
    assert result["observed"] == receipt["selected_operations"]
    reports["cases"]["partial-retry"] = {
        "native": native,
        "observation": result,
        "deltas": compare(
            tables(backup),
            tables(path),
            {identity: {"fields": {"Meaning": "partial A"}}, other_identity: {"fields": {"Meaning": "partial B"}}},
        ),
    }

    # Real mixed outcome: wrong native mapping applies content but leaves the selected tag effect unresolved.
    shutil.copyfile(backup, path)
    save_state(state_path, state)
    proposal = propose(initial, Meaning="mixed field")
    proposal["contributions"] = {"source_tags": ["source", "mixed-tag"]}
    p, receipt, csv_file, _ = handoff(
        initial, [proposal], lambda op: op.get("field") == "Meaning" or op["kind"] == "tags", "mixed"
    )
    native = native_import(path, csv_file, omit_tags=True)
    mixed = snapshot(path, notes, "mixed-after", root)
    observed = observe_updates(
        state_path, p["plan_id"], mixed, "incorrect Tags mapping; partial note", interval_confirmed=True
    )
    assert observed["observed"] == [] and observed["unresolved"] == receipt["selected_operations"]
    assert load_state(state_path)["anchors"] == state["anchors"]
    reports["cases"]["mixed"] = {
        "native": native,
        "observation": observed,
        "deltas": compare(tables(backup), tables(path), {identity: {"fields": {"Meaning": "mixed field"}}}),
    }

    # A genuine intervening destination edit, not a fabricated stale sidecar.
    shutil.copyfile(backup, path)
    save_state(state_path, state)
    p, receipt, _, _ = handoff(
        initial, [propose(initial, Meaning="handoff target")], lambda op: op.get("field") == "Meaning", "handoff"
    )
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(
            "update notes set tags=' later manual source ' where id=?", (int(initial.payload["notes"][0]["local_id"]),)
        )
    changed = snapshot(path, notes, "handoff-changed", root)
    reports["cases"]["changed-handoff"] = reject(lambda: verify_approval(p, receipt, changed, state))
    assert "stale" in reports["cases"]["changed-handoff"]

    # Native rows establish the actual card set used by the refusal boundary.
    shutil.copyfile(backup, path)
    actual_before = tables(path)
    reports["cases"]["unsupported"] = {}
    for label in ("new-slot", "guard-clear", "front-clear", "slot-content", "retire", "consolidate"):
        proposal = propose(initial, Meaning="content subset")
        fields = proposal["fields"]
        if label == "new-slot":
            slot = TEMPLATE_REGISTRY[1]
            fields.update({slot.enabled_field: "1", slot.prompt_field: "new prompt", slot.answer_field: "new answer"})
        elif label == "guard-clear":
            fields[TEMPLATE_REGISTRY[0].enabled_field] = ""
        elif label == "front-clear":
            fields[TEMPLATE_REGISTRY[0].prompt_field] = ""
        elif label == "slot-content":
            fields[TEMPLATE_REGISTRY[0].answer_field] = "changed slot answer"
        elif label == "retire":
            proposal["retire"] = True
        else:
            fields["Source ID"] = "consolidated-source"
        unsafe = compose_plan(initial, state, [proposal], PROFILE)
        ops = [op["id"] for entry in unsafe["notes"] for op in entry["operations"] if not op["supported"]]
        if not ops:
            assert label == "front-clear"
            ops = [f"{identity}/field/{TEMPLATE_REGISTRY[0].prompt_field}"]
        reports["cases"]["unsupported"][label] = reject(
            lambda unsafe=unsafe, ops=ops: approve_plan(unsafe, ops, "must refuse")
        )
        # A sibling content subset is either safely retained or refused; never import structural effects.
        content_ops = [
            op["id"] for entry in unsafe["notes"] for op in entry["operations"] if op.get("field") == "Meaning"
        ]
        if content_ops and label != "retire":
            try:
                retained = approve_plan(unsafe, content_ops, "content only, no structural permission")
            except ReconciliationRequired as error:
                reports["cases"]["unsupported"][label + "-content-subset"] = str(error)
            else:
                target = retained["targets"][identity]["fields"]
                for slot in TEMPLATE_REGISTRY:
                    for field in (slot.enabled_field, slot.prompt_field, slot.answer_field):
                        assert target[field] == initial.payload["notes"][0]["fields"][field]
                save_state(state_path, state)
                csv_file = root / f"retained-{label}.csv"
                emit_updates(
                    unsafe, retained, initial, state_path, backup, "restore then reconcile", csv_file, MODEL, DECK
                )
                native = native_import(path, csv_file)
                reports["cases"]["unsupported"][label + "-content-native"] = {
                    "native": native,
                    "deltas": compare(
                        actual_before, tables(path), {identity: {"fields": {"Meaning": "content subset"}}}
                    ),
                }
                shutil.copyfile(backup, path)
        assert actual_before == tables(path)
    # Enabled slot, actual destination card missing: approval must not repair it by import.
    missing_card = initial.payload["notes"][0]["cards"]["rows"][0]["id"]
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("delete from cards where id=?", (int(missing_card),))
    missing = snapshot(path, notes, "missing-card", root)
    missing_tables = tables(path)
    expected_card_ids = {str(card["id"]) for card in initial_tables["cards"]} - {missing_card}
    assert {str(card["id"]) for card in missing_tables["cards"]} == expected_card_ids, (
        "unexpected missing-card inventory"
    )
    unsafe = compose_plan(missing, state, [propose(missing, Meaning="must not regenerate missing card")], PROFILE)
    content_ops = [f"{identity}/field/Meaning"]
    reports["cases"]["unsupported"]["missing-card-drift"] = reject(
        lambda: approve_plan(unsafe, content_ops, "must refuse")
    )
    missing_state = adopt(missing, ownership(missing), "explicit reviewed missing-card destination")
    unsafe = compose_plan(
        missing, missing_state, [propose(missing, Meaning="must not regenerate missing card")], PROFILE
    )
    receipt = approve_plan(unsafe, content_ops, "content subset does not approve structural repair")
    save_state(state_path, missing_state)
    journal_bytes = state_path.read_bytes()
    reports["cases"]["unsupported"]["enabled-missing-card"] = reject(
        lambda: emit_updates(
            unsafe, receipt, missing, state_path, backup, "restore", root / "must-not-exist.csv", MODEL, DECK
        )
    )
    assert state_path.read_bytes() == journal_bytes and not (root / "must-not-exist.csv").exists()
    assert tables(path) == missing_tables, "rejected emission changed destination tables"
    snapshot(path, notes, "missing-card-rejection-after", root)
    reports["cases"]["missing-card-exact-set"] = {
        "expected_ids": sorted(expected_card_ids),
        "actual_ids": sorted(str(card["id"]) for card in tables(path)["cards"]),
        "full_tables_unchanged": True,
    }

    # Structural reactivation stays unsupported even when the receipt would retain a user suspension.
    shutil.copyfile(backup, path)
    data = initial.payload
    suspended_note = next(note for note in data["notes"] if any(card["suspended"] for card in note["cards"]["rows"]))
    suspended = next(card for card in suspended_note["cards"]["rows"] if card["suspended"])
    # Explicit synthetic historical provenance, not a claim that CSV performed retirement.
    suspended["retirement"] = {
        "approval": "synthetic prior retirement review",
        "pre_suspended": True,
        "tool_suspended": False,
        "observed": initial.reference,
    }
    retired = DestinationSnapshot(json.dumps(data))
    retired_state = adopt(retired, ownership(retired), "review synthetic prior lifecycle provenance")
    index = next(i for i, note in enumerate(retired.payload["notes"]) if note["identity"] == suspended_note["identity"])
    unsafe = compose_plan(retired, retired_state, [propose(retired, index)], PROFILE)
    effects = [
        effect for entry in unsafe["notes"] for effect in entry["card_effects"] if effect["effect"] == "reactivate"
    ]
    assert len(effects) == 1 and not effects[0]["unsuspend"]
    ops = [op["id"] for entry in unsafe["notes"] for op in entry["operations"] if op["kind"] == "reactivate"]
    reports["cases"]["unsupported"]["reactivate"] = reject(lambda: approve_plan(unsafe, ops, "must refuse"))
    assert tables(path) == tables(backup)

    # Native inventory proves an absent note cannot turn a managed update into create.
    with closing(sqlite3.connect(path)) as db, db:
        nid = int(initial.payload["notes"][0]["local_id"])
        db.execute("delete from revlog where cid in (select id from cards where nid=?)", (nid,))
        db.execute("delete from cards where nid=?", (nid,))
        db.execute("delete from notes where id=?", (nid,))
    absent = snapshot(path, notes, "absent-note", root)
    absent_tables = tables(path)
    reports["cases"]["absence"] = check_absent_note(absent, state, propose(initial, Meaning="cannot create"))
    assert tables(path) == absent_tables, "absence planning/approval changed destination tables"
    assert len(absent_tables["notes"]) == 1 and len(absent_tables["cards"]) == 4
    reports["cases"]["absence"]["full_tables_unchanged"] = True

    # CSS drift is acquired from the real native model, never asserted as a matching contract.
    shutil.copyfile(backup, path)
    collection = Collection(str(path))
    try:
        model = collection.models.by_name(MODEL)
        model["css"] += "\n/* independently changed styling */"
        collection.models.update(model)
    finally:
        collection.close()
    reports["cases"]["unsupported"]["css-mismatch"] = reject(lambda: snapshot(path, notes, "css-drift", root))

    # Fresh start is a different destination with new scheduling, never structural update permission.
    shutil.copyfile(backup, path)
    original_hash = digest(path)
    original_tables = tables(path)
    fresh_path = root / "fresh.anki2"

    def approve_fresh(model_id: str) -> None:
        original = BoundDestination(
            "disposable-native-original", "native-profile", initial.payload["schema"], initial.payload["managed_set"]
        )
        destination = BoundDestination(
            "disposable-native-fresh", "native-profile", schema_contract(model_id), initial.payload["managed_set"]
        )
        inventory = [
            DestinationNoteEvidence(
                n["identity"],
                MODEL,
                n["fields"]["Note Schema"],
                personal_notes="synthetic personal content",
                tags=tuple(n["tags"]),
                has_review_history=bool(n["history"]["rows"]),
                card_count=len(n["cards"]["rows"]),
            )
            for n in initial.payload["notes"]
        ]
        profile = DeckProfile.default(
            note_type="Synthetic Source",
            lexical_entry_field="Lemma",
            principal_parts_field="Forms",
            generated_note_type=MODEL,
        )
        plan = plan_destination_fresh_start(
            profile,
            inventory,
            original=original,
            destination=destination,
            selection="fresh_start",
            fresh_start=FreshStartApproval(backup, MODEL, True, True, model_id),
        )
        assert "no inherited scheduling" in plan.disclosure and plan.old_collections_retained
        write(root / "fresh-start-plan.json", asdict(plan))

    fresh_report = create(fresh_path, notes, root / "fresh.csv", before_import=approve_fresh)
    fresh_tables = tables(fresh_path)
    assert len(fresh_tables["notes"]) == 2 and len(fresh_tables["cards"]) == 8
    assert fresh_tables["revlog"] == []
    assert all(card["type"] == 0 and card["queue"] == 0 and card["reps"] == 0 for card in fresh_tables["cards"])
    assert original_hash == digest(path) and original_tables == tables(path)
    write(root / "fresh-tables.json", fresh_tables)
    reports["cases"]["fresh-start"] = {
        "native": fresh_report,
        "original_hash_unchanged": original_hash,
        "new_scheduling": "8 new cards; no inherited reviews, suspension or Personal Notes; separate destination",
    }

    # Independent tag-only branches: every managed field is exactly unchanged.
    for label, existing, proposed in (
        ("tag-add", ["manual", "source"], ["source", "new"]),
        ("tag-remove", ["manual", "source"], []),
        ("tag-final-remove", ["source"], []),
    ):
        shutil.copyfile(backup, path)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute(
                "update notes set tags=? where id=?",
                (" " + " ".join(existing) + " ", int(initial.payload["notes"][0]["local_id"])),
            )
        snap = snapshot(path, notes, label + "-before", root)
        save_state(state_path, adopt(snap, ownership(snap), "independent branch ownership"))
        proposal = propose(snap)
        proposal["contributions"] = {"source_tags": proposed}
        p, receipt, csv_file, emission = handoff(snap, [proposal], lambda op: op["kind"] == "tags", label)
        before = tables(path)
        native = native_import(path, csv_file)
        result = snapshot(path, notes, label + "-after", root)
        observed = observe_updates(state_path, p["plan_id"], result, "native tag-only probe", interval_confirmed=True)
        actual = result.payload["notes"][0]["tags"]
        target = receipt["targets"][identity]["tags"]
        assert target == sorted(set(existing) - {"source"} | set(proposed))
        deltas = compare(before, tables(path), {identity: {"tags": target}} if actual == target else {})
        assert before["notes"][0]["flds"] == tables(path)["notes"][0]["flds"]
        if actual != target:
            assert observed["observed"] == [] and observed["unresolved"] == receipt["selected_operations"]
            assert load_state(state_path)["anchors"][identity]["tags"] == existing
        reports["cases"][label] = {
            "emission": emission,
            "native": native,
            "expected_tags": target,
            "actual_tags": actual,
            "deltas": deltas,
            "observation": observed,
            "capability": "observed" if actual == target else "unsupported; unresolved",
        }
    reports["environment"] = {
        "python": sys.version,
        "os": platform.platform(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "native_api": "Collection.get_csv_metadata + Collection.import_csv; UPDATE/NOTETYPE",
        "script_sha256": digest(Path(__file__)),
    }
    package = Path(__file__).resolve().parents[1] / "src" / "latinitas_cards"
    reports["source_sha256"] = {p.name: digest(p) for p in sorted(package.glob("*.py"))}
    reports["evidence_sha256"] = {p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()}
    write(root / "report.json", reports)
    print(json.dumps(reports, ensure_ascii=False, indent=2))
    print("Native managed backend scenarios passed; client/version/fixture limits apply.")


if __name__ == "__main__":
    main()
