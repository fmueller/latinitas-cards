"""Offline handoff tests are not native Anki preservation evidence."""

import csv
import io
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from card_lifecycle_test import capture, fixture, generated
from managed_plans_test import request
from typer.testing import CliRunner

from latinitas_cards.cli import app
from latinitas_cards.destination_state import ReconciliationRequired, load_state, review_reconciliation, save_state
from latinitas_cards.managed_application import emit_updates, observe_updates
from latinitas_cards.managed_plans import approve_plan, compose_plan


def setup(tmp_path: Path) -> tuple[Any, ...]:
    snapshot, state = fixture(generated(), generated(sense="other"))
    data = snapshot.payload
    data["notes"][0]["fields"]["Lemma"] = "kept destination B"
    data["notes"][0]["tags"] = ["manual"]
    snapshot = capture(data)
    proposal = request(snapshot, meaning="selected A")
    other = {"identity": data["notes"][1]["identity"], "fields": dict(data["notes"][1]["fields"], Meaning="unapproved")}
    plan = compose_plan(
        snapshot, state, [proposal, other], {"generated_note_type": "Latinitas", "target_deck": "Latin"}
    )
    operation = next(op["id"] for op in plan["notes"][0]["operations"] if op.get("field") == "Meaning")
    approval = approve_plan(plan, [operation], "apply selected A only")
    path = tmp_path / "state.json"
    save_state(path, state)
    backup = tmp_path / "backup.colpkg"
    backup.write_bytes(b"recoverable synthetic backup")
    return snapshot, plan, approval, path, backup


def test_emitted_mapping_and_observed_only_advancement(tmp_path: Path) -> None:
    snapshot, plan, approval, path, backup = setup(tmp_path)
    before = load_state(path)
    output = tmp_path / "updates.csv"
    report = emit_updates(plan, approval, snapshot, path, backup, "restore closed backup", output, "Latinitas", "Latin")
    assert report["emitted"] == approval["selected_operations"]
    assert report["observed"] == []
    assert load_state(path)["anchors"] == before["anchors"]
    text = output.read_text()
    columns = next(
        line.removeprefix("#columns:").split(",") for line in text.splitlines() if line.startswith("#columns:")
    )
    rows = list(csv.reader(io.StringIO("\n".join(line for line in text.splitlines() if not line.startswith("#")))))
    assert len(rows) == 1
    row = dict(zip(columns, rows[0], strict=True))
    assert row["Meaning"] == "selected A"
    assert row["Lemma"] == "kept destination B"
    assert row["Tags"] == "manual"
    assert "Personal Notes" not in columns
    assert f"#tags column:{columns.index('Tags') + 1}" in text
    assert "#notetype:Latinitas" in text
    assert "#deck:Latin" in text
    unresolved = observe_updates(path, plan["plan_id"], snapshot, "no import", interval_confirmed=True)
    assert unresolved["unresolved"] == approval["selected_operations"]
    assert load_state(path)["anchors"] == before["anchors"]
    after = snapshot.payload
    after["notes"][0]["fields"]["Meaning"] = "selected A"
    observed = capture(after)
    report = observe_updates(path, plan["plan_id"], observed, "imported", interval_confirmed=True)
    assert report["observed"] == approval["selected_operations"]
    saved = load_state(path)
    assert saved["anchors"][after["notes"][0]["identity"]]["fields"]["Meaning"] == "selected A"
    observe_updates(path, plan["plan_id"], observed, "same observation", interval_confirmed=True)
    assert load_state(path) == saved
    with pytest.raises(ReconciliationRequired, match="inconsistent"):
        observe_updates(path, plan["plan_id"], snapshot, "restored backup", interval_confirmed=True)
    ownership = {
        note["identity"]: {
            "source_tags": [],
            "configured_tags": [],
            "keep_fields": [],
            "keep_tags": note["tags"],
        }
        for note in snapshot.payload["notes"]
    }
    recovered = review_reconciliation(saved, snapshot, ownership, "explicitly review restored destination")
    historical = recovered["plans"][plan["plan_id"]]
    assert historical["status"] == "abandoned"
    assert historical["operations"][after["notes"][0]["identity"]]["receipt"] is not None
    save_state(path, recovered)
    retry = compose_plan(snapshot, recovered, [request(snapshot, meaning="selected A")], plan["effective_profile"])
    renewed = approve_plan(retry, approval["selected_operations"], "renew after restored-state review")
    assert retry["plan_id"] != plan["plan_id"]
    emit_updates(retry, renewed, snapshot, path, backup, "restore", tmp_path / "renewed.csv", "Latinitas", "Latin")
    report = observe_updates(path, retry["plan_id"], observed, "reimported after recovery", interval_confirmed=True)
    assert report["observed"] == renewed["selected_operations"]


def test_stale_missing_backup_and_failed_emission(tmp_path: Path) -> None:
    snapshot, plan, approval, path, backup = setup(tmp_path)
    saved = load_state(path)
    output = tmp_path / "updates.csv"
    changed = snapshot.payload
    changed["notes"][0]["tags"].append("later edit")
    with pytest.raises(ReconciliationRequired, match="stale"):
        emit_updates(plan, approval, capture(changed), path, backup, "restore", output, "Latinitas", "Latin")
    assert load_state(path) == saved
    with pytest.raises(OSError):
        emit_updates(plan, approval, snapshot, path, tmp_path / "absent", "restore", output, "Latinitas", "Latin")
    assert load_state(path) == saved
    output.mkdir()
    report = emit_updates(plan, approval, snapshot, path, backup, "restore", output, "Latinitas", "Latin")
    assert report["failed"] == approval["selected_operations"]
    assert load_state(path)["anchors"] == saved["anchors"]
    with pytest.raises(ReconciliationRequired, match="stale|pending"):
        emit_updates(plan, approval, snapshot, path, backup, "restore", tmp_path / "retry.csv", "Latinitas", "Latin")


def test_reconciled_decision_provenance_survives_observed_application(tmp_path: Path) -> None:
    snapshot, _, _, path, backup = setup(tmp_path)
    data = snapshot.payload
    data["notes"][0]["fields"]["Meaning"] = "destination conflict"
    snapshot = capture(data)
    proposal = request(snapshot, meaning="reviewed replacement")
    proposal["decisions"] = {"field:Meaning": {"action": "accept_proposal", "approval": "explicit conflict review"}}
    plan = compose_plan(
        snapshot, load_state(path), [proposal], {"generated_note_type": "Latinitas", "target_deck": "Latin"}
    )
    identity = proposal["identity"]
    approval = approve_plan(plan, [f"{identity}/field/Meaning"], "apply selected resolved content")
    emit_updates(plan, approval, snapshot, path, backup, "restore", tmp_path / "out.csv", "Latinitas", "Latin")
    after = snapshot.payload
    after["notes"][0]["fields"]["Meaning"] = "reviewed replacement"
    observe_updates(path, plan["plan_id"], capture(after), "imported", interval_confirmed=True)
    assert load_state(path)["anchors"][identity]["decisions"]["field:Meaning"]["approval"] == "explicit conflict review"


@pytest.mark.parametrize("interval", [False, True])
def test_partial_and_interrupted_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interval: bool) -> None:
    snapshot, plan, approval, path, backup = setup(tmp_path)
    emit_updates(plan, approval, snapshot, path, backup, "restore", tmp_path / "updates.csv", "Latinitas", "Latin")
    pending = load_state(path)
    after = snapshot.payload
    after["notes"][0]["fields"]["Meaning"] = "selected A"
    observed = capture(after)
    import latinitas_cards.managed_application as module

    monkeypatch.setattr(module, "save_state", lambda *_: (_ for _ in ()).throw(OSError("interruption")))
    with pytest.raises(OSError):
        observe_updates(path, plan["plan_id"], observed, "imported", interval_confirmed=interval)
    assert load_state(path) == pending
    monkeypatch.setattr(module, "save_state", save_state)
    report = observe_updates(path, plan["plan_id"], observed, "reacquired", interval_confirmed=interval)
    assert report["observed" if interval else "unresolved"] == approval["selected_operations"]
    wrong = deepcopy(after)
    wrong["destination"] = "wrong"
    with pytest.raises(ReconciliationRequired, match="wrong"):
        observe_updates(path, plan["plan_id"], capture(wrong), "wrong destination", interval_confirmed=True)


def test_import_mapping_is_approved_and_bound(tmp_path: Path) -> None:
    snapshot, plan, approval, path, backup = setup(tmp_path)
    with pytest.raises(ReconciliationRequired, match="mapping"):
        emit_updates(plan, approval, snapshot, path, backup, "restore", tmp_path / "out.csv", "Other model", "Latin")


def test_cli_emit_observe_and_reconcile(tmp_path: Path) -> None:
    snapshot, plan, approval, path, backup = setup(tmp_path)
    handoff = tmp_path / "handoff.json"
    data = {
        "binding": load_state(path)["binding"],
        "snapshot": snapshot.payload,
        "plan": plan,
        "approval": approval,
        "backup": str(backup),
        "recovery": "restore closed backup including scheduling",
        "note_type": "Latinitas",
        "deck": "Latin",
    }
    handoff.write_text(json.dumps(data))
    runner = CliRunner()
    result = runner.invoke(
        app, ["managed", "emit", str(handoff), "--state", str(path), "--output", str(tmp_path / "out.csv")]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["pending"] == approval["selected_operations"]
    observation = tmp_path / "observation.json"
    data = {"binding": data["binding"], "snapshot": None, "plan_id": plan["plan_id"], "report": "not imported"}
    observation.write_text(json.dumps(data))
    result = runner.invoke(app, ["managed", "observe", str(observation), "--state", str(path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["unresolved"] == approval["selected_operations"]
    data.update(
        snapshot=snapshot.payload,
        ownership={
            identity: {
                key: anchor.get(key, [])
                for key in ("source_tags", "configured_tags", "lifecycle_tags", "keep_tags", "keep_fields")
            }
            for identity, anchor in load_state(path)["anchors"].items()
        },
        review="accept current destination",
    )
    identity = snapshot.payload["notes"][0]["identity"]
    data["ownership"][identity]["keep_tags"] = ["manual"]
    observation.write_text(json.dumps(data))
    result = runner.invoke(app, ["managed", "reconcile", str(observation), "--state", str(path)])
    assert result.exit_code == 0, result.output
    assert load_state(path)["plans"][plan["plan_id"]]["status"] == "abandoned"


def test_missing_output_directory_reports_failed(tmp_path: Path) -> None:
    snapshot, plan, approval, path, backup = setup(tmp_path)
    report = emit_updates(
        plan, approval, snapshot, path, backup, "restore", tmp_path / "missing" / "out.csv", "Latinitas", "Latin"
    )
    assert report["failed"] == approval["selected_operations"]


def test_mixed_field_tags_partial_notes_and_tag_only_skip(tmp_path: Path) -> None:
    snapshot, state = fixture(generated(), generated(sense="other"))
    proposals = []
    for index, note in enumerate(snapshot.payload["notes"]):
        proposals.append(
            {
                "identity": note["identity"],
                "fields": dict(note["fields"], Meaning=f"meaning-{index}"),
                "contributions": {"configured_tags": [f"configured-{index}"]},
            }
        )
    plan = compose_plan(snapshot, state, proposals, {"generated_note_type": "Latinitas", "target_deck": "Latin"})
    selected = [operation["id"] for note in plan["notes"] for operation in note["operations"]]
    approval = approve_plan(plan, selected, "approve both notes content/tags")
    path = tmp_path / "state.json"
    save_state(path, state)
    backup = tmp_path / "backup.colpkg"
    backup.write_bytes(b"synthetic")
    emit_updates(plan, approval, snapshot, path, backup, "restore", tmp_path / "out.csv", "Latinitas", "Latin")
    after = snapshot.payload
    for index, note in enumerate(after["notes"]):
        note["fields"]["Meaning"] = f"meaning-{index}"
    after["notes"][0]["tags"] = ["configured-0"]
    report = observe_updates(path, plan["plan_id"], capture(after), "partial native result", interval_confirmed=True)
    first, second = [note["identity"] for note in after["notes"]]
    assert set(report["observed"]) == {f"{first}/field/Meaning", f"{first}/tags"}
    assert set(report["unresolved"]) == {f"{second}/field/Meaning", f"{second}/tags"}
    assert load_state(path)["anchors"][second] == state["anchors"][second]
    assert load_state(path)["anchors"][first]["configured_tags"] == ["configured-0"]
    ownership = {
        note["identity"]: {
            "source_tags": [],
            "configured_tags": note["tags"],
            "keep_fields": [],
            "keep_tags": [],
        }
        for note in after["notes"]
    }
    observed = capture(after)
    reconciled = review_reconciliation(load_state(path), observed, ownership, "review mixed results before retry")
    save_state(path, reconciled)
    retry = compose_plan(observed, reconciled, proposals, plan["effective_profile"])
    pending = [operation["id"] for note in retry["notes"] for operation in note["operations"]]
    assert pending == [f"{second}/tags"]
    renewed = approve_plan(retry, pending, "approve only remaining tag effect")
    assert len(renewed["import_rows"]) == 1
    assert renewed["targets"][second]["fields"] == after["notes"][1]["fields"]
    emit_updates(retry, renewed, observed, path, backup, "restore", tmp_path / "retry.csv", "Latinitas", "Latin")
    after["notes"][1]["tags"] = ["configured-1"]
    report = observe_updates(path, retry["plan_id"], capture(after), "remaining tags imported", interval_confirmed=True)
    assert report["observed"] == pending
    # Tag-only skip never manufactures a field edit to force Anki Update.
    fresh, baseline = fixture(generated())
    note = fresh.payload["notes"][0]
    tag_plan = compose_plan(
        fresh,
        baseline,
        [{"identity": note["identity"], "fields": note["fields"], "contributions": {"source_tags": ["source"]}}],
        {"generated_note_type": "Latinitas", "target_deck": "Latin"},
    )
    tag_approval = approve_plan(tag_plan, [f"{note['identity']}/tags"], "approve tags only")
    save_state(path, baseline)
    emit_updates(tag_plan, tag_approval, fresh, path, backup, "restore", tmp_path / "tags.csv", "Latinitas", "Latin")
    assert tag_approval["targets"][note["identity"]]["fields"] == note["fields"]
    report = observe_updates(path, tag_plan["plan_id"], fresh, "GUI skipped unchanged fields", interval_confirmed=True)
    assert report["unresolved"] == tag_approval["selected_operations"]
    assert load_state(path)["anchors"] == baseline["anchors"]


def test_intervening_edit_and_unsafe_slot_subset(tmp_path: Path) -> None:
    snapshot, plan, approval, path, backup = setup(tmp_path)
    emit_updates(plan, approval, snapshot, path, backup, "restore", tmp_path / "out.csv", "Latinitas", "Latin")
    after = snapshot.payload
    after["notes"][0]["fields"]["Meaning"] = "selected A"
    after["notes"][0]["tags"].append("intervening manual tag")
    report = observe_updates(path, plan["plan_id"], capture(after), "edited during handoff", interval_confirmed=False)
    assert report["unresolved"] == approval["selected_operations"]
    assert load_state(path)["anchors"][after["notes"][0]["identity"]]["fields"]["Meaning"] == "old"
    original, baseline = fixture(generated())
    proposal = request(original)
    guard = next(name for name in proposal["fields"] if name.endswith("Enabled") and proposal["fields"][name])
    proposal["fields"][guard] = ""
    unsafe = compose_plan(original, baseline, [proposal], {"generated_note_type": "Latinitas", "target_deck": "Latin"})
    slot = next(operation["id"] for operation in unsafe["notes"][0]["operations"] if operation.get("field") == guard)
    with pytest.raises(ReconciliationRequired, match="unsupported"):
        approve_plan(unsafe, [slot], "attempt unsupported structural change")


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_backup_cannot_be_baseline(tmp_path: Path, alias: str) -> None:
    snapshot, plan, approval, path, _ = setup(tmp_path)
    backup = path if alias == "same" else tmp_path / "alias.colpkg"
    if alias == "symlink":
        backup.symlink_to(path)
    elif alias == "hardlink":
        backup.hardlink_to(path)
    saved = path.read_bytes()
    with pytest.raises(ReconciliationRequired, match="backup.*baseline"):
        emit_updates(plan, approval, snapshot, path, backup, "restore", tmp_path / "out.csv", "Latinitas", "Latin")
    assert path.read_bytes() == saved
    assert not (tmp_path / "out.csv").exists()


def test_published_csv_then_failed_journal_save_is_unknown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot, plan, approval, path, backup = setup(tmp_path)
    import latinitas_cards.managed_application as module

    calls = 0

    def fail_second_save(destination: Path, state: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("post-publication journal failure")
        save_state(destination, state)

    monkeypatch.setattr(module, "save_state", fail_second_save)
    output = tmp_path / "out.csv"
    report = emit_updates(plan, approval, snapshot, path, backup, "restore", output, "Latinitas", "Latin")
    assert output.exists()
    assert report["emission"]["status"] == "unknown"
    assert report["failed"] == []
    assert report["observed"] == []
    assert "inspect" in report["emission"]["error"]
    assert load_state(path)["plans"][plan["plan_id"]]["emission"]["status"] == "pending"
    with pytest.raises(ReconciliationRequired, match="stale|pending"):
        emit_updates(plan, approval, snapshot, path, backup, "restore", tmp_path / "retry.csv", "Latinitas", "Latin")
